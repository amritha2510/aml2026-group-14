from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT         = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH  = REPO_ROOT / "config.yaml"
MODEL_SCRIPT      = REPO_ROOT / "models" / "DualBranchConvVit" / "DualBranchConvViT.py"
OUTPUT_DIR        = REPO_ROOT / "outputs" / "dual_conv_vit"
SWEEP_DIR         = OUTPUT_DIR / "fusion_sweep"
FINAL_CSV         = SWEEP_DIR / "phase3_final_results.csv"
RANKED_CSV        = SWEEP_DIR / "phase3_final_results_ranked.csv"

LABEL_TO_ID   = {"normal": 0, "bacterial": 1, "viral": 2}
SEARCH_EPOCHS = 8          # enough for the heads (esp. the attention block) to actually train
FINAL_EPOCHS  = 20         # full epochs for Phase 3
SEARCH_TOP_K  = 3          # how many ranked configs to keep / display per search
FINAL_TOP_K   = 1          # configs per fusion carried into Phase 3 full training
FUSION_TYPES  = ["concat", "attention"]
BATCH_SIZE    = 64

# Pipeline (best-vs-best):
#   Phase 1 — hyperparameter search on CONCAT     -> best concat config(s)
#   Phase 2 — hyperparameter search on ATTENTION  -> best attention config(s)
#   Phase 3 — full-length training of each fusion's own best config(s)
#   -> comparison table + saved best models
# Each fusion is tuned to its own optimum, so Phase 3 compares each at its best.

SEARCH_GRID = {
    "learning_rate": [1e-4, 5e-5, 1e-5],
    # Shifted up to include 1e-2 — the weight decay the standalone ViT baseline
    # used. The old top of 1e-3 never tested that regularization regime.
    "weight_decay":  [1e-4, 1e-3, 1e-2],
}

# The ViT-branch dropout sits on the tokens that become the QUERIES in the
# attention fusion, so heavy dropout (0.5) is far more damaging there than in
# the concat path. Search a gentler range for attention. Both lists have the
# same length, so each fusion still gets a 3×3×3 = 27-config grid.
DROPOUT_GRID = {
    "concat":    [0.2, 0.3, 0.4],
    "attention": [0.1, 0.2, 0.3],
}


def search_csv_path(fusion_type: str) -> Path:
    return SWEEP_DIR / f"search_results_{fusion_type}.csv"


def build_run_config(
    base_config: dict,
    fusion_type: str,
    hyperparams: dict,
    epochs:      int,
    run_index:   int,
) -> Path:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    run_config = dict(base_config)
    dl_cfg     = dict(run_config.get("dual_conv_vit", {}))
    dl_cfg.pop("fusion_types", None)

    dl_cfg.update({
        "fusion_type":         fusion_type,
        "learning_rate":       hyperparams["learning_rate"],
        "noise_dropout_rates": hyperparams["noise_dropout_rates"],
        "weight_decay":        hyperparams["weight_decay"],
        "batch_size":          BATCH_SIZE,
        "epochs":              epochs,
    })

    run_config["dual_conv_vit"] = dl_cfg

    tmp_path = REPO_ROOT / f".sweep_run_{run_index:04d}.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    return tmp_path


def run_subprocess(config_path: Path, save_weights: bool = False) -> tuple[int, dict, Path | None]:
    env = os.environ.copy()
    env["DUAL_BRANCH_CONFIG_PATH"]  = str(config_path)
    env["DUAL_BRANCH_SAVE_WEIGHTS"] = "1" if save_weights else "0"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(MODEL_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        print(f"[WARNING] Subprocess stderr:\n{proc.stderr[-2000:]}")
        return proc.returncode, {}, None

    experiments_dir = OUTPUT_DIR / "experiments"
    metrics = {}
    run_dir = None
    if experiments_dir.exists():
        run_dirs = sorted(
            [d for d in experiments_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            run_dir      = run_dirs[0]
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    raw = json.load(f)
                for split in ("val", "test"):
                    if split in raw:
                        split_data = raw[split]
                        if split == "val":
                            # avg_macro_recall is more robust than single-epoch peak on a small val set
                            metrics[f"{split}_macro_recall"] = split_data.get(
                                "avg_macro_recall", split_data.get("macro_recall", -1)
                            )
                        else:
                            metrics[f"{split}_macro_recall"] = split_data.get("macro_recall", -1)
                        metrics[f"{split}_macro_f1"] = split_data.get("macro_f1", -1)
                        for cls in ("normal", "bacterial", "viral"):
                            key = f"{split}_recall_{cls}"
                            metrics[key] = (
                                raw[split]
                                .get("classification_report", {})
                                .get(cls, {})
                                .get("recall", -1)
                            )

    return proc.returncode, metrics, run_dir


def run_search(base_config: dict, fusion_type: str, phase_num: int) -> list[dict]:
    """Grid search for one fusion type (a 'phase'). Returns configs ranked by val macro recall."""
    dropout_values = DROPOUT_GRID[fusion_type]
    param_combos = list(itertools.product(
        SEARCH_GRID["learning_rate"],
        dropout_values,
        SEARCH_GRID["weight_decay"],
    ))

    n_total = len(param_combos)
    print(f"\n{'═'*68}")
    print(f"  PHASE {phase_num} — Hyperparameter Search  [{fusion_type.upper()}]  "
          f"({n_total} configs × {SEARCH_EPOCHS} epochs = {n_total * SEARCH_EPOCHS} epochs)")
    print(f"  LR={SEARCH_GRID['learning_rate']}  dropout={dropout_values}  wd={SEARCH_GRID['weight_decay']}")
    print(f"{'═'*68}\n")

    results = []

    for idx, (lr, dropout, wd) in enumerate(param_combos, start=1):
        hyperparams = {
            "learning_rate":       lr,
            "noise_dropout_rates": dropout,
            "weight_decay":        wd,
        }
        print(
            f"[Phase {phase_num} {fusion_type} | {idx:>2}/{n_total}]  "
            f"lr={lr:.0e}  dropout={dropout:.1f}  wd={wd:.0e}"
        )

        cfg_path       = build_run_config(
            base_config, fusion_type, hyperparams,
            epochs=SEARCH_EPOCHS, run_index=idx,
        )
        rc, metrics, _ = run_subprocess(cfg_path)
        cfg_path.unlink(missing_ok=True)

        row = {
            "phase":               phase_num,
            "run_index":           idx,
            "fusion_type":         fusion_type,
            "learning_rate":       lr,
            "noise_dropout_rates": dropout,
            "weight_decay":        wd,
            "epochs_trained":      SEARCH_EPOCHS,
            "val_macro_recall":    metrics.get("val_macro_recall", -1),
            "val_macro_f1":        metrics.get("val_macro_f1",     -1),
        }
        results.append(row)

        print(
            f"           → val recall={row['val_macro_recall']:.4f}  "
            f"val f1={row['val_macro_f1']:.4f}"
        )
        if rc != 0:
            print(f"           [WARNING] Run exited with code {rc}")

    results.sort(key=lambda r: (r["val_macro_recall"], r["val_macro_f1"]), reverse=True)

    csv_path = search_csv_path(fusion_type)
    _write_csv(results, csv_path)
    print(f"\n[Phase {phase_num} {fusion_type}] Full ranking saved → {csv_path}")

    print(f"\n[Phase {phase_num} {fusion_type}] Top-{SEARCH_TOP_K} configs:")
    for rank, r in enumerate(results[:SEARCH_TOP_K], 1):
        marker = "  ← to Phase 3" if rank <= FINAL_TOP_K else ""
        print(
            f"  #{rank}  lr={r['learning_rate']:.0e}  "
            f"dropout={r['noise_dropout_rates']:.1f}  "
            f"wd={r['weight_decay']:.0e}  "
            f"→  val recall={r['val_macro_recall']:.4f}{marker}"
        )

    return results


def run_final_training(
    base_config: dict,
    best_by_fusion: dict[str, list[dict]],
) -> list[dict]:
    """Phase 3 — full-length training of each fusion's own best config(s)."""
    total_runs = sum(len(best_by_fusion[f]) for f in FUSION_TYPES)
    print(f"\n{'═'*68}")
    print(f"  PHASE 3 — Final Training  (best config per fusion, full length)")
    print(f"  {total_runs} runs × {FINAL_EPOCHS} epochs")
    print(f"{'═'*68}\n")

    results = []
    run_idx = 0

    for fusion in FUSION_TYPES:
        for rank, cfg in enumerate(best_by_fusion[fusion], start=1):
            run_idx += 1
            lr      = cfg["learning_rate"]
            dropout = cfg["noise_dropout_rates"]
            wd      = cfg["weight_decay"]

            print(
                f"\n[Phase 3 | Run {run_idx:>2}/{total_runs}]  "
                f"{fusion.upper()}  |  search rank-{rank}  "
                f"(lr={lr:.0e}  dropout={dropout:.1f}  wd={wd:.0e})"
            )

            hyperparams = {
                "learning_rate":       lr,
                "noise_dropout_rates": dropout,
                "weight_decay":        wd,
            }
            cfg_path             = build_run_config(
                base_config, fusion, hyperparams,
                epochs=FINAL_EPOCHS, run_index=1000 + run_idx,
            )
            rc, metrics, run_dir = run_subprocess(cfg_path, save_weights=True)
            cfg_path.unlink(missing_ok=True)

            row = {
                "phase":                  3,
                "search_rank":            rank,
                "fusion_type":            fusion,
                "learning_rate":          lr,
                "noise_dropout_rates":    dropout,
                "weight_decay":           wd,
                "epochs_trained":         FINAL_EPOCHS,
                "val_macro_recall":       metrics.get("val_macro_recall",      -1),
                "val_macro_f1":           metrics.get("val_macro_f1",          -1),
                "test_macro_recall":      metrics.get("test_macro_recall",     -1),
                "test_macro_f1":          metrics.get("test_macro_f1",         -1),
                "test_recall_normal":     metrics.get("test_recall_normal",    -1),
                "test_recall_bacterial":  metrics.get("test_recall_bacterial", -1),
                "test_recall_viral":      metrics.get("test_recall_viral",     -1),
                "exit_code":              rc,
                "run_dir":                str(run_dir) if run_dir else "",
            }
            results.append(row)

            print(
                f"  val recall={row['val_macro_recall']:.4f}  "
                f"test recall={row['test_macro_recall']:.4f}  "
                f"test f1={row['test_macro_f1']:.4f}  "
                f"viral recall={row['test_recall_viral']:.4f}"
            )
            if rc != 0:
                print(f"  [WARNING] Run exited with code {rc}")

    return results


def print_comparison_table(results: list[dict]) -> None:
    col = {
        "config":     38,
        "fusion":     10,
        "val_rec":    10,
        "test_rec":   11,
        "test_f1":     9,
        "normal":     10,
        "bacterial":  12,
        "viral":      10,
    }
    header = (
        f"{'Config':<{col['config']}} {'Fusion':<{col['fusion']}} "
        f"{'Val Rec':>{col['val_rec']}} {'Test Rec':>{col['test_rec']}} "
        f"{'Test F1':>{col['test_f1']}} {'Normal':>{col['normal']}} "
        f"{'Bacterial':>{col['bacterial']}} {'Viral':>{col['viral']}}"
    )
    sep = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print(f"{'FUSION COMPARISON — PHASE 3 (BEST vs BEST)':^{len(header)}}")
    print(f"{'═'*len(header)}")
    print(header)
    print(sep)

    for fusion in FUSION_TYPES:
        rows = sorted(
            [r for r in results if r["fusion_type"] == fusion],
            key=lambda r: r["test_macro_recall"],
            reverse=True,
        )
        for r in rows:
            cfg_str = (
                f"lr={r['learning_rate']:.0e} "
                f"do={r['noise_dropout_rates']:.1f} "
                f"wd={r['weight_decay']:.0e}"
            )
            print(
                f"{cfg_str:<{col['config']}} {r['fusion_type']:<{col['fusion']}} "
                f"{r['val_macro_recall']:>{col['val_rec']}.4f} "
                f"{r['test_macro_recall']:>{col['test_rec']}.4f} "
                f"{r['test_macro_f1']:>{col['test_f1']}.4f} "
                f"{r['test_recall_normal']:>{col['normal']}.4f} "
                f"{r['test_recall_bacterial']:>{col['bacterial']}.4f} "
                f"{r['test_recall_viral']:>{col['viral']}.4f}"
            )
        print(sep)

    print(f"{'═'*len(header)}\n")

    # Head-to-head winner
    best_per_fusion = {}
    for fusion in FUSION_TYPES:
        rows = [r for r in results if r["fusion_type"] == fusion and r["test_macro_recall"] > 0]
        if rows:
            best_per_fusion[fusion] = max(rows, key=lambda r: r["test_macro_recall"])

    print("[INFO] Best per fusion:")
    for fusion, best in best_per_fusion.items():
        print(
            f"  {fusion.upper():10s}  "
            f"lr={best['learning_rate']:.0e}  "
            f"dropout={best['noise_dropout_rates']:.1f}  "
            f"wd={best['weight_decay']:.0e}  "
            f"→  test recall={best['test_macro_recall']:.4f}  "
            f"test f1={best['test_macro_f1']:.4f}  "
            f"viral recall={best['test_recall_viral']:.4f}"
        )

    if len(best_per_fusion) == len(FUSION_TYPES):
        c, a = best_per_fusion["concat"], best_per_fusion["attention"]
        d_rec   = a["test_macro_recall"] - c["test_macro_recall"]
        d_viral = a["test_recall_viral"] - c["test_recall_viral"]
        winner  = max(best_per_fusion.values(), key=lambda r: r["test_macro_recall"])["fusion_type"]
        print(f"\n[INFO] Attention − Concat:  Δ macro recall = {d_rec:+.4f}  |  Δ viral recall = {d_viral:+.4f}")
        print(f"[INFO] Overall winner by test macro recall: {winner.upper()}")


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_results(final_results: list[dict]) -> None:
    ranked = sorted(
        final_results,
        key=lambda r: (r["test_macro_recall"], r["test_macro_f1"]),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r["overall_rank"] = i

    _write_csv(final_results, FINAL_CSV)
    _write_csv(ranked, RANKED_CSV)
    print(f"[INFO] Phase-3 results saved → {FINAL_CSV}")
    print(f"[INFO] Ranked results saved  → {RANKED_CSV}")

    best_run_dirs: set[Path] = set()

    for fusion in FUSION_TYPES:
        fusion_rows = [r for r in ranked if r["fusion_type"] == fusion]
        if not fusion_rows:
            continue

        best     = fusion_rows[0]
        cfg_path = SWEEP_DIR / f"best_config_{fusion}.json"
        with open(cfg_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"[INFO] Best {fusion} config  → {cfg_path}")

        run_dir = Path(best.get("run_dir", ""))
        best_run_dirs.add(run_dir)

        weights_src = run_dir / "best_model_weights.pth"
        if weights_src.exists():
            shutil.copy2(weights_src, SWEEP_DIR / f"best_model_{fusion}.pth")
            print(f"[INFO] Best {fusion} weights    → {SWEEP_DIR / f'best_model_{fusion}.pth'}")
        else:
            print(f"[WARNING] Weights not found for best {fusion} run: {weights_src}")

        card_src = run_dir / "model_card.json"
        if card_src.exists():
            shutil.copy2(card_src, SWEEP_DIR / f"best_model_card_{fusion}.json")
            print(f"[INFO] Best {fusion} model card → {SWEEP_DIR / f'best_model_card_{fusion}.json'}")

    for r in ranked:
        rd = Path(r.get("run_dir", ""))
        if rd and rd not in best_run_dirs:
            leftover = rd / "best_model_weights.pth"
            if leftover.exists():
                leftover.unlink()
    print(f"[INFO] Non-winning Phase 3 weights removed — {len(FUSION_TYPES)} .pth files kept.")


def main() -> int:
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {BASE_CONFIG_PATH}")

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    n_search_configs = (
        len(SEARCH_GRID["learning_rate"])
        * len(DROPOUT_GRID["concat"])
        * len(SEARCH_GRID["weight_decay"])
    )
    total_epoch_bound = (
        2 * n_search_configs * SEARCH_EPOCHS          # two searches (concat + attention)
        + len(FUSION_TYPES) * FINAL_TOP_K * FINAL_EPOCHS  # Phase 3 full training
    )

    print(f"[INFO] DualBranchConvViT Fusion Sweep — {datetime.now().isoformat()}")
    print(f"[INFO] Phase 1: search CONCAT     — {n_search_configs} configs × {SEARCH_EPOCHS} epochs")
    print(f"[INFO] Phase 2: search ATTENTION  — {n_search_configs} configs × {SEARCH_EPOCHS} epochs")
    print(f"[INFO] Phase 3: train best {FINAL_TOP_K}/fusion × {len(FUSION_TYPES)} fusions × {FINAL_EPOCHS} epochs")
    print(f"[INFO] Total training epochs (upper bound): {total_epoch_bound}")

    # Phase 1 — concat hyperparameter search
    concat_ranked    = run_search(base_config, "concat", phase_num=1)
    # Phase 2 — attention hyperparameter search
    attention_ranked = run_search(base_config, "attention", phase_num=2)

    best_by_fusion = {
        "concat":    concat_ranked[:FINAL_TOP_K],
        "attention": attention_ranked[:FINAL_TOP_K],
    }

    # Phase 3 — full-length training of each fusion's own best config(s)
    final_results = run_final_training(base_config, best_by_fusion)

    print_comparison_table(final_results)
    save_results(final_results)

    print(f"\n[INFO] Fusion sweep complete.  All results under: {SWEEP_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())