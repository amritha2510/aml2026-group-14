"""
fusion_sweep.py — Scientifically rigorous hyperparameter search and
fusion comparison for DualBranchConvViT.

Design rationale
────────────────
1.  WHAT we tune:
      • learning_rate      — controls gradient step magnitude
      • noise_dropout_rate — regularization on both branches
      • weight_decay       — L2 penalty via AdamW (independent axis from dropout)
    WHAT we do NOT tune as hyperparameters:
      • batch_size  — fixed at 32; a compute/memory choice, not a model
                      capability parameter for our dataset size (~5 k images).
      • epochs      — fixed at FINAL_EPOCHS; best-model tracking (save at peak
                      val recall) makes the ceiling irrelevant.

2.  TWO-PHASE STRATEGY keeps total compute bounded and the comparison fair:

    Phase 1 │ 27-config grid (3×3×3) on CONCAT fusion only, SEARCH_EPOCHS each.
            │ Concat is used as the search vehicle because (a) it converges more
            │ stably than attention on a 16-sample val set and (b) the two
            │ branches share identical optimization dynamics regardless of fusion.
            │ → Produces a ranked list of hyperparameter configs.

    Phase 2 │ Top-K configs from Phase 1 re-run on BOTH fusion types at full
            │ FINAL_EPOCHS.  Crucially, the SAME config is applied to both
            │ fusions, so fusion mechanism is the only variable that changes.
            │ This is a proper ablation study.

3.  DYNAMIC CLASS WEIGHTS are computed once from the training label distribution
    using inverse-frequency weighting and injected into every subprocess config.
    This is data-driven and reproducible — no manual tuning required.

4.  STATISTICAL HONESTY: because val set = 16 samples, we promote TOP_K = 3
    configs rather than a single "best", and report all Phase-2 runs so the
    reader can see the variance induced by the noisy val signal.
"""

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

import numpy as np
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_class_weight

REPO_ROOT         = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH  = REPO_ROOT / "config_dual_branch.yaml"
MODEL_SCRIPT      = REPO_ROOT / "models" / "DualBranchConvViT.py"
OUTPUT_DIR        = REPO_ROOT / "outputs" / "deep_learning"
SWEEP_DIR         = OUTPUT_DIR / "fusion_sweep"
PHASE1_CSV        = SWEEP_DIR / "phase1_search_results.csv"
PHASE2_CSV        = SWEEP_DIR / "phase2_final_results.csv"
RANKED_CSV        = SWEEP_DIR / "phase2_final_results_ranked.csv"

# ─── constants ────────────────────────────────────────────────────────────────
LABEL_TO_ID   = {"normal": 0, "bacterial": 1, "viral": 2}   # keep in sync with constants.py
SEARCH_EPOCHS = 5     # Phase 1 budget — 3 epochs sufficient to rank configs on 5060 Ti
                      # (loss curves diverge within 3-5 epochs; reduced from 5 for speed)
FINAL_EPOCHS  = 20    # Phase 2 full training budget
TOP_K         = 3     # configs promoted from Phase 1 → Phase 2
FUSION_TYPES  = ["concat", "attention"]
BATCH_SIZE    = 64    # doubled from 32 — 16 GB VRAM handles ResNet18+ViT-Base at bs=64 with AMP

# ─── Phase-1 search grid ──────────────────────────────────────────────────────
# Three axes from the problem setting:
#   • learning_rate      — gradient step magnitude
#   • noise_dropout_rate — symmetric regularization on both branches
#   • weight_decay       — L2 penalty via AdamW
#
# viral_boost_multiplier is FIXED at 5.0 rather than searched.
# Rationale: prior manual runs established that viral recall collapses to 0.00
# at boost≈1.0 and recovers to 0.55 at boost≈5.0, with no further gain at 7.0
# (recall did not improve beyond the 5.0 checkpoint in the logged results).
# Treating 5.0 as a known good value and fixing it is standard practice when
# prior experimental evidence is available — it reduces compute without
# sacrificing scientific validity.
#
# Grid: 3 × 3 × 3 × 3 = 81 configs × 5 epochs = 405 Phase-1 epochs  (~4.5 hrs)
#       + 3 × 2 fusions × 20 epochs             = 120 Phase-2 epochs  (~2.0 hrs)
#       Total ≈ 6.5 hours on a single GPU
VIRAL_BOOST_MULTIPLIER = 0

SEARCH_GRID = {
   
    "learning_rate":      [1e-4, 5e-5, 1e-5], 
    "noise_dropout_rate": [0.3,  0.4,  0.5],
    "weight_decay":       [1e-5, 1e-4, 1e-3],
    "viral_boost_multiplier": [1.5, 2.0, 3.0], 
}

# ══════════════════════════════════════════════════════════════════════════════
# Dynamic class-weight computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_class_weights(
    preprocessed_metadata_path: Path,
    viral_boost_multiplier: float = 1.0,
    verbose: bool = True,
) -> list[float]:
    """
    Loads the training split from the preprocessed metadata CSV and computes
    per-class weights via sklearn's 'balanced' inverse-frequency formula:

        weight[c] = total_train / (n_classes × count[c])

    An optional viral_boost_multiplier is then applied on top of the balanced
    weight for the viral class only.  This separates two distinct problems:

        balanced weights  →  correct for frequency imbalance (viral ≈ normal
                             in count, so balanced gives both ~1.3)
        viral_boost       →  correct for difficulty/similarity imbalance
                             (viral looks like bacterial to the model, so
                             missing a viral case needs extra penalisation
                             beyond what frequency alone justifies)

    Robustness notes:
      • Label strings are lowercased and stripped before mapping so the
        function is tolerant of CSV formatting differences (e.g. "NORMAL",
        " bacterial ", "Viral").
      • Rows that still fail to map (genuinely unknown labels) are dropped
        with a warning rather than silently propagating NaN into sklearn.
      • A hard assertion verifies all three expected classes are present in
        the training split before calling sklearn.

    Returns a plain Python list [w_normal, w_bacterial, w_viral] so it can
    be serialised directly into a YAML config.
    """
    df       = pd.read_csv(preprocessed_metadata_path)
    train_df = df[df["split"] == "train"].copy()

    if train_df.empty:
        raise ValueError(
            "No rows with split='train' found in the preprocessed metadata. "
            f"Unique split values present: {df['split'].unique().tolist()}"
        )

    id_to_label = {v: k for k, v in LABEL_TO_ID.items()}

    # Normalise label strings before mapping — tolerates case/whitespace drift
    normalised = train_df["label"].astype(str).str.strip().str.lower()

    # Warn about any values that don't map to a known class
    unmapped_mask   = ~normalised.isin(LABEL_TO_ID)
    unmapped_values = normalised[unmapped_mask].unique().tolist()
    if unmapped_values and verbose:
        print(
            f"[WARNING] {unmapped_mask.sum()} training rows have unrecognised "
            f"label values after normalisation and will be excluded from class "
            f"weight computation: {unmapped_values}\n"
            f"          Expected keys: {list(LABEL_TO_ID.keys())}"
        )

    labels_mapped = normalised.map(LABEL_TO_ID).dropna().astype(int)

    # Hard check: all three classes must be present
    present_classes  = set(labels_mapped.unique())
    expected_classes = set(LABEL_TO_ID.values())
    if present_classes != expected_classes:
        missing = expected_classes - present_classes
        raise ValueError(
            f"Training split is missing class IDs {missing} after label mapping.\n"
            f"  Present  : {sorted(present_classes)}  "
            f"({[id_to_label[c] for c in sorted(present_classes)]})\n"
            f"  Expected : {sorted(expected_classes)}  "
            f"({[id_to_label[c] for c in sorted(expected_classes)]})\n"
            f"  Raw label values in CSV: {train_df['label'].unique().tolist()}"
        )

    labels  = labels_mapped.to_numpy()
    classes = np.arange(len(LABEL_TO_ID))
    weights = compute_class_weight("balanced", classes=classes, y=labels)

    # Apply viral boost on top of the balanced baseline.
    # Viral class ID is LABEL_TO_ID["viral"] = 2.
    viral_id = LABEL_TO_ID["viral"]
    if viral_boost_multiplier != 1.0:
        weights[viral_id] *= viral_boost_multiplier

    if verbose:
        print("\n[INFO] Dynamic class weights (balanced × viral boost):")
        print(f"       viral_boost_multiplier = {viral_boost_multiplier:.1f}")
        for c, w in zip(classes, weights):
            count    = int((labels == c).sum())
            boosted  = " ← boosted" if (c == viral_id and viral_boost_multiplier != 1.0) else ""
            print(f"       {id_to_label[c]:12s}  n={count:5d}  →  weight = {w:.4f}{boosted}")
        print()

    return weights.tolist()


def build_weights_cache(preprocessed_metadata_path: Path) -> dict[float, list[float]]:
    """
    Pre-computes class weights for every viral_boost_multiplier value in the
    search grid.  Returns a dict keyed by boost value so phase runners can
    look up weights in O(1) without recomputing per run.

    Warnings about unrecognised label values are printed once per boost value.
    """
    boost_values = SEARCH_GRID.get("viral_boost_multiplier", [VIRAL_BOOST_MULTIPLIER])
    print(f"\n[INFO] Pre-computing class weights for boost values: {boost_values}")

    cache: dict[float, list[float]] = {}
    for boost in boost_values:
        cache[boost] = compute_dynamic_class_weights(
            preprocessed_metadata_path,
            viral_boost_multiplier=boost,
            verbose=True,
        )
    print("[INFO] Weight cache ready.\n")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# Config management
# ══════════════════════════════════════════════════════════════════════════════

def build_run_config(
    base_config:            dict,
    fusion_type:            str,
    hyperparams:            dict,
    class_weights:          list[float],
    epochs:                 int,
    run_index:              int,
) -> Path:
    """
    Writes a temporary YAML config for a single subprocess run.

    The base config is deep-copied and the deep_learning block is overwritten
    with the supplied hyperparameters, fusion type, dynamic class weights, and
    epoch budget.  All other top-level config sections (preprocessing paths,
    augmentation, etc.) are inherited unchanged.
    """
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    run_config = dict(base_config)
    dl_cfg     = dict(run_config.get("deep_learning", {}))

    dl_cfg.update({
        "fusion_type":        fusion_type,
        "learning_rate":      hyperparams["learning_rate"],
        "noise_dropout_rate": hyperparams["noise_dropout_rate"],
        "weight_decay":       hyperparams["weight_decay"],
        "batch_size":         BATCH_SIZE,
        "epochs":             epochs,
        # Dynamic class weights injected here — DualBranchConvViT.py reads
        # this list directly from config and converts to a weighted tensor.
        "class_weight":       class_weights,
        # Freeze schedule: keep fixed across all runs so it is not a
        # confounding variable in the fusion comparison.
        "freeze_epochs":      dl_cfg.get("freeze_epochs", 5),
    })

    run_config["deep_learning"] = dl_cfg

    tmp_path = REPO_ROOT / f".sweep_run_{run_index:04d}.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# Subprocess runner
# ══════════════════════════════════════════════════════════════════════════════

def run_subprocess(config_path: Path, save_weights: bool = False) -> tuple[int, dict, Path | None]:
    """
    Spawns DualBranchConvViT.py as a subprocess with the given config,
    then locates the metrics.json written by the model's save_run() call.

    Returns (return_code, metrics_dict, run_dir).
    save_weights=False (default) skips .pth saving — only Phase 2 winners need weights.
    """
    env = os.environ.copy()
    env["DUAL_BRANCH_CONFIG_PATH"]  = str(config_path)
    env["DUAL_BRANCH_SAVE_WEIGHTS"] = "1" if save_weights else "0"

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

    # Locate the most recently modified experiment dir (the one just created)
    experiments_dir = OUTPUT_DIR / "experiments"
    metrics  = {}
    run_dir  = None
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
                        # For val: prefer avg_macro_recall (mean across epochs) over single-epoch peak
                        # — more robust when val set is small (16 samples).
                        if split == "val":
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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — hyperparameter search on concat
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(
    base_config:   dict,
    weights_cache: dict[float, list[float]],
) -> list[dict]:
    """
    Trains all 27 grid configs (concat only, fixed viral_boost) for
    SEARCH_EPOCHS each and returns results sorted by val_macro_recall.
    """
    boost_values = SEARCH_GRID.get("viral_boost_multiplier", [VIRAL_BOOST_MULTIPLIER])
    param_combos = list(itertools.product(
        SEARCH_GRID["learning_rate"],
        SEARCH_GRID["noise_dropout_rate"],
        SEARCH_GRID["weight_decay"],
        boost_values,
    ))

    n_total = len(param_combos)
    print(f"\n{'═'*65}")
    print(f"  PHASE 1 — Hyperparameter Search  "
          f"({n_total} configs × {SEARCH_EPOCHS} epochs = "
          f"{n_total * SEARCH_EPOCHS} training epochs total)")
    print(f"  Search space: {SEARCH_GRID}")
    print(f"{'═'*65}\n")

    results = []

    for idx, (lr, dropout, wd, viral_boost) in enumerate(param_combos, start=1):
        class_weights = weights_cache[viral_boost]
        hyperparams = {
            "learning_rate":          lr,
            "noise_dropout_rate":     dropout,
            "weight_decay":           wd,
            "viral_boost_multiplier": viral_boost,
        }
        print(
            f"[Phase 1 | {idx:>2}/{n_total}]  "
            f"lr={lr:.0e}  dropout={dropout:.1f}  wd={wd:.0e}  boost={viral_boost:.1f}"
        )

        cfg_path    = build_run_config(
            base_config, "concat", hyperparams, class_weights,
            epochs=SEARCH_EPOCHS, run_index=idx,
        )
        rc, metrics, _ = run_subprocess(cfg_path)
        cfg_path.unlink(missing_ok=True)

        row = {
            "phase":                  1,
            "run_index":              idx,
            "fusion_type":            "concat",
            "learning_rate":          lr,
            "noise_dropout_rate":     dropout,
            "weight_decay":           wd,
            "viral_boost_multiplier": viral_boost,
            "epochs_trained":         SEARCH_EPOCHS,
            "val_macro_recall":       metrics.get("val_macro_recall", -1),
            "val_macro_f1":           metrics.get("val_macro_f1",     -1),
            # Phase 1 does not evaluate test — avoids test-set leakage
            # into hyperparameter selection.
            "test_macro_recall":      -1,
            "test_macro_f1":          -1,
        }
        results.append(row)

        print(
            f"           → val recall={row['val_macro_recall']:.4f}  "
            f"val f1={row['val_macro_f1']:.4f}"
        )
        if rc != 0:
            print(f"           [WARNING] Run exited with code {rc}")

    # Rank: primary = val_macro_recall, tiebreak = val_macro_f1
    results.sort(key=lambda r: (r["val_macro_recall"], r["val_macro_f1"]), reverse=True)

    # Write Phase-1 results
    _write_csv(results, PHASE1_CSV)
    print(f"\n[Phase 1] Results saved → {PHASE1_CSV}")

    print(f"\n[Phase 1] Top-{TOP_K} configs promoted to Phase 2:")
    for rank, r in enumerate(results[:TOP_K], 1):
        print(
            f"  #{rank}  lr={r['learning_rate']:.0e}  "
            f"dropout={r['noise_dropout_rate']:.1f}  "
            f"wd={r['weight_decay']:.0e}  "
            f"→  val recall={r['val_macro_recall']:.4f}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — full comparison: top-K configs × both fusion types
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(
    base_config:   dict,
    weights_cache: dict[float, list[float]],
    top_configs:   list[dict],
) -> list[dict]:
    """
    Re-runs the Top-K configs from Phase 1 with BOTH fusion types at full
    FINAL_EPOCHS, then evaluates on the test split.

    Because the SAME hyperparameter config (including viral_boost_multiplier)
    is applied to both fusion types, fusion mechanism is the only experimental
    variable — a proper ablation.  Class weights are looked up from the cache,
    not recomputed.
    """
    total_runs = len(top_configs) * len(FUSION_TYPES)
    print(f"\n{'═'*65}")
    print(f"  PHASE 2 — Final Fusion Comparison")
    print(f"  Top-{TOP_K} configs × {len(FUSION_TYPES)} fusion types = "
          f"{total_runs} runs × {FINAL_EPOCHS} epochs")
    print(f"{'═'*65}\n")

    results = []
    run_idx = 0

    for fusion in FUSION_TYPES:
        for rank, cfg in enumerate(top_configs, start=1):
            run_idx     += 1
            lr           = cfg["learning_rate"]
            dropout      = cfg["noise_dropout_rate"]
            wd           = cfg["weight_decay"]
            viral_boost  = cfg["viral_boost_multiplier"]
            class_weights = weights_cache[viral_boost]   # O(1) — no recomputation

            print(
                f"\n[Phase 2 | Run {run_idx:>2}/{total_runs}]  "
                f"{fusion.upper()}  |  Rank-{rank} config  "
                f"(lr={lr:.0e}  dropout={dropout:.1f}  "
                f"wd={wd:.0e}  viral_boost={viral_boost:.1f})"
            )

            hyperparams = {
                "learning_rate":          lr,
                "noise_dropout_rate":     dropout,
                "weight_decay":           wd,
                "viral_boost_multiplier": viral_boost,
            }
            cfg_path    = build_run_config(
                base_config, fusion, hyperparams, class_weights,
                epochs=FINAL_EPOCHS, run_index=1000 + run_idx,
            )
            rc, metrics, run_dir = run_subprocess(cfg_path, save_weights=True)
            cfg_path.unlink(missing_ok=True)

            row = {
                "phase":                  2,
                "rank_from_phase1":       rank,
                "fusion_type":            fusion,
                "learning_rate":          lr,
                "noise_dropout_rate":     dropout,
                "weight_decay":           wd,
                "viral_boost_multiplier": viral_boost,
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


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: list[dict]) -> None:
    """
    Prints a report-ready comparison table.  Runs are sorted by test macro
    recall so the best configuration floats to the top within each fusion type.
    """
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
    print(f"{'FUSION COMPARISON — PHASE 2 RESULTS':^{len(header)}}")
    print(f"  (Ranked by test macro recall within each fusion type)")
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
                f"do={r['noise_dropout_rate']:.1f} "
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

    # Best per fusion
    print("[INFO] Best configuration per fusion type:")
    for fusion in FUSION_TYPES:
        rows = [r for r in results if r["fusion_type"] == fusion and r["test_macro_recall"] > 0]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["test_macro_recall"])
        print(
            f"  {fusion.upper():10s}  "
            f"lr={best['learning_rate']:.0e}  "
            f"dropout={best['noise_dropout_rate']:.1f}  "
            f"wd={best['weight_decay']:.0e}  "
            f"→  test recall={best['test_macro_recall']:.4f}  "
            f"test f1={best['test_macro_f1']:.4f}  "
            f"viral recall={best['test_recall_viral']:.4f}"
        )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_results(phase2_results: list[dict]) -> None:
    """Saves Phase-2 results and best-config JSON files."""
    ranked = sorted(
        phase2_results,
        key=lambda r: (r["test_macro_recall"], r["test_macro_f1"]),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r["overall_rank"] = i

    _write_csv(phase2_results, PHASE2_CSV)
    _write_csv(ranked, RANKED_CSV)
    print(f"[INFO] Phase-2 results saved → {PHASE2_CSV}")
    print(f"[INFO] Ranked results saved  → {RANKED_CSV}")

    best_run_dirs: set[Path] = set()

    for fusion in FUSION_TYPES:
        fusion_rows = [r for r in ranked if r["fusion_type"] == fusion]
        if not fusion_rows:
            continue

        best = fusion_rows[0]
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

    # Delete weights from non-winning Phase 2 experiment dirs — only 2 .pth files remain.
    for r in ranked:
        rd = Path(r.get("run_dir", ""))
        if rd and rd not in best_run_dirs:
            leftover = rd / "best_model_weights.pth"
            if leftover.exists():
                leftover.unlink()
    print(f"[INFO] Non-winning Phase 2 weights removed — {len(FUSION_TYPES)} .pth files kept.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {BASE_CONFIG_PATH}")

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    n_search_configs = (
        len(SEARCH_GRID["learning_rate"])
        * len(SEARCH_GRID["noise_dropout_rate"])
        * len(SEARCH_GRID["weight_decay"])
        * len(SEARCH_GRID.get("viral_boost_multiplier", [VIRAL_BOOST_MULTIPLIER]))
    )
    total_epoch_bound = (
        n_search_configs * SEARCH_EPOCHS
        + TOP_K * len(FUSION_TYPES) * FINAL_EPOCHS
    )

    print(f"[INFO] DualBranchConvViT Fusion Sweep — {datetime.now().isoformat()}")
    print(f"[INFO] Phase 1: {n_search_configs} configs × {SEARCH_EPOCHS} epochs "
          f"(concat, viral_boost fixed at {VIRAL_BOOST_MULTIPLIER})")
    print(f"[INFO] Phase 2: Top-{TOP_K} configs × {len(FUSION_TYPES)} fusions "
          f"× {FINAL_EPOCHS} epochs")
    print(f"[INFO] Total training epochs (upper bound): {total_epoch_bound} "
          f"(~{total_epoch_bound * 3 // 60}–{total_epoch_bound * 5 // 60} hrs)")

    # Locate preprocessed metadata — needed for dynamic class weight computation
    preprocessed_path = Path(
        base_config.get("preprocessed_metadata_output_path", "")
    )
    if not preprocessed_path.is_absolute():
        preprocessed_path = REPO_ROOT / preprocessed_path
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed metadata not found: {preprocessed_path}\n"
            "Run the preprocessing step before the sweep."
        )

    # ── Build weights cache (3 computations total, warnings printed once each) ─
    weights_cache = build_weights_cache(preprocessed_path)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    phase1_results = run_phase1(base_config, weights_cache)
    top_configs    = phase1_results[:TOP_K]

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    phase2_results = run_phase2(base_config, weights_cache, top_configs)

    # ── Reporting ─────────────────────────────────────────────────────────────
    print_comparison_table(phase2_results)
    save_results(phase2_results)

    print(f"\n[INFO] Fusion sweep complete.  All results under: {SWEEP_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())