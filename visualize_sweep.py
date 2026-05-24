"""Visualize hyperparameter sweep and ablation results.

Generates four publication-ready PNGs into a `sweep_visualizations` folder:
  1) Phase-1 hyperparameter impact heatmap (learning_rate × noise_dropout_rate)
  2) Phase-2 ablation grouped bar chart (concat vs attention, recall + F1)
  3) Val-recall vs test-recall scatter — illustrates noisy val signal
  4) Class-level recall breakdown for the best Phase-2 model

Usage: python visualize_sweep.py
Or:    python visualize_sweep.py --phase1 path/to/p1.csv --phase2 path/to/p2.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DEFAULT_PHASE1       = Path("outputs/deep_learning/fusion_sweep/phase1_search_results.csv")
DEFAULT_PHASE2       = Path("outputs/deep_learning/fusion_sweep/phase2_final_results_ranked.csv")
DEFAULT_EXPERIMENTS  = Path("outputs/deep_learning/experiments")
OUT_DIR              = Path("sweep_visualizations")

CLASS_MAP = {
    "normal":     "test_recall_normal",
    "bacterial":  "test_recall_bacterial",
    "viral":      "test_recall_viral",
}


def ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def style() -> None:
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.titlesize":   14,
        "axes.labelsize":   12,
        "legend.fontsize":  11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })


# ─── per-class backfill ───────────────────────────────────────────────────────

def backfill_per_class(df: pd.DataFrame, experiments_dir: Path) -> pd.DataFrame:
    """
    Reads experiment JSON dirs and fills in per-class test recalls for rows
    where the CSV columns are -1.  This handles runs made before the
    classification_report key fix was applied to the sweep runner.

    Matches each CSV row to an experiment dir by comparing fusion_type,
    learning_rate, noise_dropout_rate, and weight_decay from config.json.
    """
    per_class_cols = list(CLASS_MAP.values())
    needs_backfill = df[per_class_cols].eq(-1).all(axis=1)
    if not needs_backfill.any() or not experiments_dir.exists():
        return df

    print(f"[backfill] {needs_backfill.sum()} rows have per-class=-1, "
          f"scanning {experiments_dir} …")

    exp_data = []
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        cfg_file  = exp_dir / "config.json"
        met_file  = exp_dir / "metrics.json"
        if not (cfg_file.exists() and met_file.exists()):
            continue
        with open(cfg_file) as f:
            cfg = json.load(f)
        with open(met_file) as f:
            mets = json.load(f)
        exp_data.append({"cfg": cfg, "metrics": mets})

    df = df.copy()
    df[per_class_cols] = df[per_class_cols].astype(float)
    for idx, row in df[needs_backfill].iterrows():
        fusion  = row["fusion_type"]
        lr      = float(row["learning_rate"])
        dropout = float(row["noise_dropout_rate"])
        wd      = float(row["weight_decay"])

        for exp in exp_data:
            c = exp["cfg"]
            if (
                c.get("fusion_type") == fusion
                and abs(float(c.get("learning_rate",      0)) - lr)      < 1e-9
                and abs(float(c.get("noise_dropout_rate", 0)) - dropout) < 1e-9
                and abs(float(c.get("weight_decay",       0)) - wd)      < 1e-9
            ):
                report = exp["metrics"].get("test", {}).get("classification_report", {})
                for cls, col in CLASS_MAP.items():
                    df.at[idx, col] = report.get(cls, {}).get("recall", -1)
                break

    filled = df[per_class_cols].ne(-1).any(axis=1) & needs_backfill
    print(f"[backfill] Filled {filled.sum()}/{needs_backfill.sum()} rows from experiment dirs.")
    return df


# ─── Plot 1: Phase-1 heatmap ─────────────────────────────────────────────────

def plot_phase1_heatmap(df_phase1: pd.DataFrame, out_path: Path) -> None:
    df = df_phase1.copy()
    df["learning_rate"]      = df["learning_rate"].astype(float)
    df["noise_dropout_rate"] = df["noise_dropout_rate"].astype(float)

    pivot = (
        df.groupby(["noise_dropout_rate", "learning_rate"])["val_macro_recall"]
        .mean()
        .reset_index()
        .pivot(index="noise_dropout_rate", columns="learning_rate", values="val_macro_recall")
    )
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(8, 5.2))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                     cbar_kws={"label": "val macro recall (mean over weight_decay)"})
    ax.set_title("Phase 1 — Hyperparameter impact on val macro recall")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Noise dropout rate")
    ax.set_xticklabels([f"{v:.0e}" for v in pivot.columns])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# ─── Plot 2: Phase-2 ablation bar chart ──────────────────────────────────────

def plot_phase2_ablation(df_phase2: pd.DataFrame, top_k: int, out_path: Path) -> None:
    df = df_phase2.copy()
    if "rank_from_phase1" in df.columns:
        df["rank_from_phase1"] = df["rank_from_phase1"].astype(int)
        sel = df[df["rank_from_phase1"].between(1, top_k)].copy()
    else:
        sel = df.nlargest(top_k * 2, "val_macro_recall").copy()

    # Melt recall + F1 into long form for side-by-side bars
    sel["config"] = sel.apply(
        lambda r: f"lr={r['learning_rate']:.0e}\ndo={r['noise_dropout_rate']:.1f}  wd={r['weight_decay']:.0e}",
        axis=1,
    )
    long = pd.melt(
        sel,
        id_vars=["config", "fusion_type", "rank_from_phase1"],
        value_vars=["test_macro_recall", "test_macro_f1"],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(
        {"test_macro_recall": "Test Recall", "test_macro_f1": "Test F1"}
    )
    long["label"] = long["fusion_type"].str.capitalize() + " — " + long["metric"]

    _, ax = plt.subplots(figsize=(10, 5))
    palette = {
        "Concat — Test Recall":    "#4c72b0",
        "Concat — Test F1":        "#9fb8d8",
        "Attention — Test Recall": "#c44e52",
        "Attention — Test F1":     "#e8a09a",
    }
    sns.barplot(
        data=long, x="rank_from_phase1", y="score", hue="label",
        palette=palette, ax=ax,
    )
    ax.set_title(f"Phase 2 — Concat vs Attention (Top-{top_k} configs from Phase 1)")
    ax.set_xlabel("Rank from Phase 1")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0.05:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)

    ax.legend(title="Fusion — Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── Plot 3: Val vs Test scatter ──────────────────────────────────────────────

def plot_val_vs_test(df_phase2: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter of val_macro_recall vs test_macro_recall for every Phase-2 run,
    coloured by fusion type.  Illustrates how noisy the 16-sample val signal is —
    all runs cluster at the same val recall while test varies widely.
    """
    df = df_phase2.copy()

    _, ax = plt.subplots(figsize=(7, 5))
    palette = {"concat": "#4c72b0", "attention": "#c44e52"}

    for fusion, grp in df.groupby("fusion_type"):
        ax.scatter(
            grp["val_macro_recall"], grp["test_macro_recall"],
            label=fusion.capitalize(), color=palette.get(fusion, "grey"),
            s=120, zorder=3, edgecolors="white", linewidth=0.8,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                f"lr={row['learning_rate']:.0e}",
                (row["val_macro_recall"], row["test_macro_recall"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5,
            )

    # Diagonal reference line (perfect val–test agreement)
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="grey", linewidth=0.8, alpha=0.5, label="val = test")

    ax.set_title("Phase 2 — Val macro recall vs Test macro recall\n"
                 "(cluster width on x shows val-set noise with only 16 samples)")
    ax.set_xlabel("Val macro recall (16-sample set)")
    ax.set_ylabel("Test macro recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# ─── Plot 4: Best-model class breakdown ──────────────────────────────────────

def plot_best_model_class_breakdown(
    df_phase2: pd.DataFrame,
    out_path: Path,
    viral_boost_multiplier: float = 5.0,
) -> None:
    df = df_phase2.copy()
    sort_cols = [c for c in ["test_macro_recall", "test_macro_f1"] if c in df.columns]
    best = df.sort_values(sort_cols, ascending=False).iloc[0]

    class_labels = ["Normal", "Bacterial", "Viral"]
    col_keys     = ["test_recall_normal", "test_recall_bacterial", "test_recall_viral"]
    values       = [float(best.get(k, -1)) for k in col_keys]

    # Replace sentinel -1 with NaN so missing bars are obvious
    values_plot = [v if v >= 0 else np.nan for v in values]
    all_missing = all(np.isnan(v) for v in values_plot)

    _, ax = plt.subplots(figsize=(7, 4.8))

    if all_missing:
        ax.text(0.5, 0.5,
                "Per-class recalls not available for this run.\n"
                "Re-run the sweep to populate these values.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11,
                color="grey")
    else:
        colors = ["#4c72b0", "#55a868", "#c44e52"]
        bars = ax.bar(class_labels, values_plot, color=colors)
        for bar, v in zip(bars, values_plot):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.05,
                        "N/A", ha="center", va="bottom", fontsize=10, color="grey")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Recall")

    fusion = best.get("fusion_type", "N/A")
    rank   = best.get("rank_from_phase1", "N/A")
    lr     = best.get("learning_rate", "N/A")
    title  = (f"Class-level recall — Best Phase-2 model\n"
              f"fusion={fusion}  rank={rank}  lr={lr:.0e}  "
              f"viral_boost={viral_boost_multiplier:.1f}")
    ax.set_title(title)

    note = textwrap.fill(
        f"viral_boost_multiplier={viral_boost_multiplier:.1f} upweights "
        "viral loss to rescue viral recall beyond frequency-based balancing.",
        width=44,
    )
    ax.text(1.02, 0.5, note, transform=ax.transAxes, fontsize=9, va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize hyperparameter sweep results")
    parser.add_argument("--phase1",       type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2",       type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--experiments",  type=Path, default=DEFAULT_EXPERIMENTS,
                        help="Experiment output dir used to backfill per-class recalls")
    parser.add_argument("--top_k",        type=int,  default=3)
    parser.add_argument("--out",          type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    ensure_dirs(args.out)
    style()

    if not args.phase1.exists():
        raise FileNotFoundError(f"Phase-1 CSV not found: {args.phase1}")
    if not args.phase2.exists():
        raise FileNotFoundError(f"Phase-2 CSV not found: {args.phase2}")

    df1 = pd.read_csv(args.phase1)
    df2 = pd.read_csv(args.phase2)

    # Backfill per-class recalls from experiment dirs if columns are -1
    df2 = backfill_per_class(df2, args.experiments)

    plot_phase1_heatmap(df1, args.out / "phase1_hyperparam_heatmap.png")
    plot_phase2_ablation(df2, args.top_k, args.out / "phase2_ablation_grouped.png")
    plot_val_vs_test(df2, args.out / "phase2_val_vs_test_scatter.png")

    viral_boost = 5.0
    if "viral_boost_multiplier" in df1.columns:
        try:
            viral_boost = float(df1["viral_boost_multiplier"].dropna().unique()[0])
        except Exception:
            pass
    plot_best_model_class_breakdown(df2, args.out / "phase2_best_class_breakdown.png",
                                    viral_boost_multiplier=viral_boost)


if __name__ == "__main__":
    main()
