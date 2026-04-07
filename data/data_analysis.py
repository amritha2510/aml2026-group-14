import json
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def inspect_raw_image(filepath: str) -> Dict[str, object]:
    """
    Inspect the original image without modifying it.
    """
    try:
        with Image.open(filepath) as img:
            arr = np.array(img)

            if arr.ndim == 2:
                n_channels = 1
            elif arr.ndim == 3:
                n_channels = int(arr.shape[2])
            else:
                n_channels = None

            return {
                "ok": True,
                "width": img.width,
                "height": img.height,
                "original_mode": img.mode,
                "n_channels": n_channels,
                "error": None,
            }
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return {
            "ok": False,
            "width": None,
            "height": None,
            "original_mode": None,
            "n_channels": None,
            "error": str(e),
        }


def build_analysis_table(
    metadata_df: pd.DataFrame,
    progress_every: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a self-contained analysis table from reader metadata.

    Input:
        metadata_df with at least: split, filepath, filename, label

    Output:
        original metadata + analysis columns such as:
        ok, width, height, original_mode, n_channels, error
    """
    rows = []

    for i, row in metadata_df.iterrows():
        if verbose and i % progress_every == 0:
            print(f"[INFO] Inspecting raw image {i + 1}/{len(metadata_df)}")

        info = inspect_raw_image(row["filepath"])
        rows.append({**row.to_dict(), **info})

    return pd.DataFrame(rows)


def print_basic_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("\nTotal images:", len(df))
    print("\nImages by split:")
    print(df["split"].value_counts().sort_index())

    print("\nImages by label:")
    print(df["label"].value_counts())

    print("\nImages by split x label:")
    print(pd.crosstab(df["split"], df["label"]))

    if "ok" in df.columns:
        corrupted = (~df["ok"]).sum()
        print(f"\nCorrupted / unreadable images: {corrupted}")

        if corrupted > 0:
            print("\nSample corrupted files:")
            print(df.loc[~df["ok"], ["filepath", "error"]].head(10).to_string(index=False))


def print_image_mode_summary(df: pd.DataFrame) -> None:
    if "original_mode" not in df.columns:
        raise ValueError("DataFrame must contain an 'original_mode' column.")

    print("\nOriginal image modes:")
    print(df["original_mode"].value_counts(dropna=False))

    if "n_channels" in df.columns:
        print("\nChannel count summary:")
        print(df["n_channels"].value_counts(dropna=False).sort_index())


def print_resize_suggestion(df: pd.DataFrame) -> None:
    if "ok" not in df.columns:
        raise ValueError("DataFrame must contain an 'ok' column.")

    ok_df = df[df["ok"]].copy()
    if ok_df.empty:
        raise ValueError("No valid images available to compute resize suggestion.")

    median_width = int(ok_df["width"].median())
    median_height = int(ok_df["height"].median())

    print("\nSuggested resize reference:")
    print(f"Median size = ({median_height}, {median_width})")
    print("Common downstream choices: 64x64, 96x96, 128x128, 224x224.")


def show_sample_rgb_files(df: pd.DataFrame, n: int = 10, random_state: int = 42) -> None:
    if "original_mode" not in df.columns:
        raise ValueError("DataFrame must contain an 'original_mode' column.")

    rgb_df = df[df["original_mode"] == "RGB"].copy()
    print(f"\nFound {len(rgb_df)} RGB images. Showing up to {n} sampled filepaths:\n")

    if rgb_df.empty:
        return

    sample = rgb_df.sample(min(n, len(rgb_df)), random_state=random_state)
    for path in sample["filepath"].tolist():
        print(path)


def save_summary_tables(df: pd.DataFrame, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_dir / "analysis_metadata_full.csv", index=False)

    class_counts = pd.crosstab(df["split"], df["label"])
    class_counts.to_csv(save_dir / "class_distribution.csv")

    if {"split", "original_mode"}.issubset(df.columns):
        mode_counts = pd.crosstab(df["split"], df["original_mode"])
        mode_counts.to_csv(save_dir / "image_mode_distribution.csv")

    summary = (
        df.groupby(["split", "label"])
        .agg(
            count=("filepath", "count"),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(save_dir / "per_split_label_stats.csv", index=False)

    print(f"\n[INFO] Saved analysis CSV reports to: {save_dir}")


def plot_class_distribution(df: pd.DataFrame, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ctab = pd.crosstab(df["split"], df["label"])
    ax = ctab.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=200)
    plt.close()


def plot_image_mode_distribution(df: pd.DataFrame, save_dir: str | Path) -> None:
    if "original_mode" not in df.columns:
        raise ValueError("DataFrame must contain an 'original_mode' column.")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ctab = pd.crosstab(df["split"], df["original_mode"])
    ax = ctab.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Original Image Mode Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "image_mode_distribution.png", dpi=200)
    plt.close()


def plot_image_size_distributions(df: pd.DataFrame, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ok_df = df[df["ok"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(ok_df["width"], bins=30)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Count")

    axes[1].hist(ok_df["height"], bins=30)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_dir / "image_size_distributions.png", dpi=200)
    plt.close()


def show_sample_images(
    df: pd.DataFrame,
    save_dir: str | Path,
    split: str = "train",
    max_sample_images_per_class: int = 6,
    random_seed: int = 42,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    subset = df[(df["split"] == split) & (df["ok"])].copy()
    labels = [lbl for lbl in sorted(subset["label"].unique()) if lbl != "unknown"]

    if not labels:
        print("[WARNING] No valid labels found for sample plotting.")
        return

    random.seed(random_seed)
    np.random.seed(random_seed)

    n_rows = len(labels)
    n_cols = max_sample_images_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, label in enumerate(labels):
        label_subset = subset[subset["label"] == label]
        sampled = label_subset.sample(
            min(n_cols, len(label_subset)),
            random_state=random_seed,
        )

        sampled_paths = sampled["filepath"].tolist()

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(sampled_paths):
                with Image.open(sampled_paths[col_idx]) as img:
                    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
                    ax.set_title(f"{label}\n{img.mode}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"sample_images_{split}.png", dpi=200)
    plt.close()


def compute_class_weights(df: pd.DataFrame) -> Dict[str, float]:
    train_df = df[
        (df["split"] == "train") & (df["label"] != "unknown") & (df["ok"])
    ].copy()

    counts = train_df["label"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)

    if total == 0 or n_classes == 0:
        return {}

    return {cls: total / (n_classes * count) for cls, count in counts.items()}


def save_class_weights(df: pd.DataFrame, save_dir: str | Path) -> Dict[str, float]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    class_weights = compute_class_weights(df)
    with open(save_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    return class_weights


def print_class_proportions(df):
    print("\nClass proportions per split (normalized):")
    proportions = (
        df.groupby("split")["label"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    print(proportions.to_string(index=False))
    
def print_aspect_ratio_summary(df):
    if not {"width", "height"}.issubset(df.columns):
        raise ValueError("DataFrame must contain width and height.")

    df = df[df["ok"]].copy()
    df["aspect_ratio"] = df["width"] / df["height"]

    print("\nAspect ratio summary:")
    print(df["aspect_ratio"].describe())
    
def plot_aspect_ratio_distribution(df, save_dir):
    import matplotlib.pyplot as plt
    from pathlib import Path

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["ok"]].copy()
    df["aspect_ratio"] = df["width"] / df["height"]

    plt.hist(df["aspect_ratio"], bins=30)
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Width / Height")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_dir / "aspect_ratio_distribution.png", dpi=200)
    plt.close()
    
    
def plot_pixel_intensity_distribution(df, save_dir, sample_size=500):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["ok"]].copy()

    if len(df) == 0:
        print("[WARNING] No valid images.")
        return

    sample_df = df.sample(min(sample_size, len(df)), random_state=42)

    pixels = []

    for path in sample_df["filepath"]:
        with Image.open(path) as img:
            arr = np.array(img.convert("L"))  # force grayscale
            pixels.append(arr.flatten())

    pixels = np.concatenate(pixels)

    plt.hist(pixels, bins=50)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Intensity (0-255)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_dir / "pixel_intensity_distribution.png", dpi=200)
    plt.close()        