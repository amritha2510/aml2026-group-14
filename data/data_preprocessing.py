from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError


def filter_valid_images(df: pd.DataFrame) -> pd.DataFrame:
    if "ok" not in df.columns:
        raise ValueError("DataFrame must contain an 'ok' column.")
    return df[df["ok"]].copy()


def filter_known_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")
    return df[df["label"] != "unknown"].copy()


def get_clean_training_data(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    required_cols = {"split", "label", "ok"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    return df[(df["split"] == split) & (df["label"] != "unknown") & (df["ok"])].copy()


def preprocess_and_save_image(
    src_path: str | Path,
    dst_path: str | Path,
    convert_to_grayscale: bool = True,
    resize_to: Optional[tuple[int, int]] = None,
) -> None:
    """
    Actual preprocessing:
    - opens source image
    - converts to grayscale if requested
    - resizes if requested
    - saves the transformed image
    """
    src_path = Path(src_path).resolve()
    dst_path = Path(dst_path).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        if convert_to_grayscale and img.mode != "L":
            img = img.convert("L")

        if resize_to is not None:
            width, height = resize_to
            img = img.resize((width, height))

        img.save(dst_path)


def build_preprocessed_filepath(
    original_filepath: str | Path,
    source_root: str | Path,
    target_root: str | Path,
) -> Path:
    """
    Mirror the original directory structure under the preprocessing output root.
    """
    original_filepath = Path(original_filepath).resolve()
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()

    rel_path = original_filepath.relative_to(source_root)
    return target_root / rel_path


def preprocess_dataset(
    metadata_df: pd.DataFrame,
    source_root: str | Path,
    target_root: str | Path,
    convert_to_grayscale: bool = True,
    resize_to: Optional[tuple[int, int]] = None,
    progress_every: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Self-contained preprocessing runner.

    Input:
        metadata_df from the data reader only.

    Output:
        new metadata table pointing to transformed files.
    """
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()

    rows = []

    for i, row in metadata_df.iterrows():
        if verbose and i % progress_every == 0:
            print(f"[INFO] Preprocessing image {i + 1}/{len(metadata_df)}")

        src_path = Path(row["filepath"])
        if not src_path.is_absolute():
            src_path = src_path.resolve()

        try:
            dst_path = build_preprocessed_filepath(
                original_filepath=src_path,
                source_root=source_root,
                target_root=target_root,
            )

            preprocess_and_save_image(
                src_path=src_path,
                dst_path=dst_path,
                convert_to_grayscale=convert_to_grayscale,
                resize_to=resize_to,
            )

            new_row = row.to_dict()
            new_row["original_filepath"] = row["filepath"]
            new_row["filepath"] = str(dst_path)
            new_row["is_preprocessed"] = True
            new_row["preprocessed_to_grayscale"] = bool(convert_to_grayscale)
            new_row["resize_width"] = resize_to[0] if resize_to is not None else None
            new_row["resize_height"] = resize_to[1] if resize_to is not None else None
            new_row["preprocessing_error"] = None
            rows.append(new_row)

        except (UnidentifiedImageError, OSError, ValueError) as e:
            new_row = row.to_dict()
            new_row["original_filepath"] = row["filepath"]
            new_row["filepath"] = None
            new_row["is_preprocessed"] = False
            new_row["preprocessed_to_grayscale"] = bool(convert_to_grayscale)
            new_row["resize_width"] = resize_to[0] if resize_to is not None else None
            new_row["resize_height"] = resize_to[1] if resize_to is not None else None
            new_row["preprocessing_error"] = str(e)
            rows.append(new_row)

    return pd.DataFrame(rows)


def save_preprocessed_metadata(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)