import os
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import yaml


DEFAULT_SPLITS = ("train", "val", "test")


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping/object.")

    return config


def _resolve_path(base_dir: Path, value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    return (base_dir / path).resolve()


def infer_label_from_path(filepath: Path) -> str:
    parent_name = filepath.parent.name.lower()
    filename = filepath.name.lower()

    if parent_name == "normal":
        return "normal"

    if parent_name == "pneumonia":
        if "bacteria" in filename:
            return "bacterial"
        if "virus" in filename:
            return "viral"
        return "unknown"

    if "normal" in parent_name or "normal" in filename:
        return "normal"
    if "bacteria" in filename:
        return "bacterial"
    if "virus" in filename:
        return "viral"

    return "unknown"


def collect_image_paths(split_path: Path) -> List[Path]:
    files: List[Path] = []

    for root, _, filenames in os.walk(split_path):
        for fname in filenames:
            path = Path(root) / fname
            if path.is_file():
                files.append(path.resolve())

    return sorted(files)


def save_metadata(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return pd.read_csv(metadata_path)


class ChestXrayDataReader:
    """
    Dataset reader (no config logic, fixed splits).
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.splits = DEFAULT_SPLITS

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ChestXrayDataReader":
        config_path = Path(config_path).resolve()
        config = load_config(config_path)
        config_dir = config_path.parent

        data_root = _resolve_path(config_dir, config.get("data_root"))
        if data_root is None:
            raise ValueError("data_root is missing in config.yaml")

        return cls(data_root=data_root)

    def get_split_path(self, split: str) -> Path:
        return self.data_root / split

    def list_split_images(self, split: str) -> List[Path]:
        split_path = self.get_split_path(split)
        if not split_path.exists():
            return []
        return collect_image_paths(split_path)

    def build_metadata_table(self, verbose: bool = True) -> pd.DataFrame:
        records = []

        for split in self.splits:
            split_path = self.get_split_path(split)

            if not split_path.exists():
                if verbose:
                    print(f"[WARNING] Split folder missing: {split_path}")
                continue

            image_paths = self.list_split_images(split)

            if verbose:
                print(f"[INFO] Found {len(image_paths)} images in split '{split}'")

            for img_path in image_paths:
                label = infer_label_from_path(img_path)
                records.append(
                    {
                        "split": split,
                        "filepath": str(img_path),
                        "filename": img_path.name,
                        "label": label,
                    }
                )

        return pd.DataFrame(records)

    def read_metadata(self, verbose: bool = True) -> pd.DataFrame:
        return self.build_metadata_table(verbose=verbose)


def get_required_config_path(config: dict[str, Any], config_path: str | Path, key: str) -> Path:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    resolved = _resolve_path(config_dir, config.get(key))
    if resolved is None:
        raise ValueError(f"{key} is missing in config.yaml")

    return resolved


def get_preprocessing_grayscale(config: dict[str, Any]) -> bool:
    preprocessing_cfg = config.get("preprocessing", {}) or {}
    return bool(preprocessing_cfg.get("grayscale", True))


def get_preprocessing_resize(config: dict[str, Any]) -> Optional[tuple[int, int]]:
    preprocessing_cfg = config.get("preprocessing", {}) or {}
    resize_cfg = preprocessing_cfg.get("resize", {}) or {}

    if not resize_cfg.get("enabled", False):
        return None

    size = resize_cfg.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Invalid resize size in config.yaml")

    return (int(size[0]), int(size[1]))