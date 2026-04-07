import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constants import ID_TO_LABEL, LABEL_TO_ID
from data.data_reader import get_required_config_path, load_config, load_metadata
from evaluation.metrics import ClassificationEvaluator


CONFIG_PATH = Path("config.yaml")


def get_lr_config(config: dict) -> dict:
    if "logistic_regression" not in config:
        raise ValueError("Missing 'logistic_regression' section in config.yaml")

    lr_cfg = config["logistic_regression"]

    required_keys = [
        "output_dir",
        "experiment_name",
        "max_iter",
        "class_weight",
        "pca_n_components",
        "random_state",
        "solver",
    ]

    missing = [k for k in required_keys if k not in lr_cfg]
    if missing:
        raise ValueError(f"Missing keys in logistic_regression config: {missing}")

    return {
        "output_dir": lr_cfg["output_dir"],
        "experiment_name": lr_cfg["experiment_name"],
        "max_iter": int(lr_cfg["max_iter"]),
        "class_weight": lr_cfg["class_weight"],
        "pca_n_components": lr_cfg["pca_n_components"],
        "random_state": int(lr_cfg["random_state"]),
        "solver": lr_cfg["solver"],
    }


def load_preprocessed_metadata(config: dict) -> pd.DataFrame:
    preprocessed_metadata_path = get_required_config_path(
        config,
        CONFIG_PATH,
        "preprocessed_metadata_output_path",
    )
    return load_metadata(preprocessed_metadata_path)


def filter_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"split", "filepath", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean_df = df.copy()
    clean_df = clean_df[clean_df["label"].isin(LABEL_TO_ID.keys())].copy()
    clean_df = clean_df[clean_df["filepath"].notna()].copy()
    clean_df["filepath"] = clean_df["filepath"].astype(str)
    clean_df = clean_df[clean_df["filepath"].map(lambda p: Path(p).exists())].copy()

    if len(clean_df) == 0:
        raise ValueError("No usable rows found after filtering known labels and valid filepaths.")

    return clean_df


def load_flattened_images(df: pd.DataFrame) -> np.ndarray:
    features = []

    for i, path in enumerate(df["filepath"].tolist()):
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32)

            if arr.ndim != 2:
                raise ValueError(
                    f"Expected preprocessed grayscale image, but got shape {arr.shape} for {path}"
                )

            arr = arr / 255.0
            features.append(arr.flatten())

        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"[INFO] Loaded {i + 1}/{len(df)} images")

    return np.vstack(features)


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)


def build_pipeline(lr_cfg: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=lr_cfg["pca_n_components"],
                    random_state=lr_cfg["random_state"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=lr_cfg["max_iter"],
                    class_weight=lr_cfg["class_weight"],
                    random_state=lr_cfg["random_state"],
                    solver=lr_cfg["solver"],
                    multi_class="auto",
                ),
            ),
        ]
    )


def main() -> None:
    config = load_config(CONFIG_PATH)
    lr_cfg = get_lr_config(config)
    evaluator = ClassificationEvaluator(model_name="logistic_regression")

    output_dir = get_required_config_path(
        {"output_dir": lr_cfg["output_dir"]},
        CONFIG_PATH,
        "output_dir",
    )

    df = load_preprocessed_metadata(config)
    df = filter_model_rows(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("Train split contains no usable rows.")
    if len(test_df) == 0:
        raise ValueError("Test split contains no usable rows.")

    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Val rows: {len(val_df)}")
    print(f"[INFO] Test rows: {len(test_df)}")

    X_train = load_flattened_images(train_df)
    y_train = encode_labels(train_df)

    X_val = load_flattened_images(val_df) if len(val_df) > 0 else None
    y_val = encode_labels(val_df) if len(val_df) > 0 else None

    X_test = load_flattened_images(test_df)
    y_test = encode_labels(test_df)

    pipeline = build_pipeline(lr_cfg)
    pipeline.fit(X_train, y_train)

    metrics = {
        "train": evaluator.evaluate_split(y_train, pipeline.predict(X_train), "train"),
        "test": evaluator.evaluate_split(y_test, pipeline.predict(X_test), "test"),
    }

    if X_val is not None and y_val is not None and len(y_val) > 0:
        metrics["val"] = evaluator.evaluate_split(y_val, pipeline.predict(X_val), "val")

    pca = pipeline.named_steps["pca"]
    clf = pipeline.named_steps["clf"]

    extra_artifacts = {
        "pca_info.json": {
            "n_components_": int(pca.n_components_) if hasattr(pca, "n_components_") else None,
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "classes_": [ID_TO_LABEL[int(c)] for c in clf.classes_],
        },
        "dataset_summary.json": {
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "labels": LABEL_TO_ID,
        },
    }

    run_dir = evaluator.save_run(
        base_output_dir=output_dir,
        config=lr_cfg,
        metrics_by_split=metrics,
        experiment_name=lr_cfg["experiment_name"],
        extra_artifacts=extra_artifacts,
    )

    joblib.dump(pipeline, run_dir / "logistic_regression_pipeline.joblib")

    print(f"\n[INFO] Saved run to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()