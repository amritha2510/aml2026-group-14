import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_reader import get_required_config_path, load_config, load_metadata


CONFIG_PATH = Path("config.yaml")

LABEL_TO_ID = {
    "normal": 0,
    "bacterial": 1,
    "viral": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def get_lr_config(config: dict) -> dict:
    lr_cfg = config.get("logistic_regression", {}) or {}
    if not isinstance(lr_cfg, dict):
        raise ValueError("'logistic_regression' must be a YAML mapping/object.")

    return {
        "output_dir": lr_cfg.get("output_dir", "./outputs/logistic_regression"),
        "max_iter": int(lr_cfg.get("max_iter", 1000)),
        "class_weight": lr_cfg.get("class_weight", "balanced"),
        "pca_n_components": lr_cfg.get("pca_n_components", 0.95),
        "random_state": int(lr_cfg.get("random_state", 42)),
        "solver": lr_cfg.get("solver", "lbfgs"),
    }


def load_preprocessed_metadata_from_config(config: dict) -> pd.DataFrame:
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

        if i % 500 == 0:
            print(f"[INFO] Loaded {i + 1}/{len(df)} images")

    return np.vstack(features)


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)


def build_pipeline(lr_cfg: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(
                n_components=lr_cfg["pca_n_components"],
                random_state=lr_cfg["random_state"],
            )),
            ("clf", LogisticRegression(
                max_iter=lr_cfg["max_iter"],
                class_weight=lr_cfg["class_weight"],
                random_state=lr_cfg["random_state"],
                solver=lr_cfg["solver"],
                multi_class="auto",
            )),
        ]
    )


def evaluate_split(model: Pipeline, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
    y_pred = model.predict(X)

    macro_recall = recall_score(y, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y,
        y_pred,
        target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL.keys())],
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y, y_pred, labels=sorted(ID_TO_LABEL.keys()))

    print(f"\n=== {split_name.upper()} RESULTS ===")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL.keys())],
            zero_division=0,
        )
    )

    return {
        "split": split_name,
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_results(
    output_dir: Path,
    pipeline: Pipeline,
    metrics: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_dir / "logistic_regression_pipeline.joblib")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "labels": LABEL_TO_ID,
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pca = pipeline.named_steps["pca"]
    clf = pipeline.named_steps["clf"]

    pca_info = {
        "n_components_": int(pca.n_components_) if hasattr(pca, "n_components_") else None,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "classes_": [ID_TO_LABEL[int(c)] for c in clf.classes_],
    }

    with open(output_dir / "pca_info.json", "w", encoding="utf-8") as f:
        json.dump(pca_info, f, indent=2)

    print(f"\n[INFO] Saved Logistic Regression outputs to: {output_dir.resolve()}")


def main() -> None:
    config = load_config(CONFIG_PATH)
    lr_cfg = get_lr_config(config)

    output_dir = get_required_config_path(
        {"output_dir": lr_cfg["output_dir"]},
        CONFIG_PATH,
        "output_dir",
    )

    df = load_preprocessed_metadata_from_config(config)
    df = filter_model_rows(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train and test splits must both contain usable rows.")

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
        "config": lr_cfg,
        "train": evaluate_split(pipeline, X_train, y_train, "train"),
        "test": evaluate_split(pipeline, X_test, y_test, "test"),
    }

    if X_val is not None and y_val is not None and len(y_val) > 0:
        metrics["val"] = evaluate_split(pipeline, X_val, y_val, "val")

    save_results(
        output_dir=output_dir,
        pipeline=pipeline,
        metrics=metrics,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()