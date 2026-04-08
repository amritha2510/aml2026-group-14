import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import itertools
import json
import joblib
import matplotlib.pyplot as plt
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
        "regularization_strength",
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
        "regularization_strength": lr_cfg["regularization_strength"],
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


def infer_actual_input_size(df: pd.DataFrame) -> dict:
    """
    Read the actual input size from a real preprocessed image file.
    This is the source of truth for what the model consumed.
    """
    if len(df) == 0:
        raise ValueError("Cannot infer input size from an empty DataFrame.")

    sample_path = Path(df.iloc[0]["filepath"])
    if not sample_path.exists():
        raise FileNotFoundError(f"Cannot infer input size because file does not exist: {sample_path}")

    with Image.open(sample_path) as img:
        arr = np.array(img)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D image when inferring input size, but got shape {arr.shape} "
            f"for {sample_path}"
        )

    height, width = arr.shape
    return {
        "input_width": int(width),
        "input_height": int(height),
        "input_shape": [int(height), int(width)],
        "n_input_features": int(height * width),
    }


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
    steps = [("scaler", StandardScaler())]

    pca_n_components = lr_cfg["pca_n_components"]
    use_pca = pca_n_components not in [None, 0, "0", "none", "None"]

    if use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_n_components,
                    random_state=lr_cfg["random_state"],
                ),
            )
        )

    steps.append(
        (
            "clf",
            LogisticRegression(
                C=float(lr_cfg["regularization_strength"]),
                max_iter=lr_cfg["max_iter"],
                class_weight=lr_cfg["class_weight"],
                random_state=lr_cfg["random_state"],
                solver=lr_cfg["solver"],
                multi_class="auto",
            ),
        )
    )

    return Pipeline(steps=steps)


def get_search_space(lr_cfg: dict) -> list[dict]:
    pca_values = lr_cfg["pca_n_components"]
    reg_values = lr_cfg["regularization_strength"]

    if not isinstance(pca_values, list):
        pca_values = [pca_values]
    if not isinstance(reg_values, list):
        reg_values = [reg_values]

    configs = []
    for pca_n, reg_c in itertools.product(pca_values, reg_values):
        cfg = lr_cfg.copy()
        cfg["pca_n_components"] = pca_n
        cfg["regularization_strength"] = float(reg_c)
        configs.append(cfg)

    return configs


def save_model_selection_outputs(
    search_results: list[dict],
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not search_results:
        raise ValueError("No search results to save.")

    results_df = pd.DataFrame(search_results).copy()

    results_csv_path = output_dir / "model_selection_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    ranked_df = results_df.sort_values(
        by=["val_macro_recall", "val_macro_f1", "train_macro_recall", "pca_n_components", "regularization_strength"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    ranked_df.insert(0, "rank", np.arange(1, len(ranked_df) + 1))

    ranked_csv_path = output_dir / "model_selection_results_ranked.csv"
    ranked_df.to_csv(ranked_csv_path, index=False)

    best_row = ranked_df.iloc[0].to_dict()
    best_json_path = output_dir / "best_config_from_model_selection.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    if len(ranked_df) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(ranked_df["rank"], ranked_df["val_macro_recall"], marker="o")
        plt.xlabel("Candidate rank")
        plt.ylabel("Validation macro recall")
        plt.title("Validation Macro Recall by Ranked Candidate")
        plt.tight_layout()
        plt.savefig(output_dir / "model_selection_rank_vs_val_macro_recall.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        for reg_strength in sorted(ranked_df["regularization_strength"].unique()):
            subset = ranked_df[ranked_df["regularization_strength"] == reg_strength].sort_values("candidate_index")
            plt.plot(
                subset["candidate_index"],
                subset["val_macro_recall"],
                marker="o",
                label=f"C={reg_strength}",
            )
        plt.xlabel("Candidate index")
        plt.ylabel("Validation macro recall")
        plt.title("Validation Macro Recall Across Candidates")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "model_selection_candidates_vs_val_macro_recall.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(ranked_df["rank"], ranked_df["val_macro_f1"], marker="o")
        plt.xlabel("Candidate rank")
        plt.ylabel("Validation macro F1")
        plt.title("Validation Macro F1 by Ranked Candidate")
        plt.tight_layout()
        plt.savefig(output_dir / "model_selection_rank_vs_val_macro_f1.png", dpi=200)
        plt.close()

    return {
        "results_csv_path": str(results_csv_path),
        "ranked_csv_path": str(ranked_csv_path),
        "best_json_path": str(best_json_path),
        "best_row": best_row,
    }


def select_best_model(
    candidate_configs: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    evaluator: ClassificationEvaluator,
) -> tuple[Pipeline, dict, dict, list[dict]]:
    if X_val is None or y_val is None or len(y_val) == 0:
        best_cfg = candidate_configs[0]
        best_pipeline = build_pipeline(best_cfg)
        best_pipeline.fit(X_train, y_train)

        train_metrics = evaluator.evaluate_split(y_train, best_pipeline.predict(X_train), "train")
        return best_pipeline, best_cfg, {"train": train_metrics}, []

    best_pipeline = None
    best_cfg = None
    best_val_metrics = None
    best_score = -np.inf
    search_results = []

    for i, cfg in enumerate(candidate_configs, start=1):
        pca_label = cfg["pca_n_components"]
        if pca_label in [None, 0, "0", "none", "None"]:
            pca_label = "disabled"

        print(
            f"[INFO] Candidate {i}/{len(candidate_configs)} | "
            f"PCA={pca_label} | C={cfg['regularization_strength']}"
        )

        pipeline = build_pipeline(cfg)
        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)

        train_metrics = evaluator.evaluate_split(y_train, train_pred, "train")
        val_metrics = evaluator.evaluate_split(y_val, val_pred, "val")

        score = val_metrics["macro_recall"]

        result_row = {
            "candidate_index": i,
            "pca_n_components": cfg["pca_n_components"],
            "regularization_strength": float(cfg["regularization_strength"]),
            "train_macro_recall": float(train_metrics["macro_recall"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        search_results.append(result_row)

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_cfg = cfg
            best_val_metrics = val_metrics

    final_train_metrics = evaluator.evaluate_split(y_train, best_pipeline.predict(X_train), "train")
    return best_pipeline, best_cfg, {"train": final_train_metrics, "val": best_val_metrics}, search_results


def save_prediction_examples(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    n_correct_per_class: int = 5,
    n_wrong_per_pair: int = 5,
) -> None:
    import shutil

    output_dir = Path(output_dir)
    correct_dir = output_dir / "samples" / "correct"
    wrong_dir = output_dir / "samples" / "incorrect"

    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = df.copy().reset_index(drop=True)
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = y_pred
    analysis_df["true_label"] = analysis_df["y_true"].map(ID_TO_LABEL)
    analysis_df["pred_label"] = analysis_df["y_pred"].map(ID_TO_LABEL)
    analysis_df["is_correct"] = analysis_df["y_true"] == analysis_df["y_pred"]

    for class_id, class_name in ID_TO_LABEL.items():
        class_correct = analysis_df[
            (analysis_df["y_true"] == class_id) & (analysis_df["is_correct"])
        ].head(n_correct_per_class)

        for i, row in enumerate(class_correct.itertuples(index=False), start=1):
            src = Path(row.filepath)
            dst = correct_dir / f"{class_name}_{i}_{src.name}"
            shutil.copy2(src, dst)

    wrong_df = analysis_df[~analysis_df["is_correct"]].copy()

    for true_id, true_name in ID_TO_LABEL.items():
        for pred_id, pred_name in ID_TO_LABEL.items():
            if true_id == pred_id:
                continue

            pair_df = wrong_df[
                (wrong_df["y_true"] == true_id) & (wrong_df["y_pred"] == pred_id)
            ].head(n_wrong_per_pair)

            for i, row in enumerate(pair_df.itertuples(index=False), start=1):
                src = Path(row.filepath)
                dst = wrong_dir / f"{true_name}_as_{pred_name}_{i}_{src.name}"
                shutil.copy2(src, dst)

    summary = analysis_df[["filepath", "true_label", "pred_label", "is_correct"]].copy()
    summary.to_csv(output_dir / "samples" / "prediction_examples.csv", index=False)

    print(f"[INFO] Saved prediction examples to: {output_dir / 'samples'}")


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

    run_metadata = infer_actual_input_size(train_df)
    print(
        f"[INFO] Actual model input size: "
        f"{run_metadata['input_height']}x{run_metadata['input_width']} "
        f"({run_metadata['n_input_features']} features)"
    )

    X_train = load_flattened_images(train_df)
    y_train = encode_labels(train_df)

    X_val = load_flattened_images(val_df) if len(val_df) > 0 else None
    y_val = encode_labels(val_df) if len(val_df) > 0 else None

    X_test = load_flattened_images(test_df)
    y_test = encode_labels(test_df)

    if X_train.shape[1] != run_metadata["n_input_features"]:
        raise ValueError(
            f"Mismatch between inferred input features ({run_metadata['n_input_features']}) "
            f"and loaded training feature width ({X_train.shape[1]})."
        )

    candidate_configs = get_search_space(lr_cfg)

    best_pipeline, best_cfg, metrics, search_results = select_best_model(
        candidate_configs=candidate_configs,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        evaluator=evaluator,
    )

    selection_artifacts = {}
    if search_results:
        selection_output_dir = Path(output_dir) / "model_selection"
        selection_artifacts = save_model_selection_outputs(
            search_results=search_results,
            output_dir=selection_output_dir,
        )

        print("\n[INFO] Top ranked validation configs:")
        ranked_preview = (
            pd.DataFrame(search_results)
            .sort_values(
                by=["val_macro_recall", "val_macro_f1", "train_macro_recall", "pca_n_components", "regularization_strength"],
                ascending=[False, False, False, True, True],
            )
            .head(10)
        )
        print(ranked_preview.to_string(index=False))

        print("\n[INFO] Best config selected from validation:")
        print(json.dumps(selection_artifacts["best_row"], indent=2))
    else:
        print("[INFO] No validation split available; model selection search was skipped.")

    y_test_pred = best_pipeline.predict(X_test)
    metrics["test"] = evaluator.evaluate_split(y_test, y_test_pred, "test")

    pca = best_pipeline.named_steps.get("pca")
    clf = best_pipeline.named_steps["clf"]

    extra_artifacts = {
        "pca_info.json": {
            "pca_enabled": pca is not None,
            "n_components_": int(pca.n_components_) if pca is not None and hasattr(pca, "n_components_") else None,
            "explained_variance_ratio_sum": (
                float(np.sum(pca.explained_variance_ratio_))
                if pca is not None and hasattr(pca, "explained_variance_ratio_")
                else None
            ),
            "classes_": [ID_TO_LABEL[int(c)] for c in clf.classes_],
        },
        "dataset_summary.json": {
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "labels": LABEL_TO_ID,
        },
    }

    if search_results:
        extra_artifacts["model_selection_summary.json"] = {
            "n_candidates": int(len(search_results)),
            "selection_metric": "val_macro_recall",
            "best_candidate": selection_artifacts["best_row"],
            "results_csv_path": selection_artifacts["results_csv_path"],
            "ranked_csv_path": selection_artifacts["ranked_csv_path"],
        }

    run_dir = evaluator.save_run(
        base_output_dir=output_dir,
        config=best_cfg,
        metrics_by_split=metrics,
        experiment_name=best_cfg["experiment_name"],
        extra_artifacts=extra_artifacts,
        run_metadata=run_metadata,
    )

    save_prediction_examples(
        df=test_df,
        y_true=y_test,
        y_pred=y_test_pred,
        output_dir=run_dir,
        n_correct_per_class=5,
        n_wrong_per_pair=5,
    )

    joblib.dump(best_pipeline, run_dir / "logistic_regression_pipeline.joblib")

    print(f"\n[INFO] Saved run to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()