import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from constants import CLASS_NAMES, ID_TO_LABEL, LABEL_TO_ID


class ClassificationEvaluator:
    def __init__(self, model_name: str, label_to_id=None):
        self.model_name = model_name
        self.label_to_id = label_to_id or LABEL_TO_ID
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.label_ids = sorted(self.id_to_label.keys())
        self.label_names = [self.id_to_label[i] for i in self.label_ids]

    def compute_metrics(self, y_true, y_pred):
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        report = classification_report(
            y_true,
            y_pred,
            labels=self.label_ids,
            target_names=self.label_names,
            zero_division=0,
            output_dict=True,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=self.label_ids,
        )

        return {
            "primary_metric": "macro_recall",
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def print_metrics(self, y_true, y_pred, split_name: str):
        metrics = self.compute_metrics(y_true, y_pred)

        print(f"\n=== {self.model_name.upper()} | {split_name.upper()} RESULTS ===")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])
        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=self.label_ids,
                target_names=self.label_names,
                zero_division=0,
            )
        )

        return metrics

    def evaluate_split(self, y_true, y_pred, split_name: str):
        return {
            "model_name": self.model_name,
            "split": split_name,
            **self.print_metrics(y_true, y_pred, split_name),
        }

    def create_run_dir(self, base_output_dir: str | Path, experiment_name: str | None = None) -> tuple[Path, str]:
        base_output_dir = Path(base_output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if experiment_name:
            run_name = f"{experiment_name}_{timestamp}"
        else:
            run_name = f"{self.model_name}_{timestamp}"

        run_dir = base_output_dir / "experiments" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, timestamp

    def save_json(self, data: dict, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _flatten_dict(self, data: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))

        return dict(items)

    def _build_experiment_row(
        self,
        timestamp: str,
        config: dict,
        metrics_by_split: dict,
    ) -> dict:
        row = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "primary_metric": "macro_recall",
        }

        flat_config = self._flatten_dict(config)
        for key, value in flat_config.items():
            row[f"config.{key}"] = value

        for split_name, split_metrics in metrics_by_split.items():
            if "macro_recall" in split_metrics:
                row[f"{split_name}.macro_recall"] = split_metrics["macro_recall"]
            if "macro_f1" in split_metrics:
                row[f"{split_name}.macro_f1"] = split_metrics["macro_f1"]

        return row

    def append_experiment_row(
        self,
        base_output_dir: str | Path,
        row: dict,
        filename: str = "experiment_results.csv",
    ) -> Path:
        base_output_dir = Path(base_output_dir)
        csv_path = base_output_dir / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame([row])

        if csv_path.exists():
            old_df = pd.read_csv(csv_path)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(csv_path, index=False)
        return csv_path

    def save_run(
        self,
        base_output_dir: str | Path,
        config: dict,
        metrics_by_split: dict,
        experiment_name: str | None = None,
        extra_artifacts: dict | None = None,
    ) -> Path:
        run_dir, timestamp = self.create_run_dir(base_output_dir, experiment_name)

        self.save_json(config, run_dir / "config.json")
        self.save_json(metrics_by_split, run_dir / "metrics.json")

        run_summary = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "primary_metric": "macro_recall",
            "class_names": CLASS_NAMES,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "saved_splits": list(metrics_by_split.keys()),
            "run_dir": str(run_dir),
        }
        self.save_json(run_summary, run_dir / "run_summary.json")

        if extra_artifacts:
            for filename, data in extra_artifacts.items():
                self.save_json(data, run_dir / filename)

        experiment_row = self._build_experiment_row(
            timestamp=timestamp,
            config=config,
            metrics_by_split=metrics_by_split,
        )
        self.append_experiment_row(base_output_dir, experiment_row)

        return run_dir