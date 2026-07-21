from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import PLOTS_DIR
from utils import RESULTS_DIR
from utils import ensure_directories
from utils import load_json
from utils import save_json


MODEL_LABELS = {
    "random_forest": "Random Forest",
    "isolation_forest": "Isolation Forest",
    "lstm_autoencoder": "LSTM-Autoencoder",
}


def metric_row(
    model_key: str,
    payload: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    metrics = payload[split]
    return {
        "model": MODEL_LABELS[model_key],
        "split": split,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "false_positive_rate": metrics["false_positive_rate"],
        "accuracy": metrics["accuracy"],
        "normal_support": metrics["support"]["normal"],
        "anomaly_support": metrics["support"]["anomaly"],
        "anomaly_prevalence": metrics["anomaly_prevalence"],
        "threshold": payload["threshold"],
        "threshold_percentile": payload["threshold_percentile"],
    }


def plot_metric(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    *,
    y_max: float = 1.0,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(frame["model"], frame[metric], color=["#2c7fb8", "#7fcdbb", "#fdae61"])
    axis.set_ylim(0, y_max)
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, frame[metric], strict=True):
        if pd.notna(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(float(value) + 0.02 * y_max, 0.98 * y_max),
                f"{float(value):.3f}",
                ha="center",
                fontsize=9,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    ensure_directories()
    baseline_path = RESULTS_DIR / "metrics_baseline.json"
    autoencoder_path = RESULTS_DIR / "metrics_autoencoder.json"
    if not baseline_path.exists() or not autoencoder_path.exists():
        raise FileNotFoundError("Run baseline and LSTM training before evaluation")

    baseline = load_json(baseline_path)
    autoencoder = load_json(autoencoder_path)
    baseline_outputs = pd.read_csv(RESULTS_DIR / "baseline_outputs.csv")
    autoencoder_outputs = pd.read_csv(RESULTS_DIR / "autoencoder_outputs.csv")
    cohort_columns = ["user_id", "timestamp", "y_true"]
    baseline_cohort = baseline_outputs[cohort_columns].copy()
    autoencoder_cohort = autoencoder_outputs[cohort_columns].copy()
    for cohort in (baseline_cohort, autoencoder_cohort):
        cohort["timestamp"] = pd.to_datetime(cohort["timestamp"])
        cohort.sort_values(["user_id", "timestamp"], inplace=True, ignore_index=True)
    if not baseline_cohort.equals(autoencoder_cohort):
        raise ValueError(
            "Baseline and LSTM evaluation cohorts differ; cross-model comparison is invalid"
        )
    models = {
        "random_forest": baseline["random_forest"],
        "isolation_forest": baseline["isolation_forest"],
        "lstm_autoencoder": autoencoder["lstm_autoencoder"],
    }
    validation = pd.DataFrame(
        [metric_row(key, payload, split="validation") for key, payload in models.items()]
    )
    test = pd.DataFrame(
        [metric_row(key, payload, split="test") for key, payload in models.items()]
    )
    validation.to_csv(RESULTS_DIR / "metrics_validation_summary.csv", index=False)
    test.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    plot_metric(test, "f1", "Test F1 by Model", PLOTS_DIR / "f1_score_comparison.png")
    plot_metric(test, "roc_auc", "Test ROC-AUC by Model", PLOTS_DIR / "roc_auc_comparison.png")
    plot_metric(test, "pr_auc", "Test PR-AUC by Model", PLOTS_DIR / "pr_auc_comparison.png")
    plot_metric(
        test,
        "false_positive_rate",
        "Test False-Positive Rate by Model",
        PLOTS_DIR / "false_positive_rate_comparison.png",
        y_max=max(0.1, float(test["false_positive_rate"].max()) * 1.25),
    )

    best_validation = validation.sort_values(
        ["f1", "recall", "false_positive_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    best_test = test.sort_values("f1", ascending=False).iloc[0]
    save_json(
        {
            "protocol_version": "1",
            "evaluation_cohort": "sequence_endpoints",
            "evaluation_cohort_rows": int(len(baseline_cohort)),
            "evaluation_cohort_verified_equal": True,
            "selection_rule": "validation F1, recall, then lower false-positive rate",
            "selected_model_by_validation": best_validation["model"],
            "selected_model_validation_f1": float(best_validation["f1"]),
            "descriptive_best_test_model": best_test["model"],
            "descriptive_best_test_f1": float(best_test["f1"]),
            "warning": (
                "The descriptive test ranking is reported after evaluation and must not "
                "be used to retune the current release."
            ),
        },
        RESULTS_DIR / "evaluation_summary.json",
    )

    print("Validation metrics:")
    print(validation.to_string(index=False))
    print("\nTest metrics:")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
