from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix

from experiment_tracking import create_run_directory
from experiment_tracking import finalize_run_artifacts
from experiment_tracking import save_run_results
import joblib
from utils import DEFAULT_CONFIG_PATH
from utils import PLOTS_DIR
from utils import PROCESSED_DIR
from utils import RAW_DIR
from utils import RESULTS_DIR
from utils import compute_classification_metrics
from utils import ensure_directories
from utils import load_experiment_config
from utils import load_json
from utils import measure_inference
from utils import save_json
from utils import select_best_threshold
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic tabular baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["Normal", "Anomaly"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc(
    y_true: np.ndarray,
    random_forest_scores: np.ndarray,
    isolation_forest_scores: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_true,
        random_forest_scores,
        name="Random Forest",
        ax=axis,
    )
    RocCurveDisplay.from_predictions(
        y_true,
        isolation_forest_scores,
        name="Isolation Forest",
        ax=axis,
    )
    axis.set_title("ROC Curve: Synthetic Baselines")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def flatten_candidates(model: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "percentile",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
        "accuracy",
        "anomaly_prevalence",
    )
    return [{"model": model, **{key: row.get(key) for key in keys}} for row in candidates]


def build_error_analysis(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    result = meta.reset_index(drop=True).copy()
    result["y_true"] = y_true
    result["prediction"] = predictions
    result["score"] = scores
    result["error_type"] = np.select(
        [
            (result["y_true"] == 0) & (result["prediction"] == 1),
            (result["y_true"] == 1) & (result["prediction"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    return result[result["error_type"] != "correct"].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    seed = int(config["seed"])
    baseline_config = config["baseline"]
    timing_config = config["timing"]
    ensure_directories()
    set_seed(seed)

    dataset_path = PROCESSED_DIR / "baseline_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Baseline dataset not found. Run `python3 src/preprocessing.py` first."
        )
    frame = pd.read_csv(dataset_path)
    feature_columns = list(load_json(PROCESSED_DIR / "feature_columns.json"))
    feature_config = load_json(PROCESSED_DIR / "feature_config.json")
    sequence_length = int(feature_config["sequence_length"])
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError("Processed feature columns are missing: " + ", ".join(missing))

    split_frames = {
        name: frame[frame["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    if any(split.empty for split in split_frames.values()):
        raise ValueError("Processed dataset must contain train, validation, and test rows")

    X_train = split_frames["train"][feature_columns]
    y_train = split_frames["train"]["label"].astype(int).to_numpy()
    evaluation_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("validation", "test"):
        split_frame = split_frames[split_name]
        endpoint_mask = (
            split_frame.sort_values(["user_id", "timestamp"])
            .groupby("user_id", sort=False)
            .cumcount()
            .ge(sequence_length - 1)
        )
        evaluation_frames[split_name] = split_frame.loc[endpoint_mask].reset_index(drop=True)

    X_validation = evaluation_frames["validation"][feature_columns]
    y_validation = evaluation_frames["validation"]["label"].astype(int).to_numpy()
    X_test = evaluation_frames["test"][feature_columns]
    y_test = evaluation_frames["test"]["label"].astype(int).to_numpy()

    rf_config = baseline_config["random_forest"]
    random_forest = RandomForestClassifier(
        n_estimators=int(rf_config["n_estimators"]),
        max_depth=int(rf_config["max_depth"]),
        class_weight=str(rf_config["class_weight"]),
        random_state=seed,
        n_jobs=-1,
    )
    started = time.perf_counter()
    random_forest.fit(X_train, y_train)
    rf_training_seconds = time.perf_counter() - started
    rf_validation_scores = random_forest.predict_proba(X_validation)[:, 1]
    rf_threshold, rf_percentile, rf_validation_metrics, rf_candidates = select_best_threshold(
        rf_validation_scores,
        y_validation,
        baseline_config["threshold_percentiles"],
    )
    rf_test_scores, rf_timing = measure_inference(
        lambda: random_forest.predict_proba(X_test)[:, 1],
        record_count=len(X_test),
        warmup_runs=int(timing_config["warmup_runs"]),
        measured_runs=int(timing_config["measured_runs"]),
    )
    rf_test_predictions = (rf_test_scores >= rf_threshold).astype(int)
    rf_test_metrics = compute_classification_metrics(
        y_test,
        rf_test_predictions,
        rf_test_scores,
    )

    isolation_config = baseline_config["isolation_forest"]
    isolation_forest = IsolationForest(
        n_estimators=int(isolation_config["n_estimators"]),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    X_train_normal = X_train[y_train == 0]
    started = time.perf_counter()
    isolation_forest.fit(X_train_normal)
    if_training_seconds = time.perf_counter() - started
    if_validation_scores = -isolation_forest.score_samples(X_validation)
    if_threshold, if_percentile, if_validation_metrics, if_candidates = select_best_threshold(
        if_validation_scores,
        y_validation,
        baseline_config["threshold_percentiles"],
    )
    if_test_scores, if_timing = measure_inference(
        lambda: -isolation_forest.score_samples(X_test),
        record_count=len(X_test),
        warmup_runs=int(timing_config["warmup_runs"]),
        measured_runs=int(timing_config["measured_runs"]),
    )
    if_test_predictions = (if_test_scores >= if_threshold).astype(int)
    if_test_metrics = compute_classification_metrics(
        y_test,
        if_test_predictions,
        if_test_scores,
    )

    rf_plot = PLOTS_DIR / "confusion_matrix_random_forest.png"
    if_plot = PLOTS_DIR / "confusion_matrix_isolation_forest.png"
    roc_plot = PLOTS_DIR / "roc_curve_baseline.png"
    plot_confusion(y_test, rf_test_predictions, "Random Forest: Test", rf_plot)
    plot_confusion(y_test, if_test_predictions, "Isolation Forest: Test", if_plot)
    plot_roc(y_test, rf_test_scores, if_test_scores, roc_plot)

    metrics_payload = {
        "protocol_version": "1",
        "split_policy": "synthetic-chronological-60-20-20",
        "evaluation_cohort": "sequence_endpoints",
        "sequence_length": sequence_length,
        "random_forest": {
            "threshold": rf_threshold,
            "threshold_percentile": rf_percentile,
            "threshold_calibration": "labeled-validation",
            "validation": rf_validation_metrics,
            "test": rf_test_metrics,
            "hyperparameters": rf_config,
        },
        "isolation_forest": {
            "threshold": if_threshold,
            "threshold_percentile": if_percentile,
            "threshold_calibration": "labeled-validation",
            "validation": if_validation_metrics,
            "test": if_test_metrics,
            "hyperparameters": isolation_config,
            "fit_rows": int(len(X_train_normal)),
        },
    }
    save_json(metrics_payload, RESULTS_DIR / "metrics_baseline.json")

    meta_columns = [
        "timestamp",
        "user_id",
        "source_ip",
        "destination_ip",
        "event_type",
        "anomaly_type",
        "bytes_sent",
        "session_duration",
        "failed_logins",
        "distinct_destinations",
        "privileged_action",
        "login_status",
        "location",
    ]
    test_meta = evaluation_frames["test"][meta_columns]
    outputs = test_meta.copy()
    outputs["y_true"] = y_test
    outputs["rf_score"] = rf_test_scores
    outputs["rf_pred"] = rf_test_predictions
    outputs["iforest_score"] = if_test_scores
    outputs["iforest_pred"] = if_test_predictions
    outputs.to_csv(RESULTS_DIR / "baseline_outputs.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "feature_importance_random_forest.csv", index=False)

    threshold_rows = [
        *flatten_candidates("Random Forest", rf_candidates),
        *flatten_candidates("Isolation Forest", if_candidates),
    ]
    pd.DataFrame(threshold_rows).to_csv(
        RESULTS_DIR / "baseline_validation_thresholds.csv",
        index=False,
    )
    timing_payload = {
        "random_forest": {"training_seconds": rf_training_seconds, **rf_timing},
        "isolation_forest": {"training_seconds": if_training_seconds, **if_timing},
    }
    save_json(timing_payload, RESULTS_DIR / "timing_baseline.json")
    save_json(
        {
            "seed": seed,
            "feature_count": len(feature_columns),
            "split_sizes": {name: len(value) for name, value in split_frames.items()},
            "evaluation_cohort": "sequence_endpoints",
            "sequence_length": sequence_length,
            "validation_evaluation_rows": len(evaluation_frames["validation"]),
            "test_evaluation_rows": len(evaluation_frames["test"]),
            "threshold_selection": "validation-percentile-search",
            "threshold_percentiles": baseline_config["threshold_percentiles"],
            "random_forest_threshold": rf_threshold,
            "isolation_forest_threshold": if_threshold,
        },
        RESULTS_DIR / "baseline_training_summary.json",
    )

    run_inputs = {
        "scenario": "synthetic",
        "config_version": config["config_version"],
        "seed": seed,
        "split_policy": metrics_payload["split_policy"],
    }
    for model_name, model_config, validation_metrics, test_metrics, candidates, timing, errors, plots in (
        (
            "random_forest",
            {**run_inputs, "model": rf_config, "threshold_percentiles": baseline_config["threshold_percentiles"]},
            rf_validation_metrics,
            rf_test_metrics,
            rf_candidates,
            timing_payload["random_forest"],
            build_error_analysis(test_meta, y_test, rf_test_predictions, rf_test_scores),
            (rf_plot, roc_plot),
        ),
        (
            "isolation_forest",
            {**run_inputs, "model": isolation_config, "threshold_percentiles": baseline_config["threshold_percentiles"]},
            if_validation_metrics,
            if_test_metrics,
            if_candidates,
            timing_payload["isolation_forest"],
            build_error_analysis(test_meta, y_test, if_test_predictions, if_test_scores),
            (if_plot, roc_plot),
        ),
    ):
        run_dir, _ = create_run_directory(
            scenario="synthetic",
            model=model_name,
            resolved_config=model_config,
            seed=seed,
            data_path=RAW_DIR / "synthetic_logs.csv",
            split_report_path=PROCESSED_DIR / "split_report.json",
            feature_manifest_path=PROCESSED_DIR / "feature_manifest.json",
        )
        save_run_results(
            run_dir,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            threshold_candidates=candidates,
            timing=timing,
            error_analysis=errors,
            plot_paths=plots,
        )

        fitted_model = random_forest if model_name == "random_forest" else isolation_forest
        finalize_run_artifacts(
            run_dir, predictions=outputs,
            model_writer=lambda path: joblib.dump(fitted_model, path),
            model_filename="model.joblib",
            inference_contract={
                "model": model_name, "feature_columns": feature_columns,
                "threshold": rf_threshold if model_name == "random_forest" else if_threshold,
                "score": "predict_proba_class_1" if model_name == "random_forest" else "negative_score_samples",
                "classes": random_forest.classes_.tolist() if model_name == "random_forest" else None,
                "decision": "score >= threshold",
            },
            feature_state_path=PROCESSED_DIR / "feature_state.json",
        )

    print(f"Saved baseline metrics to: {RESULTS_DIR / 'metrics_baseline.json'}")
    print(f"Random Forest test F1: {rf_test_metrics['f1_score']:.4f}")
    print(f"Isolation Forest test F1: {if_test_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
