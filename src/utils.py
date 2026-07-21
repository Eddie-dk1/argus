from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from data_paths import get_scenario_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_PATHS = get_scenario_paths("synthetic")
DATA_DIR = SYNTHETIC_PATHS.root.parent
RAW_DIR = SYNTHETIC_PATHS.raw
PROCESSED_DIR = SYNTHETIC_PATHS.processed
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_v0.2.json"


def ensure_directories() -> None:
    for path in (
        SYNTHETIC_PATHS.raw,
        SYNTHETIC_PATHS.interim,
        SYNTHETIC_PATHS.processed,
        SYNTHETIC_PATHS.manifests,
        RESULTS_DIR,
        PLOTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in confusion.ravel()
    )
    negative_support = true_negative + false_positive
    positive_support = true_positive + false_negative

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_positive_rate": (
            float(false_positive / negative_support) if negative_support else None
        ),
        "confusion_matrix": confusion.tolist(),
        "support": {
            "normal": negative_support,
            "anomaly": positive_support,
            "total": int(len(y_true)),
        },
        "anomaly_prevalence": float(y_true.mean()) if len(y_true) else None,
        "roc_auc": None,
        "pr_auc": None,
    }

    has_both_classes = len(np.unique(y_true)) > 1
    if scores is not None and has_both_classes:
        scores = np.asarray(scores, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))

    return metrics


def load_experiment_config(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    payload = load_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must contain a JSON object: {config_path}")
    return payload


def select_best_threshold(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    percentiles: Sequence[float],
) -> tuple[float, float, dict[str, Any], list[dict[str, Any]]]:
    scores = np.asarray(validation_scores, dtype=float)
    labels = np.asarray(validation_labels, dtype=int)
    if not len(scores) or len(scores) != len(labels):
        raise ValueError("Validation scores and labels must be non-empty and aligned.")

    candidates: list[dict[str, Any]] = []
    for percentile in sorted(set(float(value) for value in percentiles)):
        if not 0 <= percentile <= 100:
            raise ValueError(f"Threshold percentile is outside [0, 100]: {percentile}")
        threshold = float(np.percentile(scores, percentile))
        predictions = (scores >= threshold).astype(np.int64)
        metrics = compute_classification_metrics(labels, predictions, scores)
        candidates.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                **metrics,
            }
        )

    if not candidates:
        raise ValueError("At least one threshold percentile is required.")

    def selection_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
        false_positive_rate = candidate["false_positive_rate"]
        return (
            float(candidate["f1_score"]),
            float(candidate["recall"]),
            -float(false_positive_rate if false_positive_rate is not None else 1.0),
            -float(candidate["threshold"]),
        )

    best = max(candidates, key=selection_key)
    return (
        float(best["threshold"]),
        float(best["percentile"]),
        {key: value for key, value in best.items() if key not in {"threshold", "percentile"}},
        candidates,
    )


def measure_inference(
    inference: Callable[[], np.ndarray],
    *,
    record_count: int,
    warmup_runs: int = 1,
    measured_runs: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    if record_count < 0 or warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("Invalid inference timing parameters.")

    latest: np.ndarray | None = None
    for _ in range(warmup_runs):
        latest = np.asarray(inference())

    durations: list[float] = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        latest = np.asarray(inference())
        durations.append(time.perf_counter() - started)

    if latest is None:
        latest = np.asarray(inference())
    median_seconds = float(np.median(durations))
    return latest, {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "durations_seconds": durations,
        "median_seconds": median_seconds,
        "records": int(record_count),
        "records_per_second": (
            float(record_count / median_seconds) if median_seconds > 0 else None
        ),
    }
