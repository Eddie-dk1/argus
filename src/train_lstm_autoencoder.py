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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is not installed. Run `pip install -r requirements.txt` before training."
    ) from error

from experiment_tracking import create_run_directory
from experiment_tracking import finalize_run_artifacts
from experiment_tracking import save_run_results
from features import build_sequences
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


DEFAULT_HIDDEN_SIZE = 32
DEFAULT_BATCH_SIZE = 128
BASELINE_REFERENCE_DIR = RESULTS_DIR / "baseline_reference"


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.encoder(inputs)
        repeated_hidden = hidden_state[-1].unsqueeze(1).repeat(1, inputs.size(1), 1)
        decoded, _ = self.decoder(repeated_hidden)
        return self.output_layer(decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the synthetic LSTM Autoencoder.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threshold-percentiles", nargs="+", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--selection-metric", choices=("f1", "roc_auc"))
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--artifact-suffix", default="")
    return parser.parse_args()


def get_device() -> torch.device:
    return torch.device("cpu")


def score_reconstruction_errors(
    model: nn.Module,
    sequences: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(sequences).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            errors.append(((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(errors).astype(np.float32) if errors else np.empty(0, dtype=np.float32)


def normalize_artifact_suffix(value: str) -> str:
    suffix = value.strip()
    if "/" in suffix or "\\" in suffix:
        raise ValueError("Artifact suffix must not contain path separators")
    return suffix


def suffixed(path: Path, suffix: str) -> Path | None:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}") if suffix else None


def load_sequence_splits(
    *,
    sequence_length: int,
    feature_columns: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    splits: dict[str, dict[str, np.ndarray]] = {}
    stats: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        path = PROCESSED_DIR / f"events_{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed split not found: {path}. Run preprocessing first."
            )
        frame = pd.read_csv(path)
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(
                f"LSTM features missing from {split_name}: " + ", ".join(missing)
            )
        sequences = build_sequences(
            frame,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
        )
        if not len(sequences["X"]):
            raise ValueError(f"No {split_name} sequences were generated")
        splits[split_name] = sequences
        stats[split_name] = {
            "sequences": int(len(sequences["X"])),
            "anomalies": int(sequences["y"].sum()),
            "anomaly_prevalence": float(sequences["y"].mean()),
        }
    return splits, stats


def selection_key(
    metrics: dict[str, Any],
    *,
    selection_metric: str,
) -> tuple[float, float, float, float]:
    fpr = metrics["false_positive_rate"]
    if selection_metric == "roc_auc":
        primary = metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0
    else:
        primary = metrics["f1_score"]
    return (
        float(primary),
        float(metrics["recall"]),
        -float(fpr if fpr is not None else 1.0),
        float(metrics["f1_score"]),
    )


def plot_reconstruction_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(scores[y_true == 0], bins=30, alpha=0.7, label="Normal", density=True)
    axis.hist(scores[y_true == 1], bins=30, alpha=0.7, label="Anomaly", density=True)
    axis.axvline(threshold, color="red", linestyle="--", label="Threshold")
    axis.set_title("LSTM Reconstruction Error: Test")
    axis.set_xlabel("Reconstruction error")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["Normal", "Anomaly"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title("LSTM-Autoencoder: Test")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc(y_true: np.ndarray, scores: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, scores, name="LSTM-Autoencoder", ax=axis)
    axis.set_title("ROC Curve: LSTM-Autoencoder Test")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def load_reference_metrics() -> dict[str, Any] | None:
    path = BASELINE_REFERENCE_DIR / "metrics_autoencoder.json"
    if not path.exists():
        return None
    payload = load_json(path).get("lstm_autoencoder", {})
    return payload.get("test", payload)


def append_experiment_record(record: dict[str, Any]) -> None:
    path = RESULTS_DIR / "autoencoder_experiments.csv"
    row = pd.DataFrame([record])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    configured = config["lstm_autoencoder"]
    feature_config = load_json(PROCESSED_DIR / "feature_config.json")

    sequence_length = int(args.sequence_length or feature_config["sequence_length"])
    hidden_size = int(args.hidden_size or configured["hidden_size"])
    epochs = int(args.epochs or configured["epochs"])
    batch_size = int(args.batch_size or configured["batch_size"])
    learning_rate = float(args.learning_rate or configured["learning_rate"])
    threshold_percentiles = list(
        args.threshold_percentiles or configured["threshold_percentiles"]
    )
    early_stopping_patience = int(
        configured["early_stopping_patience"]
        if args.early_stopping_patience is None
        else args.early_stopping_patience
    )
    selection_metric = str(args.selection_metric or configured["selection_metric"])
    suffix = normalize_artifact_suffix(args.artifact_suffix)
    seed = int(config["seed"])
    if min(sequence_length, hidden_size, epochs, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("Sequence length and training parameters must be positive")
    if early_stopping_patience < 0:
        raise ValueError("Early stopping patience must be non-negative")

    ensure_directories()
    set_seed(seed)
    feature_columns = list(feature_config["feature_columns_lstm"])
    sequence_splits, sequence_stats = load_sequence_splits(
        sequence_length=sequence_length,
        feature_columns=feature_columns,
    )
    X_train = sequence_splits["train"]["X"].astype(np.float32)
    y_train = sequence_splits["train"]["y"].astype(np.int64)
    X_validation = sequence_splits["validation"]["X"].astype(np.float32)
    y_validation = sequence_splits["validation"]["y"].astype(np.int64)
    X_test = sequence_splits["test"]["X"].astype(np.float32)
    y_test = sequence_splits["test"]["y"].astype(np.int64)
    X_train_normal = X_train[y_train == 0]
    if not len(X_train_normal):
        raise ValueError("No normal train sequences available")

    device = get_device()
    model = LSTMAutoencoder(input_size=X_train.shape[2], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_normal).float()),
        batch_size=batch_size,
        shuffle=True,
    )

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = 0.0
    best_percentile = 0.0
    best_validation_metrics: dict[str, Any] | None = None
    best_candidates: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for (batch,) in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        validation_errors = score_reconstruction_errors(
            model,
            X_validation,
            device=device,
            batch_size=batch_size,
        )
        threshold, percentile, validation_metrics, candidates = select_best_threshold(
            validation_errors,
            y_validation,
            threshold_percentiles,
        )
        current_key = selection_key(validation_metrics, selection_metric=selection_metric)
        improved = best_key is None or current_key > best_key
        if improved:
            best_key = current_key
            best_epoch = epoch
            best_threshold = threshold
            best_percentile = percentile
            best_validation_metrics = validation_metrics
            best_candidates = candidates
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1_score"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_false_positive_rate": validation_metrics["false_positive_rate"],
                "validation_threshold": threshold,
                "validation_threshold_percentile": percentile,
                "improved": int(improved),
            }
        )
        print(
            f"Epoch {epoch}/{epochs}: loss={history[-1]['train_loss']:.6f}, "
            f"val_f1={validation_metrics['f1_score']:.4f}, "
            f"val_recall={validation_metrics['recall']:.4f}"
        )
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            break

    training_seconds = time.perf_counter() - training_started
    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("Training finished without a valid checkpoint")
    model.load_state_dict(best_state)

    timing_config = config["timing"]
    test_errors, inference_timing = measure_inference(
        lambda: score_reconstruction_errors(
            model,
            X_test,
            device=device,
            batch_size=batch_size,
        ),
        record_count=len(X_test),
        warmup_runs=int(timing_config["warmup_runs"]),
        measured_runs=int(timing_config["measured_runs"]),
    )
    test_predictions = (test_errors >= best_threshold).astype(np.int64)
    test_metrics = compute_classification_metrics(y_test, test_predictions, test_errors)

    distribution_plot = PLOTS_DIR / "reconstruction_error_distribution.png"
    confusion_plot = PLOTS_DIR / "confusion_matrix_autoencoder.png"
    roc_plot = PLOTS_DIR / "roc_curve_autoencoder.png"
    plot_reconstruction_distribution(y_test, test_errors, best_threshold, distribution_plot)
    plot_confusion(y_test, test_predictions, confusion_plot)
    plot_roc(y_test, test_errors, roc_plot)

    hyperparameters = {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "early_stopping_patience": early_stopping_patience,
        "selection_metric": selection_metric,
    }
    metrics_payload = {
        "protocol_version": "1",
        "split_policy": "synthetic-chronological-60-20-20",
        "evaluation_cohort": "sequence_endpoints",
        "lstm_autoencoder": {
            "threshold": best_threshold,
            "threshold_percentile": best_percentile,
            "threshold_calibration": "labeled-validation",
            "validation": best_validation_metrics,
            "test": test_metrics,
            "hyperparameters": hyperparameters,
            "best_epoch": best_epoch,
            "device": str(device),
        },
    }
    metrics_path = RESULTS_DIR / "metrics_autoencoder.json"
    save_json(metrics_payload, metrics_path)
    if (target := suffixed(metrics_path, suffix)) is not None:
        save_json(metrics_payload, target)

    model_path = RESULTS_DIR / "lstm_autoencoder.pt"
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "autoencoder_history.csv", index=False)
    pd.DataFrame(best_candidates).to_csv(
        RESULTS_DIR / "autoencoder_validation_thresholds.csv",
        index=False,
    )
    outputs = pd.DataFrame(
        {
            "user_id": sequence_splits["test"]["user_ids"],
            "timestamp": sequence_splits["test"]["timestamps"],
            "y_true": y_test,
            "reconstruction_error": test_errors,
            "pred": test_predictions,
        }
    )
    outputs.to_csv(RESULTS_DIR / "autoencoder_outputs.csv", index=False)
    errors = outputs[outputs["y_true"] != outputs["pred"]].copy()
    errors["error_type"] = np.where(
        errors["y_true"] == 0,
        "false_positive",
        "false_negative",
    )

    timing_payload = {"training_seconds": training_seconds, **inference_timing}
    save_json(timing_payload, RESULTS_DIR / "timing_autoencoder.json")
    reference = load_reference_metrics()
    metadata = {
        "experiment_tag": args.experiment_tag,
        "artifact_suffix": suffix,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        **hyperparameters,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "stopped_early": len(history) < epochs,
        "threshold": best_threshold,
        "threshold_percentile": best_percentile,
        "threshold_percentiles_checked": threshold_percentiles,
        "threshold_selection_method": "labeled-validation-percentile-search",
        "split_sizes": {name: stats["sequences"] for name, stats in sequence_stats.items()},
        "train_size": sequence_stats["train"]["sequences"],
        "train_normal_count": int(len(X_train_normal)),
        "validation_size": sequence_stats["validation"]["sequences"],
        "test_size": sequence_stats["test"]["sequences"],
        "validation_anomaly_ratio": sequence_stats["validation"]["anomaly_prevalence"],
        "sequence_stats": sequence_stats,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "reference_metrics": reference,
        "device": str(device),
    }
    metadata_path = RESULTS_DIR / "autoencoder_metadata.json"
    save_json(metadata, metadata_path)
    if (target := suffixed(metadata_path, suffix)) is not None:
        save_json(metadata, target)

    append_experiment_record(
        {
            "run_label": pd.Timestamp.now(tz="UTC").isoformat(),
            "experiment_tag": args.experiment_tag,
            "artifact_suffix": suffix,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "selected_threshold_percentile": best_percentile,
            "threshold": best_threshold,
            "train_size": sequence_stats["train"]["sequences"],
            "validation_size": sequence_stats["validation"]["sequences"],
            "test_size": sequence_stats["test"]["sequences"],
            "sequence_anomaly_ratio": sequence_stats["test"]["anomaly_prevalence"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "false_positive_rate": test_metrics["false_positive_rate"],
            "validation_precision": best_validation_metrics["precision"],
            "validation_recall": best_validation_metrics["recall"],
            "validation_f1": best_validation_metrics["f1_score"],
            "validation_roc_auc": best_validation_metrics["roc_auc"],
            "validation_pr_auc": best_validation_metrics["pr_auc"],
            "stopped_early": int(len(history) < epochs),
        }
    )

    resolved_config = {
        "scenario": "synthetic",
        "config_version": config["config_version"],
        "seed": seed,
        "model": "lstm_autoencoder",
        **hyperparameters,
        "threshold_percentiles": threshold_percentiles,
        "feature_columns": feature_columns,
    }
    run_dir, _ = create_run_directory(
        scenario="synthetic",
        model="lstm_autoencoder",
        resolved_config=resolved_config,
        seed=seed,
        data_path=RAW_DIR / "synthetic_logs.csv",
        split_report_path=PROCESSED_DIR / "split_report.json",
        feature_manifest_path=PROCESSED_DIR / "feature_manifest.json",
    )
    save_run_results(
        run_dir,
        validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
        threshold_candidates=best_candidates,
        timing=timing_payload,
        error_analysis=errors,
        plot_paths=(distribution_plot, confusion_plot, roc_plot),
    )

    finalize_run_artifacts(
        run_dir, predictions=outputs,
        model_writer=lambda path: torch.save(model.state_dict(), path),
        model_filename="model.pt",
        inference_contract={
            "model": "lstm_autoencoder", "feature_columns": feature_columns,
            "input_size": len(feature_columns), "hidden_size": hidden_size,
            "sequence_length": sequence_length, "threshold": best_threshold,
            "score": "mean_squared_reconstruction_error",
            "decision": "score >= threshold",
        },
        feature_state_path=PROCESSED_DIR / "feature_state.json",
    )

    print(f"Saved LSTM-Autoencoder metrics to: {metrics_path}")
    print(f"Best epoch: {best_epoch}; test F1: {test_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
