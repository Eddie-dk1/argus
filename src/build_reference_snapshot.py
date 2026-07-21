from __future__ import annotations

import shutil
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pandas as pd

from experiment_tracking import build_data_manifest
from experiment_tracking import environment_metadata
from experiment_tracking import payload_sha256
from utils import DEFAULT_CONFIG_PATH
from utils import PROCESSED_DIR
from utils import RAW_DIR
from utils import RESULTS_DIR
from utils import load_json
from utils import save_json


REFERENCE_DIR = RESULTS_DIR / "baseline_reference"


def copy(source: Path, destination_name: str | None = None) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Reference input is missing: {source}")
    destination = REFERENCE_DIR / (destination_name or source.name)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    copied = [
        copy(DEFAULT_CONFIG_PATH, "synthetic_config.json"),
        copy(PROCESSED_DIR / "feature_columns.json"),
        copy(PROCESSED_DIR / "feature_config.json"),
        copy(PROCESSED_DIR / "feature_manifest.json"),
        copy(PROCESSED_DIR / "split_report.json"),
        copy(RESULTS_DIR / "metrics_baseline.json"),
        copy(RESULTS_DIR / "metrics_autoencoder.json"),
        copy(RESULTS_DIR / "metrics_summary.csv"),
        copy(RESULTS_DIR / "metrics_validation_summary.csv"),
        copy(RESULTS_DIR / "baseline_training_summary.json"),
        copy(RESULTS_DIR / "autoencoder_metadata.json"),
        copy(RESULTS_DIR / "evaluation_summary.json"),
        copy(RESULTS_DIR / "timing_baseline.json"),
        copy(RESULTS_DIR / "timing_autoencoder.json"),
        copy(RESULTS_DIR / "feature_importance_random_forest.csv"),
    ]

    test_metrics = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
    validation_metrics = pd.read_csv(RESULTS_DIR / "metrics_validation_summary.csv")
    data_manifest = build_data_manifest(RAW_DIR / "synthetic_logs.csv")
    data_manifest["path"] = "data/synthetic/raw/synthetic_logs.csv"
    config = load_json(DEFAULT_CONFIG_PATH)
    environment = environment_metadata()
    environment.pop("python_executable", None)
    environment.pop("git", None)
    snapshot = {
        "release": "argus-v0.2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "synthetic",
        "config_version": config["config_version"],
        "protocol_version": "1",
        "evaluation_cohort": "sequence_endpoints",
        "config_sha256": payload_sha256(config),
        "data_manifest": data_manifest,
        "environment": environment,
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "test_metrics": test_metrics.to_dict(orient="records"),
        "files": sorted(path.name for path in copied),
        "limitations": [
            "Synthetic data is a controlled engineering benchmark, not production evidence.",
            "Near-perfect Random Forest ranking reflects strong generator separability.",
            "Public dataset scenarios have not been downloaded or evaluated in this release.",
            "Final multi-seed model stability remains a later diploma-stage task.",
        ],
    }
    save_json(snapshot, REFERENCE_DIR / "reference_snapshot.json")
    print(f"Updated reference snapshot: {REFERENCE_DIR}")


if __name__ == "__main__":
    main()
