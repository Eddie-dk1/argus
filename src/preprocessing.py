from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features import apply_feature_state
from features import build_sequences
from features import fit_feature_state
from features import sort_events
from utils import DEFAULT_CONFIG_PATH
from utils import PROCESSED_DIR
from utils import RAW_DIR
from utils import ensure_directories
from utils import load_experiment_config
from utils import save_json


SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the synthetic Argus scenario.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, pd.DataFrame]:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0 for fraction in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.0")

    sorted_frame = sort_events(frame)
    train_stop = int(len(sorted_frame) * train_fraction)
    validation_stop = train_stop + int(len(sorted_frame) * validation_fraction)
    return {
        "train": sorted_frame.iloc[:train_stop].reset_index(drop=True),
        "validation": sorted_frame.iloc[train_stop:validation_stop].reset_index(drop=True),
        "test": sorted_frame.iloc[validation_stop:].reset_index(drop=True),
    }


def split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["timestamp"])
    return {
        "rows": int(len(frame)),
        "normal": int((frame["label"] == 0).sum()),
        "anomaly": int((frame["label"] == 1).sum()),
        "anomaly_prevalence": float(frame["label"].mean()),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
        "users": int(frame["user_id"].nunique()),
        "anomaly_types": {
            str(key): int(value)
            for key, value in frame["anomaly_type"].value_counts().sort_index().items()
        },
    }


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    expected_anomaly_types = {
        "brute_force_login",
        "unusual_night_activity",
        "data_exfiltration",
        "unusual_ip_change",
    }
    for split_name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Split {split_name!r} is empty")
        if set(frame["label"].unique()) != {0, 1}:
            raise ValueError(f"Split {split_name!r} must contain normal and anomaly labels")
        observed = set(frame.loc[frame["label"] == 1, "anomaly_type"].astype(str))
        missing = sorted(expected_anomaly_types - observed)
        if missing:
            raise ValueError(
                f"Split {split_name!r} is missing anomaly families: {', '.join(missing)}"
            )

    train_end = pd.to_datetime(splits["train"]["timestamp"]).max()
    validation_start = pd.to_datetime(splits["validation"]["timestamp"]).min()
    validation_end = pd.to_datetime(splits["validation"]["timestamp"]).max()
    test_start = pd.to_datetime(splits["test"]["timestamp"]).min()
    if train_end > validation_start or validation_end > test_start:
        raise ValueError("Chronological split boundaries overlap")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    ensure_directories()

    input_path = RAW_DIR / "synthetic_logs.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "Raw data not found. Run `python3 src/generate_data.py` first."
        )
    raw_frame = pd.read_csv(input_path)
    split_config = config["split"]
    splits = chronological_split(
        raw_frame,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
    )
    validate_splits(splits)

    feature_config = config["features"]
    rolling_window = int(feature_config["rolling_window_events"])
    sequence_length = int(feature_config["sequence_length"])
    state = fit_feature_state(splits["train"], rolling_window=rolling_window)

    processed_splits: dict[str, pd.DataFrame] = {}
    sequence_splits: dict[str, dict[str, np.ndarray]] = {}
    feature_columns: list[str] | None = None
    lstm_feature_columns: list[str] | None = None

    for split_name in SPLIT_NAMES:
        processed, current_feature_columns, current_lstm_columns = apply_feature_state(
            splits[split_name],
            state,
        )
        processed["split"] = split_name
        processed_splits[split_name] = processed
        if feature_columns is None:
            feature_columns = current_feature_columns
            lstm_feature_columns = current_lstm_columns
        elif current_feature_columns != feature_columns or current_lstm_columns != lstm_feature_columns:
            raise RuntimeError("Feature columns differ between splits")

        sequence_splits[split_name] = build_sequences(
            processed,
            feature_columns=current_lstm_columns,
            sequence_length=sequence_length,
        )

    assert feature_columns is not None
    assert lstm_feature_columns is not None
    baseline_dataset = pd.concat(
        [processed_splits[name] for name in SPLIT_NAMES],
        ignore_index=True,
    )
    baseline_dataset_path = PROCESSED_DIR / "baseline_dataset.csv"
    baseline_dataset.to_csv(baseline_dataset_path, index=False)

    for split_name, processed in processed_splits.items():
        processed.to_csv(PROCESSED_DIR / f"events_{split_name}.csv", index=False)
        np.savez_compressed(
            PROCESSED_DIR / f"sequences_{split_name}.npz",
            **sequence_splits[split_name],
        )

    combined_sequences = np.concatenate(
        [sequence_splits[name]["X"] for name in SPLIT_NAMES],
        axis=0,
    )
    combined_labels = np.concatenate(
        [sequence_splits[name]["y"] for name in SPLIT_NAMES],
        axis=0,
    )
    np.save(PROCESSED_DIR / "sequences.npy", combined_sequences)
    np.save(PROCESSED_DIR / "sequence_labels.npy", combined_labels)

    save_json(feature_columns, PROCESSED_DIR / "feature_columns.json")
    save_json(state, PROCESSED_DIR / "feature_state.json")

    split_report = {
        "scenario": "synthetic",
        "strategy": "chronological",
        "fractions": {
            "train": float(split_config["train_fraction"]),
            "validation": float(split_config["validation_fraction"]),
            "test": float(split_config["test_fraction"]),
        },
        "splits": {name: split_summary(splits[name]) for name in SPLIT_NAMES},
        "entity_overlap": {
            "train_validation_users": len(
                set(splits["train"]["user_id"]) & set(splits["validation"]["user_id"])
            ),
            "train_test_users": len(
                set(splits["train"]["user_id"]) & set(splits["test"]["user_id"])
            ),
            "policy": "user overlap allowed; chronological periods do not overlap",
        },
        "validated": True,
    }
    save_json(split_report, PROCESSED_DIR / "split_report.json")

    feature_manifest = {
        "scenario": "synthetic",
        "feature_set": "synthetic_v0.2",
        "feature_columns": feature_columns,
        "lstm_feature_columns": lstm_feature_columns,
        "feature_count": len(feature_columns),
        "lstm_feature_count": len(lstm_feature_columns),
        "fit_split": "train",
        "profile_fit_policy": state["profile_fit_policy"],
        "rolling_window_events": rolling_window,
        "transformations": {
            "continuous": "train mean/std after log/ratio/window derivation",
            "categorical": "train vocabulary one-hot with unknown bucket",
            "hour": "sine/cosine",
            "identifiers": "provenance only; IPs transformed to novelty/context",
        },
        "forbidden_columns": ["label", "anomaly_type", "user_id", "source_ip", "destination_ip"],
    }
    save_json(feature_manifest, PROCESSED_DIR / "feature_manifest.json")

    save_json(
        {
            "config_version": config["config_version"],
            "feature_columns": feature_columns,
            "feature_columns_lstm": lstm_feature_columns,
            "sequence_length": sequence_length,
            "split_rows": {name: int(len(processed_splits[name])) for name in SPLIT_NAMES},
            "split_sequences": {
                name: int(len(sequence_splits[name]["X"])) for name in SPLIT_NAMES
            },
            "baseline_rows": int(len(baseline_dataset)),
        },
        PROCESSED_DIR / "feature_config.json",
    )

    print(f"Saved baseline dataset to: {baseline_dataset_path}")
    print(f"Baseline dataset shape: {baseline_dataset.shape}")
    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: rows={len(processed_splits[split_name])}, "
            f"sequences={len(sequence_splits[split_name]['X'])}"
        )


if __name__ == "__main__":
    main()
