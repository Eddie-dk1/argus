from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from features import apply_feature_state
from features import build_sequences
from features import fit_feature_state
from generate_data import ANOMALY_TYPES
from generate_data import generate_synthetic_logs
from preprocessing import chronological_split
from preprocessing import validate_splits
from utils import compute_classification_metrics
from utils import select_best_threshold


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = generate_synthetic_logs(
            n_events=600,
            n_users=12,
            n_days=30,
            anomaly_ratio=0.10,
            start_timestamp="2025-01-01 00:00:00",
            split_fractions=(0.6, 0.2, 0.2),
            seed=42,
        )

    def test_generator_is_deterministic(self) -> None:
        repeated = generate_synthetic_logs(
            n_events=600,
            n_users=12,
            n_days=30,
            anomaly_ratio=0.10,
            start_timestamp="2025-01-01 00:00:00",
            split_fractions=(0.6, 0.2, 0.2),
            seed=42,
        )
        pd.testing.assert_frame_equal(self.frame, repeated)

    def test_chronological_splits_contain_all_anomaly_families(self) -> None:
        splits = chronological_split(
            self.frame,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        validate_splits(splits)
        for frame in splits.values():
            observed = set(frame.loc[frame["label"] == 1, "anomaly_type"])
            self.assertEqual(observed, set(ANOMALY_TYPES))

    def test_features_exclude_raw_identifiers_and_handle_unknown_category(self) -> None:
        splits = chronological_split(
            self.frame,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        state = fit_feature_state(splits["train"], rolling_window=5)
        validation = splits["validation"].copy()
        validation.loc[validation.index[0], "protocol"] = "UNSEEN_PROTOCOL"
        processed, feature_columns, lstm_columns = apply_feature_state(validation, state)

        self.assertFalse(any(column == "user_id" for column in feature_columns))
        self.assertFalse(any("source_ip_encoded" in column for column in feature_columns))
        self.assertEqual(processed.loc[0, "feat_protocol__unknown"], 1.0)
        self.assertTrue(set(lstm_columns).issubset(feature_columns))
        self.assertTrue(np.isfinite(processed[feature_columns].to_numpy()).all())

    def test_metrics_include_false_positive_rate_and_support(self) -> None:
        metrics = compute_classification_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 0]),
            np.asarray([0.1, 0.8, 0.9, 0.2]),
        )
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["support"]["anomaly"], 2)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_threshold_is_selected_from_validation_scores(self) -> None:
        threshold, percentile, metrics, candidates = select_best_threshold(
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            np.asarray([0, 0, 1, 1]),
            [25, 50, 75],
        )
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(percentile, 50.0)
        self.assertEqual(len(candidates), 3)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)

    def test_sequence_target_is_endpoint_event(self) -> None:
        frame = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "label": [1, 0, 0],
                "feat_value": [1.0, 2.0, 3.0],
            }
        )
        sequences = build_sequences(frame, ["feat_value"], sequence_length=2)
        self.assertEqual(sequences["y"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
