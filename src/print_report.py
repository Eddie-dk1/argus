from __future__ import annotations

import pandas as pd

from utils import RESULTS_DIR
from utils import load_json


def main() -> None:
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    evaluation_path = RESULTS_DIR / "evaluation_summary.json"
    anomaly_path = RESULTS_DIR / "anomaly_examples.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Metrics summary is missing; run evaluate.py first")

    metrics = pd.read_csv(metrics_path)
    columns = [
        "model",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
    ]
    print("\nARGUS SYNTHETIC PIPELINE — TEST METRICS")
    print(metrics[columns].round(4).to_string(index=False))

    if evaluation_path.exists():
        evaluation = load_json(evaluation_path)
        print(
            "\nModel selected by validation: "
            f"{evaluation['selected_model_by_validation']} "
            f"(validation F1={evaluation['selected_model_validation_f1']:.4f})"
        )
        print(evaluation["warning"])

    if anomaly_path.exists():
        examples = pd.read_csv(anomaly_path).head(5)
        print("\nFIRST MODEL-FLAG EXPLANATIONS")
        preferred = ["timestamp", "user_id", "flagged_by", "probable_reason"]
        print(examples[[column for column in preferred if column in examples]].to_string(index=False))


if __name__ == "__main__":
    main()
