from __future__ import annotations

import pandas as pd

from utils import PROCESSED_DIR
from utils import RESULTS_DIR
from utils import ensure_directories


def build_normal_profiles(train: pd.DataFrame) -> dict[str, object]:
    normal = train[train["label"] == 0].copy()
    return {
        "source_ips": normal.groupby("user_id")["source_ip"].agg(lambda values: set(values.astype(str))).to_dict(),
        "bytes_median": normal.groupby("user_id")["bytes_sent"].median().to_dict(),
        "bytes_high": float(normal["bytes_sent"].quantile(0.99)),
        "duration_high": float(normal["session_duration"].quantile(0.99)),
    }


def explain_row(row: pd.Series, profiles: dict[str, object]) -> str:
    reasons: list[str] = []
    user_id = str(row["user_id"])
    failed_logins = int(row.get("failed_logins", 0))
    hour = pd.Timestamp(row["timestamp"]).hour
    bytes_sent = float(row.get("bytes_sent", 0.0))
    duration = float(row.get("session_duration", 0.0))
    source_ip = str(row.get("source_ip", ""))

    if failed_logins >= 5:
        reasons.append("burst of failed logins")
    if hour <= 5 or hour >= 22:
        reasons.append("off-hours activity")

    bytes_median = float(profiles["bytes_median"].get(user_id, 0.0))
    if bytes_sent >= float(profiles["bytes_high"]) or (
        bytes_median > 0 and bytes_sent >= bytes_median * 4
    ):
        reasons.append("outbound traffic above the user baseline")
    if duration >= float(profiles["duration_high"]):
        reasons.append("unusually long session")

    known_ips = profiles["source_ips"].get(user_id, set())
    if known_ips and source_ip not in known_ips:
        reasons.append("source IP not observed in normal train history")
    if int(row.get("distinct_destinations", 0)) >= 8:
        reasons.append("high destination diversity")
    if int(row.get("privileged_action", 0)) == 1:
        reasons.append("privileged action")

    return "; ".join(reasons) if reasons else "combined model features deviated from the train profile"


def main() -> None:
    ensure_directories()
    train_path = PROCESSED_DIR / "events_train.csv"
    outputs_path = RESULTS_DIR / "baseline_outputs.csv"
    if not train_path.exists() or not outputs_path.exists():
        raise FileNotFoundError("Run preprocessing and baseline training before explanations")

    train = pd.read_csv(train_path)
    outputs = pd.read_csv(outputs_path)
    profiles = build_normal_profiles(train)
    flagged = outputs[(outputs["rf_pred"] == 1) | (outputs["iforest_pred"] == 1)].copy()
    flagged["probable_reason"] = flagged.apply(explain_row, axis=1, profiles=profiles)
    flagged["flagged_by"] = flagged.apply(
        lambda row: ",".join(
            model
            for model, prediction in (
                ("random_forest", row["rf_pred"]),
                ("isolation_forest", row["iforest_pred"]),
            )
            if int(prediction) == 1
        ),
        axis=1,
    )
    flagged = flagged.sort_values(
        ["rf_score", "iforest_score"],
        ascending=False,
    )
    columns = [
        "user_id",
        "timestamp",
        "flagged_by",
        "probable_reason",
        "rf_score",
        "iforest_score",
        "y_true",
        "anomaly_type",
        "bytes_sent",
        "failed_logins",
        "source_ip",
        "event_type",
    ]
    result = flagged[[column for column in columns if column in flagged.columns]]
    output_path = RESULTS_DIR / "anomaly_examples.csv"
    result.to_csv(output_path, index=False)
    print(f"Saved model-flag explanations to: {output_path}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
