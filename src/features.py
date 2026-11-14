from __future__ import annotations

from collections import Counter
from collections import deque
from typing import Any

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = [
    "event_type",
    "login_status",
    "protocol",
    "location",
    "dst_service",
    "department",
    "user_role",
    "device_type",
]

CONTINUOUS_FEATURES = [
    "log_bytes_sent",
    "log_bytes_received",
    "bytes_ratio",
    "log_session_duration",
    "failed_logins",
    "distinct_destinations",
    "prior_event_count",
    "prior_failed_login_sum",
    "prior_log_bytes_sent_sum",
    "prior_log_bytes_received_sum",
    "prior_unique_source_ips",
    "prior_unique_destination_networks",
    "recent_source_ip_change_count",
]

BINARY_AND_CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_off_hours",
    "is_login",
    "is_login_failure",
    "privileged_action",
    "source_ip_novel_for_user",
    "destination_network_novel_for_user",
    "insufficient_history",
]

LSTM_FEATURE_COLUMNS = [
    *(f"feat_{name}" for name in CONTINUOUS_FEATURES),
    *(f"feat_{name}" for name in BINARY_AND_CYCLICAL_FEATURES),
]


def sort_events(data_frame: pd.DataFrame) -> pd.DataFrame:
    frame = data_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values(["timestamp", "user_id"] + (["event_id"] if "event_id" in frame else []), kind="stable").reset_index(drop=True)
    frame["hour"] = frame["timestamp"].dt.hour.astype(int)
    frame["is_weekend"] = (frame["timestamp"].dt.dayofweek >= 5).astype(int)
    frame["label"] = frame["label"].astype(int)
    return frame


def destination_network(value: object) -> str:
    text = str(value)
    parts = text.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return ".".join(parts[:3]) + ".0/24"
    return "unknown"


def causal_rolling_unique(
    values: pd.Series,
    users: pd.Series,
    window: int,
) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float32)
    positions_by_user: dict[str, list[int]] = {}
    for position, user_id in enumerate(users.astype(str)):
        positions_by_user.setdefault(user_id, []).append(position)

    raw_values = values.astype(str).to_numpy()
    for positions in positions_by_user.values():
        recent: deque[str] = deque()
        counts: Counter[str] = Counter()
        for position in positions:
            output[position] = float(len(counts))
            current = raw_values[position]
            recent.append(current)
            counts[current] += 1
            if len(recent) > window:
                removed = recent.popleft()
                counts[removed] -= 1
                if counts[removed] == 0:
                    del counts[removed]
    return output


def add_causal_context(frame: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")

    result = sort_events(frame)
    result["destination_network"] = result["destination_ip"].map(destination_network)
    result["log_bytes_sent"] = np.log1p(result["bytes_sent"].clip(lower=0).astype(float))
    result["log_bytes_received"] = np.log1p(result["bytes_received"].clip(lower=0).astype(float))
    result["bytes_ratio"] = (
        result["bytes_sent"].astype(float)
        / result["bytes_received"].astype(float).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["log_session_duration"] = np.log1p(
        result["session_duration"].clip(lower=0).astype(float)
    )
    radians = 2 * np.pi * result["hour"].astype(float) / 24.0
    result["hour_sin"] = np.sin(radians)
    result["hour_cos"] = np.cos(radians)
    result["is_off_hours"] = ((result["hour"] < 7) | (result["hour"] >= 20)).astype(int)
    result["is_login"] = (result["event_type"] == "login").astype(int)
    result["is_login_failure"] = (
        (result["event_type"] == "login") & (result["login_status"] == "failure")
    ).astype(int)

    grouped = result.groupby("user_id", sort=False)
    result["prior_event_count"] = grouped.cumcount().clip(upper=rolling_window).astype(float)
    result["prior_failed_login_sum"] = grouped["failed_logins"].transform(
        lambda values: values.shift(1).rolling(rolling_window, min_periods=1).sum()
    ).fillna(0.0)
    result["prior_log_bytes_sent_sum"] = grouped["log_bytes_sent"].transform(
        lambda values: values.shift(1).rolling(rolling_window, min_periods=1).sum()
    ).fillna(0.0)
    result["prior_log_bytes_received_sum"] = grouped["log_bytes_received"].transform(
        lambda values: values.shift(1).rolling(rolling_window, min_periods=1).sum()
    ).fillna(0.0)
    result["prior_unique_source_ips"] = causal_rolling_unique(
        result["source_ip"],
        result["user_id"],
        rolling_window,
    )
    result["prior_unique_destination_networks"] = causal_rolling_unique(
        result["destination_network"],
        result["user_id"],
        rolling_window,
    )
    previous_source = grouped["source_ip"].shift(1)
    result["source_ip_changed"] = (
        previous_source.notna() & (result["source_ip"] != previous_source)
    ).astype(int)
    result["recent_source_ip_change_count"] = result.groupby("user_id", sort=False)[
        "source_ip_changed"
    ].transform(lambda values: values.rolling(rolling_window, min_periods=1).sum())
    result["insufficient_history"] = (result["prior_event_count"] < rolling_window).astype(int)
    return result


def fit_feature_state(
    train_frame: pd.DataFrame,
    *,
    rolling_window: int,
) -> dict[str, Any]:
    context = add_causal_context(train_frame, rolling_window)
    normal = context[context["label"] == 0]
    if normal.empty:
        raise ValueError("Synthetic train split contains no normal events.")

    known_source_ips = {
        str(user_id): sorted(set(group["source_ip"].astype(str)))
        for user_id, group in normal.groupby("user_id")
    }
    known_destination_networks = {
        str(user_id): sorted(set(group["destination_network"].astype(str)))
        for user_id, group in normal.groupby("user_id")
    }
    categories = {
        column: sorted(context[column].astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }

    continuous = context[CONTINUOUS_FEATURES].astype(float)
    means = continuous.mean()
    scales = continuous.std(ddof=0).replace(0.0, 1.0)
    return {
        "rolling_window_events": rolling_window,
        "continuous_features": list(CONTINUOUS_FEATURES),
        "binary_and_cyclical_features": list(BINARY_AND_CYCLICAL_FEATURES),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "categories": categories,
        "continuous_mean": {key: float(value) for key, value in means.items()},
        "continuous_scale": {key: float(value) for key, value in scales.items()},
        "known_source_ips": known_source_ips,
        "known_destination_networks": known_destination_networks,
        "profile_fit_rows": int(len(normal)),
        "profile_fit_policy": "normal-train-events-only",
    }


def safe_category_token(value: str) -> str:
    token = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return token.strip("_") or "empty"


def apply_feature_state(
    data_frame: pd.DataFrame,
    state: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    context = add_causal_context(
        data_frame,
        int(state["rolling_window_events"]),
    )
    known_sources = {
        user: set(values) for user, values in state["known_source_ips"].items()
    }
    known_destinations = {
        user: set(values) for user, values in state["known_destination_networks"].items()
    }
    context["source_ip_novel_for_user"] = [
        int(str(source_ip) not in known_sources.get(str(user_id), set()))
        for user_id, source_ip in zip(context["user_id"], context["source_ip"], strict=True)
    ]
    context["destination_network_novel_for_user"] = [
        int(str(network) not in known_destinations.get(str(user_id), set()))
        for user_id, network in zip(
            context["user_id"],
            context["destination_network"],
            strict=True,
        )
    ]

    feature_frame = pd.DataFrame(index=context.index)
    for column in state["continuous_features"]:
        mean = float(state["continuous_mean"][column])
        scale = float(state["continuous_scale"][column])
        feature_frame[f"feat_{column}"] = (
            (context[column].astype(float) - mean) / scale
        ).astype(np.float32)
    for column in state["binary_and_cyclical_features"]:
        feature_frame[f"feat_{column}"] = context[column].astype(np.float32)

    for column in state["categorical_columns"]:
        values = context[column].astype(str)
        known_values = set(state["categories"][column])
        for category in state["categories"][column]:
            output_name = f"feat_{column}__{safe_category_token(category)}"
            feature_frame[output_name] = (values == category).astype(np.float32)
        feature_frame[f"feat_{column}__unknown"] = (~values.isin(known_values)).astype(np.float32)

    feature_columns = feature_frame.columns.tolist()
    lstm_columns = [column for column in LSTM_FEATURE_COLUMNS if column in feature_columns]
    lstm_columns.extend(
        column
        for column in feature_columns
        if column.startswith("feat_event_type__")
        or column.startswith("feat_login_status__")
        or column.startswith("feat_protocol__")
    )

    raw_columns = [column for column in sort_events(data_frame).columns if column not in feature_frame.columns]
    result = pd.concat(
        [
            context[raw_columns].reset_index(drop=True),
            feature_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    return result, feature_columns, list(dict.fromkeys(lstm_columns))


def build_sequences(
    data_frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 10,
) -> dict[str, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    user_ids: list[str] = []
    timestamps: list[str] = []
    item_ids: list[str] = []
    member_ids: list[list[str]] = []

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    data_frame = data_frame.copy()
    if "event_id" not in data_frame:
        data_frame["event_id"] = [f"fixture:{i}" for i in range(len(data_frame))]
    if data_frame["event_id"].isna().any() or data_frame["event_id"].duplicated().any():
        raise ValueError("Event IDs must be unique and non-null")
    sorted_frame = data_frame.sort_values(["user_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    sorted_frame["timestamp"] = pd.to_datetime(sorted_frame["timestamp"])

    for user_id, user_frame in sorted_frame.groupby("user_id"):
        if len(user_frame) < sequence_length:
            continue
        values = user_frame[feature_columns].to_numpy(dtype=np.float32)
        window_labels = user_frame["label"].to_numpy(dtype=np.int64)
        window_timestamps = user_frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        for start_index in range(0, len(user_frame) - sequence_length + 1):
            stop_index = start_index + sequence_length
            sequences.append(values[start_index:stop_index])
            members = user_frame["event_id"].iloc[start_index:stop_index].astype(str).tolist()
            item_ids.append(members[-1])
            member_ids.append(members)
            # The sequence predicts whether its endpoint event is anomalous.
            # This keeps sequence and tabular evaluation targets aligned.
            labels.append(int(window_labels[stop_index - 1]))
            user_ids.append(str(user_id))
            timestamps.append(str(window_timestamps[stop_index - 1]))

    if not sequences:
        return {
            "item_ids": np.empty((0,), dtype="<U1"),
            "member_ids": np.empty((0, sequence_length), dtype="<U1"),
            "X": np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "user_ids": np.empty((0,), dtype="<U1"),
            "timestamps": np.empty((0,), dtype="<U1"),
        }
    return {
        "item_ids": np.asarray(item_ids),
        "member_ids": np.asarray(member_ids),
        "X": np.stack(sequences).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "user_ids": np.asarray(user_ids),
        "timestamps": np.asarray(timestamps),
    }
