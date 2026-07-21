from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import DEFAULT_CONFIG_PATH
from utils import RAW_DIR
from utils import ensure_directories
from utils import load_experiment_config
from utils import set_seed


NORMAL_ANOMALY_TYPE = "normal"
ANOMALY_TYPES = (
    "brute_force_login",
    "unusual_night_activity",
    "data_exfiltration",
    "unusual_ip_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic security logs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-events", type=int, help="Override generator event count.")
    parser.add_argument("--seed", type=int, help="Override the configured seed.")
    return parser.parse_args()


def random_ip(rng: np.random.Generator, network: str) -> str:
    base_network = ipaddress.ip_network(network)
    host_index = int(rng.integers(10, base_network.num_addresses - 10))
    return str(base_network.network_address + host_index)


def build_user_profiles(
    user_ids: list[str],
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    departments = ["it", "finance", "hr", "sales", "security"]
    roles = ["analyst", "engineer", "manager", "admin"]
    devices = ["laptop", "workstation", "server"]
    locations = ["office", "vpn", "remote"]

    profiles: dict[str, dict[str, Any]] = {}
    for user_id in user_ids:
        department = str(rng.choice(departments, p=[0.28, 0.18, 0.14, 0.22, 0.18]))
        role = str(rng.choice(roles, p=[0.34, 0.32, 0.24, 0.10]))
        work_start = int(rng.choice([7, 8, 9, 10], p=[0.10, 0.40, 0.40, 0.10]))
        work_end = work_start + int(rng.choice([8, 9, 10], p=[0.30, 0.55, 0.15]))
        usual_ips = sorted(
            {
                random_ip(rng, "10.10.0.0/16"),
                random_ip(rng, "10.20.0.0/16"),
            }
        )
        profiles[user_id] = {
            "department": department,
            "user_role": role,
            "device_type": str(rng.choice(devices, p=[0.50, 0.35, 0.15])),
            "usual_source_ips": usual_ips,
            "rare_source_ip": random_ip(rng, "185.220.100.0/24"),
            "default_location": str(rng.choice(locations, p=[0.65, 0.20, 0.15])),
            "work_start": work_start,
            "work_end": work_end,
            "traffic_scale": float(rng.lognormal(mean=0.0, sigma=0.25)),
            "session_scale": float(rng.lognormal(mean=0.0, sigma=0.20)),
            "privileged_probability": 0.18 if role == "admin" else 0.05,
        }
    return profiles


def make_normal_timestamp(
    rng: np.random.Generator,
    start_timestamp: pd.Timestamp,
    n_days: int,
    profile: dict[str, Any],
) -> pd.Timestamp:
    days = np.arange(n_days)
    weekday_weights = np.asarray(
        [1.0 if (start_timestamp + pd.Timedelta(days=int(day))).dayofweek < 5 else 0.22 for day in days],
        dtype=float,
    )
    day_offset = int(rng.choice(days, p=weekday_weights / weekday_weights.sum()))

    if rng.random() < 0.96:
        hour = int(
            np.clip(
                round(rng.normal((profile["work_start"] + profile["work_end"]) / 2, 2.0)),
                profile["work_start"],
                profile["work_end"],
            )
        )
    else:
        hour = int(rng.choice([5, 6, 7, 19, 20, 21]))

    return (
        start_timestamp
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(int(rng.integers(0, 60)), unit="m")
        + pd.to_timedelta(int(rng.integers(0, 60)), unit="s")
    )


def build_temporally_stratified_schedule(
    n_events: int,
    anomaly_ratio: float,
    split_fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    schedule = np.full(n_events, NORMAL_ANOMALY_TYPE, dtype=object)
    train_stop = int(n_events * split_fractions[0])
    validation_stop = train_stop + int(n_events * split_fractions[1])
    boundaries = ((0, train_stop), (train_stop, validation_stop), (validation_stop, n_events))

    for start, stop in boundaries:
        segment_size = stop - start
        anomaly_count = max(len(ANOMALY_TYPES), int(round(segment_size * anomaly_ratio)))
        anomaly_count = min(anomaly_count, segment_size)
        anomaly_indices = rng.choice(np.arange(start, stop), size=anomaly_count, replace=False)
        anomaly_labels = np.resize(np.asarray(ANOMALY_TYPES, dtype=object), anomaly_count)
        rng.shuffle(anomaly_labels)
        schedule[anomaly_indices] = anomaly_labels
    return schedule


def generate_normal_event(
    rng: np.random.Generator,
    user_id: str,
    profile: dict[str, Any],
    timestamp: pd.Timestamp,
) -> dict[str, Any]:
    event_type = str(
        rng.choice(
            ["login", "file_access", "web_request", "dns_query", "email_access"],
            p=[0.24, 0.22, 0.28, 0.14, 0.12],
        )
    )
    login_status = "success" if event_type == "login" else "not_applicable"
    if event_type == "login" and rng.random() < 0.06:
        login_status = "failure"

    destination_networks = {
        "login": "10.0.1.0/24",
        "file_access": "10.0.10.0/24",
        "web_request": "172.16.20.0/24",
        "dns_query": "10.0.53.0/24",
        "email_access": "172.16.30.0/24",
    }
    protocol = {
        "login": str(rng.choice(["SSH", "HTTPS"])),
        "file_access": str(rng.choice(["SMB", "HTTPS"])),
        "web_request": "HTTPS",
        "dns_query": "DNS",
        "email_access": "HTTPS",
    }[event_type]
    service = {
        "login": "auth_server",
        "file_access": "fileshare",
        "web_request": "intranet",
        "dns_query": "dns_resolver",
        "email_access": "mail",
    }[event_type]

    traffic_scale = float(profile["traffic_scale"])
    session_scale = float(profile["session_scale"])
    return {
        "timestamp": timestamp,
        "user_id": user_id,
        "source_ip": str(rng.choice(profile["usual_source_ips"])),
        "destination_ip": random_ip(rng, destination_networks[event_type]),
        "event_type": event_type,
        "login_status": login_status,
        "bytes_sent": round(float(rng.lognormal(7.0, 0.42) * traffic_scale), 2),
        "bytes_received": round(float(rng.lognormal(7.3, 0.38) * traffic_scale), 2),
        "protocol": protocol,
        "hour": int(timestamp.hour),
        "is_weekend": int(timestamp.dayofweek >= 5),
        "label": 0,
        "anomaly_type": NORMAL_ANOMALY_TYPE,
        "department": profile["department"],
        "user_role": profile["user_role"],
        "device_type": profile["device_type"],
        "location": profile["default_location"],
        "dst_service": service,
        "session_duration": round(float(rng.integers(60, 3600) * session_scale), 2),
        "failed_logins": 1 if login_status == "failure" else 0,
        "distinct_destinations": int(rng.integers(1, 5)),
        "privileged_action": int(rng.random() < profile["privileged_probability"]),
    }


def generate_anomaly_event(
    rng: np.random.Generator,
    user_id: str,
    profile: dict[str, Any],
    timestamp: pd.Timestamp,
    anomaly_type: str,
) -> dict[str, Any]:
    if anomaly_type == "unusual_night_activity":
        timestamp = timestamp.normalize() + pd.Timedelta(
            hours=int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23])),
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )

    event = generate_normal_event(rng, user_id, profile, timestamp)
    event.update({"label": 1, "anomaly_type": anomaly_type})

    if anomaly_type == "brute_force_login":
        event.update(
            {
                "event_type": "login",
                "login_status": "failure",
                "destination_ip": random_ip(rng, "10.0.1.0/24"),
                "protocol": str(rng.choice(["SSH", "RDP", "HTTPS"])),
                "bytes_sent": round(float(rng.lognormal(6.4, 0.28)), 2),
                "bytes_received": round(float(rng.lognormal(6.2, 0.25)), 2),
                "dst_service": "auth_server",
                "session_duration": round(float(rng.integers(15, 240)), 2),
                "failed_logins": int(rng.integers(6, 20)),
                "distinct_destinations": int(rng.integers(1, 3)),
                "privileged_action": 0,
            }
        )
    elif anomaly_type == "unusual_night_activity":
        event.update(
            {
                "event_type": str(rng.choice(["file_access", "web_request", "email_access"])),
                "login_status": "not_applicable",
                "bytes_sent": round(float(rng.lognormal(8.1, 0.45)), 2),
                "bytes_received": round(float(rng.lognormal(8.0, 0.40)), 2),
                "session_duration": round(float(rng.integers(1800, 10800)), 2),
                "distinct_destinations": int(rng.integers(4, 10)),
                "privileged_action": int(rng.random() < 0.45),
                "location": str(rng.choice(["remote", "vpn"])),
                "dst_service": str(rng.choice(["fileshare", "intranet", "mail"])),
            }
        )
    elif anomaly_type == "data_exfiltration":
        event.update(
            {
                "event_type": "file_access",
                "login_status": "not_applicable",
                "destination_ip": random_ip(rng, "203.0.113.0/24"),
                "protocol": str(rng.choice(["HTTPS", "SFTP"])),
                "bytes_sent": round(float(rng.lognormal(11.1, 0.55)), 2),
                "bytes_received": round(float(rng.lognormal(7.4, 0.35)), 2),
                "session_duration": round(float(rng.integers(2400, 14000)), 2),
                "failed_logins": 0,
                "distinct_destinations": int(rng.integers(8, 18)),
                "privileged_action": 1,
                "location": str(rng.choice(["vpn", "remote"])),
                "dst_service": "external_storage",
            }
        )
    elif anomaly_type == "unusual_ip_change":
        event.update(
            {
                "event_type": "login",
                "login_status": "success",
                "source_ip": profile["rare_source_ip"],
                "destination_ip": random_ip(rng, "10.0.1.0/24"),
                "protocol": str(rng.choice(["HTTPS", "SSH"])),
                "session_duration": round(float(rng.integers(120, 2400)), 2),
                "failed_logins": 0,
                "location": "external",
                "dst_service": "auth_server",
            }
        )

    event["hour"] = int(pd.Timestamp(event["timestamp"]).hour)
    event["is_weekend"] = int(pd.Timestamp(event["timestamp"]).dayofweek >= 5)
    return event


def generate_synthetic_logs(
    *,
    n_events: int,
    n_users: int,
    n_days: int,
    anomaly_ratio: float,
    start_timestamp: str,
    split_fractions: tuple[float, float, float],
    seed: int,
) -> pd.DataFrame:
    if n_events <= 0 or n_users <= 0 or n_days <= 0:
        raise ValueError("n_events, n_users, and n_days must be positive")
    if not 0 < anomaly_ratio < 1:
        raise ValueError("anomaly_ratio must be between zero and one")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    user_ids = [f"user_{index:03d}" for index in range(1, n_users + 1)]
    profiles = build_user_profiles(user_ids, rng)
    start = pd.Timestamp(start_timestamp)

    contexts: list[tuple[pd.Timestamp, str]] = []
    for _ in range(n_events):
        user_id = str(rng.choice(user_ids))
        timestamp = make_normal_timestamp(rng, start, n_days, profiles[user_id])
        contexts.append((timestamp, user_id))
    contexts.sort(key=lambda item: item[0])

    schedule = build_temporally_stratified_schedule(
        n_events,
        anomaly_ratio,
        split_fractions,
        rng,
    )
    events: list[dict[str, Any]] = []
    for (timestamp, user_id), anomaly_type in zip(contexts, schedule, strict=True):
        profile = profiles[user_id]
        if anomaly_type == NORMAL_ANOMALY_TYPE:
            event = generate_normal_event(rng, user_id, profile, timestamp)
        else:
            event = generate_anomaly_event(
                rng,
                user_id,
                profile,
                timestamp,
                str(anomaly_type),
            )
        events.append(event)

    return pd.DataFrame.from_records(events).sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    generator_config = config["generator"]
    split_config = config["split"]
    seed = int(args.seed if args.seed is not None else config["seed"])
    n_events = int(args.n_events if args.n_events is not None else generator_config["n_events"])

    ensure_directories()
    output_path = RAW_DIR / "synthetic_logs.csv"
    frame = generate_synthetic_logs(
        n_events=n_events,
        n_users=int(generator_config["n_users"]),
        n_days=int(generator_config["n_days"]),
        anomaly_ratio=float(generator_config["anomaly_ratio"]),
        start_timestamp=str(generator_config["start_timestamp"]),
        split_fractions=(
            float(split_config["train_fraction"]),
            float(split_config["validation_fraction"]),
            float(split_config["test_fraction"]),
        ),
        seed=seed,
    )
    frame.to_csv(output_path, index=False)

    print(f"Saved synthetic logs to: {output_path}")
    print(f"Dataset size: {frame.shape}")
    print(frame["label"].value_counts(dropna=False).sort_index())
    print(frame["anomaly_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
