"""Print a read-only CERT answer-index summary; do not label raw data or train."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from datasets.cert_labels import POLICY_ID, build_label_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--psychometric", type=Path, required=True)
    args = parser.parse_args()
    with args.psychometric.open(newline="") as stream:
        accounts = {row["user_id"] for row in csv.DictReader(stream)}
    if not accounts or "" in accounts:
        raise ValueError("Empty/invalid account population")
    policy_path = Path(__file__).resolve().parents[1] / "configs/cert_r4_2_labels_v1.json"
    policy_bytes = policy_path.read_bytes()
    if json.loads(policy_bytes)["policy_id"] != POLICY_ID:
        raise ValueError("Policy config and implementation version differ")
    index = build_label_index(args.answers, accounts=accounts)
    summary = index.summary()
    summary["policy_config_sha256"] = hashlib.sha256(policy_bytes).hexdigest()
    summary["account_population_sha256"] = hashlib.sha256(
        json.dumps(sorted(accounts), separators=(",", ":")).encode()).hexdigest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
