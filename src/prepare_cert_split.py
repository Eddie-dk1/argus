"""Freeze a CERT account-group assignment; never overwrite an existing manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from datasets.cert_labels import POLICY_ID, build_label_index
from datasets.cert_split import allocate_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--psychometric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy_path = Path(__file__).resolve().parents[1] / "configs/cert_r4_2_labels_v1.json"
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    if policy["policy_id"] != POLICY_ID:
        raise ValueError("Policy version differs")
    with args.psychometric.open(newline="", encoding="utf-8") as stream:
        users = [row["user_id"] for row in csv.DictReader(stream)]
    if not users or any(not u for u in users) or len(set(users)) != len(users):
        raise ValueError("Empty, missing or duplicate account identity")
    index = build_label_index(args.answers, accounts=set(users))
    result = allocate_groups(index, policy["split_fractions"], policy["split_seed"])
    result["policy_sha256"] = hashlib.sha256(policy_bytes).hexdigest()
    result["account_population_sha256"] = hashlib.sha256(
        json.dumps(sorted(users), separators=(",", ":")).encode()).hexdigest()
    result["implementation_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).parent / "datasets/cert_split.py",
            Path(__file__).parent / "datasets/cert_labels.py")}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    # Output parent must already exist; raw source trees are never created/changed.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
