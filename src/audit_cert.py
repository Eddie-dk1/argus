"""Read-only, chunked acceptance audit of publisher CERT r4.2 inputs.

Reports and temporary ID-sort files live in the selected SSD data root. This
does not generate model labels/features, select splits, or train any model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import pandas as pd

from data_paths import get_scenario_paths
from datasets.inspection import inspect_scenario, save_inspection_report

SOURCES = ("logon", "device", "file", "http", "email")
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def load_answers(answers: Path) -> tuple[list[dict], dict]:
    with (answers / "insiders.csv").open(newline="") as stream:
        incidents = [r for r in csv.DictReader(stream) if r["dataset"] == "4.2"]
    if not incidents:
        raise ValueError("No r4.2 incidents in official answer key")
    expected = {name: defaultdict(set) for name in SOURCES}
    for incident in incidents:
        start = datetime.strptime(incident["start"], DATE_FORMAT)
        end = datetime.strptime(incident["end"], DATE_FORMAT)
        if start > end:
            raise ValueError("Invalid incident interval")
        matches = list(answers.rglob(incident["details"]))
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous incident file: {incident['details']}")
        incident["observable_rows"] = 0
        incident["outside_master_interval"] = 0
        incident["other_user_observable_rows"] = 0
        observable_users = Counter()
        with matches[0].open(newline="") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                source, fields = row[0], tuple(row[1:])
                if source not in expected or len(fields) < 4:
                    raise ValueError("Unknown answer source or malformed row")
                incident["other_user_observable_rows"] += int(fields[2] != incident["user"])
                observable_users[fields[2]] += 1
                date = datetime.strptime(fields[1], DATE_FORMAT)
                incident["outside_master_interval"] += int(not start <= date <= end)
                expected[source][fields[0]].add(fields)
                incident["observable_rows"] += 1
        incident["observable_users"] = dict(observable_users)
    return incidents, expected


def count_duplicate_ids(path: Path, work: Path) -> int:
    """Exact external sort, bounded RAM; IDs are source-qualified by caller."""
    ordered = path.with_suffix(".sorted")
    subprocess.run(["sort", "-S", "128M", "--parallel=2", "-T", str(work),
                    "-o", str(ordered), str(path)],
                   env={**os.environ, "LC_ALL": "C"}, check=True)
    previous, duplicates = None, 0
    with ordered.open() as stream:
        for line in stream:
            duplicates += int(line == previous)
            previous = line
    ordered.unlink()
    path.unlink()
    return duplicates


def audit(source_dir: Path, answers: Path, output: Path, work: Path,
          chunk_size: int = 100_000) -> dict:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if shutil.disk_usage(work).free < 10 * 2**30:
        raise ValueError("Audit needs at least 10 GiB free for bounded external sorting")
    schema = inspect_scenario("cert_r4_2", source_dir, sample_rows=10_000)
    save_inspection_report(schema, output.parent / "schema_inspection.json")
    if not schema["valid_schema"]:
        raise ValueError("Core schema validation failed; see schema_inspection.json")
    incidents, expected = load_answers(answers)
    psych = pd.read_csv(source_dir / "psychometric.csv", dtype=str, keep_default_na=False)
    known_users = set(psych["user_id"])
    ldap_users = set()
    ldap_files = sorted(p for p in (source_dir / "LDAP").glob("*.csv")
                        if not p.name.startswith("._"))
    for path in ldap_files:
        ldap = pd.read_csv(path, dtype=str, keep_default_na=False)
        ldap_users.update(ldap["user_id"])
    report = {"scenario": "cert_r4_2", "scan": "all rows of five activity CSVs",
              "chunk_size": chunk_size, "sources": {},
              "schema_valid": True, "incidents": incidents,
              "incident_users": len({r["user"] for r in incidents}),
              "incidents_by_scenario": dict(Counter(r["scenario"] for r in incidents)),
              "psychometric_users": len(known_users), "ldap_files": len(ldap_files),
              "ldap_users": len(ldap_users),
              "license_status": "review-required: KiltHub CC BY 4.0 vs bundled ExactData terms",
              "label_policy_status": "review-required: incident observables can include other users; no labels generated",
              "readiness": "audit_in_progress", "not_performed": [
                  "training", "split creation", "feature generation",
                  "content-level near-duplicate detection", "operational representativeness validation"]}
    for source in SOURCES:
        path = source_dir / f"{source}.csv"
        stats = {"rows": 0, "missing_by_column": Counter(), "invalid_dates": 0,
                 "missing_required_cells": 0,
                 "repeated_headers": 0, "nonfinite_core_tokens": 0,
                 "invalid_numeric_cells": 0, "negative_numeric_cells": 0,
                 "date_min": None, "date_max": None, "activities": Counter()}
        users, pcs, matched = set(), set(), set()
        last = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"cert-audit-{source}-", dir=work) as directory:
            temp = Path(directory)
            ids = temp / "ids.txt"
            with ids.open("w") as idstream:
                for frame in pd.read_csv(path, chunksize=chunk_size, dtype=str,
                                         keep_default_na=False, on_bad_lines="error"):
                    stats["rows"] += len(frame)
                    stats["missing_by_column"].update((frame.eq("") | frame.isna()).sum().to_dict())
                    dates = pd.to_datetime(frame["date"], format=DATE_FORMAT, errors="coerce")
                    stats["invalid_dates"] += int(dates.isna().sum())
                    if dates.notna().any():
                        lo, hi = dates.min().isoformat(), dates.max().isoformat()
                        stats["date_min"] = min(stats["date_min"] or lo, lo)
                        stats["date_max"] = max(stats["date_max"] or hi, hi)
                    users.update(frame["user"])
                    pcs.update(frame["pc"])
                    idstream.write(frame["id"].str.cat(sep="\n") + "\n")
                    core = frame[["id", "date", "user", "pc"]]
                    required = ["id", "date", "user", "pc"] + {
                        "logon": ["activity"], "device": ["activity"],
                        "file": ["filename"], "http": ["url"],
                        "email": ["to", "from", "size", "attachments"]}[source]
                    stats["missing_required_cells"] += int(frame[required].eq("").sum().sum())
                    stats["repeated_headers"] += int(core.eq(pd.Series({c: c for c in core})).all(axis=1).sum())
                    stats["nonfinite_core_tokens"] += int(core.isin(["inf", "-inf", "Infinity", "NaN"]).sum().sum())
                    if "activity" in frame:
                        stats["activities"].update(frame["activity"].value_counts().to_dict())
                    for column in ("size", "attachments") if source == "email" else ():
                        numeric = pd.to_numeric(frame[column], errors="coerce")
                        stats["invalid_numeric_cells"] += int((
                            numeric.isna() | numeric.isin([float("inf"), float("-inf")])
                        ).sum())
                        stats["negative_numeric_cells"] += int((numeric < 0).sum())
                    # Compare complete variable-width official observables against raw rows.
                    subset = frame[frame["id"].isin(expected[source])]
                    for row in subset.itertuples(index=False, name=None):
                        row = tuple(row)
                        if row in expected[source][row[0]]:
                            matched.add(row)
                    if time.monotonic() - last > 15:
                        print(json.dumps({"source": source, "rows_scanned": stats["rows"]}), flush=True)
                        last = time.monotonic()
            stats["duplicate_ids_within_source"] = count_duplicate_ids(ids, temp)
        stats.update(users=len(users), pcs=len(pcs), unknown_psychometric_users=sorted(users-known_users),
                     unknown_ldap_users=sorted(users-ldap_users),
                     expected_observables=sum(len(x) for x in expected[source].values()),
                     matched_observables=len(matched))
        stats["unmatched_observables"] = stats["expected_observables"] - len(matched)
        report["sources"][source] = stats
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps({"source": source, "complete": True, "rows": stats["rows"],
                          "unmatched_observables": stats["unmatched_observables"]}), flush=True)
    report["total_events"] = sum(s["rows"] for s in report["sources"].values())
    report["blocking_data_issues"] = {
        name: {key: stats[key] for key in (
            "invalid_dates", "missing_required_cells", "nonfinite_core_tokens",
            "repeated_headers", "duplicate_ids_within_source",
            "invalid_numeric_cells", "negative_numeric_cells", "unmatched_observables",
            "unknown_psychometric_users", "unknown_ldap_users") if stats[key]}
        for name, stats in report["sources"].items()}
    report["blocking_data_issues"] = {k: v for k, v in report["blocking_data_issues"].items() if v}
    report["readiness"] = "audit_complete_review_findings_before_preprocessing"
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    paths = get_scenario_paths("cert_r4_2")
    raw, answers = paths.raw / "r4.2/r4.2", paths.raw / "answers/answers"
    paths.manifests.mkdir(parents=True, exist_ok=True)
    paths.interim.mkdir(parents=True, exist_ok=True)
    report = audit(raw, answers, paths.manifests / "cert_acceptance_audit.json",
                   paths.interim, args.chunk_size)
    print(json.dumps({"complete": True, "total_events": report["total_events"],
                      "blocking_data_issues": report["blocking_data_issues"]}), flush=True)


if __name__ == "__main__":
    main()
