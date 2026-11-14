"""CERT v1 benchmark targets; answer metadata must never become model inputs.

This builds an answer index, not features or a train/test allocation. Raw rows
must be passed in publisher column order. No source files are modified.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


POLICY_ID = "cert_r4_2_observable_account_day_v1"
SOURCES = {"logon": 5, "device": 5, "file": 6, "http": 6, "email": 11}
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


@dataclass(frozen=True)
class Attribution:
    incident_id: str
    incident_user: str
    scenario: str


@dataclass
class LabelIndex:
    rows: dict[tuple[str, str], tuple[str, ...]]
    attributions: dict[tuple[str, str], tuple[Attribution, ...]]
    account_groups: dict[str, str]
    input_hashes: dict[str, str]

    def event_label(self, source: str, row: tuple[str, ...]) -> int:
        """0 means unlisted, not independently verified benign."""
        validate_row(source, row)
        expected = self.rows.get((source, row[0]))
        if expected is None:
            return 0
        if tuple(row) != expected:
            raise ValueError("Observable key matches but complete raw row differs")
        return 1

    def positive_days(self) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
        days: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for key, row in sorted(self.rows.items()):
            date = datetime.strptime(row[1], DATE_FORMAT).date().isoformat()
            days.setdefault((row[2], date), []).append(key)
        return {key: tuple(events) for key, events in days.items()}

    def summary(self) -> dict:
        other = Counter()
        for key, row in self.rows.items():
            for item in self.attributions[key]:
                if row[2] != item.incident_user:
                    other[f"scenario_{item.scenario}/{key[0]}"] += 1
        group_sizes = Counter(Counter(self.account_groups.values()).values())
        return {
            "policy_id": POLICY_ID,
            "input_sha256": self.input_hashes,
            "observable_events": len(self.rows),
            "observable_events_by_source": dict(sorted(Counter(k[0] for k in self.rows).items())),
            "positive_account_days": len(self.positive_days()),
            "positive_accounts": len({row[2] for row in self.rows.values()}),
            "cross_account_observables_by_scenario_source": dict(sorted(other.items())),
            "accounts_in_group_map": len(self.account_groups),
            "group_size_histogram": dict(sorted(group_sizes.items())),
            "scope": "answer index only; raw matching requires acceptance audit; no split or training",
        }


def validate_row(source: str, row: tuple[str, ...]) -> None:
    if source not in SOURCES or len(row) != SOURCES[source]:
        raise ValueError("Unknown source or unexpected publisher row width")
    if not all(row[:4]):
        raise ValueError("Missing observable identity")
    datetime.strptime(row[1], DATE_FORMAT)


def build_label_index(answers: Path, *, accounts: set[str] | None = None) -> LabelIndex:
    answers = answers.resolve()
    hashes: dict[str, str] = {}

    def read_rows(path: Path) -> list[list[str]]:
        data = path.read_bytes()
        hashes[path.relative_to(answers).as_posix()] = hashlib.sha256(data).hexdigest()
        # newline='' preserves CSV embedded newlines for exact raw comparison.
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream))

    master = read_rows(answers / "insiders.csv")
    if not master or len(set(master[0])) != len(master[0]):
        raise ValueError("Invalid master header")
    required = {"dataset", "scenario", "details", "user", "start", "end"}
    if not required.issubset(master[0]):
        raise ValueError("Missing master columns")
    incidents = []
    for row in master[1:]:
        if len(row) != len(master[0]):
            raise ValueError("Invalid master row width")
        item = dict(zip(master[0], row))
        if item["dataset"] == "4.2":
            incidents.append(item)
    if not incidents:
        raise ValueError("No r4.2 incidents")
    parents = {account: account for account in (accounts or set())}

    def root(account: str) -> str:
        if accounts is not None and account not in accounts:
            raise ValueError("Unknown incident or observable account")
        parents.setdefault(account, account)
        while parents[account] != account:
            parents[account] = parents[parents[account]]
            account = parents[account]
        return account

    rows = {}
    attribution: dict[tuple[str, str], set[Attribution]] = {}
    seen_incidents = set()
    for incident in sorted(incidents, key=lambda i: i["details"]):
        name = incident["details"]
        if Path(name).name != name or name in seen_incidents:
            raise ValueError("Invalid or duplicate incident detail name")
        seen_incidents.add(name)
        if incident["scenario"] not in {"1", "2", "3"}:
            raise ValueError("Unexpected r4.2 scenario")
        start = datetime.strptime(incident["start"], DATE_FORMAT)
        end = datetime.strptime(incident["end"], DATE_FORMAT)
        if start > end:
            raise ValueError("Reversed incident interval")
        matches = [p for p in answers.rglob(name) if not p.name.startswith("._")]
        if len(matches) != 1 or not matches[0].resolve().is_relative_to(answers):
            raise ValueError("Missing/ambiguous/outside-root incident file")
        detail_rows = read_rows(matches[0])
        if not detail_rows or not any(detail_rows):
            raise ValueError("Empty incident")
        root(incident["user"])
        for fields in detail_rows:
            if not fields:
                continue
            source, row = fields[0], tuple(fields[1:])
            validate_row(source, row)
            if not start <= datetime.strptime(row[1], DATE_FORMAT) <= end:
                raise ValueError("Observable outside incident interval")
            key = (source, row[0])
            if key in rows and rows[key] != row:
                raise ValueError("Conflicting rows for source-qualified ID")
            rows[key] = row
            attribution.setdefault(key, set()).add(
                Attribution(name, incident["user"], incident["scenario"]))
            left, right = root(incident["user"]), root(row[2])
            parents[max(left, right)] = min(left, right)
    groups = {account: root(account) for account in sorted(parents)}
    return LabelIndex(rows, {k: tuple(sorted(v, key=lambda a: a.incident_id))
                             for k, v in attribution.items()}, groups, hashes)
