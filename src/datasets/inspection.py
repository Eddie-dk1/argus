from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import SCENARIO_CONTRACTS
from .contracts import ColumnContract
from .contracts import ScenarioContract
from .contracts import SourceContract
from .contracts import normalize_column_name


NON_FINITE_TOKENS = {
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


@dataclass
class FileInspection:
    path: str
    size_bytes: int
    sampled_rows: int
    original_columns: list[str]
    canonical_columns: dict[str, str]
    missing_required_columns: list[str]
    duplicate_normalized_columns: list[str]
    sample_missing_cells: int
    sample_non_finite_tokens: int
    sample_duplicate_rows: int
    sample_repeated_header_rows: int
    errors: list[str]

    @property
    def valid_schema(self) -> bool:
        return not (
            self.missing_required_columns
            or self.duplicate_normalized_columns
            or self.errors
        )


def matches_source(path: Path, raw_dir: Path, source: SourceContract) -> bool:
    relative = path.relative_to(raw_dir).as_posix().lower()
    name = path.name.lower()
    return any(
        fnmatch.fnmatch(relative, pattern.lower())
        or fnmatch.fnmatch(name, pattern.lower())
        for pattern in source.patterns
    )


def build_normalized_column_map(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    by_normalized: dict[str, list[str]] = {}
    for original in columns:
        normalized = normalize_column_name(original)
        by_normalized.setdefault(normalized, []).append(original)

    duplicates = sorted(
        normalized
        for normalized, originals in by_normalized.items()
        if len(originals) > 1
    )
    unique_map = {
        normalized: originals[0]
        for normalized, originals in by_normalized.items()
    }
    return unique_map, duplicates


def resolve_contract_columns(
    available: dict[str, str],
    contract_columns: tuple[ColumnContract, ...],
) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for contract_column in contract_columns:
        match = next(
            (
                available[accepted]
                for accepted in contract_column.accepted_names
                if accepted in available
            ),
            None,
        )
        if match is not None:
            resolved[contract_column.canonical_name] = match
        elif contract_column.required:
            missing.append(contract_column.canonical_name)

    return resolved, missing


def count_repeated_headers(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0

    normalized_headers = [normalize_column_name(column) for column in frame.columns]
    repeated = 0
    for row in frame.itertuples(index=False, name=None):
        normalized_values = [normalize_column_name(value) for value in row]
        if normalized_values == normalized_headers:
            repeated += 1
    return repeated


def count_non_finite_tokens(frame: pd.DataFrame) -> int:
    count = 0
    for column in frame.columns:
        values = frame[column].dropna().astype(str).str.strip().str.lower()
        count += int(values.isin(NON_FINITE_TOKENS).sum())
    return count


def inspect_file(
    path: Path,
    raw_dir: Path,
    source: SourceContract,
    *,
    sample_rows: int,
) -> FileInspection:
    errors: list[str] = []
    try:
        frame = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    except Exception as error:
        return FileInspection(
            path=path.relative_to(raw_dir).as_posix(),
            size_bytes=path.stat().st_size,
            sampled_rows=0,
            original_columns=[],
            canonical_columns={},
            missing_required_columns=[
                item.canonical_name for item in source.columns if item.required
            ],
            duplicate_normalized_columns=[],
            sample_missing_cells=0,
            sample_non_finite_tokens=0,
            sample_duplicate_rows=0,
            sample_repeated_header_rows=0,
            errors=[f"{type(error).__name__}: {error}"],
        )

    original_columns = [str(column) for column in frame.columns]
    available, duplicate_columns = build_normalized_column_map(original_columns)
    resolved, missing = resolve_contract_columns(available, source.columns)

    return FileInspection(
        path=path.relative_to(raw_dir).as_posix(),
        size_bytes=path.stat().st_size,
        sampled_rows=int(len(frame)),
        original_columns=original_columns,
        canonical_columns=resolved,
        missing_required_columns=missing,
        duplicate_normalized_columns=duplicate_columns,
        sample_missing_cells=int(frame.isna().sum().sum()),
        sample_non_finite_tokens=count_non_finite_tokens(frame),
        sample_duplicate_rows=int(frame.duplicated().sum()),
        sample_repeated_header_rows=count_repeated_headers(frame),
        errors=errors,
    )


def inspect_scenario(
    scenario: str,
    raw_dir: Path,
    *,
    sample_rows: int = 10_000,
) -> dict[str, Any]:
    if scenario not in SCENARIO_CONTRACTS:
        supported = ", ".join(sorted(SCENARIO_CONTRACTS))
        raise ValueError(f"Unknown scenario {scenario!r}. Supported: {supported}")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")

    contract: ScenarioContract = SCENARIO_CONTRACTS[scenario]
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw path is not a directory: {raw_dir}")

    csv_files = sorted(
        path for path in raw_dir.rglob("*.csv")
        if path.is_file() and not path.name.startswith("._")
    )
    source_reports: list[dict[str, Any]] = []
    matched_paths: set[Path] = set()
    missing_source_groups: list[str] = []

    for source in contract.sources:
        matches = [path for path in csv_files if matches_source(path, raw_dir, source)]
        matched_paths.update(matches)
        if len(matches) < source.minimum_files:
            missing_source_groups.append(source.name)

        file_reports = [
            inspect_file(
                path,
                raw_dir,
                source,
                sample_rows=sample_rows,
            )
            for path in matches
        ]
        source_reports.append(
            {
                "source": source.name,
                "patterns": list(source.patterns),
                "minimum_files": source.minimum_files,
                "matched_files": len(matches),
                "files": [
                    {**asdict(report), "valid_schema": report.valid_schema}
                    for report in file_reports
                ],
            }
        )

    all_file_reports = [
        file_report
        for source_report in source_reports
        for file_report in source_report["files"]
    ]
    invalid_files = [
        file_report["path"]
        for file_report in all_file_reports
        if not file_report["valid_schema"]
    ]
    unmatched_files = [
        path.relative_to(raw_dir).as_posix()
        for path in csv_files
        if path not in matched_paths
    ]

    return {
        "scenario": scenario,
        "schema_status": contract.schema_status,
        "contract_notes": contract.notes,
        "raw_dir": str(raw_dir),
        "sample_rows_per_file": sample_rows,
        "discovered_csv_files": len(csv_files),
        "missing_source_groups": missing_source_groups,
        "invalid_files": invalid_files,
        "unmatched_csv_files": unmatched_files,
        "valid_schema": not missing_source_groups and not invalid_files,
        "sources": source_reports,
    }


def save_inspection_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
