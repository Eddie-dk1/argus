from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .contracts import SCENARIO_CONTRACTS
from .contracts import SourceContract
from .contracts import normalize_column_name
from .inspection import build_normalized_column_map
from .inspection import matches_source
from .inspection import resolve_contract_columns


PROVENANCE_COLUMNS = (
    "__source_file",
    "__source_group",
    "__source_row",
)


def build_rename_map(
    original_columns: list[str],
    source: SourceContract,
) -> dict[str, str]:
    available, duplicates = build_normalized_column_map(original_columns)
    if duplicates:
        raise ValueError(
            "Columns collide after normalization: " + ", ".join(duplicates)
        )

    resolved, missing = resolve_contract_columns(available, source.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    original_to_canonical = {
        original: canonical for canonical, original in resolved.items()
    }
    rename_map: dict[str, str] = {}
    used_names: set[str] = set(PROVENANCE_COLUMNS)

    for original in original_columns:
        normalized = original_to_canonical.get(
            original,
            normalize_column_name(original),
        )
        if normalized in used_names:
            raise ValueError(
                f"Column {original!r} resolves to duplicate name {normalized!r}"
            )
        used_names.add(normalized)
        rename_map[original] = normalized

    return rename_map


def iter_csv_chunks(
    path: Path,
    raw_dir: Path,
    source: SourceContract,
    *,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    header = pd.read_csv(path, nrows=0)
    original_columns = [str(column) for column in header.columns]
    rename_map = build_rename_map(original_columns, source)
    source_row = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        frame = chunk.rename(columns=rename_map)
        frame.insert(0, "__source_row", range(source_row, source_row + len(frame)))
        frame.insert(0, "__source_group", source.name)
        frame.insert(0, "__source_file", path.relative_to(raw_dir).as_posix())
        source_row += len(frame)
        yield frame


def iter_scenario_chunks(
    scenario: str,
    raw_dir: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    if scenario not in SCENARIO_CONTRACTS:
        supported = ", ".join(sorted(SCENARIO_CONTRACTS))
        raise ValueError(f"Unknown scenario {scenario!r}. Supported: {supported}")

    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw directory does not exist: {raw_dir}")

    contract = SCENARIO_CONTRACTS[scenario]
    csv_files = sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())

    for source in contract.sources:
        matches = [path for path in csv_files if matches_source(path, raw_dir, source)]
        if len(matches) < source.minimum_files:
            raise FileNotFoundError(
                f"Scenario {scenario!r} requires source group {source.name!r}; "
                f"expected patterns: {', '.join(source.patterns)}"
            )
        for path in matches:
            yield from iter_csv_chunks(
                path,
                raw_dir,
                source,
                chunk_size=chunk_size,
            )
