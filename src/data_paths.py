from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT_ENV = "ARGUS_DATA_ROOT"

SCENARIOS = (
    "synthetic",
    "cert_r4_2",
    "cicids2017",
    "cse_cic_ids2018",
)


@dataclass(frozen=True)
class ScenarioPaths:
    name: str
    root: Path
    raw: Path
    interim: Path
    processed: Path
    manifests: Path


def resolve_data_root() -> Path:
    configured_root = os.environ.get(DATA_ROOT_ENV)
    if not configured_root:
        return DEFAULT_DATA_ROOT

    data_root = Path(configured_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(
            f"{DATA_ROOT_ENV} points to a missing path: {data_root}. "
            "Mount the external drive and create its Argus data directory "
            "before running the pipeline."
        )
    if not data_root.is_dir():
        raise NotADirectoryError(
            f"{DATA_ROOT_ENV} must point to a directory: {data_root}"
        )
    return data_root


def get_scenario_paths(
    scenario: str,
    *,
    data_root: Path | None = None,
) -> ScenarioPaths:
    if scenario not in SCENARIOS:
        supported = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown scenario {scenario!r}. Supported: {supported}")

    root = (data_root or resolve_data_root()) / scenario
    return ScenarioPaths(
        name=scenario,
        root=root,
        raw=root / "raw",
        interim=root / "interim",
        processed=root / "processed",
        manifests=root / "manifests",
    )


def ensure_scenario_directories(paths: ScenarioPaths) -> None:
    for path in (
        paths.raw,
        paths.interim,
        paths.processed,
        paths.manifests,
    ):
        path.mkdir(parents=True, exist_ok=True)
