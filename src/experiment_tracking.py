from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils import PROJECT_ROOT
from utils import RESULTS_DIR
from utils import load_json
from utils import save_json


TRACKED_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "torch",
    "tqdm",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty_worktree": bool(status) if status is not None else None,
    }


def environment_metadata() -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for dependency in TRACKED_DEPENDENCIES:
        try:
            dependencies[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            dependencies[dependency] = None

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": dependencies,
        "git": git_metadata(),
    }


def build_data_manifest(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def create_run_directory(
    *,
    scenario: str,
    model: str,
    resolved_config: dict[str, Any],
    seed: int,
    data_path: Path,
    split_report_path: Path,
    feature_manifest_path: Path,
) -> tuple[Path, dict[str, Any]]:
    config_hash = payload_sha256(resolved_config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = "".join(character if character.isalnum() else "_" for character in model).strip("_")
    run_id = f"{timestamp}_{safe_model}_{config_hash[:8]}"
    run_dir = RESULTS_DIR / "experiments" / scenario / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = RESULTS_DIR / "experiments" / scenario / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (run_dir / "plots").mkdir()

    data_manifest = build_data_manifest(data_path)
    split_report = load_json(split_report_path)
    feature_manifest = load_json(feature_manifest_path)
    metadata = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "model": model,
        "seed": seed,
        "config_sha256": config_hash,
        "data_manifest_sha256": payload_sha256(data_manifest),
        "split_report_sha256": payload_sha256(split_report),
        "feature_manifest_sha256": payload_sha256(feature_manifest),
    }

    save_json(resolved_config, run_dir / "resolved_config.json")
    save_json(environment_metadata(), run_dir / "environment.json")
    save_json(data_manifest, run_dir / "data_manifest.json")
    save_json(split_report, run_dir / "split_report.json")
    save_json(feature_manifest, run_dir / "feature_manifest.json")
    save_json(metadata, run_dir / "run_metadata.json")
    return run_dir, metadata


def save_run_results(
    run_dir: Path,
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    threshold_candidates: list[dict[str, Any]],
    timing: dict[str, Any],
    error_analysis: pd.DataFrame,
    plot_paths: tuple[Path, ...] = (),
) -> None:
    save_json(validation_metrics, run_dir / "metrics_validation.json")
    save_json(test_metrics, run_dir / "metrics_test.json")
    save_json(timing, run_dir / "timing.json")
    pd.DataFrame(threshold_candidates).to_csv(
        run_dir / "threshold_search.csv",
        index=False,
    )
    error_analysis.to_csv(run_dir / "error_analysis.csv", index=False)
    for plot_path in plot_paths:
        if plot_path.exists():
            shutil.copy2(plot_path, run_dir / "plots" / plot_path.name)
