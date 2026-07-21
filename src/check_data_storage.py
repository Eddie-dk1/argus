from __future__ import annotations

import argparse
import shutil
import sys

from data_paths import DATA_ROOT_ENV
from data_paths import SCENARIOS
from data_paths import ensure_scenario_directories
from data_paths import get_scenario_paths
from data_paths import resolve_data_root


BYTES_PER_GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the configured Argus data root before acquisition.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Optionally inspect or create directories for one scenario.",
    )
    parser.add_argument(
        "--required-gib",
        type=float,
        default=0.0,
        help="Fail if less than this many GiB are currently available.",
    )
    parser.add_argument(
        "--create-scenario-directories",
        action="store_true",
        help="Create raw/interim/processed/manifests for the selected scenario.",
    )
    return parser.parse_args()


def format_gib(size_bytes: int) -> str:
    return f"{size_bytes / BYTES_PER_GIB:.2f} GiB"


def main() -> int:
    args = parse_args()
    if args.required_gib < 0:
        raise ValueError("--required-gib must be zero or positive.")
    if args.create_scenario_directories and not args.scenario:
        raise ValueError(
            "--create-scenario-directories requires --scenario to avoid "
            "creating unintended directories."
        )

    data_root = resolve_data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_root)

    print(f"{DATA_ROOT_ENV}: {data_root}")
    print(f"Checked path: {data_root}")
    print(f"Capacity: {format_gib(usage.total)}")
    print(f"Used: {format_gib(usage.used)}")
    print(f"Available: {format_gib(usage.free)}")

    if args.scenario:
        paths = get_scenario_paths(args.scenario, data_root=data_root)
        if args.create_scenario_directories:
            ensure_scenario_directories(paths)
        print(f"Scenario: {paths.name}")
        print(f"Scenario root: {paths.root}")
        print(
            "Scenario directories: "
            + ("ready" if paths.root.exists() else "not created")
        )

    required_bytes = int(args.required_gib * BYTES_PER_GIB)
    if usage.free < required_bytes:
        print(
            f"ERROR: required {args.required_gib:.2f} GiB, but only "
            f"{format_gib(usage.free)} is available.",
            file=sys.stderr,
        )
        return 2

    if required_bytes:
        print(f"Free-space check: PASS ({args.required_gib:.2f} GiB required)")
    return 0


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
