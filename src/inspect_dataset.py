from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_paths import SCENARIOS
from data_paths import get_scenario_paths
from datasets.inspection import inspect_scenario
from datasets.inspection import save_inspection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a dataset directory against an Argus schema contract.",
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Override the scenario raw directory (useful for fixtures).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10_000,
        help="Maximum rows sampled from each CSV for lightweight quality checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path. Defaults to the scenario manifests directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_scenario_paths(args.scenario)
    raw_dir = args.raw_dir or paths.raw
    output = args.output or paths.manifests / "schema_inspection.json"

    report = inspect_scenario(
        args.scenario,
        raw_dir,
        sample_rows=args.sample_rows,
    )
    save_inspection_report(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved inspection report to: {output}")
    return 0 if report["valid_schema"] else 2


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
