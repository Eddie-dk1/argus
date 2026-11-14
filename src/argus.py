"""Argus frozen runtime: inspect, exact replay, CERT diagnosis and raw synthetic scenarios."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
CPU_THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', type=Path, required=True, help='Existing output root; every command creates a new UUID directory')
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'replay', 'scenario'):
        command = commands.add_parser(name)
        command.add_argument('--run', type=Path, required=True, help='Sealed isolated candidate or operating-point bundle')
        command.add_argument('--profile', help='Mandatory for an operating bundle; no automatic fallback')
        command.add_argument('--expected-manifest', help='Optional expected model complete.json SHA256')
        if name in ('inspect', 'replay'):
            command.add_argument('--source', type=Path, help='Sealed saved-input artifact; replay defaults to the model itself')
            command.add_argument('--role', choices=('validation', 'calibration', 'assessment'))
            command.add_argument('--expected-source-manifest', help='Optional saved-input complete.json SHA256')
        else:
            command.add_argument('--preparation', type=Path, required=True)
            command.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
            command.add_argument('--position', choices=('endpoint', 'earlier'), default='endpoint')
            command.add_argument('--operation', choices=('noop', 'double', 'zero', 'set'), default='noop')
            command.add_argument('--field', default='bytes_sent', help='Only bytes_sent is supported')
            command.add_argument('--value', type=float, help='Finite nonnegative number; operation=set only')
            command.add_argument('--item-id', help='Optional eligible endpoint; default selection uses IDs/time, no labels or scores')
    command = commands.add_parser('diagnose', help='Run the pinned, full S3 CERT replay/diagnosis protocol')
    command.add_argument('--model-variants', type=Path, required=True)
    command.add_argument('--operating-points', type=Path, required=True)
    command.add_argument('--config', type=Path, default=PROJECT / 'configs/s3_frozen_diagnostics_v1.json')
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Before importing numpy/sklearn/PyTorch, retain the original CPU replay settings.
    for name in CPU_THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.command == 'diagnose':
        # Keep the established test-access/write audit guard in its own process.
        return subprocess.run([sys.executable, str(PROJECT / 'src/run_frozen_diagnostics.py'),
            '--root', str(args.root), '--model-variants', str(args.model_variants),
            '--operating-points', str(args.operating_points), '--config', str(args.config)], check=False).returncode
    from inference import load_frozen_run
    run = load_frozen_run(args.run, profile=args.profile, expected_manifest_sha256=args.expected_manifest)
    if args.command == 'scenario':
        from synthetic_scenarios import run_scenario
        output = run_scenario(run, args.preparation, args.root, split=args.split, position=args.position,
            operation=args.operation, field=args.field, value=args.value, item_id=args.item_id)
    else:
        from frozen_runtime import inspect_run, replay_run
        if args.command == 'inspect' and args.source is None and (args.role or args.expected_source_manifest):
            raise ValueError('Input inspection with --role/--expected-source-manifest requires --source')
        action = inspect_run if args.command == 'inspect' else replay_run
        output = action(run, args.root, source=args.source, role=args.role,
                        expected_source_manifest=args.expected_source_manifest)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
