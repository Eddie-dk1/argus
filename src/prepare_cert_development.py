"""Prepare and audit v2/v3 on primary development only; never train a model."""
import argparse
from collections import defaultdict, deque
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from datasets.cert_experiments import (CAUSAL_POLICY, GLOBAL_POLICY, HISTORY_COLUMNS,
                                      prepare_cert_frame, read_context)
from experiment import allocate, finish, verify
from experiment_tracking import environment_metadata, file_sha256
from utils import load_json, save_json

PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = 'cert_r4_2_causal_history_development_v3'


def audit_dependencies(path, frame):
    """Independently check calendar cutoffs and the expanding-history chain."""
    histories, previous = defaultdict(deque), {}
    report = {role: {'days': 0, 'cold_start_days': 0, 'personal_profile_days': 0}
              for role in sorted(frame['split'].unique())}
    dates = pd.to_datetime(frame['date']).to_numpy(dtype='datetime64[D]').astype(int)
    rows = frame[['account_id', 'date', 'item_id', 'split']].itertuples(index=False, name=None)
    with gzip.open(path, 'rt') as stream:
        for (account, date, item, role), day, line in zip(rows, dates, stream, strict=True):
            dependency = json.loads(line)
            history = histories[account]
            while history and history[0][0] < day - 30:
                history.popleft()
            earlier = previous.get(account)
            expected = dict(item_id=item, account_id=account, cutoff_exclusive=date,
                            rolling30_members=[r[1] for r in history],
                            rolling7_members=[r[1] for r in history if r[0] >= day - 7],
                            previous_active_item_id=earlier[1] if earlier else None,
                            previous_active_date=earlier[2] if earlier else None,
                            profile_reference='rolling30_members' if len(history) >= 5 else 'frozen global train profile',
                            novelty_reference='all earlier same-account observed days' if earlier else 'frozen global train vocabulary')
            if any(dependency.get(key) != value for key, value in expected.items()):
                raise ValueError('Causal history dependency mismatch')
            if earlier is not None and earlier[0] >= day:
                raise ValueError('History includes current or future observations')
            report[role]['days'] += 1
            report[role]['cold_start_days'] += int(earlier is None)
            report[role]['personal_profile_days'] += int(len(history) >= 5)
            history.append((day, item))
            previous[account] = (day, item, date)
    return {'checked_rows': sum(row['days'] for row in report.values()), 'roles': report,
            'calendar_cutoffs_and_memberships': True, 'expanding_vocabulary_predecessor_chain': True}


def audit_pair(preparations, *, roles=('train', 'validation')):
    """Require identical populations and base inputs across the single factor."""
    paths = [preparations[key] for key in ('global_only_v2', 'causal_all_v3')]
    states = [load_json(path / 'feature_state.json') for path in paths]
    if states[0]['columns'] != states[1]['columns']:
        raise ValueError('Feature schema differs between policies')
    for key in ('volume_median', 'volume_iqr', 'vocabulary'):
        if states[0][key] != states[1][key]:
            raise ValueError('Global train reference changed between policies')
    columns = states[0]['columns']
    base = [name for name in columns if name not in HISTORY_COLUMNS]
    for key in ('mean', 'scale'):
        indices = [columns.index(name) for name in base]
        if not np.array_equal(np.asarray(states[0][key])[indices], np.asarray(states[1][key])[indices]):
            raise ValueError('Base feature scaling changed between policies')
    cohorts = [load_json(path / 'cohort_report.json') for path in paths]
    if cohorts[0] != cohorts[1]:
        raise ValueError('Different development cohorts')
    for role in roles:
        left, right = [pd.read_csv(path / (role + '.csv')) for path in paths]
        if (not left[['event_id', 'user_id', 'timestamp', 'label']].equals(
                right[['event_id', 'user_id', 'timestamp', 'label']])
                or not np.array_equal(left[base].to_numpy(), right[base].to_numpy())):
            raise ValueError('Policy changed population, target or base inputs')
        if any(not np.isfinite(part[columns].to_numpy()).all() for part in (left, right)):
            raise ValueError('Non-finite model inputs')
    return {'same_ordered_items_targets_and_base_inputs': True,
            'same_frozen_global_reference': True, 'feature_count': len(columns),
            'base_feature_count': len(base), 'history_feature_count': len(HISTORY_COLUMNS),
            'cohorts': cohorts[0]}


def run(days_path, root, plan_path):
    days_path, root, plan_path = Path(days_path).resolve(), Path(root).resolve(), Path(plan_path).resolve()
    plan = load_json(plan_path)
    if plan['plan_id'] != PLAN_ID:
        raise ValueError('Unsupported development preparation plan')
    if not root.is_dir() or shutil.disk_usage(root).free < plan['minimum_free_gib'] * 1024**3:
        raise ValueError('Existing artifact root with sufficient free space required')
    parent_hash = file_sha256(days_path / 'complete.json')
    if days_path.name != plan['daily_preparation_id'] or parent_hash != plan['daily_manifest_sha256']:
        raise ValueError('Different sealed daily source')
    expected = {'global_only_v2': GLOBAL_POLICY, 'causal_all_v3': CAUSAL_POLICY}
    if set(plan['configs']) != set(expected):
        raise ValueError('Both declared history policies are required')
    configs = {}
    for key, settings in plan['configs'].items():
        path = PROJECT / settings['path']
        if file_sha256(path) != settings['sha256']:
            raise ValueError('Configuration hash mismatch')
        configs[key] = load_json(path)
        if configs[key]['features']['history_policy'] != expected[key]:
            raise ValueError('Different history factor')
    comparable = []
    for config in configs.values():
        value = json.loads(json.dumps(config))
        value.pop('config_version')
        value['features'].pop('history_policy')
        comparable.append(value)
    if comparable[0] != comparable[1]:
        raise ValueError('Only history policy may change')
    # The sealed daily CSV co-locates all original roles. Verification/loading
    # reads that source, but test rows are excluded before any feature building.
    verify(days_path)
    daily_contract = load_json(days_path / 'contract.json')
    frame = pd.read_csv(days_path / 'account_days.csv')
    if set(frame['split']) != {'train', 'validation', 'test'}:
        raise ValueError('Expected the original three primary roles')
    if (frame['item_id'].duplicated().any()
            or frame.groupby('account_id')['split'].nunique().max() != 1
            or frame.groupby('group')['split'].nunique().max() != 1):
        raise ValueError('Daily identity or split leakage')
    output = allocate(root, 'cert_history_development')
    print('PREPARATION ' + str(output), flush=True)
    try:
        shutil.copy2(plan_path, output / 'plan.json')
        source_hashes = {}
        for source in sorted((PROJECT / 'src').rglob('*.py')):
            relative = source.relative_to(PROJECT / 'src')
            target = output / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[str(relative)] = file_sha256(source)
        save_json(source_hashes, output / 'execution_source_hashes.json')
        save_json(environment_metadata(), output / 'environment.json')
        context_path = output / 'development_context.jsonl.gz'
        keep = frame['split'].isin(['train', 'validation'])
        with gzip.open(context_path, 'wt', compresslevel=1) as stream:
            for include, context in zip(keep, read_context(days_path / 'day_context.jsonl.gz', frame), strict=True):
                if include:
                    stream.write(json.dumps(context) + '\n')
        frame = frame.loc[keep].reset_index(drop=True)
        provenance = {'scope': 'primary_development', 'source_population': 'original primary train and validation only',
                      'excluded_role': 'test', 'model_fitting_or_selection': False,
                      'used_test_status': 'consumed by the previous frozen final; no new confirmatory claim'}
        preparations = {}
        for key, config in configs.items():
            preparations[key] = prepare_cert_frame(frame, context_path, daily_contract, config,
                                                   days_path, output, development=provenance)
            verify(preparations[key])
            if (preparations[key] / 'test.csv').exists() or (preparations[key] / 'assessment.csv').exists():
                raise ValueError('Unexpected assessment or test output')
            print(json.dumps({'policy': key, 'prepared': preparations[key].name}), flush=True)
        pair = audit_pair(preparations)
        dependencies = audit_dependencies(preparations['causal_all_v3'] / 'history_dependencies.jsonl.gz', frame)
        for relative, digest in source_hashes.items():
            if file_sha256(PROJECT / 'src' / relative) != digest:
                raise ValueError('Source changed during frozen preparation')
        if file_sha256(days_path / 'complete.json') != parent_hash:
            raise ValueError('Parent manifest changed during preparation')
        summary = {'plan_id': PLAN_ID, 'daily_manifest_sha256': parent_hash,
                   'preparations': {key: {'id': path.name, 'manifest_sha256': file_sha256(path / 'complete.json')}
                                    for key, path in preparations.items()},
                   'pair_audit': pair, 'causal_dependency_audit': dependencies,
                   'model_fitting': False, 'threshold_selection': False, 'test_feature_building_or_scoring': False,
                   'interpretation': 'technical preparation evidence only; detection improvement has not been measured'}
        save_json(summary, output / 'summary.json')
        finish(output)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': str(error)}, output / 'status.json')
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.days, args.root, args.plan))


if __name__ == '__main__':
    main()
