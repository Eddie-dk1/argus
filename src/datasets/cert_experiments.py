"""Frozen train state and causal personal history for CERT daily preparations."""
from collections import defaultdict, deque
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.cert_days import FEATURES, FEATURE_SET
from datasets import cert_file_usb

POLICY = 'cert_frozen_global_causal_train_v1'
GLOBAL_POLICY = 'cert_frozen_global_only_v2'
CAUSAL_POLICY = 'cert_frozen_global_causal_all_v3'
POLICIES = (POLICY, GLOBAL_POLICY, CAUSAL_POLICY, *cert_file_usb.POLICIES)
CAUSAL_POLICIES = (CAUSAL_POLICY, *cert_file_usb.POLICIES)
HISTORY_COLUMNS = ['prior7_events_log1p', 'prior30_events_log1p', 'prior30_days_log1p',
                   'history_missing', 'cold_start', 'volume_robust_deviation',
                   'pc_novel_fraction', 'domain_novel_fraction', 'recipient_novel_fraction',
                   'days_since_previous_active_log1p']


def base_values(frame):
    values = {}
    for name in FEATURES:
        column = frame[name].to_numpy(dtype=float)
        if not np.isfinite(column).all() or (column < 0).any():
            raise ValueError('Invalid daily numeric values')
        if name in ('first_second', 'last_second'):
            values[name + '_sin'] = np.sin(column * (2 * np.pi / 86400))
            values[name + '_cos'] = np.cos(column * (2 * np.pi / 86400))
        else:
            values[name] = column if name.endswith('_fraction') or name == 'is_weekend' else np.log1p(column)
    return pd.DataFrame(values, index=frame.index)


def read_context(path, frame):
    with gzip.open(path, 'rt') as stream:
        for key, line in zip(frame['day_key'], stream, strict=True):
            value = json.loads(line)
            if value['day_key'] != key:
                raise ValueError('Daily context order/identity mismatch')
            yield value


def fit_global(frame, context_path, policy=POLICY):
    if policy not in POLICIES:
        raise ValueError('Unsupported CERT history policy')
    train = frame['split'].eq('train').to_numpy()
    if not train.any():
        raise ValueError('Empty train population')
    volumes = np.log1p(frame.loc[train, 'total_events'].to_numpy(float))
    q25, median, q75 = np.quantile(volumes, [.25, .5, .75])
    vocabulary = {key: set() for key in ('pcs', 'domains', 'recipients')}
    for keep, context in zip(train, read_context(context_path, frame), strict=True):
        if keep:
            for key in vocabulary:
                vocabulary[key].update(context[key])
    state = {'policy': policy, 'volume_median': float(median), 'volume_iqr': max(float(q75 - q25), 1e-6),
            'vocabulary': {k: sorted(v) for k, v in vocabulary.items()},
            'profile_fit_population': 'all train days, labels unused',
            'held_out_policy': 'frozen global profile/vocabulary; no personal history adaptation'}
    if policy in CAUSAL_POLICIES:
        state.update(held_out_policy='same-account past observations only; update after emitting each day; labels unused',
                     global_reference_scope='offline all-train reference; not a temporal forecasting protocol',
                     personal_history_policy='all roles; rolling [date-30,date) profile and [date-7,date) counts; expanding past vocabulary')
    if policy in cert_file_usb.POLICIES:
        state['file_usb_profile'] = cert_file_usb.fit_global(frame, policy)
    return state


def history_values(frame, context_path, state, dependency_path=None):
    """Replay observed days in order; emit before updating personal history.

    Global state stays frozen. v3 uses the same personal update rule in every
    role, including positive days: no label-based cleaning of held-out history.
    Each call replays the supplied history from scratch, not a serving session.
    """
    if state['policy'] not in POLICIES:
        raise ValueError('Unsupported CERT history policy')
    global_vocab = {k: set(v) for k, v in state['vocabulary'].items()}
    histories = defaultdict(deque)
    vocabularies = {}
    previous_dates = {}
    previous_items = {}
    result = []
    dates = pd.to_datetime(frame['date']).to_numpy(dtype='datetime64[D]').astype(int)
    stream = gzip.open(dependency_path, 'wt') if dependency_path else None
    try:
        rows = frame[['account_id', 'date', 'split', 'item_id', 'total_events']].itertuples(index=False, name=None)
        for row, day, context in zip(rows, dates, read_context(context_path, frame), strict=True):
            account, timestamp, split, item_id, total = row
            previous = previous_dates.get(account)
            if previous is not None and day <= previous:
                raise ValueError('Duplicate or nonchronological account-day')
            gap = 0 if previous is None else day - previous
            previous_dates[account] = day
            history = histories[account]
            while history and history[0][0] < day - 30:
                history.popleft()
            personal = state['policy'] in CAUSAL_POLICIES or (split == 'train' and state['policy'] == POLICY)
            enough = personal and len(history) >= 5
            volumes = np.log1p([r[1] for r in history]) if enough else None
            if enough:
                low, median, high = np.quantile(volumes, [.25, .5, .75])
                spread = max(float(high - low), 1e-6)
            else:
                median, spread = state['volume_median'], state['volume_iqr']
            prior7 = sum(r[1] for r in history if r[0] >= day - 7) if personal else 0
            prior30 = sum(r[1] for r in history) if personal else 0
            personal_vocab = vocabularies.get(account)
            reference = personal_vocab if personal and personal_vocab is not None else global_vocab
            novelty = [len(set(context[key]) - reference[key]) / max(len(set(context[key])), 1)
                       for key in ('pcs', 'domains', 'recipients')]
            # Robust deviations are clipped by a predeclared constant, not held-out quantiles.
            deviation = float(np.clip((np.log1p(total) - median) / spread, -20, 20))
            result.append([np.log1p(prior7), np.log1p(prior30), np.log1p(len(history) if personal else 0),
                           int(not enough), int(not personal or personal_vocab is None), deviation,
                           *novelty, np.log1p(gap)])
            if stream:
                dependency = {'item_id': item_id,
                    'rolling30_members': [r[2] for r in history] if personal else [],
                    'rolling7_members': [r[2] for r in history if r[0] >= day - 7] if personal else [],
                    'novelty_reference': 'all earlier same-account train days' if personal_vocab is not None and personal else 'frozen global train vocabulary',
                    'profile_reference': 'rolling30_members' if enough else 'frozen global train profile',
                    'cutoff_exclusive': timestamp, 'account_id': account,
                    'previous_active_date': None if previous is None else str(np.datetime64(int(previous), 'D'))}
                if state['policy'] in CAUSAL_POLICIES:
                    dependency.update(
                        novelty_reference='all earlier same-account observed days' if personal_vocab is not None else 'frozen global train vocabulary',
                        # The predecessor chain records expanding vocabulary
                        # dependencies without a quadratic list of all past days.
                        previous_active_item_id=previous_items.get(account),
                        update_policy='emit then observe current day; every role; labels unused')
                stream.write(json.dumps(dependency) + '\n')
            if personal:
                history.append((day, total, item_id))
                if personal_vocab is None:
                    personal_vocab = {k: set() for k in global_vocab}
                    vocabularies[account] = personal_vocab
                for key in personal_vocab:
                    personal_vocab[key].update(context[key])
            previous_items[account] = item_id
    finally:
        if stream:
            stream.close()
    return pd.DataFrame(result, columns=HISTORY_COLUMNS, index=frame.index)


def feature_values(frame, context_path, state, dependency_path=None, file_usb_dependency_path=None):
    values = pd.concat([base_values(frame), history_values(frame, context_path, state, dependency_path)], axis=1)
    if state['policy'] in cert_file_usb.POLICIES:
        values = pd.concat([values, cert_file_usb.values(frame, state, file_usb_dependency_path)], axis=1)
    return values


def transform(frame, context_path, state, dependency_path=None):
    values = feature_values(frame, context_path, state, dependency_path)
    if list(values.columns) != state['columns']:
        raise ValueError('Feature order changed')
    array = (values.to_numpy(float) - np.array(state['mean'])) / np.array(state['scale'])
    if not np.isfinite(array).all():
        raise ValueError('Non-finite transformed values')
    return pd.DataFrame(array.astype(np.float32), columns=state['columns'], index=frame.index)


def prepare_cert(days_path, config_path, root):
    # Local import avoids coupling the pure feature functions to experiment tooling.
    from experiment import verify
    from utils import load_json
    days_path = Path(days_path).resolve()
    verify(days_path)
    daily_contract = load_json(days_path / 'contract.json')
    config = load_json(config_path)
    if (config['scenario'] != 'cert_r4_2' or config['features']['history_policy'] not in POLICIES
            or daily_contract['feature_set'] != FEATURE_SET or daily_contract['feature_columns'] != FEATURES):
        raise ValueError('Unsupported CERT preparation contract')
    frame = pd.read_csv(days_path / 'account_days.csv')
    return prepare_cert_frame(frame, days_path / 'day_context.jsonl.gz', daily_contract,
                              config, days_path, root)


def prepare_cert_frame(frame, context_path, daily_contract, config, days_path, root, *, development=None):
    """Fit one frozen preparation; CV assessments are separate from real test.

    Callers verify the parent manifest and filter original train before CV.
    Fitting uses only rows whose current role is train.
    """
    from experiment import allocate, finish
    from experiment_tracking import environment_metadata, file_sha256
    from utils import save_json
    frame = frame.reset_index(drop=True)
    policy = config['features']['history_policy']
    if (config['scenario'] != 'cert_r4_2' or policy not in POLICIES
            or daily_contract['feature_set'] != FEATURE_SET or daily_contract['feature_columns'] != FEATURES):
        raise ValueError('Unsupported CERT preparation contract')
    length = config['features']['sequence_length']
    if not isinstance(length, int) or length < 1:
        raise ValueError('Sequence length must be a positive integer')
    scope = development.get('scope', 'development_group_cv') if development is not None else 'primary_account_holdout'
    scopes = {'primary_account_holdout': ('train', 'validation', 'test'),
              'development_group_cv': ('train', 'validation', 'assessment'),
              'primary_development': ('train', 'validation')}
    if scope not in scopes:
        raise ValueError('Unsupported CERT preparation scope')
    splits = scopes[scope]
    if (frame['item_id'].isna().any() or frame['item_id'].duplicated().any()
            or frame.duplicated(['account_id', 'date']).any()
            or not frame['label'].isin([0, 1]).all()
            or set(frame['split']) != set(splits)):
        raise ValueError('Invalid daily identity/target/splits')
    if (frame.groupby('account_id')['split'].nunique().max() != 1
            or frame.groupby('group')['split'].nunique().max() != 1):
        raise ValueError('Account/group split leakage')
    # read_context enforces the same date/account order as the sealed parent artifact.
    path = allocate(root, 'cert_preparation')
    try:
        state = fit_global(frame, context_path, policy)
        values = feature_values(frame, context_path, state, path / 'history_dependencies.jsonl.gz',
                                path / 'file_usb_dependencies.jsonl.gz')
        train_values = values.loc[frame['split'].eq('train')].to_numpy(float)
        state.update(columns=list(values.columns), mean=train_values.mean(axis=0).tolist(),
                     scale=np.where(train_values.std(axis=0) > 1e-12, train_values.std(axis=0), 1).tolist(),
                     scaler_fit_population='all train rows only', clip_deviation=20,
                     minimum_personal_days=5, rolling_calendar_days=[7, 30])
        array = (values.to_numpy(float) - np.array(state['mean'])) / np.array(state['scale'])
        if not np.isfinite(array).all():
            raise ValueError('Non-finite model inputs')
        transformed = pd.DataFrame(array.astype(np.float32), columns=state['columns'])
        transformed.insert(0, 'label', frame['label'].to_numpy())
        transformed.insert(0, 'timestamp', frame['date'].to_numpy())
        transformed.insert(0, 'user_id', frame['account_id'].to_numpy())
        transformed.insert(0, 'event_id', frame['item_id'].to_numpy())
        report = {}
        for split in splits:
            mask = frame['split'].eq(split)
            part = transformed.loc[mask].sort_values(['user_id', 'timestamp', 'event_id'])
            part.to_csv(path / f'{split}.csv', index=False)
            eligible = part.groupby('user_id').cumcount() >= length - 1
            if not eligible.any():
                raise ValueError('Empty sequence endpoint cohort')
            report[split] = {'days': len(part), 'endpoints': int(eligible.sum()),
                             'excluded_warmup_days': int((~eligible).sum()),
                             'positive_endpoints': int(part.loc[eligible, 'label'].sum())}
        save_json(config, path / 'config.json')
        save_json(state, path / 'feature_state.json')
        save_json({'scenario': 'cert_r4_2', 'snapshot_id': daily_contract['snapshot_id'],
                   'preparation_id': path.name, 'sequence_length': length,
                   'tabular_columns': state['columns'], 'sequence_columns': state['columns'],
                   'target': 'endpoint_account_day_observable', 'history_policy': policy,
                   'sequence_policy': 'last N active account-days; no padding; calendar gaps retained as feature',
                   'normal_window_policy': 'all_members_background',
                   'daily_source': str(days_path), 'daily_manifest_sha256': file_sha256(days_path / 'complete.json'),
                   'scope': scope},
                  path / 'contract.json')
        if development is not None:
            save_json(development, path / 'development.json')
        lineage = {'daily_source': str(days_path), 'daily_manifest_sha256': file_sha256(days_path / 'complete.json'),
                   'implementation_sha256': file_sha256(Path(__file__)),
                   'dependencies': 'daily item_id joins parent account_days and source membership; history file records extra dependencies'}
        if policy in cert_file_usb.POLICIES:
            lineage['file_usb_implementation_sha256'] = file_sha256(Path(cert_file_usb.__file__))
        save_json(lineage, path / 'lineage.json')
        save_json(report, path / 'cohort_report.json')
        save_json(environment_metadata(), path / 'environment.json')
        finish(path)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': str(error)}, path / 'status.json')
        raise
    return path
