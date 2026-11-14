"""Independent preparation checks and post-inference adaptation diagnostics."""
from collections import defaultdict, deque
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.cert_file_usb import COLUMNS, RAW_COLUMNS, FROZEN_POLICY
from utils import load_json, save_json


def audit_preparations(preparations, frame):
    baseline = preparations['causal_all_v3']
    old_state = load_json(baseline / 'feature_state.json')
    common = old_state['columns']
    cohorts = load_json(baseline / 'cohort_report.json')
    report = {}
    for policy in ('file_usb_rolling_v4', 'file_usb_frozen_v4'):
        path = preparations[policy]
        state = load_json(path / 'feature_state.json')
        if state['columns'] != common + COLUMNS or load_json(path / 'cohort_report.json') != cohorts:
            raise ValueError('File/USB ablation changed schema or cohort')
        for key in ('volume_median', 'volume_iqr', 'vocabulary'):
            if state[key] != old_state[key]:
                raise ValueError('File/USB ablation changed the common global reference')
        for key in ('mean', 'scale'):
            if state[key][:len(common)] != old_state[key]:
                raise ValueError('File/USB ablation changed common feature scaling')
        profile = state['file_usb_profile']
        raw = np.log1p(frame.loc[frame.split.eq('train'), list(RAW_COLUMNS)].to_numpy(float))
        low, median, high = np.percentile(raw, [25, 50, 75], axis=0)
        if (not np.array_equal(median, profile['median'])
                or not np.array_equal(np.maximum(high - low, 1.0), profile['iqr'])):
            raise ValueError('File/USB fallback did not fit inner train only')
        # Compute expected values from the recorded members independently of
        # the feature builder; membership is checked against calendar replay.
        history, initial, previous = defaultdict(deque), defaultdict(list), {}
        lookup = dict(zip(frame.item_id, np.log1p(frame[list(RAW_COLUMNS)].to_numpy(float)), strict=True))
        expected, counts = [], defaultdict(lambda: dict(days=0, fallback_days=0))
        days = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
        with gzip.open(path / 'file_usb_dependencies.jsonl.gz', 'rt') as stream:
            for row, day, line in zip(frame.itertuples(index=False), days, stream, strict=True):
                dependency = json.loads(line)
                while history[row.account_id] and history[row.account_id][0][0] < day - 30:
                    history[row.account_id].popleft()
                members = (initial[row.account_id] if state['policy'] == FROZEN_POLICY
                           else [entry[1] for entry in history[row.account_id]])
                enough = len(members) >= 5
                required = dict(item_id=row.item_id, account_id=row.account_id, cutoff_exclusive=row.date,
                    reference=profile['reference'] if enough else 'frozen_global_train',
                    profile_members=members if enough else [], available_personal_members=members,
                    update_policy='emit then observe current day in every role; never clean by labels')
                if dependency != required or (row.account_id in previous and day <= previous[row.account_id]):
                    raise ValueError('Noncausal or incorrect file/USB reference membership')
                if enough:
                    values = np.asarray([lookup[item] for item in members])
                    lo, med, hi = np.percentile(values, [25, 50, 75], axis=0)
                    spread = np.maximum(hi - lo, 1.0)
                else:
                    med, spread = np.asarray(profile['median']), np.asarray(profile['iqr'])
                current = np.log1p(np.asarray([getattr(row, name) for name in RAW_COLUMNS], float))
                expected.append([*np.clip((current - med) / spread, -20, 20), int(not enough)])
                counts[row.split]['days'] += 1
                counts[row.split]['fallback_days'] += int(not enough)
                history[row.account_id].append((day, row.item_id))
                if len(initial[row.account_id]) < 5:
                    initial[row.account_id].append(row.item_id)
                previous[row.account_id] = day
        expected = np.asarray(expected)
        indices = [state['columns'].index(name) for name in COLUMNS]
        scaled = ((expected - np.asarray(state['mean'])[indices]) / np.asarray(state['scale'])[indices]).astype(np.float32)
        indexed = pd.DataFrame(scaled, columns=COLUMNS, index=frame.item_id)
        for role in ('train', 'validation', 'assessment'):
            old, new = [pd.read_csv(part / (role + '.csv')) for part in (baseline, path)]
            if (not old[['event_id', 'user_id', 'timestamp', 'label'] + common].equals(
                    new[['event_id', 'user_id', 'timestamp', 'label'] + common])
                    or not np.array_equal(new[COLUMNS].to_numpy(np.float32), indexed.loc[new.event_id].to_numpy())):
                raise ValueError('File/USB ablation changed common inputs or has incorrect deviations')
        report[policy] = dict(checked_days=len(expected), roles=dict(counts), feature_count=len(state['columns']),
                             same_common_inputs_labels_and_cohort=True, inner_fit_fallback_verified=True,
                             causal_memberships_and_numerical_deviations_verified=True)
    return report


def adaptation_report(results, primary, output):
    """Labels are used here only after every candidate/assessment is frozen.

    Prior observable-positive counts are a descriptive stress stratum, not
    incident onset or independently established malicious duration.
    """
    history, strata = defaultdict(deque), {}
    days = pd.to_datetime(primary.date).to_numpy(dtype='datetime64[D]').astype(int)
    for row, day in zip(primary.itertuples(index=False), days, strict=True):
        past = history[row.account_id]
        while past and past[0][0] < day - 30:
            past.popleft()
        count = sum(entry[1] for entry in past)
        strata[row.item_id] = '0' if count == 0 else '1_to_4' if count < 5 else '5_or_more'
        past.append((day, row.label))
    predictions, deviations = {}, {}
    seen = set()
    for result in results:
        key = result['policy'], result['model']
        with np.load(result['assessment'] + '/predictions_assessment.npz', allow_pickle=False) as arrays:
            rows = pd.DataFrame(dict(item_id=arrays['item_ids'], label=arrays['y'], prediction=arrays['predictions']))
        rows['stratum'] = rows.item_id.map(strata)
        predictions.setdefault(key, []).append(rows)
        prep = result['preparation']
        if result['policy'] == 'causal_all_v3' or prep in seen:
            continue
        seen.add(prep)
        state = load_json(Path(prep) / 'feature_state.json')
        part = pd.read_csv(prep + '/assessment.csv')
        length = load_json(Path(prep) / 'contract.json')['sequence_length']
        part = part.loc[part.groupby('user_id').cumcount() >= length - 1].copy()
        indices = [state['columns'].index(name) for name in COLUMNS]
        raw = part[COLUMNS].to_numpy() * np.asarray(state['scale'])[indices] + np.asarray(state['mean'])[indices]
        for index, name in enumerate(COLUMNS):
            part[name] = raw[:, index]
        part['stratum'] = part.event_id.map(strata)
        deviations.setdefault(result['policy'], []).append(part[['event_id', 'label', 'stratum'] + COLUMNS])
    report = dict(stratum='number of previous observable-positive active days in [date-30,date); reporting only',
                  bins=['0', '1_to_4', '5_or_more'], predictions=[], deviations=[])
    for (policy, model), parts in predictions.items():
        rows = pd.concat(parts, ignore_index=True)
        for (label, stratum), part in rows.groupby(['label', 'stratum']):
            report['predictions'].append(dict(policy=policy, model=model, label=int(label), stratum=stratum,
                days=len(part), alarms=int(part.prediction.sum()), alarm_fraction=float(part.prediction.mean())))
    for policy, parts in deviations.items():
        rows = pd.concat(parts, ignore_index=True)
        for (label, stratum), part in rows.groupby(['label', 'stratum']):
            report['deviations'].append(dict(policy=policy, label=int(label), stratum=stratum, days=len(part),
                features={name: dict(median=float(part[name].median()), mean=float(part[name].mean()),
                                    positive_fraction=float(part[name].gt(1e-6).mean())) for name in COLUMNS[:3]}))
    save_json(report, output / 'reports' / 'adaptation.json')
    return report
