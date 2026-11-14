"""Descriptive diagnostics for frozen decisions; labels stay outside inference."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from datasets.cert_reporting import error_report
from utils import compute_classification_metrics


RANK_METRICS = ('pr_auc', 'roc_auc')
METRICS = ('precision', 'recall', 'f1_score', 'pr_auc', 'roc_auc', 'false_positive_rate')
HISTORY_BUCKETS = ('0', '1-4', '5-9', '10+')


def descriptive_metrics(y, predictions, scores=None):
    """Keep legacy metrics for comparison, but name undefined quantities honestly."""
    y, predictions = np.asarray(y), np.asarray(predictions)
    if (y.ndim != 1 or predictions.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isin(predictions, [0, 1]).all()):
        raise ValueError('Aligned binary labels and decisions required for diagnostics')
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != y.shape or not np.isfinite(scores).all():
            raise ValueError('Aligned finite diagnostic scores required')
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    tp = int(((y == 1) & (predictions == 1)).sum())
    if len(y):
        metrics = compute_classification_metrics(y, predictions, scores)
    else:
        metrics = dict(precision=None, recall=None, f1_score=None, accuracy=None,
            false_positive_rate=None, confusion_matrix=[[0, 0], [0, 0]],
            support=dict(normal=0, anomaly=0, total=0), anomaly_prevalence=None, pr_auc=None, roc_auc=None)
    undefined = {}
    for name, denominator, reason in (
        ('precision', tp + fp, 'no predicted positives'),
        ('recall', tp + fn, 'no positive support'),
        ('f1_score', 2 * tp + fp + fn, 'no positive support or predicted positives'),
        ('false_positive_rate', tn + fp, 'no background support')):
        if not denominator:
            metrics[name] = None
            undefined[name] = reason
    if not len(y):
        undefined.update(accuracy='empty cohort', anomaly_prevalence='empty cohort')
    if scores is None or not (tp + fn and tn + fp):
        for name in RANK_METRICS:
            metrics[name] = None
            undefined[name] = 'cross-fold ranking not defined' if scores is None else 'both classes required'
    metrics.update(tp=tp, fp=fp, fn=fn, tn=tn,
                   fp_per_1000_background=1000 * fp / (tn + fp) if tn + fp else None)
    return metrics, undefined


def statistics(values):
    valid = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(valid).all():
        raise ValueError('Undefined statistics require None, not nonfinite values')
    identical = len(valid) > 0 and (valid == valid[0]).all()
    return dict(mean=float(valid[0] if identical else valid.mean()) if len(valid) else None,
        sample_std=(0. if identical else float(valid.std(ddof=1))) if len(valid) > 1 else None,
        min=float(valid.min()) if len(valid) else None, max=float(valid.max()) if len(valid) else None,
        defined=len(valid), total=len(values))


def observed_history(frame):
    """Count actually available earlier active days, separately from feature policy."""
    if frame.event_id.isna().any() or frame.event_id.duplicated().any():
        raise ValueError('Unique history event IDs required')
    ordered = frame[['event_id', 'user_id', 'timestamp']].sort_values(['user_id', 'timestamp', 'event_id'])
    result = pd.Series(index=ordered.event_id, dtype=np.int64)
    for _, account in ordered.groupby('user_id', sort=False):
        days = pd.to_datetime(account.timestamp).to_numpy(dtype='datetime64[D]').astype(np.int64)
        if len(days) > 1 and (np.diff(days) <= 0).any():
            raise ValueError('CERT account-days must be strictly ordered and unique')
        counts = np.arange(len(days)) - np.searchsorted(days, days - 30, side='left')
        result.loc[account.event_id] = counts
    return result.astype(np.int64)


def history_flags(frame, state, *, inverse_atol=1e-5):
    """Invert saved train scaling for binary reporting fields; never refit it."""
    flags = {}
    for name in ('history_missing', 'cold_start', 'file_usb_profile_missing'):
        if name not in state['columns']:
            flags[name] = None
            continue
        index = state['columns'].index(name)
        values = frame[name].to_numpy(float) * state['scale'][index] + state['mean'][index]
        restored = np.rint(values)
        if (not np.isfinite(values).all() or not np.isin(restored, [0, 1]).all()
                or (np.abs(values - restored) > inverse_atol).any()):
            raise ValueError(f'Cannot restore frozen binary history flag: {name}')
        flags[name] = pd.Series(restored.astype(np.int64), index=frame.event_id)
    return flags


def _stratum(y, predictions, scores, metadata, mask):
    metrics, undefined = descriptive_metrics(y[mask], predictions[mask], None if scores is None else scores[mask])
    part = metadata.loc[mask]
    return dict(metrics=metrics, undefined_metrics=undefined, accounts=int(part.account_id.nunique()),
                independent_groups=int(part.group.nunique()))


def cohort_diagnostics(ids, y, scores, predictions, metadata, evidence, *, history=None, flags=None):
    """One role/fold/profile only. Pooled cross-fold reports omit score rankings."""
    ids, y, predictions = map(np.asarray, (ids, y, predictions))
    scores = None if scores is None else np.asarray(scores)
    if len(np.unique(ids)) != len(ids) or metadata.item_id.duplicated().any():
        raise ValueError('Duplicate diagnostic items')
    selected = metadata.set_index('item_id').loc[ids].reset_index()
    if not np.array_equal(selected.label.to_numpy(), y):
        raise ValueError('Diagnostic metadata labels differ from saved inputs')
    rows = pd.DataFrame(dict(item_id=ids, y_true=y, score=np.zeros(len(y)) if scores is None else scores,
                             prediction=predictions))
    old = error_report(rows, metadata, evidence)
    metrics, undefined = descriptive_metrics(y, predictions, scores)
    background = selected.loc[y == 0].groupby('account_id').size().to_dict()
    fp_counts = old['false_positives_by_account']
    account_groups = selected.groupby('account_id')['group'].first().to_dict()
    ordered = sorted(fp_counts.items(), key=lambda item: (-item[1], item[0]))
    concentration = dict(accounts_with_fp=len(ordered),
        shares={str(n): sum(v for _, v in ordered[:n]) / metrics['fp'] if metrics['fp'] else None for n in (1, 3, 10)},
        accounts=[dict(account_id=k, group=account_groups[k], fp=v, background_days=int(background[k]),
                       fp_per_1000_background=1000 * v / background[k]) for k, v in ordered])
    strata = {}
    if history is None:
        strata['observed_prior30_days'] = dict(available=False, reason='history metadata unavailable', buckets=None)
    else:
        counts = history.loc[ids].to_numpy(np.int64)
        masks = (counts == 0, (counts >= 1) & (counts < 5), (counts >= 5) & (counts < 10), counts >= 10)
        strata['observed_prior30_days'] = dict(available=True,
            definition='earlier active days in [date-30,date); independent of feature fallback policy',
            buckets={name: _stratum(y, predictions, scores, selected, mask) for name, mask in zip(HISTORY_BUCKETS, masks)})
    for name in ('history_missing', 'cold_start', 'file_usb_profile_missing'):
        value = (flags or {}).get(name)
        if value is None:
            strata[name] = dict(available=False, reason='not a feature of this policy', buckets=None)
        else:
            values = value.loc[ids].to_numpy(np.int64)
            strata[name] = dict(available=True, definition='saved binary feature inverted through frozen train scaler',
                buckets={str(v): _stratum(y, predictions, scores, selected, values == v) for v in (0, 1)})
    return dict(metrics=metrics, undefined_metrics=undefined,
        population=dict(accounts=int(selected.account_id.nunique()), independent_groups=int(selected.group.nunique()),
            positive_accounts=int(selected.loc[y == 1, 'account_id'].nunique()),
            positive_independent_groups=int(selected.loc[y == 1, 'group'].nunique())),
        scenarios=old['scenarios'], positive_days_with_multiple_scenarios=old['positive_days_with_multiple_scenarios'],
        score_distributions=old['scores'] if scores is not None else None,
        score_distribution_unavailable_reason='different fitted fold score scales' if scores is None else None,
        false_alarm_concentration=concentration, history=strata)


def variation_reports(records):
    """Separate fixed-fold seed variation from fixed-seed fold variation, never CI."""
    seed_groups, fold_groups = defaultdict(list), defaultdict(list)
    for row in records:
        for role in ('calibration', 'assessment'):
            common = (row['series'], row['method'], row['feature_policy'], row['profile'], role)
            seed_groups[(*common, row['fold'])].append(row)
            fold_groups[(*common, row['seed'])].append(row)
    result = {}
    for axis, groups in (('model_seed_at_fixed_fold', seed_groups), ('fold_at_fixed_model_seed', fold_groups)):
        rows = []
        for key, values in sorted(groups.items()):
            series, method, policy, profile, role, fixed = key
            rows.append(dict(series=series, method=method, feature_policy=policy, profile=profile, role=role,
                fixed_fold=fixed if axis == 'model_seed_at_fixed_fold' else None,
                fixed_model_seed=fixed if axis == 'fold_at_fixed_model_seed' else None,
                metrics={m: statistics([r[role]['metrics'][m] for r in values]) for m in METRICS},
                observation_count=len(values), independent_confirmation=False,
                interpretation='descriptive sample SD; not a confidence interval or independent sample count'))
        result[axis] = rows
    return result
