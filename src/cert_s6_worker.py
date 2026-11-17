"""S6 v2 CERT feature, fit and selection contract."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from datasets.cert_file_usb import robust_profile
from experiment_tracking import payload_sha256

RAW = ('off_hours_events', 'logon_off_hours_events', 'device_off_hours_events',
       'email_events', 'email_external_message_count', 'email_external_recipient_count',
       'email_size_bytes_sum', 'email_attachment_count', 'email_with_attachment_count',
       'http_events', 'http_distinct_domains', 'http_distinct_urls')
COLUMNS = ([x + '_behavior_deviation' for x in RAW] + [x + '_behavior_trend' for x in RAW]
           + ['behavior_profile_missing', 'behavior_trend_missing', 'behavior_prior7_days_log1p'])


def history(frame):
    """One role. Each dependency list is the ordered item_id slice [start, row)."""
    if frame.empty or not frame.item_id.is_unique or frame[['account_id', 'date', 'item_id']].isna().any().any():
        raise ValueError('Missing or duplicate daily identity')
    ordered = frame.sort_values(['account_id', 'date'])
    if not ordered.item_id.equals(frame.item_id) or frame.duplicated(['account_id', 'date']).any():
        raise ValueError('Expected unique sorted account-days')
    raw = frame[list(RAW)].to_numpy(float)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Invalid behavior count')
    values = np.log1p(raw)
    days = pd.to_datetime(frame.date, errors='raise').to_numpy('datetime64[D]').astype(int)
    idx = np.arange(len(frame))
    starts = {}
    for width in (7, 30):
        start = np.empty_like(idx)
        for rows in frame.groupby('account_id', sort=False).indices.values():
            start[rows] = rows[0] + days[rows].searchsorted(days[rows] - width)
        starts[width] = start
    return values, starts


def personal(values, starts, median, spread):
    """Evaluate equal-length past windows in batches to bound temporary memory."""
    idx = np.arange(len(values))
    n30, n7 = idx - starts[30], idx - starts[7]
    center = np.broadcast_to(median, values.shape).copy()
    scale = np.broadcast_to(spread, values.shape).copy()
    short = np.zeros_like(values)
    for width, counts in ((30, n30), (7, n7)):
        usable = (n30 >= 5) if width == 30 else ((n30 >= 5) & (n7 >= 2))
        for count in np.unique(counts[usable]):
            rows = idx[usable & (counts == count)]
            for offset in range(0, len(rows), 1024):
                batch = rows[offset:offset + 1024]
                past = values[starts[width][batch, None] + np.arange(count)]
                if width == 30:
                    low, center[batch], high = np.quantile(past, [.25, .5, .75], axis=1)
                    scale[batch] = np.maximum(high - low, 1)
                else:
                    short[batch] = (np.median(past, axis=1) - center[batch]) / scale[batch]
    return np.column_stack((np.clip((values - center) / scale, -20, 20),
                            np.clip(short, -20, 20), n30 < 5, (n30 < 5) | (n7 < 2), np.log1p(n7)))


def fit_personal(frame):
    values, starts = history(frame)
    median, spread = robust_profile(values)
    raw = personal(values, starts, median, spread)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std[std <= 1e-12] = 1.
    state = dict(columns=COLUMNS, median=median.tolist(), spread=spread.tolist(),
                 mean=mean.tolist(), std=std.tolist())
    return state, (raw - mean) / std, starts


def apply_personal(frame, state):
    values, starts = history(frame)
    raw = personal(values, starts, state['median'], state['spread'])
    return (raw - state['mean']) / state['std'], starts


def group_weights(labels, groups):
    cells = pd.DataFrame({'label': labels, 'group': groups})
    if cells.isna().any().any() or set(cells.label) != {0, 1}:
        raise ValueError('Fit requires present groups and both binary classes')
    size = cells.groupby(['label', 'group']).label.transform('size')
    count = cells.groupby('label').group.transform('nunique')
    return (len(cells) / (2 * count * size)).to_numpy()


def matrix(variant, base, extra):
    if variant not in ('Q0', 'Q1', 'Q2', 'Q3') or base.ndim != 2 or base.shape[1] != 75:
        raise ValueError('Invalid S6 variant or base features')
    if variant in ('Q1', 'Q3'):
        if extra.shape != (len(base), 27) or not np.isfinite(extra).all():
            raise ValueError('Invalid personal27 features')
        return np.column_stack((base, extra))
    return base


def fit_args(variant, saved, groups, extra, binding):
    ids, labels = saved['item_ids'], saved['y']
    for name, actual in [('fit_ids', ids), ('labels', labels), ('groups', groups)]:
        if payload_sha256(actual.tolist()) != binding[name + '_sha256']:
            raise ValueError('Fit row binding mismatch: ' + name)
    weights = None
    if variant in ('Q2', 'Q3'):
        weights = group_weights(labels, groups)
        if payload_sha256(weights.tolist()) != binding['weights_sha256']:
            raise ValueError('Weight binding mismatch')
    return matrix(variant, saved['X'], extra), labels, weights


def train_variant(variant, inputs, protocol):
    """Called only for a new, already-reserved research slot."""
    if variant not in ('Q1', 'Q2', 'Q3'):
        raise ValueError('Q0 requires exact historical reuse')
    x, y, weights = inputs
    spec, = [v for v in protocol['variants'] if v['id'] == variant]
    if (weights is None) != (spec['sample_weight'] is None):
        raise ValueError('Incompatible weighting policy')
    model = HistGradientBoostingClassifier(**protocol['models']['base'],
        random_state=protocol['models']['model_seed'], class_weight=spec['class_weight'])
    with threadpool_limits(limits=1):
        return model.fit(x, y, sample_weight=weights)


def select_variant(reports):
    """One fold. Mapping variant -> rotation number -> complete FP1 metrics."""
    ranked = []
    for index, variant in enumerate(('Q0', 'Q1', 'Q2', 'Q3')):
        rotations = reports.get(variant, {})
        if set(rotations) != {1, 2, 3}:
            continue
        rows = list(rotations.values())
        if any(r.get('status') != 'complete' for r in rows):
            continue
        data = [[r['metrics'].get(k) for k in ('recall', 'pr_auc', 'fp_per_1000_background')] for r in rows]
        if any(v is None or not np.isfinite(v) for row in data for v in row):
            continue
        recall, ap, fp = np.array(data).T
        ranked.append(((sum((fp <= 1) & (recall >= .5)), sum(fp <= 1), min(recall),
                        ap.mean(), recall.mean(), -max(fp), -index), variant))
    return max(ranked)[1] if ranked else None
