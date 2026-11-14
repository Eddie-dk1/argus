"""Post-selection CERT diagnostics; answer attributions never enter inference."""
from collections import defaultdict
import json

import numpy as np


def evidence_index(path):
    index = defaultdict(lambda: {'scenarios': set(), 'incidents': set(),
                                 'incidents_by_scenario': defaultdict(set)})
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            for attribution in row['attributions']:
                scenario = str(attribution['scenario'])
                if scenario not in {'1', '2', '3'}:
                    raise ValueError('Unknown evidence scenario')
                index[row['day_key']]['scenarios'].add(scenario)
                index[row['day_key']]['incidents'].add(attribution['incident_id'])
                index[row['day_key']]['incidents_by_scenario'][scenario].add(attribution['incident_id'])
    return dict(index)


def error_report(predictions, metadata, evidence):
    if predictions.item_id.duplicated().any() or metadata.item_id.duplicated().any():
        raise ValueError('Duplicate report items')
    selected = metadata.set_index('item_id').loc[predictions.item_id].reset_index(drop=True)
    if not np.array_equal(selected.label, predictions.y_true):
        raise ValueError('Reporting metadata and evaluation labels differ')
    rows = predictions.reset_index(drop=True).copy()
    for column in ('day_key', 'account_id', 'group'):
        rows[column] = selected[column]
    membership = rows.day_key.map(lambda key: key in evidence).to_numpy()
    if not np.array_equal(membership.astype(int), rows.y_true):
        raise ValueError('Positive evidence and target labels differ')
    scenarios = {}
    for scenario in ('1', '2', '3'):
        mask = rows.day_key.map(lambda key: scenario in evidence.get(key, {}).get('scenarios', set()))
        part = rows.loc[mask]
        support = len(part)
        found = int(part.prediction.sum())
        scenarios[scenario] = {'positive_days': support, 'true_positive_days': found,
                              'false_negative_days': support - found,
                              'recall': found / support if support else None,
                              'accounts': int(part.account_id.nunique()),
                              'independent_groups': int(part.group.nunique()),
                              'incidents': len(set().union(*(evidence[key]['incidents_by_scenario'][scenario]
                                                           for key in part.day_key))) if support else 0}
    # A positive day can belong to multiple scenarios; do not force a single label.
    overlapping = sum(len(evidence[key]['scenarios']) > 1 for key in rows.loc[rows.y_true.eq(1), 'day_key'])
    background = rows.loc[rows.y_true.eq(0)]
    false_positive = background.loc[background.prediction.eq(1)]
    counts = false_positive.groupby('account_id').size().sort_values(ascending=False)
    score_summary = {}
    for label, name in ((0, 'background'), (1, 'observable_positive')):
        values = rows.loc[rows.y_true.eq(label), 'score'].to_numpy(float)
        score_summary[name] = {'count': len(values), 'mean': float(values.mean()) if len(values) else None,
            'quantiles': dict(zip(['min','p25','median','p75','p90','p99','max'],
                np.quantile(values, [0,.25,.5,.75,.9,.99,1]).tolist())) if len(values) else None}
    return {'scenarios': scenarios, 'positive_days_with_multiple_scenarios': overlapping,
            'background': {'days': len(background), 'false_positive_days': len(false_positive),
                           'accounts_with_false_positives': int(len(counts)),
                           'share_fp_in_top10_accounts': float(counts.head(10).sum() / len(false_positive)) if len(false_positive) else None},
            'false_positives_by_account': {str(key): int(value) for key, value in counts.items()},
            'scores': score_summary}
