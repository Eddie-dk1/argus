"""Named calibration-only operating points; no changes to legacy F1 selection."""
from fractions import Fraction

import numpy as np
import pandas as pd

from utils import compute_classification_metrics


OBJECTIVE = 'max_recall_under_background_fp_budget_v1'
PROFILES = {
    'fp1_per_1000': {'max_fp_per_1000': 1, 'minimum_recall': .5},
    'fp2_per_1000': {'max_fp_per_1000': 2, 'minimum_recall': .5},
}
PRIMARY_PROFILE = 'fp1_per_1000'


def fp_budget(background_count, max_fp_per_1000):
    rate = Fraction(str(max_fp_per_1000)) / 1000
    if background_count < 1 or not 0 <= rate <= 1:
        raise ValueError('Positive background support and a valid FP budget are required')
    return (int(background_count) * rate.numerator) // rate.denominator


def threshold_table(scores, labels):
    """Every distinct calibration decision, including reject-all, without tie splitting."""
    values, y = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if (values.ndim != 1 or y.ndim != 1 or len(values) != len(y) or not len(y)
            or not np.isfinite(values).all() or not np.isin(y, [0, 1]).all()
            or set(np.unique(y)) != {0, 1}):
        raise ValueError('Aligned finite calibration scores with both binary classes required')
    order = np.argsort(-values, kind='stable')
    sorted_values, sorted_y = values[order], y[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(sorted_values[:-1] != sorted_values[1:]), len(y) - 1]
    tp = np.r_[0, np.cumsum(sorted_y)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    reject = np.nextafter(values.max(), np.inf)
    if not np.isfinite(reject):
        raise ValueError('No finite reject-all threshold')
    positive, background = int(y.sum()), int((y == 0).sum())
    return pd.DataFrame(dict(threshold=np.r_[reject, sorted_values[ends]], tp=tp, fp=fp,
                             fn=positive - tp, tn=background - fp,
                             recall=tp / positive, false_positive_rate=fp / background))


def select_operating_points(scores, labels, profiles=None):
    """Max TP, then min FP, then higher threshold; failed recall never relaxes budget."""
    profiles = PROFILES if profiles is None else profiles
    if not profiles:
        raise ValueError('At least one declared operating profile required')
    table = threshold_table(scores, labels)
    background = int((np.asarray(labels) == 0).sum())
    selections = {}
    for name, spec in profiles.items():
        if set(spec) != {'max_fp_per_1000', 'minimum_recall'} or not 0 <= spec['minimum_recall'] <= 1:
            raise ValueError('Invalid operating profile')
        limit = fp_budget(background, spec['max_fp_per_1000'])
        allowed = table.loc[table.fp.le(limit)]
        best = allowed.sort_values(['tp', 'fp', 'threshold'], ascending=[False, True, False]).iloc[0]
        threshold = float(best.threshold)
        metrics = compute_classification_metrics(labels, np.asarray(scores) >= threshold, scores)
        selections[name] = dict(objective=OBJECTIVE, threshold=threshold, **spec,
            calibration_background_count=background, calibration_maximum_fp=limit,
            calibration_metrics=metrics, calibration_minimum_recall_met=metrics['recall'] >= spec['minimum_recall'],
            calibration_budget_met=metrics['confusion_matrix'][0][1] <= limit,
            eligible_on_calibration=metrics['recall'] >= spec['minimum_recall'],
            infeasible_action='retain best budget-respecting threshold; mark ineligible; no fallback or refit')
    return selections, table


def check_operating_point(metrics, spec):
    """Descriptive assessment check; never used to recalibrate or pick a seed."""
    budget_met = metrics['confusion_matrix'][0][1] <= fp_budget(metrics['support']['normal'], spec['max_fp_per_1000'])
    recall_met = metrics['recall'] >= spec['minimum_recall']
    return dict(budget_met=budget_met, minimum_recall_met=recall_met, both_met=budget_met and recall_met)
