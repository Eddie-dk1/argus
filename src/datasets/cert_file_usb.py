"""Generic file/device volume deviations; no labels or answer-key inputs."""
from collections import defaultdict, deque
import gzip
import json

import numpy as np
import pandas as pd

ROLLING_POLICY = 'cert_causal_file_usb_rolling_v4'
FROZEN_POLICY = 'cert_causal_file_usb_frozen_v4'
POLICIES = (ROLLING_POLICY, FROZEN_POLICY)
RAW_COLUMNS = ('file_events', 'connect_count', 'device_paired_seconds')
COLUMNS = [name + '_personal_deviation' for name in RAW_COLUMNS] + ['file_usb_profile_missing']
MINIMUM_DAYS = 5
WINDOW_DAYS = 30
IQR_FLOOR = 1.0
CLIP = 20.0
DEFINITION = dict(raw_columns=list(RAW_COLUMNS), added_columns=COLUMNS, minimum_past_active_days=MINIMUM_DAYS,
                  rolling_calendar_days=WINDOW_DAYS, frozen_initial_active_days=MINIMUM_DAYS,
                  log1p_iqr_floor=IQR_FLOOR, deviation_clip=CLIP,
                  fallback='all inner-fit active days, including zero counts; labels unused',
                  update='emit before observe; all roles; no label-based history cleaning')


def robust_profile(values):
    low, median, high = np.quantile(np.asarray(values, dtype=float), [.25, .5, .75], axis=0)
    return median, np.maximum(high - low, IQR_FLOOR)


def fit_global(frame, policy):
    if policy not in POLICIES:
        raise ValueError('Unsupported file/USB policy')
    values = frame.loc[frame.split.eq('train'), list(RAW_COLUMNS)].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid file/USB inner-fit inputs')
    median, spread = robust_profile(np.log1p(values))
    return dict(raw_columns=list(RAW_COLUMNS), median=median.tolist(), iqr=spread.tolist(),
                reference='rolling30' if policy == ROLLING_POLICY else 'first5_frozen',
                minimum_days=MINIMUM_DAYS, calendar_window_days=WINDOW_DAYS,
                iqr_floor=IQR_FLOOR, clip=CLIP,
                fit_population='all inner-fit/train active days including zeros; labels unused',
                update_policy='emit then observe current day in every role; never clean by labels')


def values(frame, state, dependency_path=None):
    """Both references are past-only; frozen references never backfill early rows."""
    if state['policy'] not in POLICIES:
        raise ValueError('Unsupported file/USB policy')
    settings = state['file_usb_profile']
    expected = dict(raw_columns=list(RAW_COLUMNS), minimum_days=MINIMUM_DAYS,
                    calendar_window_days=WINDOW_DAYS, iqr_floor=IQR_FLOOR, clip=CLIP,
                    reference='rolling30' if state['policy'] == ROLLING_POLICY else 'first5_frozen')
    if any(settings[key] != value for key, value in expected.items()):
        raise ValueError('File/USB feature contract changed')
    numeric = frame[list(RAW_COLUMNS)].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError('Invalid file/USB daily inputs')
    numeric = np.log1p(numeric)
    days = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    histories, initial, previous, frozen = defaultdict(deque), defaultdict(list), {}, {}
    result = []
    stream = gzip.open(dependency_path, 'wt') if dependency_path else None
    try:
        rows = frame[['account_id', 'date', 'item_id']].itertuples(index=False, name=None)
        for (account, date, item), day, current in zip(rows, days, numeric, strict=True):
            if account in previous and day <= previous[account]:
                raise ValueError('Duplicate or nonchronological account-day')
            previous[account] = day
            history = histories[account]
            while history and history[0][0] < day - WINDOW_DAYS:
                history.popleft()
            reference = history if settings['reference'] == 'rolling30' else initial[account]
            enough = len(reference) >= MINIMUM_DAYS
            if enough:
                if settings['reference'] == 'first5_frozen':
                    median, spread = frozen[account]
                else:
                    median, spread = robust_profile([row[2] for row in reference])
            else:
                median, spread = settings['median'], settings['iqr']
            deviation = np.clip((current - median) / spread, -CLIP, CLIP)
            result.append([*deviation, int(not enough)])
            if stream:
                stream.write(json.dumps(dict(item_id=item, account_id=account, cutoff_exclusive=date,
                    reference=settings['reference'] if enough else 'frozen_global_train',
                    profile_members=[row[1] for row in reference] if enough else [],
                    available_personal_members=[row[1] for row in reference],
                    update_policy=settings['update_policy'])) + '\n')
            # Current input is observed only after its deviations have been emitted.
            if settings['reference'] == 'rolling30':
                history.append((day, item, current))
            elif len(initial[account]) < MINIMUM_DAYS:
                initial[account].append((day, item, current))
                if len(initial[account]) == MINIMUM_DAYS:
                    frozen[account] = robust_profile([row[2] for row in initial[account]])
    finally:
        if stream:
            stream.close()
    return pd.DataFrame(result, columns=COLUMNS, index=frame.index)
