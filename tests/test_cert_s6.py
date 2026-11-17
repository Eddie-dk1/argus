import unittest

import numpy as np
import pandas as pd

from cert_s6_worker import (RAW, apply_personal, fit_args, fit_personal, group_weights, history,
                            matrix, personal, select_variant, train_variant)
from experiment_tracking import payload_sha256


def days(offsets, values, account='alice'):
    frame = pd.DataFrame({name: values for name in RAW})
    frame['date'] = pd.Timestamp('2020-01-01') + pd.to_timedelta(offsets, unit='D')
    frame['account_id'] = account
    frame['item_id'] = [f'{account}-{day}' for day in offsets]
    return frame


class PersonalContract(unittest.TestCase):
    def test_calendar_oracle_and_fallback(self):
        frame = days([0, 1, 2, 23, 24, 29, 30, 31, 38, 90], np.expm1(np.arange(10.)))
        values, starts = history(frame)
        x = personal(values, starts, np.full(12, 5.), np.full(12, 2.))
        for i, today in enumerate(frame.date):
            past = (today - frame.date).dt.days
            old = np.flatnonzero((past > 0) & (past <= 30))
            recent = np.flatnonzero((past > 0) & (past <= 7))
            np.testing.assert_array_equal(np.arange(starts[30][i], i), old)
            np.testing.assert_array_equal(np.arange(starts[7][i], i), recent)
            med, spread = 5., 2.
            if len(old) >= 5:
                lo, med, hi = np.percentile(old, [25, 50, 75])
                spread = max(hi - lo, 1.)
            trend = (np.median(recent) - med) / spread if len(old) >= 5 and len(recent) >= 2 else 0.
            np.testing.assert_allclose(x[i, :12], (i-med)/spread)
            np.testing.assert_allclose(x[i, 12:24], trend)
            np.testing.assert_allclose(x[i, -3:], [len(old)<5, len(old)<5 or len(recent)<2, np.log1p(len(recent))])

    def test_frozen_state_causality_and_role_isolation(self):
        fit = days(range(8), [0,1,2,3,4,5,6,7])
        state, scaled, _ = fit_personal(fit)
        np.testing.assert_allclose(scaled, apply_personal(fit, state)[0])
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-14)
        held = days(range(7,15), [10,0,1,2,3,4,5,6], 'bob')
        first = apply_personal(held, state)[0]
        held['label'] = 1
        np.testing.assert_array_equal(first, apply_personal(held, state)[0])
        held.loc[7, list(RAW)] = 900
        np.testing.assert_array_equal(first[:-1], apply_personal(held, state)[0][:-1])
        combined = pd.concat([fit, held], ignore_index=True)
        np.testing.assert_array_equal(apply_personal(held,state)[0], apply_personal(combined,state)[0][8:])
        np.testing.assert_allclose(state['median'], np.median(np.log1p(np.arange(8))))
        for bad in (-1, np.nan, np.inf):
            malformed = fit.astype({RAW[0]:float})
            malformed.loc[0, RAW[0]] = bad
            with self.assertRaises(ValueError):
                history(malformed)
        with self.assertRaises(ValueError):
            history(fit.iloc[::-1])
        with self.assertRaises(ValueError):
            history(pd.concat([fit, fit]))

    def test_weights_matrices_and_reuse(self):
        labels = np.array([0,0,0,1,1,1])
        groups = np.array(['a','a','b','b','c','d'])
        weights = group_weights(labels,groups)
        np.testing.assert_array_equal(weights,[.75,.75,1.5,1,1,1])
        base = np.arange(450).reshape(6,75)
        saved = dict(X=base, item_ids=np.arange(6), y=labels)
        binding = {k+'_sha256':payload_sha256(v.tolist()) for k,v in
                   [('weights',weights),('fit_ids',saved['item_ids']),('groups',groups),('labels',labels)]}
        extra = np.ones((6,27))
        for variant in ('Q0','Q1','Q2','Q3'):
            x,y,w = fit_args(variant,saved,groups,extra,binding)
            np.testing.assert_array_equal(x[:,:75],base)
            self.assertEqual(x.shape[1],102 if variant in ('Q1','Q3') else 75)
            self.assertEqual(w is None,variant in ('Q0','Q1'))
            if w is not None:
                np.testing.assert_array_equal(w,weights)
        with self.assertRaises(ValueError):
            fit_args('Q3',saved,groups[::-1],extra,binding)
        with self.assertRaises(ValueError):
            train_variant('Q0',None,{})
        with self.assertRaises(ValueError):
            matrix('Q1',base,extra[:-1])
        with self.assertRaises(ValueError):
            group_weights(labels*0,groups)

    def test_selection_order_and_undefined(self):
        def reports(recall,fp):
            return {i:dict(status='complete',metrics=dict(recall=recall,pr_auc=.7,fp_per_1000_background=fp),
                           eligible_on_calibration=False) for i in (1,2,3)}
        a,b = reports(.6, 2),reports(.7,3)
        self.assertEqual(select_variant({'Q0':a,'Q2':b}), 'Q2')
        self.assertEqual(select_variant({'Q1':a,'Q2':a}), 'Q1')
        b[1]['metrics']['pr_auc'] = None
        self.assertIsNone(select_variant({'Q0':{},'Q2':b}))
