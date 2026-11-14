from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from cert_model_variants import BOOST_MODEL
from cert_operating_point import (OBJECTIVE, PRIMARY_PROFILE, PROFILES, check_operating_point,
                                 fp_budget, select_operating_points, threshold_table)
from datasets.cert_experiments import prepare_cert_frame
from datasets.cert_validation import assess_development, development_folds
from experiment import candidate, finish, verify
from experiment_tracking import file_sha256
from run_cert_operating_point import (PLAN_ID, SEEDS, SHORTLIST, assess_operating_candidate,
                                      freeze_operating_candidate, read_plan, run)
from utils import compute_classification_metrics, load_json, save_json
from test_cert_model_variants import BOOST
import test_cert_file_usb as fixtures


class CertOperatingPointTests(unittest.TestCase):
    def test_ties_integer_budget_and_infeasible_recall_do_not_relax_constraints(self):
        # At .9, one negative and two positives tie: budget zero must reject all.
        scores = np.r_[.9, .9, .9, np.zeros(999)]
        y = np.r_[1, 1, np.zeros(1000, int)]
        profiles = {'zero': dict(max_fp_per_1000=0, minimum_recall=.5),
                    'one': dict(max_fp_per_1000=1, minimum_recall=.5)}
        choices, table = select_operating_points(scores, y, profiles)
        self.assertFalse(choices['zero']['eligible_on_calibration'])
        self.assertTrue(choices['zero']['calibration_budget_met'])
        self.assertEqual(choices['zero']['calibration_metrics']['recall'], 0)
        self.assertTrue(choices['one']['eligible_on_calibration'])
        self.assertEqual(choices['one']['threshold'], .9)
        self.assertEqual(fp_budget(999, 1), 0)
        self.assertEqual(fp_budget(1000, 1), 1)
        for row in table.itertuples():
            metrics = compute_classification_metrics(y, scores >= row.threshold)
            self.assertEqual(metrics['confusion_matrix'], [[row.tn, row.fp], [row.fn, row.tp]])
        self.assertEqual(len(table), 3)

    def test_selection_matches_brute_force_and_rejects_invalid_inputs(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            scores, y = rng.integers(0, 10, 200) / 10, rng.integers(0, 2, 200)
            choices, table = select_operating_points(scores, y, {'p': dict(max_fp_per_1000=50, minimum_recall=.5)})
            allowed = []
            for threshold in np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]:
                m = compute_classification_metrics(y, scores >= threshold)
                if m['confusion_matrix'][0][1] <= fp_budget(int((y == 0).sum()), 50):
                    allowed.append((m['recall'], -m['confusion_matrix'][0][1], threshold))
            self.assertEqual(choices['p']['threshold'], max(allowed)[2])
            self.assertEqual(len(table), len(np.unique(scores)) + 1)
        for scores, y in [([], []), ([np.nan, .2], [0, 1]), ([.1, .2], [1, 1]), ([.1, .2], [0, 2])]:
            with self.assertRaises(ValueError):
                threshold_table(scores, y)

    def prep(self, root):
        source, frame, context, contract, config = fixtures.CertFileUSBTests().fixture(root)
        folded, provenance = next(development_folds(frame, n_splits=2, calibration_splits=3))
        prep = prepare_cert_frame(folded, context, contract, config, source, root, development=provenance)
        return source, prep, config

    def test_freeze_cannot_read_outer_and_assessment_cannot_fit_or_reselect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prep, _ = self.prep(root)
            trained = candidate(prep, root, 'random_forest')
            digest = file_sha256(trained / 'complete.json')
            outer, hidden = prep / 'assessment.csv', prep / 'hidden.csv'
            outer.rename(hidden)
            try:
                frozen = freeze_operating_candidate(trained, prep, root)
            finally:
                hidden.rename(outer)
            frozen_digest = file_sha256(frozen / 'complete.json')
            with (patch('run_cert_operating_point.select_operating_points', side_effect=AssertionError('selection')),
                  patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('fit'))):
                assessed = assess_operating_candidate(frozen, prep, root)
            self.assertEqual(file_sha256(frozen / 'complete.json'), frozen_digest)
            self.assertEqual(file_sha256(trained / 'complete.json'), digest)
            verify(frozen)
            verify(assessed)
            with np.load(assessed / 'assessment_inputs.npz') as a:
                x = a['X']
            model = joblib.load(frozen / 'model.joblib')
            with np.load(assessed / 'predictions_assessment.npz') as a:
                np.testing.assert_array_equal(a['scores'], model.predict_proba(x)[:, 1])
                for profile, spec in load_json(frozen / 'operating_points.json').items():
                    np.testing.assert_array_equal(a[profile], a['scores'] >= spec['threshold'])

    def test_calibration_budget_is_not_an_assessment_guarantee(self):
        scores = np.r_[.9, .8, np.zeros(1000)]
        y = np.r_[1, 1, np.zeros(1000, int)]
        choices, _ = select_operating_points(scores, y)
        spec = choices[PRIMARY_PROFILE]
        self.assertTrue(spec['eligible_on_calibration'])
        shifted = compute_classification_metrics(np.r_[1, np.zeros(1000)], np.r_[1, np.ones(3), np.zeros(997)])
        self.assertFalse(check_operating_point(shifted, spec)['budget_met'])
        self.assertTrue(check_operating_point(shifted, spec)['minimum_recall_met'])

    def fixture_plan(self, root):
        source, frame, context, contract, base = fixtures.CertFileUSBTests().fixture(root)
        cv = dict(outer_folds=2, outer_seed=142, inner_calibration_folds=3, inner_calibration_seed_base=242)
        previous = root / 'previous'
        previous.mkdir()
        (previous / 'folds').mkdir()
        settings, controls = {}, []
        for name, (_, model, history) in SHORTLIST.items():
            config = deepcopy(base)
            config['features']['history_policy'] = history
            if model == BOOST_MODEL:
                config['baseline'][model] = BOOST
            path = root / (name + '.json')
            save_json(config, path)
            settings[name] = dict(path=str(path), sha256=file_sha256(path))
        for folded, provenance in development_folds(frame, n_splits=2, calibration_splits=3):
            save_json(provenance, previous / 'folds' / f"fold_{provenance['fold']}.json")
            for name, (policy, model, _) in SHORTLIST.items():
                config = load_json(Path(settings[name]['path']))
                prep = prepare_cert_frame(folded, context, contract, config, source, root, development=provenance)
                trained = candidate(prep, root, model)
                assessed = assess_development(trained, prep, root)
                controls.append(dict(policy=policy, model=model, fold=provenance['fold'],
                    candidate=str(trained), preparation=str(prep),
                    candidate_manifest_sha256=file_sha256(trained / 'complete.json'),
                    preparation_manifest_sha256=file_sha256(prep / 'complete.json'),
                    assessment_metrics=load_json(assessed / 'metrics_assessment.json')))
        save_json(dict(daily_manifest_sha256=file_sha256(source / 'complete.json'), cv=cv), previous / 'plan.json')
        save_json(dict(results=controls), previous / 'summary.json')
        finish(previous)
        plan = dict(plan_id=PLAN_ID, source=dict(id=previous.name, manifest_sha256=file_sha256(previous / 'complete.json')),
            daily_preparation_id=source.name, daily_manifest_sha256=file_sha256(source / 'complete.json'),
            shortlist=settings, model_seeds=SEEDS, reference_seed=42, primary_candidate='hgb_rolling',
            primary_profile=PRIMARY_PROFILE, profiles=PROFILES, objective=OBJECTIVE, cv=cv,
            budget=dict(new_fits=24, reused_candidates=6, frozen_bundles=30, decision_assessments=90,
                        search_trials=0, cpu_threads=1, minimum_free_gib=0))
        path = root / 'plan.json'
        save_json(plan, path)
        return source, path

    def test_complete_five_seed_runner_and_frozen_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, path = self.fixture_plan(root)
            output = run(source, root, path)
            verify(output)
            summary = load_json(output / 'summary.json')
            self.assertEqual(len(summary['results']), 30)
            self.assertEqual(len(summary['per_fold_profiles']), 90)
            self.assertEqual(len(summary['pooled_oof']), 45)
            self.assertFalse(summary['independent_confirmation']['confirmatory_claim'])
            for row in summary['pooled_oof']:
                self.assertEqual(row['metrics']['support']['total'], 144)
                self.assertIsNone(row['metrics']['pr_auc'])
            self.assertEqual(sum(r['reused'] for r in summary['results']), 6)
            plan = load_json(path)
            plan['primary_profile'] = 'fp2_per_1000'
            save_json(plan, path)
            with self.assertRaisesRegex(ValueError, 'bounded step-5'):
                read_plan(path, root)


if __name__ == '__main__':
    unittest.main()
