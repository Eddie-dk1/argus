from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from cert_model_variants import ENDPOINT_SCORE
from datasets.cert_experiments import prepare_cert_frame
from datasets.cert_reporting import evidence_index
from datasets.cert_validation import assess_development, development_folds
from experiment import candidate, finish
from experiment_tracking import file_sha256
from frozen_diagnostics import (cohort_diagnostics, descriptive_metrics, history_flags,
                                observed_history, statistics, variation_reports)
from run_cert_operating_point import assess_operating_candidate, freeze_operating_candidate
from run_frozen_diagnostics import PreparationContext, no_training, process_entry, validate_coverage
from utils import load_json


class FrozenDiagnosticStatisticsTests(unittest.TestCase):
    def test_undefined_metrics_empty_and_single_class_have_reasons_and_json_null(self):
        cases = [([], []), ([0, 0], [0, 0]), ([1, 1], [0, 0]), ([1, 1], [1, 1])]
        for y, p in cases:
            metrics, undefined = descriptive_metrics(y, p, np.zeros(len(y)))
            self.assertIsNone(metrics['pr_auc'])
            self.assertIsNone(metrics['roc_auc'])
            self.assertIn('pr_auc', undefined)
            json.dumps(dict(metrics=metrics, undefined=undefined), allow_nan=False)
        m, reasons = descriptive_metrics([0, 0], [0, 0], [0., 0.])
        self.assertIsNone(m['precision'])
        self.assertIsNone(m['recall'])
        self.assertIsNone(m['f1_score'])
        self.assertEqual(m['fp'], 0)
        self.assertEqual(m['fp_per_1000_background'], 0.)
        self.assertIn('no predicted positives', reasons['precision'])
        with self.assertRaises(ValueError):
            descriptive_metrics([0, 2], [0, 1], [0., 1.])
        with self.assertRaises(ValueError):
            descriptive_metrics([0, 1], [0, 1], [0., np.nan])

    def test_observed_history_keeps_thirty_day_boundary_and_user_isolation(self):
        frame = pd.DataFrame(dict(event_id=['a', 'b', 'c', 'd', 'other'], user_id=['u'] * 4 + ['v'],
            timestamp=['2020-01-01', '2020-01-02', '2020-01-31', '2020-02-01', '2020-02-01']))
        counts = observed_history(frame.iloc[::-1])
        self.assertEqual(counts.loc[['a', 'b', 'c', 'd', 'other']].tolist(), [0, 1, 2, 2, 0])
        with self.assertRaisesRegex(ValueError, 'strictly ordered'):
            observed_history(pd.concat([frame, frame.iloc[[0]].assign(event_id='same-day')]))

    def test_binary_flags_use_frozen_scaler_and_missing_feature_is_explicit(self):
        frame = pd.DataFrame(dict(event_id=['a', 'b'], history_missing=[-.5, 1.5], cold_start=[0., 1.]))
        state = dict(columns=['history_missing', 'cold_start'], mean=[.25, 0.], scale=[.5, 1.])
        flags = history_flags(frame, state)
        self.assertEqual(flags['history_missing'].tolist(), [0, 1])
        self.assertIsNone(flags['file_usb_profile_missing'])
        frame.loc[0, 'history_missing'] = .123
        with self.assertRaisesRegex(ValueError, 'restore'):
            history_flags(frame, state)

    def test_error_account_concentration_scenario_overlap_and_pooled_no_ranking(self):
        ids = np.array(['a', 'b', 'c', 'd'])
        metadata = pd.DataFrame(dict(item_id=ids, day_key=ids, account_id=['u', 'u', 'v', 'w'],
                                     group=['g1', 'g1', 'g2', 'g3'], label=[0, 0, 1, 1]))
        evidence = {'c': dict(scenarios={'1', '2'}, incidents_by_scenario={'1': {'i1'}, '2': {'i2'}}),
                    'd': dict(scenarios={'3'}, incidents_by_scenario={'3': {'i3'}})}
        report = cohort_diagnostics(ids, np.array([0, 0, 1, 1]), np.array([.7, .8, .9, .1]),
            np.array([1, 1, 1, 0]), metadata, evidence)
        self.assertEqual(report['false_alarm_concentration']['shares']['1'], 1.)
        self.assertEqual(report['false_alarm_concentration']['accounts'][0]['background_days'], 2)
        self.assertEqual(report['positive_days_with_multiple_scenarios'], 1)
        self.assertEqual(report['scenarios']['1']['recall'], 1.)
        self.assertEqual(report['scenarios']['2']['recall'], 1.)
        self.assertEqual(report['scenarios']['3']['recall'], 0.)
        self.assertFalse(report['history']['file_usb_profile_missing']['available'])
        pooled = cohort_diagnostics(ids, np.array([0, 0, 1, 1]), None, np.array([1, 1, 1, 0]), metadata, evidence)
        self.assertIsNone(pooled['metrics']['pr_auc'])
        self.assertIsNone(pooled['score_distributions'])
        self.assertEqual(pooled['undefined_metrics']['pr_auc'], 'cross-fold ranking not defined')
        self.assertEqual(pooled['metrics']['fp'], 2)

    def test_variation_separates_seeds_and_folds_and_missing_coverage_fails(self):
        identical = 0.43278688524590164
        summary = statistics([identical] * 5)
        self.assertEqual(summary['sample_std'], 0.)
        self.assertEqual(summary['mean'], identical)
        rows = []
        for fold in (1, 2):
            for seed in (7, 42):
                rows.append(dict(series='operating_points', method='hgb', feature_policy='p', profile='f',
                    fold=fold, seed=seed, **{role: dict(metrics={m: fold / 10 for m in (
                        'precision', 'recall', 'f1_score', 'pr_auc', 'roc_auc', 'false_positive_rate')})
                        for role in ('calibration', 'assessment')}))
        result = variation_reports(rows)
        for row in result['model_seed_at_fixed_fold']:
            self.assertEqual(row['metrics']['f1_score']['sample_std'], 0.)
            self.assertFalse(row['independent_confirmation'])
            self.assertEqual(row['observation_count'], 2)
        for row in result['fold_at_fixed_model_seed']:
            self.assertGreater(row['metrics']['f1_score']['sample_std'], 0.)
        validate_coverage(rows, {(fold, seed) for fold in (1, 2) for seed in (7, 42)}, lambda r: (r['fold'], r['seed']))
        for broken in (rows[:-1], rows + rows[:1]):
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                validate_coverage(broken, {(fold, seed) for fold in (1, 2) for seed in (7, 42)}, lambda r: (r['fold'], r['seed']))


class FrozenReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_cert_file_usb import CertFileUSBTests
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source, frame, context, contract, config = CertFileUSBTests().fixture(cls.root)
        folded, cls.provenance = next(development_folds(frame, n_splits=2, calibration_splits=3))
        cls.prep = prepare_cert_frame(folded, context, contract, config, source, cls.root, development=cls.provenance)
        cls.primary = frame.loc[frame.split.eq('train')].copy()
        cls.evidence = evidence_index(source / 'positive_evidence.jsonl')
        cls.entries = []
        config['lstm_autoencoder']['score_policy'] = ENDPOINT_SCORE
        setting = cls.root / 'endpoint.json'
        setting.write_text(json.dumps(config))
        for name in ('random_forest', 'isolation_forest', 'lstm_autoencoder'):
            trained = candidate(cls.prep, cls.root, name, setting if name == 'lstm_autoencoder' else None)
            assessed = assess_development(trained, cls.prep, cls.root)
            selection = load_json(trained / 'selection.json')
            row = dict(job_id='fixture_' + name, model='lstm_endpoint' if name == 'lstm_autoencoder' else name,
                seed=config['seed'], fold=1, policy='fixture', candidate=str(trained), assessment=str(assessed),
                preparation=str(cls.prep), preparation_manifest_sha256=file_sha256(cls.prep / 'complete.json'),
                candidate_manifest_sha256=file_sha256(trained / 'complete.json'), threshold=selection['threshold'],
                assessment_manifest_sha256=file_sha256(assessed / 'complete.json'),
                validation_metrics=load_json(trained / 'metrics_validation.json'),
                assessment_metrics=load_json(assessed / 'metrics_assessment.json'))
            cls.entries.append(dict(series='model_variants', row=row))
        trained = Path(cls.entries[0]['row']['candidate'])
        frozen = freeze_operating_candidate(trained, cls.prep, cls.root)
        assessed = assess_operating_candidate(frozen, cls.prep, cls.root)
        row = dict(cls.entries[0]['row'], job_id='fixture_operating', name='rf_fixture', frozen_candidate=str(frozen),
            frozen_candidate_manifest_sha256=file_sha256(frozen / 'complete.json'), assessment=str(assessed),
            assessment_manifest_sha256=file_sha256(assessed / 'complete.json'),
            operating_points=load_json(frozen / 'operating_points.json'),
            assessment_metrics=load_json(assessed / 'metrics_assessment.json'),
            operating_checks=load_json(assessed / 'operating_checks.json'))
        cls.entries.append(dict(series='operating_points', row=row))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def context(self):
        return PreparationContext(self.prep, file_sha256(self.prep / 'complete.json'), self.primary, self.provenance)

    def test_real_producers_replay_exactly_and_keep_all_profiles_without_training(self):
        frozen = {str(p): file_sha256(p) for p in self.root.rglob('*') if p.is_file()}
        output = self.root / 'replay'
        output.mkdir()
        context = self.context()
        with no_training():
            records = [r for entry in self.entries for r in process_entry(entry, context, self.primary, self.evidence, output)]
        self.assertEqual(len(records), 6)
        self.assertEqual({r['profile'] for r in records}, {'legacy_f1', 'fp1_per_1000', 'fp2_per_1000'})
        for row in records:
            self.assertTrue(row['replay']['scores_exact'])
            self.assertTrue(row['replay']['ordered_window_members_exact'])
            self.assertEqual(row['model']['scenario'], 'cert_r4_2')
            json.dumps(row, allow_nan=False)
        self.assertEqual(frozen, {name: file_sha256(Path(name)) for name in frozen})

    def test_resealed_changed_members_or_scores_are_rejected(self):
        original = self.entries[0]
        for filename, key in (('assessment_inputs.npz', 'member_ids'), ('predictions_assessment.npz', 'scores')):
            with tempfile.TemporaryDirectory() as tmp:
                entry = deepcopy(original)
                clone = Path(shutil.copytree(entry['row']['assessment'], Path(tmp) / 'assessment'))
                with np.load(clone / filename, allow_pickle=False) as a:
                    arrays = {k: a[k] for k in a.files}
                if key == 'member_ids':
                    arrays[key][0, :2] = arrays[key][0, :2][::-1]
                else:
                    arrays[key][0] = np.nextafter(arrays[key][0], np.inf)
                np.savez_compressed(clone / filename, **arrays)
                finish(clone)
                entry['row'].update(assessment=str(clone), assessment_manifest_sha256=file_sha256(clone / 'complete.json'))
                output = Path(tmp) / 'output'
                output.mkdir()
                with self.assertRaisesRegex(ValueError, 'Exact replay mismatch'):
                    process_entry(entry, self.context(), self.primary, self.evidence, output)

    def test_other_model_seeds_are_diagnosed_without_a_new_inference_claim(self):
        entry = deepcopy(self.entries[-1])
        output = self.root / 'cached'
        output.mkdir()
        with no_training():
            rows = process_entry(entry, self.context(), self.primary, self.evidence, output, reference_seed=7)
        self.assertTrue(all(not r['replay']['performed'] for r in rows))
        self.assertTrue(all(r['replay']['scores_exact'] is None for r in rows))
        self.assertTrue(all(r['replay']['numerical_inputs_exact'] for r in rows))
        self.assertFalse((output / 'replays').exists())


if __name__ == '__main__':
    unittest.main()
