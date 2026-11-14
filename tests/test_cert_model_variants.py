from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE, endpoint_errors
from datasets.cert_experiments import prepare_cert_frame
from datasets.cert_validation import assess_development, development_folds
from experiment import candidate, configured_score, finish, inputs, score, verify
from experiment_tracking import file_sha256
from run_cert_model_variants import METHODS, OLD_MODELS, PLAN_ID, POLICIES, read_plan, run
from utils import load_json, save_json
import test_cert_file_usb as fixtures


BOOST = dict(loss='log_loss', learning_rate=.05, max_iter=3, max_leaf_nodes=4, max_depth=2,
             min_samples_leaf=2, l2_regularization=1., max_bins=255, early_stopping=False, class_weight='balanced')


class CertModelVariantTests(unittest.TestCase):
    def preparation(self, root):
        source, frame, context, contract, config = fixtures.CertFileUSBTests().fixture(root)
        folded, provenance = next(development_folds(frame, n_splits=2, calibration_splits=3))
        prep = prepare_cert_frame(folded, context, contract, config, source, root, development=provenance)
        return source, prep, config

    def test_endpoint_formula_excludes_prefix_errors_and_preserves_empty_dtype(self):
        class Zeros(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)
        x = np.zeros((2, 5, 3), np.float32)
        x[0, :4] = 10
        x[1, -1] = 10
        np.testing.assert_array_equal(endpoint_errors(Zeros(), x, batch_size=1), np.array([0, 100], np.float32))
        legacy = score(Zeros(), x, 'lstm_autoencoder')
        np.testing.assert_array_equal(legacy, [80, 20])
        self.assertEqual(endpoint_errors(Zeros(), x[:0]).dtype, np.float32)
        self.assertEqual(len(endpoint_errors(Zeros(), x[:0])), 0)
        with self.assertRaisesRegex(ValueError, 'time/feature'):
            endpoint_errors(Zeros(), x[:, :0])
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            endpoint_errors(Zeros(), x)

    def test_default_configuration_delegates_to_exact_frozen_scorer(self):
        value = np.array([.25], np.float32)
        for name in OLD_MODELS:
            with patch('experiment.score', return_value=value) as old:
                model, x = object(), np.zeros((1, 1))
                self.assertIs(configured_score(model, x, name, {'lstm_autoencoder': {}}), value)
                old.assert_called_once_with(model, x, name)
        with self.assertRaisesRegex(ValueError, 'Unsupported LSTM'):
            configured_score(None, None, 'lstm_autoencoder', {'lstm_autoencoder': {'score_policy': 'typo'}})

    def test_score_does_not_change_training_loss_or_one_epoch_weights_and_uses_inner_selection(self):
        from train_lstm_autoencoder import LSTMAutoencoder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prep, config = self.preparation(root)
            first = candidate(prep, root, 'lstm_autoencoder', diagnostics=True)
            changed = deepcopy(config)
            changed['lstm_autoencoder']['score_policy'] = ENDPOINT_SCORE
            setting = root / 'endpoint.json'
            save_json(changed, setting)
            # The outer file may not exist while selecting the new checkpoint.
            outer, hidden = prep / 'assessment.csv', prep / 'hidden.csv'
            outer.rename(hidden)
            try:
                second = candidate(prep, root, 'lstm_autoencoder', setting, diagnostics=True)
            finally:
                hidden.rename(outer)
            before = torch.load(first / 'model.pt', map_location='cpu', weights_only=True)
            after = torch.load(second / 'model.pt', map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.equal(before[key], after[key]) for key in before))
            h1, h2 = [load_json(path / 'history.json') for path in (first, second)]
            self.assertEqual(h1['epochs'][0]['train_optimization_mse'], h2['epochs'][0]['train_optimization_mse'])
            self.assertEqual(load_json(second / 'selection.json')['score_policy'], ENDPOINT_SCORE)
            frozen = file_sha256(second / 'complete.json')
            with (patch('torch.optim.Adam', side_effect=AssertionError('fit')),
                  patch('experiment.select_best_threshold', side_effect=AssertionError('recalibration'))):
                assessment = assess_development(second, prep, root)
            self.assertEqual(file_sha256(second / 'complete.json'), frozen)
            with np.load(assessment / 'assessment_inputs.npz', allow_pickle=False) as arrays:
                x = arrays['X']
            model = LSTMAutoencoder(x.shape[2], config['lstm_autoencoder']['hidden_size'])
            model.load_state_dict(after)
            with np.load(assessment / 'predictions_assessment.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['scores'], endpoint_errors(model, x))

    def test_boosting_uses_all_inner_fit_labels_and_no_random_early_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prep, config = self.preparation(root)
            config['baseline'][BOOST_MODEL] = BOOST
            setting = root / 'boost.json'
            save_json(config, setting)
            trained = candidate(prep, root, BOOST_MODEL, setting)
            model = joblib.load(trained / 'model.joblib')
            self.assertFalse(model.do_early_stopping_)
            self.assertEqual(model.n_iter_, 3)
            train = pd.read_csv(prep / 'train.csv')
            with np.load(trained / 'train_inputs.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['item_ids'], train.event_id)
                np.testing.assert_array_equal(arrays['y'], train.label)
                self.assertGreater(arrays['y'].sum(), 0)
            with (patch('sklearn.ensemble.HistGradientBoostingClassifier.fit', side_effect=AssertionError('fit')),
                  patch('experiment.select_best_threshold', side_effect=AssertionError('threshold'))):
                assessed = assess_development(trained, prep, root)
            with np.load(assessed / 'assessment_inputs.npz', allow_pickle=False) as arrays:
                x = arrays['X']
            with np.load(assessed / 'predictions_assessment.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['scores'], model.predict_proba(x)[:, 1])
            config['baseline'][BOOST_MODEL] = dict(BOOST, early_stopping=True)
            save_json(config, setting)
            with self.assertRaisesRegex(ValueError, 'random validation split'):
                candidate(prep, root, BOOST_MODEL, setting)

    def plan(self, root):
        source, frame, context, contract, base = fixtures.CertFileUSBTests().fixture(root)
        cv = dict(outer_folds=2, outer_seed=142, inner_calibration_folds=3, inner_calibration_seed_base=242, model_seed=42)
        previous = {key: root / key for key in ('group_cv_v3', 'file_usb_v4')}
        for path in previous.values():
            path.mkdir()
            (path / 'folds').mkdir()
        settings = {}
        for policy, history in POLICIES.items():
            settings[policy] = {}
            original = deepcopy(base)
            original['features']['history_policy'] = history
            original['config_version'] = 'fixture_' + policy
            for method in ('baseline', 'lstm_endpoint', BOOST_MODEL):
                config = deepcopy(original)
                if method == 'lstm_endpoint':
                    config['lstm_autoencoder']['score_policy'] = ENDPOINT_SCORE
                elif method == BOOST_MODEL:
                    config['baseline'][BOOST_MODEL] = BOOST
                path = root / (policy + '_' + method + '.json')
                save_json(config, path)
                settings[policy][method] = dict(path=str(path), sha256=file_sha256(path))
        controls = {key: [] for key in previous}
        for folded, provenance in development_folds(frame, n_splits=2, calibration_splits=3):
            fold = provenance['fold']
            for path in previous.values():
                save_json(provenance, path / 'folds' / f'fold_{fold}.json')
            for policy in POLICIES:
                config = load_json(Path(settings[policy]['baseline']['path']))
                prep = prepare_cert_frame(folded, context, contract, config, source, root, development=provenance)
                for model in OLD_MODELS:
                    trained = candidate(prep, root, model, diagnostics=model == 'lstm_autoencoder')
                    assessed = assess_development(trained, prep, root)
                    with np.load(trained / 'validation_inputs.npz', allow_pickle=False) as arrays:
                        cohort = hashlib.sha256(json.dumps(list(zip(arrays['item_ids'].tolist(), arrays['y'].tolist()))).encode()).hexdigest()
                    row = dict(job_id=f'cv_{fold}_{policy}_{model}', policy=policy, model=model, seed=42, phase='cv', fold=fold,
                        candidate=str(trained), preparation=str(prep), assessment=str(assessed),
                        threshold=load_json(trained / 'selection.json')['threshold'],
                        assessment_metrics=load_json(assessed / 'metrics_assessment.json'), cohort_sha256=cohort,
                        config_sha256=file_sha256(trained / 'config.json'),
                        candidate_manifest_sha256=file_sha256(trained / 'complete.json'),
                        preparation_manifest_sha256=file_sha256(prep / 'complete.json'),
                        assessment_manifest_sha256=file_sha256(assessed / 'complete.json'),
                        checks=dict(assessment_open_guard=True, assessment_after_candidate_complete=True,
                                    assessment_open_attempts_during_fit=[], test_open_attempts=[]))
                    controls['group_cv_v3' if policy == 'global_only_v2' else 'file_usb_v4'].append(row)
        for key, path in previous.items():
            save_json(dict(daily_manifest_sha256=file_sha256(source / 'complete.json'), cv=cv), path / 'plan.json')
            save_json(dict(results=controls[key]), path / 'summary.json')
            finish(path)
        plan = dict(plan_id=PLAN_ID, daily_preparation_id=source.name, daily_manifest_sha256=file_sha256(source / 'complete.json'),
                    methods=list(METHODS), configs=settings, cv=cv, boosting=BOOST, objective='inner_calibration_f1',
                    sources={key: dict(id=path.name, manifest_sha256=file_sha256(path / 'complete.json')) for key, path in previous.items()},
                    budget=dict(new_fits=16, reused_candidates=24, assessments=40, max_lstm_epochs=1,
                                max_boosting_iterations=3, cpu_threads=1, search_trials=0, minimum_free_gib=0))
        path = root / 'plan.json'
        save_json(plan, path)
        return source, path

    def test_full_runner_all_four_policies_freezes_selection_and_rejects_extra_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, path = self.plan(root)
            output = run(source, root, path)
            verify(output)
            summary = load_json(output / 'summary.json')
            self.assertEqual(summary['completed_candidates'], 40)
            self.assertEqual(summary['new_fits'], 16)
            self.assertEqual(summary['reused_candidates'], 24)
            self.assertEqual(len(summary['paired_method_differences']), 16)
            for result in summary['pooled_oof_decisions']:
                self.assertEqual(result['metrics']['support']['total'], 144)
                self.assertIsNone(result['metrics']['pr_auc'])
            for row in summary['results']:
                self.assertTrue(row['checks']['assessment_open_guard'])
                self.assertEqual(row['checks']['assessment_open_attempts_during_fit'], [])
            plan = load_json(path)
            wrong = deepcopy(plan)
            wrong['boosting']['early_stopping'] = True
            save_json(wrong, path)
            with self.assertRaisesRegex(ValueError, 'without random early stopping'):
                read_plan(path, source, root)
            setting = Path(plan['configs']['causal_all_v3']['lstm_endpoint']['path'])
            config = load_json(setting)
            config['lstm_autoencoder']['hidden_size'] += 1
            save_json(config, setting)
            plan['configs']['causal_all_v3']['lstm_endpoint']['sha256'] = file_sha256(setting)
            save_json(plan, path)
            with self.assertRaisesRegex(ValueError, 'undeclared extra'):
                read_plan(path, source, root)


if __name__ == '__main__':
    unittest.main()
