from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from datasets.cert_experiments import CAUSAL_POLICY, prepare_cert_frame
from datasets.cert_validation import assess_development, development_folds
from experiment import candidate, inputs, score, seal, verify
from experiment_tracking import file_sha256
from run_cert_group_cv import MODELS, PLAN_ID, read_plan, run, summarize
import test_cert_validation as fixtures


class CertGroupCVTests(unittest.TestCase):
    def fixture(self, root):
        source, frame, context, contract, config = fixtures.CertValidationTests().fixture(root)
        for model in ('random_forest', 'isolation_forest'):
            config['baseline'][model]['n_estimators'] = 3
        evidence = [dict(day_key=row.day_key, attributions=[dict(
            scenario=str(int(row.account_id[1:]) % 3 + 1), incident_id=row.group)])
            for row in frame.loc[frame.label.eq(1)].itertuples()]
        (source / 'positive_evidence.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in evidence))
        seal(source)
        return source, frame, context, contract, config

    def plan(self, root, source, config):
        previous = root / 'technical_development'
        previous.mkdir()
        (previous / 'summary.json').write_text(json.dumps(dict(daily_manifest_sha256=file_sha256(source / 'complete.json'))))
        seal(previous)
        configs = {}
        for key in ('global_only_v2', 'causal_all_v3'):
            settings = deepcopy(config)
            if key == 'causal_all_v3':
                settings['config_version'] = 'fixture_v3'
                settings['features']['history_policy'] = CAUSAL_POLICY
            path = root / (key + '.json')
            path.write_text(json.dumps(settings))
            configs[key] = dict(path=str(path), sha256=file_sha256(path))
        plan = dict(plan_id=PLAN_ID, daily_preparation_id=source.name,
            daily_manifest_sha256=file_sha256(source / 'complete.json'),
            development_preparation_id=previous.name,
            development_preparation_manifest_sha256=file_sha256(previous / 'complete.json'),
            configs=configs, models=list(MODELS), objective='inner_calibration_f1',
            cv=dict(outer_folds=2, outer_seed=142, inner_calibration_folds=3,
                    inner_calibration_seed_base=242, model_seed=42),
            budget=dict(candidates=12, cpu_threads=1, search_trials=0, max_lstm_epochs=1, minimum_free_gib=0))
        path = root / 'cv_plan.json'
        path.write_text(json.dumps(plan))
        return path

    def test_frozen_assessment_all_models_never_fit_or_reselect_and_keep_exact_inputs(self):
        import torch
        from train_lstm_autoencoder import LSTMAutoencoder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frame, context, contract, config = self.fixture(root)
            part, provenance = next(development_folds(frame, n_splits=3, calibration_splits=3))
            prep = prepare_cert_frame(part, context, contract, config, source, root, development=provenance)
            expected_ids = None
            for name in MODELS:
                with self.subTest(model=name):
                    trained = candidate(prep, root, name)
                    digest = file_sha256(trained / 'complete.json')
                    with (patch('experiment.RandomForestClassifier.fit', side_effect=AssertionError('fit')),
                          patch('experiment.IsolationForest.fit', side_effect=AssertionError('fit')),
                          patch('torch.optim.Adam', side_effect=AssertionError('optimizer')),
                          patch('experiment.select_best_threshold', side_effect=AssertionError('threshold'))):
                        assessed = assess_development(trained, prep, root)
                    self.assertEqual(file_sha256(trained / 'complete.json'), digest)
                    verify(assessed)
                    x, y, ids, members = inputs(pd.read_csv(prep / 'assessment.csv'),
                                               json.loads((prep / 'contract.json').read_text()), name)
                    with np.load(assessed / 'assessment_inputs.npz', allow_pickle=False) as arrays:
                        for key, value in dict(X=x, y=y, item_ids=ids, member_ids=members).items():
                            np.testing.assert_array_equal(arrays[key], value)
                    if expected_ids is None:
                        expected_ids = ids
                    np.testing.assert_array_equal(expected_ids, ids)
                    if name == 'lstm_autoencoder':
                        model = LSTMAutoencoder(x.shape[2], config[name]['hidden_size'])
                        model.load_state_dict(torch.load(trained / 'model.pt', map_location='cpu', weights_only=True))
                    else:
                        model = joblib.load(trained / 'model.joblib')
                    selection = json.loads((trained / 'selection.json').read_text())
                    values = score(model, x, name)
                    with np.load(assessed / 'predictions_assessment.npz', allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(arrays['scores'], values)
                        np.testing.assert_array_equal(arrays['predictions'], values >= selection['threshold'])

    def test_assessment_rejects_a_different_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frame, context, contract, config = self.fixture(root)
            part, provenance = next(development_folds(frame, n_splits=3, calibration_splits=3))
            first = prepare_cert_frame(part, context, contract, config, source, root, development=provenance)
            second = prepare_cert_frame(part, context, contract, config, source, root, development=provenance)
            trained = candidate(first, root, 'random_forest')
            with self.assertRaisesRegex(ValueError, 'matching development-only'):
                assess_development(trained, second, root)

    def test_worker_refuses_assessment_reads_during_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frame, context, contract, config = self.fixture(root)
            part, provenance = next(development_folds(frame, n_splits=3, calibration_splits=3))
            prep = prepare_cert_frame(part, context, contract, config, source, root, development=provenance)
            config_path = root / 'worker_config.json'
            config_path.write_text(json.dumps(config))
            job = root / 'job.json'
            job.write_text(json.dumps(dict(preparation=str(prep), config=str(config_path),
                                          output_root=str(root), model='random_forest')))
            code = ("import sys; from pathlib import Path; from unittest.mock import patch; "
                    "from run_cert_robustness import worker; "
                    "blocked=lambda *a, **k: Path(sys.argv[2]).read_text(); "
                    "p=patch('experiment.candidate',side_effect=blocked); p.start(); "
                    "worker(Path(sys.argv[1]),guard_assessment=True)")
            completed = subprocess.run([sys.executable, '-c', code, str(job), str(prep / 'assessment.csv')],
                                       capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Outer assessment is closed while fitting', completed.stderr)

    def test_plan_rejects_an_extra_factor_or_missing_declared_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, _, config = self.fixture(root)
            path = self.plan(root, source, config)
            plan = json.loads(path.read_text())
            broken = deepcopy(plan)
            broken['models'] = ['random_forest'] * 3
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'Unsupported bounded'):
                read_plan(path, source, root)
            changed_path = Path(plan['configs']['causal_all_v3']['path'])
            changed = json.loads(changed_path.read_text())
            changed['lstm_autoencoder']['hidden_size'] += 1
            changed_path.write_text(json.dumps(changed))
            plan['configs']['causal_all_v3']['sha256'] = file_sha256(changed_path)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Only the history factor'):
                read_plan(path, source, root)

    def test_bounded_runner_all_models_and_policies_preserves_full_oof_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frame, _, _, config = self.fixture(root)
            plan_path = self.plan(root, source, config)
            output = run(source, root, plan_path)
            verify(output)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['completed_candidates'], 12)
            self.assertEqual(len(summary['fold_macro_aggregates']), 6)
            self.assertTrue(summary['every_endpoint_assessed_once_per_policy_model'])
            self.assertEqual(summary['population']['accounts'], 18)
            self.assertEqual(summary['population']['endpoints'], 18 * (12 - 4))
            for result in summary['results']:
                self.assertEqual(result['checks']['assessment_open_attempts_during_fit'], [])
                self.assertEqual(result['checks']['test_open_attempts'], [])
                self.assertTrue(result['checks']['assessment_after_candidate_complete'])
            for pooled in summary['pooled_oof_decisions']:
                self.assertEqual(pooled['metrics']['support']['total'], 144)
                self.assertIsNone(pooled['metrics']['pr_auc'])
            primary = frame.loc[frame.split.eq('train')]
            # A failed or omitted fold may never silently disappear from averages.
            with self.assertRaisesRegex(ValueError, 'every declared run'):
                summarize(summary['results'][:-1], json.loads(plan_path.read_text()), primary, source, output)
            with self.assertRaisesRegex(ValueError, 'every declared run'):
                summarize(summary['results'] + [summary['results'][0]], json.loads(plan_path.read_text()), primary, source, output)


if __name__ == '__main__':
    unittest.main()
