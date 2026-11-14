from contextlib import ExitStack
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from argus import main
from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE
from datasets.cert_validation import assess_development
from experiment import candidate, finish, prepare, verify
from experiment_tracking import file_sha256
from features import add_causal_context
from frozen_runtime import inspect_run, load_saved_inputs, replay_run
from inference import load_frozen_run
from run_cert_operating_point import assess_operating_candidate, freeze_operating_candidate
from synthetic_scenarios import (NOOP_TOLERANCES, load_scenario_source, numeric_check,
    run_scenario, select_window)
from utils import load_json, save_json


PROJECT = Path(__file__).resolve().parents[1]


class SyntheticScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        config = load_json(PROJECT / 'configs/synthetic_v0.2.json')
        config['generator'].update(n_events=1200, n_users=8, n_days=30, anomaly_ratio=.1)
        for model in ('random_forest', 'isolation_forest'):
            config['baseline'][model]['n_estimators'] = 3
        config['baseline'][BOOST_MODEL] = dict(max_iter=3, max_leaf_nodes=4, min_samples_leaf=2,
                                             early_stopping=False, class_weight='balanced')
        config['lstm_autoencoder'].update(epochs=1, hidden_size=4, batch_size=16)
        source = cls.root / 'fixture.json'
        save_json(config, source)
        cls.preparation = prepare(source, cls.root)
        cls.candidates = {name: candidate(cls.preparation, cls.root, name) for name in
                          ('random_forest', 'isolation_forest', BOOST_MODEL, 'lstm_autoencoder')}
        config['lstm_autoencoder']['score_policy'] = ENDPOINT_SCORE
        save_json(config, source)
        cls.candidates['lstm_endpoint'] = candidate(cls.preparation, cls.root, 'lstm_autoencoder', source)
        # Existing producers also test CERT's supported prepared-input route.
        from test_cert_model_variants import CertModelVariantTests
        _, cls.cert_preparation, _ = CertModelVariantTests().preparation(cls.root)
        cls.cert_candidate = candidate(cls.cert_preparation, cls.root, 'random_forest')
        cls.cert_assessment = assess_development(cls.cert_candidate, cls.cert_preparation, cls.root)
        cls.operating = freeze_operating_candidate(cls.cert_candidate, cls.cert_preparation, cls.root)
        cls.operating_assessment = assess_operating_candidate(cls.operating, cls.cert_preparation, cls.root)
        cls.sources = [cls.preparation, *cls.candidates.values(), cls.cert_preparation,
                       cls.cert_candidate, cls.cert_assessment, cls.operating, cls.operating_assessment]
        cls.initial_hashes = cls.hashes()

    @classmethod
    def hashes(cls):
        return {str(p): file_sha256(p) for source in cls.sources for p in source.rglob('*') if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_rf(self):
        return load_frozen_run(self.candidates['random_forest'])

    def no_fit(self):
        stack = ExitStack()
        for target in ('features.fit_feature_state', 'sklearn.ensemble.RandomForestClassifier.fit',
                       'sklearn.ensemble.IsolationForest.fit', 'sklearn.ensemble.HistGradientBoostingClassifier.fit',
                       'experiment.select_best_threshold', 'utils.select_best_threshold',
                       'cert_operating_point.select_operating_points', 'torch.optim.Adam'):
            stack.enter_context(patch(target, side_effect=AssertionError('Frozen path attempted fit/selection')))
        return stack

    def test_formulas_and_causality_have_independent_expected_values(self):
        source = load_scenario_source(self.run_rf(), self.preparation, 'validation')
        rows = pd.concat([source.raw.iloc[:1]] * 10, ignore_index=True)
        rows['user_id'] = ['u'] * 7 + ['v'] * 3
        rows['event_id'] = [f'e{i:02d}' for i in range(10)]
        rows['timestamp'] = pd.date_range('2025-01-01', periods=10, freq='h')
        rows['bytes_sent'] = [1., 3., 0., 7., 2., 5., 6., 1., 1., 1.]
        rows['bytes_received'] = [1., 0., 2., 4., 1., 1., 1., 1., 1., 1.]
        base = add_causal_context(rows, 2).set_index('event_id')
        np.testing.assert_allclose(base.loc[['e00', 'e01', 'e02', 'e03'], 'log_bytes_sent'],
                                   [np.log(2), np.log(4), 0., np.log(8)], rtol=1e-15, atol=1e-15)
        np.testing.assert_array_equal(base.loc[['e00', 'e01', 'e02', 'e03'], 'bytes_ratio'], [1., 0., 0., 1.75])
        np.testing.assert_allclose(base.loc[['e00', 'e01', 'e02', 'e03'], 'prior_log_bytes_sent_sum'],
                                   [0., np.log(2), np.log(2) + np.log(4), np.log(4)], rtol=1e-15, atol=1e-15)
        modified = rows.copy()
        modified.loc[3, 'bytes_sent'] = 14.
        variant = add_causal_context(modified, 2).set_index('event_id')
        self.assertEqual(base.loc['e03', 'prior_log_bytes_sent_sum'], variant.loc['e03', 'prior_log_bytes_sent_sum'])
        self.assertEqual(variant.loc['e03', 'log_bytes_sent'], np.log(15))
        self.assertEqual(variant.loc['e03', 'bytes_ratio'], 3.5)
        delta = np.log(15) - np.log(8)
        for event in ('e04', 'e05'):
            self.assertAlmostEqual(variant.loc[event, 'prior_log_bytes_sent_sum'] -
                                   base.loc[event, 'prior_log_bytes_sent_sum'], delta, places=14)
        self.assertAlmostEqual(variant.loc['e06', 'prior_log_bytes_sent_sum'],
                               base.loc['e06', 'prior_log_bytes_sent_sum'], places=14)
        pd.testing.assert_frame_equal(base.loc[['e00', 'e01', 'e02']], variant.loc[['e00', 'e01', 'e02']])
        pd.testing.assert_frame_equal(base.loc[['e07', 'e08', 'e09']], variant.loc[['e07', 'e08', 'e09']])

    def test_selection_ignores_labels_scores_and_input_row_order(self):
        source = load_scenario_source(self.run_rf(), self.preparation, 'validation')
        kwargs = dict(sequence_length=5, rolling_window=10)
        first = select_window(source.raw, **kwargs)
        altered = source.raw.sample(frac=1, random_state=7).copy()
        altered['label'], altered['anomaly_type'], altered['score'] = 'unused', 'unused', np.nan
        self.assertEqual(first, select_window(altered, **kwargs))
        self.assertEqual(first, select_window(altered, item_id=first['item_id'], **kwargs))
        self.assertGreaterEqual(first['earliest_member_prior_events'], 10)
        self.assertEqual(first['member_ids'][-1], first['item_id'])
        with self.assertRaisesRegex(ValueError, 'No eligible window'):
            select_window(source.raw, item_id='missing', **kwargs)
        with self.assertRaisesRegex(ValueError, 'No eligible window'):
            select_window(source.raw.iloc[:5], **kwargs)

    def test_all_five_frozen_variants_six_cases_match_full_split_without_fit(self):
        selections = []
        with self.no_fit():
            for name, artifact in self.candidates.items():
                run = load_frozen_run(artifact)
                for position in ('endpoint', 'earlier'):
                    for operation in ('noop', 'double', 'zero'):
                        with self.subTest(model=name, position=position, operation=operation):
                            output = run_scenario(run, self.preparation, self.root,
                                                  position=position, operation=operation)
                            verify(output)
                            summary, lineage = [load_json(output / n) for n in ('summary.json', 'lineage.json')]
                            self.assertTrue(summary['checks']['full_split_oracle_exact'])
                            self.assertTrue(summary['checks']['other_users_unchanged'])
                            self.assertEqual(lineage['variant_ground_truth'], 'unknown')
                            self.assertFalse(lineage['include_in_benchmark_metrics'])
                            self.assertFalse((output / 'metrics.json').exists())
                            selections.append(summary['selected']['item_id'])
                            self.assertTrue(set(summary['changed_feature_names']) <= {
                                'feat_log_bytes_sent', 'feat_bytes_ratio', 'feat_prior_log_bytes_sent_sum'})
                            with (np.load(output / 'variant_inputs.npz', allow_pickle=False) as variant,
                                  np.load(output / 'oracle_inputs.npz', allow_pickle=False) as oracle,
                                  np.load(output / 'base_inputs.npz', allow_pickle=False) as base):
                                self.assertEqual(set(variant.files), {'X', 'item_ids', 'member_ids'})
                                for key in variant.files:
                                    np.testing.assert_array_equal(variant[key], oracle[key])
                                np.testing.assert_array_equal(base['member_ids'], variant['member_ids'])
                                if operation == 'noop':
                                    np.testing.assert_allclose(base['X'], variant['X'], **NOOP_TOLERANCES[str(base['X'].dtype)])
                                    self.assertEqual(summary['changed_decision_items'], 0)
                            raw_base, raw_variant = [pd.read_csv(output / n) for n in ('raw_base.csv', 'raw_variant.csv')]
                            self.assertNotIn('label', raw_variant)
                            pd.testing.assert_frame_equal(raw_base.drop(columns='bytes_sent'), raw_variant.drop(columns='bytes_sent'))
                            event = summary['patch']['event_id']
                            if operation != 'noop':
                                self.assertGreater(summary['changed_input_items'], 0)
                                changed = pd.read_csv(output / 'changed_features.csv')
                                current_prior = changed[(changed.event_id == event) &
                                                        (changed.feature == 'feat_prior_log_bytes_sent_sum')]
                                self.assertTrue(current_prior.empty)  # No own contribution to prior sum.
                                if position == 'earlier':
                                    future_prior = changed[(changed.event_id == summary['selected']['item_id']) &
                                                           (changed.feature == 'feat_prior_log_bytes_sent_sum')]
                                    self.assertEqual(len(future_prior), 1)
        self.assertEqual(len(set(selections)), 1)
        self.assertEqual(self.hashes(), self.initial_hashes)

    def test_invalid_fields_values_and_roots_never_overwrite_sources(self):
        run = self.run_rf()
        for field in ('label', 'user_id', 'timestamp', 'bytes_received'):
            with self.assertRaisesRegex(ValueError, 'Only the raw field'):
                run_scenario(run, self.preparation, self.root, field=field)
        for value in (-1, np.nan, np.inf, -np.inf, True, '5', None, 10 ** 400):
            with self.subTest(value=str(value)), self.assertRaisesRegex(ValueError, 'finite nonnegative'):
                run_scenario(run, self.preparation, self.root, operation='set', value=value)
        for kwargs in (dict(position='random'), dict(operation='unknown'), dict(operation='double', value=2)):
            with self.assertRaises(ValueError):
                run_scenario(run, self.preparation, self.root, **kwargs)
        for root in (self.preparation, self.candidates['random_forest']):
            with self.assertRaisesRegex(ValueError, 'outside every source'):
                run_scenario(run, self.preparation, root)
        absent = self.root / 'missing-mount'
        with self.assertRaisesRegex(ValueError, 'must already exist'):
            run_scenario(run, self.preparation, absent)
        self.assertFalse(absent.exists())
        self.assertEqual(self.hashes(), self.initial_hashes)

    def test_oracle_and_noop_tolerances_fail_closed(self):
        run = self.run_rf()
        with patch('synthetic_scenarios.recompute_user', side_effect=lambda source, *_: source.prepared.copy()):
            with self.assertRaisesRegex(ValueError, 'full-split oracle'):
                run_scenario(run, self.preparation, self.root, operation='double')
        for dtype in (np.float32, np.float64):
            base = np.asarray([0., 1.], dtype=dtype)
            numeric_check(base, base.copy(), 'control')
            changed = base.copy()
            changed[0] = 1e-4
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                numeric_check(base, changed, 'drift')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            numeric_check(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float64), 'dtype')
        self.assertEqual(self.hashes(), self.initial_hashes)

    def test_prepared_cert_replay_profiles_and_raw_refusal(self):
        with self.no_fit():
            for artifact, source in ((self.cert_candidate, self.cert_assessment),
                                     (self.operating, self.operating_assessment)):
                profiles = list(load_json(artifact / 'operating_points.json')) if artifact == self.operating else [None]
                for profile in profiles:
                    run = load_frozen_run(artifact, profile=profile)
                    inspected = inspect_run(run, self.root, source=source, role='assessment')
                    self.assertFalse(load_json(inspected / 'inspection.json')['capabilities']['cert_raw_what_if'])
                    output = replay_run(run, self.root, source=source, role='assessment')
                    self.assertTrue(load_json(output / 'replay.json')['scores_exact'])
                    replay_run(run, self.root)  # Existing validation or operating calibration.
                    with self.assertRaisesRegex(ValueError, 'reaggregation contract'):
                        run_scenario(run, self.cert_preparation, self.root)
        self.assertEqual(self.hashes(), self.initial_hashes)

    def test_replay_rejects_damaged_reordered_or_mismatched_artifacts(self):
        run = self.run_rf()
        with patch('frozen_runtime._sealed_files', side_effect=AssertionError('Opened a closed test artifact')):
            with self.assertRaisesRegex(ValueError, 'final/test artifacts'):
                load_saved_inputs(load_frozen_run(self.cert_candidate), self.preparation)
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            load_saved_inputs(run, expected_manifest='0' * 64)
        with self.assertRaisesRegex(ValueError, 'lineage differs'):
            load_saved_inputs(run, self.cert_assessment, 'assessment')
        clone = Path(shutil.copytree(self.candidates['random_forest'], self.root / 'corrupt-candidate'))
        with np.load(clone / 'validation_inputs.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays['member_ids'] = arrays['member_ids'].copy()
        arrays['member_ids'][:, :2] = arrays['member_ids'][:, 1::-1]
        np.savez_compressed(clone / 'validation_inputs.npz', **arrays)
        finish(clone)
        with self.assertRaisesRegex(ValueError, 'membership differ'):
            load_saved_inputs(load_frozen_run(clone))
        (clone / 'model.joblib').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Artifact changed'):
            load_frozen_run(clone)

    def test_cli_inspect_replay_scenario_and_diagnose_dispatch(self):
        env = dict(os.environ, PYTHONPATH=str(PROJECT / 'src'))
        common = [sys.executable, str(PROJECT / 'src/argus.py'), '--root', str(self.root)]
        for command in ('inspect', 'replay', 'scenario'):
            arguments = [*common, command, '--run', str(self.candidates['random_forest'])]
            if command == 'scenario':
                arguments += ['--preparation', str(self.preparation), '--position', 'earlier', '--operation', 'zero']
            result = subprocess.run(arguments, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            verify(Path(result.stdout.strip().splitlines()[-1]))
        with patch('argus.subprocess.run') as child:
            child.return_value.returncode = 0
            code = main(['--root', str(self.root), 'diagnose', '--model-variants', 'variants', '--operating-points', 'points'])
            self.assertEqual(code, 0)
            self.assertEqual(child.call_args.args[0][1], str(PROJECT / 'src/run_frozen_diagnostics.py'))
            self.assertIn('--operating-points', child.call_args.args[0])
        for command in ('inspect', 'replay', 'diagnose', 'scenario'):
            result = subprocess.run([*common, command, '--help'], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_tabular_cli_paths_work_without_importing_torch(self):
        code = '''
import builtins, sys
original = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Tabular CLI imported torch')
    return original(name, *args, **kwargs)
builtins.__import__ = no_torch
from argus import main
root, run, preparation = sys.argv[1:]
for command in ('inspect', 'replay', 'scenario'):
    args = ['--root', root, command, '--run', run]
    if command == 'scenario': args += ['--preparation', preparation]
    assert main(args) == 0
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.root),
                                 str(self.candidates['random_forest']), str(self.preparation)],
            env=dict(os.environ, PYTHONPATH=str(PROJECT / 'src')), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
