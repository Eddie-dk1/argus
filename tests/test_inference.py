from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE, WINDOW_SCORE
from experiment import candidate, configured_score, finish, prepare
from experiment_tracking import file_sha256
from inference import PreparedBatch, batch_from_frame, load_frozen_run
from run_cert_operating_point import freeze_operating_candidate
from train_lstm_autoencoder import LSTMAutoencoder
from utils import load_json, save_json


class FrozenInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        project = Path(__file__).resolve().parents[1]
        config = load_json(project / 'configs/synthetic_v0.2.json')
        config['generator'].update(n_events=600, n_users=12, n_days=30, anomaly_ratio=.1)
        for model in ('random_forest', 'isolation_forest'):
            config['baseline'][model]['n_estimators'] = 3
        config['baseline'][BOOST_MODEL] = dict(max_iter=3, max_leaf_nodes=4, min_samples_leaf=2,
                                             early_stopping=False, class_weight='balanced')
        config['lstm_autoencoder'].update(epochs=1, hidden_size=4, batch_size=16)
        source = cls.root / 'config.json'
        save_json(config, source)
        cls.preparation = prepare(source, cls.root)
        cls.runs = {model: candidate(cls.preparation, cls.root, model) for model in (
            'random_forest', 'isolation_forest', BOOST_MODEL, 'lstm_autoencoder')}
        config['lstm_autoencoder']['score_policy'] = ENDPOINT_SCORE
        save_json(config, source)
        cls.runs['lstm_endpoint'] = candidate(cls.preparation, cls.root, 'lstm_autoencoder', source)
        # Exercise an actual operating bundle from the existing producer.
        from test_cert_model_variants import CertModelVariantTests
        _, cert_preparation, config = CertModelVariantTests().preparation(cls.root)
        trained = candidate(cert_preparation, cls.root, 'random_forest')
        cls.operating = freeze_operating_candidate(trained, cert_preparation, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def batch(self, runtime, role='validation_inputs.npz', path=None):
        with np.load((path or runtime.artifact_path) / role, allow_pickle=False) as arrays:
            return PreparedBatch(X=arrays['X'], feature_columns=runtime.spec.feature_columns,
                item_ids=arrays['item_ids'], member_ids=arrays['member_ids'],
                preparation_id=runtime.spec.preparation_id,
                preparation_manifest_sha256=runtime.spec.preparation_manifest_sha256,
                feature_state_sha256=runtime.spec.feature_state_sha256)

    def clone(self, destination, key='random_forest'):
        return Path(shutil.copytree(self.runs[key], destination / 'candidate'))

    def edit(self, artifact, filename, change):
        data = load_json(artifact / filename)
        change(data)
        save_json(data, artifact / filename)
        finish(artifact)

    def test_all_families_reuse_exact_scores_and_inputs_without_fit_or_selection(self):
        hashes = {str(p): file_sha256(p) for r in self.runs.values() for p in r.iterdir() if p.is_file()}
        for key, artifact in self.runs.items():
            with self.subTest(model=key):
                runtime = load_frozen_run(artifact, expected_manifest_sha256=file_sha256(artifact / 'complete.json'))
                batch = self.batch(runtime)
                frame = pd.read_csv(artifact / 'validation.csv')
                built = batch_from_frame(frame.drop(columns='label'), contract=load_json(artifact / 'contract.json'),
                    model=runtime.spec.model, preparation_manifest_sha256=runtime.spec.preparation_manifest_sha256,
                    feature_state_sha256=runtime.spec.feature_state_sha256)
                for field in ('X', 'item_ids', 'member_ids'):
                    np.testing.assert_array_equal(getattr(built, field), getattr(batch, field))
                if runtime.spec.model == 'lstm_autoencoder':
                    model = LSTMAutoencoder(batch.X.shape[-1], 4)
                    model.load_state_dict(torch.load(artifact / 'model.pt', weights_only=True, map_location='cpu'))
                else:
                    model = joblib.load(artifact / 'model.joblib')
                expected = configured_score(model, batch.X, runtime.spec.model, load_json(artifact / 'config.json'))
                with (patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('fit')),
                      patch('sklearn.ensemble.IsolationForest.fit', side_effect=AssertionError('fit')),
                      patch('sklearn.ensemble.HistGradientBoostingClassifier.fit', side_effect=AssertionError('fit')),
                      patch('torch.optim.Adam', side_effect=AssertionError('training')),
                      patch('experiment.select_best_threshold', side_effect=AssertionError('calibration')),
                      patch('utils.select_best_threshold', side_effect=AssertionError('calibration'))):
                    again = load_frozen_run(artifact)
                    np.testing.assert_array_equal(again.score(batch), expected)
                    np.testing.assert_array_equal(again.predict(batch), expected >= again.spec.threshold)
                    self.assertEqual(again.score(batch).dtype, expected.dtype)
        self.assertEqual({str(p): file_sha256(p) for r in self.runs.values() for p in r.iterdir() if p.is_file()}, hashes)

    def test_full_reconstruction_agrees_with_both_lstm_score_formulas(self):
        for key, policy in (('lstm_autoencoder', WINDOW_SCORE), ('lstm_endpoint', ENDPOINT_SCORE)):
            runtime = load_frozen_run(self.runs[key])
            batch = self.batch(runtime)
            reconstruction = runtime.reconstruct(batch)
            self.assertEqual(reconstruction.shape, batch.X.shape)
            residuals = (torch.from_numpy(reconstruction) - torch.from_numpy(batch.X)) ** 2
            axes = (1, 2) if policy == WINDOW_SCORE else 1
            errors = residuals.mean(dim=axes) if policy == WINDOW_SCORE else residuals[:, -1].mean(dim=axes)
            np.testing.assert_array_equal(runtime.score(batch), errors.numpy())
            self.assertEqual(runtime.spec.batch_size, 128)  # Fit batch size was 16.
            self.assertEqual(runtime.spec.score_policy, policy)

    def test_profiles_are_explicit_and_never_fall_back_from_ineligible_choice(self):
        with self.assertRaisesRegex(ValueError, 'Explicit operating profile'):
            load_frozen_run(self.operating)
        with self.assertRaisesRegex(ValueError, 'Unknown operating profile'):
            load_frozen_run(self.operating, profile='best')
        choices = load_json(self.operating / 'operating_points.json')
        with np.load(self.operating / 'predictions_calibration.npz', allow_pickle=False) as reference:
            for profile, choice in choices.items():
                runtime = load_frozen_run(self.operating, profile=profile)
                self.assertEqual(runtime.spec.profile, profile)
                self.assertEqual(runtime.spec.threshold, choice['threshold'])
                np.testing.assert_array_equal(runtime.score(self.batch(runtime)), reference['scores'])
                np.testing.assert_array_equal(runtime.predict(self.batch(runtime)), reference[profile])
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(shutil.copytree(self.operating, Path(tmp) / 'bundle'))
            self.edit(clone, 'operating_points.json', lambda d: d['fp1_per_1000'].update(
                threshold=1.25, eligible_on_calibration=False))
            runtime = load_frozen_run(clone, profile='fp1_per_1000')
            self.assertEqual(runtime.spec.threshold, 1.25)
            self.assertFalse(runtime.predict(self.batch(runtime)).any())
            self.edit(clone, 'operating_points.json', lambda d: d['fp2_per_1000'].update(threshold=float('nan')))
            with self.assertRaisesRegex(ValueError, 'finite numeric'):
                load_frozen_run(clone, profile='legacy_f1')

    def test_threshold_ties_use_inclusive_comparison_and_metadata_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self.clone(Path(tmp))
            first = load_frozen_run(artifact)
            batch = self.batch(first)
            tie = float(first.score(batch)[0])
            self.edit(artifact, 'selection.json', lambda d: d.update(threshold=tie))
            runtime = load_frozen_run(artifact)
            self.assertEqual(runtime.predict(batch)[0], 1)
            self.assertEqual(runtime.spec.decision_operator, '>=')
            self.assertEqual(runtime.spec.score_direction, 'higher_is_more_anomalous')
            with self.assertRaises(FrozenInstanceError):
                runtime.spec.threshold = 0
            with self.assertRaises(TypeError):
                runtime.artifact_hashes['model.joblib'] = 'changed'
            metadata = runtime.metadata
            metadata['artifact_hashes'].clear()
            self.assertIn('model.joblib', runtime.metadata['artifact_hashes'])

    def test_invalid_batches_and_provenance_are_rejected_before_scoring(self):
        runtime = load_frozen_run(self.runs['random_forest'])
        batch = self.batch(runtime)
        members = batch.member_ids.copy()
        members[0, 0] = members[0, -1]
        invalid = [dict(feature_columns=tuple(reversed(batch.feature_columns))),
                   dict(preparation_id='other'), dict(preparation_manifest_sha256='0' * 64),
                   dict(feature_state_sha256='0' * 64), dict(X=batch.X[:, :-1]),
                   dict(X=batch.X.astype(object)), dict(X=np.full_like(batch.X, np.nan)),
                   dict(X=np.full_like(batch.X, np.inf)), dict(X=batch.X[:, None]),
                   dict(item_ids=batch.item_ids[::-1]), dict(item_ids=np.repeat(batch.item_ids[:1], len(batch.X))),
                   dict(member_ids=batch.member_ids[:, :-1]), dict(member_ids=members),
                   dict(item_ids=batch.item_ids.astype(object))]
        for change in invalid:
            with self.subTest(change=list(change)):
                with patch('inference.configured_score', side_effect=AssertionError('invalid input reached scorer')):
                    with self.assertRaises(ValueError):
                        runtime.score(replace(batch, **change))
        with self.assertRaisesRegex(TypeError, 'PreparedBatch'):
            runtime.predict({'X': batch.X, 'labels': np.zeros(len(batch.X))})
        with self.assertRaises(TypeError):
            replace(batch, labels=np.zeros(len(batch.X)))
        lstm = load_frozen_run(self.runs['lstm_autoencoder'])
        sequential = self.batch(lstm)
        for x in (sequential.X.astype(np.float64), sequential.X[:, :-1]):
            with self.assertRaisesRegex(ValueError, 'LSTM requires'):
                lstm.score(replace(sequential, X=x))

    def test_unknown_policies_and_inconsistent_sealed_contracts_are_rejected(self):
        cases = [('selection.json', lambda d: d.update(model='unrecognized'), 'Unsupported frozen model'),
                 ('selection.json', lambda d: d.update(score_policy='invented'), 'score_policy'),
                 ('selection.json', lambda d: d.update(score_direction='lower_is_more_anomalous'), 'score_direction'),
                 ('selection.json', lambda d: d.update(decision_operator='>'), 'decision_operator'),
                 ('selection.json', lambda d: d.update(threshold=float('inf')), 'finite numeric'),
                 ('selection.json', lambda d: d.update(threshold=True), 'finite numeric'),
                 ('selection.json', lambda d: d.pop('threshold'), 'finite numeric'),
                 ('selection.json', lambda d: d.update(preparation_id='other'), 'preparation identities'),
                 ('contract.json', lambda d: d.update(sequence_length=7), 'sequence lengths'),
                 ('contract.json', lambda d: d['tabular_columns'].reverse(), 'ordered columns'),
                 ('contract.json', lambda d: d['tabular_columns'].__setitem__(0, 'label'), 'provenance'),
                 ('feature_state.json', lambda d: d.update(policy='unknown'), 'feature/history policy'),
                 ('config.json', lambda d: d.update(scenario='cicids2017'), 'Unsupported prepared scenario')]
        for filename, mutation, message in cases:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(filename=filename, message=message):
                clone = self.clone(Path(tmp))
                self.edit(clone, filename, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    load_frozen_run(clone)
        with self.assertRaisesRegex(ValueError, 'only the legacy_f1'):
            load_frozen_run(self.runs['random_forest'], profile='fp1_per_1000')
        with tempfile.TemporaryDirectory() as tmp:
            clone = self.clone(Path(tmp), 'lstm_endpoint')
            self.edit(clone, 'config.json', lambda d: d['lstm_autoencoder'].update(score_policy='unknown'))
            with self.assertRaisesRegex(ValueError, 'score policy'):
                load_frozen_run(clone)
        cert_cases = [('contract.json', lambda d: d.update(history_policy='unknown'), 'history policy'),
                      ('contract.json', lambda d: d.update(sequence_policy='CSV order'), 'sequence policy'),
                      ('feature_state.json', lambda d: d.update(policy='unknown'), 'history policy differs'),
                      ('feature_state.json', lambda d: d['scale'].__setitem__(0, 0), 'feature scaler'),
                      ('lineage.json', lambda d: d.update(checkpoint_sha256='0' * 64), 'checkpoint identity'),
                      ('lineage.json', lambda d: d.update(no_refit=False), 'freeze policy')]
        for filename, mutation, message in cert_cases:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(cert_field=filename, message=message):
                clone = Path(shutil.copytree(self.operating, Path(tmp) / 'bundle'))
                self.edit(clone, filename, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    load_frozen_run(clone, profile='legacy_f1')

    def test_corruption_incomplete_artifacts_and_wrong_checkpoint_are_rejected(self):
        for filename in ('model.joblib', 'feature_state.json', 'selection.json'):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(filename=filename):
                clone = self.clone(Path(tmp))
                with (clone / filename).open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                    load_frozen_run(clone)
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            load_frozen_run(self.runs['random_forest'], expected_manifest_sha256='0' * 64)
        for change, message in ((lambda p: (p / 'model.joblib').unlink(), 'Missing frozen weights'),
                                (lambda p: save_json({'status': 'running'}, p / 'status.json'), 'status must be complete')):
            with tempfile.TemporaryDirectory() as tmp:
                clone = self.clone(Path(tmp))
                change(clone)
                from experiment import seal
                seal(clone)
                with self.assertRaisesRegex(ValueError, message):
                    load_frozen_run(clone)
        with tempfile.TemporaryDirectory() as tmp:
            clone = self.clone(Path(tmp))
            shutil.copy2(self.runs['isolation_forest'] / 'model.joblib', clone / 'model.joblib')
            finish(clone)
            with self.assertRaisesRegex(ValueError, 'model family'):
                load_frozen_run(clone)
        with tempfile.TemporaryDirectory() as tmp:
            clone = self.clone(Path(tmp), 'lstm_autoencoder')
            torch.save({'unknown': torch.zeros(1)}, clone / 'model.pt')
            finish(clone)
            with self.assertRaisesRegex(ValueError, 'Invalid frozen inference artifact'):
                load_frozen_run(clone)

    def test_label_free_builder_preserves_order_membership_and_rejects_mixed_splits(self):
        runtime = load_frozen_run(self.runs['random_forest'])
        contract = load_json(runtime.artifact_path / 'contract.json')
        frame = pd.read_csv(runtime.artifact_path / 'validation.csv')
        kwargs = dict(contract=contract, model=runtime.spec.model,
                      preparation_manifest_sha256=runtime.spec.preparation_manifest_sha256,
                      feature_state_sha256=runtime.spec.feature_state_sha256)
        original = batch_from_frame(frame, **kwargs)
        frame['label'] = 'this reporting field is never read'
        changed = batch_from_frame(frame.iloc[::-1], **kwargs)
        for field in ('X', 'item_ids', 'member_ids'):
            np.testing.assert_array_equal(getattr(original, field), getattr(changed, field))
        frame['split'] = ['train', *(['validation'] * (len(frame) - 1))]
        with self.assertRaisesRegex(ValueError, 'one split'):
            batch_from_frame(frame, **kwargs)

    def test_empty_batches_retain_shapes_and_dtypes(self):
        for artifact in self.runs.values():
            runtime = load_frozen_run(artifact)
            batch = self.batch(runtime)
            empty = replace(batch, X=batch.X[:0], item_ids=batch.item_ids[:0], member_ids=batch.member_ids[:0])
            self.assertEqual(runtime.score(empty).shape, (0,))
            self.assertEqual(runtime.predict(empty).dtype, np.int64)
            if runtime.spec.model == 'lstm_autoencoder':
                self.assertEqual(runtime.reconstruct(empty).shape, empty.X.shape)
                self.assertEqual(runtime.score(empty).dtype, np.float32)

    def test_tabular_loading_and_scoring_never_import_torch_or_open_original_preparation(self):
        code = '''
import builtins, sys
original_import = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Tabular inference imported PyTorch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_torch
import numpy as np
from pathlib import Path
from inference import PreparedBatch, load_frozen_run
for path in sys.argv[1:]:
    runtime = load_frozen_run(path, profile='legacy_f1' if (Path(path) / 'operating_points.json').exists() else None)
    with np.load(runtime.artifact_path / 'validation_inputs.npz', allow_pickle=False) as a:
        batch = PreparedBatch(X=a['X'], feature_columns=runtime.spec.feature_columns,
            item_ids=a['item_ids'], member_ids=a['member_ids'], preparation_id=runtime.spec.preparation_id,
            preparation_manifest_sha256=runtime.spec.preparation_manifest_sha256,
            feature_state_sha256=runtime.spec.feature_state_sha256)
    assert runtime.score(batch).shape == (len(batch.X),)
    assert runtime.predict(batch).shape == (len(batch.X),)
assert 'torch' not in sys.modules
'''
        # A relocated, complete bundle is sufficient; the producer directory is closed.
        with tempfile.TemporaryDirectory() as tmp:
            copies = [str(shutil.copytree(self.runs[key], Path(tmp) / key)) for key in (
                'random_forest', 'isolation_forest', BOOST_MODEL)]
            copies.append(str(shutil.copytree(self.operating, Path(tmp) / 'cert_operating')))
            hidden = self.preparation.with_name('hidden_preparation')
            self.preparation.rename(hidden)
            try:
                result = subprocess.run([sys.executable, '-c', code, *copies],
                    env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src')),
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            finally:
                hidden.rename(self.preparation)


if __name__ == '__main__':
    unittest.main()
