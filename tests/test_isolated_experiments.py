import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from experiment import prepare, candidate, final_evaluate, compare, verify, seal
from features import build_sequences


class IsolatedExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/synthetic_v0.2.json').read_text())
        config['generator'].update(n_events=600, n_users=12, n_days=30, anomaly_ratio=.1)
        for name in ('random_forest', 'isolation_forest'):
            config['baseline'][name]['n_estimators'] = 3
        config['lstm_autoencoder'].update(epochs=1, hidden_size=4)
        source = cls.root / 'config.json'
        source.write_text(json.dumps(config))
        cls.preparation = prepare(source, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_models_candidate_without_test_then_frozen_final(self):
        prep = self.preparation
        test = prep / 'test.csv'
        hidden = prep / 'hidden-test.csv'
        runs = []
        test.rename(hidden)
        try:
            for model in ('random_forest', 'isolation_forest', 'lstm_autoencoder'):
                run = candidate(prep, self.root, model)
                self.assertFalse((run / 'metrics_test.json').exists())
                self.assertFalse(json.loads((run / 'selection.json').read_text())['test_accessed'])
                # IDs must be plain Unicode arrays, readable without pickle on pandas 3 too.
                with np.load(run / 'train_inputs.npz', allow_pickle=False) as arrays:
                    self.assertIn(arrays['item_ids'].dtype.kind, ('U', 'S'))
                runs.append(run)
        finally:
            hidden.rename(test)
        evaluations = [final_evaluate(run, prep, self.root) for run in runs]
        self.assertEqual(len(compare(evaluations)), 3)
        for run, result in zip(runs, evaluations):
            again = final_evaluate(run, prep, self.root)
            a = pd.read_csv(result / 'predictions_test.csv')
            b = pd.read_csv(again / 'predictions_test.csv')
            np.testing.assert_allclose(a['score'], b['score'], rtol=1e-6, atol=1e-7)
            self.assertEqual(a['prediction'].tolist(), b['prediction'].tolist())
            arrays = np.load(result / 'test_inputs.npz')
            self.assertEqual(arrays['item_ids'].tolist(), arrays['member_ids'][:, -1].tolist())
            verify(run)
        # Even a self-consistent re-sealed artifact with a different cohort is rejected.
        target = evaluations[-1]
        frame = pd.read_csv(target / 'predictions_test.csv').iloc[1:]
        frame.to_csv(target / 'predictions_test.csv', index=False)
        seal(target)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            compare(evaluations)

    def test_changed_input_is_rejected(self):
        file = self.preparation / 'train.csv'
        original = file.read_bytes()
        try:
            file.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                candidate(self.preparation, self.root, 'random_forest')
        finally:
            file.write_bytes(original)

    def test_membership_is_stable_with_tied_timestamps_and_reordering(self):
        frame = pd.DataFrame({'event_id': ['c', 'a', 'b'], 'timestamp': ['2020-01-01'] * 3,
                              'user_id': ['u'] * 3, 'label': [0, 1, 0], 'feat': [3., 1., 2.]})
        a = build_sequences(frame, ['feat'], 2)
        b = build_sequences(frame.iloc[::-1], ['feat'], 2)
        np.testing.assert_array_equal(a['X'], b['X'])
        self.assertEqual(a['member_ids'].tolist(), [['a', 'b'], ['b', 'c']])
