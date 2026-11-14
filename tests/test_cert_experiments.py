import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from datasets.cert_days import Day, FEATURES, FEATURE_SET
from datasets.cert_experiments import (fit_global, history_values, prepare_cert,
                                      transform, POLICY, GLOBAL_POLICY)
from experiment import seal, candidate, final_evaluate, compare, verify


class CertExperimentTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'days'
        source.mkdir()
        rows, contexts = [], []
        for day_index in range(16):
            date = str((pd.Timestamp('2010-01-01') + pd.Timedelta(days=day_index * 2)).date())
            for split in ('train', 'validation', 'test'):
                for index in range(2):
                    user = f'{split}{index}'
                    label = int(day_index in (7, 12))
                    day = Day()
                    for n in range(1 + day_index % 4 + label * 5):
                        day.add('logon', (str(n), '', user, f'P{index}', 'Logon'), 28800 + n, label)
                    key = user + ':' + date
                    rows.append(dict(item_id='snapshot:' + key, day_key=key, account_id=user,
                                     date=date, split=split, group=user, label=label, **day.features(date)))
                    contexts.append(dict(day_key=key, pcs=[f'P{index}'], domains=[f'{split}.example'], recipients=[]))
        frame = pd.DataFrame(rows).sort_values(['date', 'account_id']).reset_index(drop=True)
        context_map = {c['day_key']: c for c in contexts}
        self.write_context(source, [context_map[k] for k in frame['day_key']])
        frame.to_csv(source / 'account_days.csv', index=False)
        (source / 'contract.json').write_text(json.dumps(dict(feature_set=FEATURE_SET, feature_columns=FEATURES, snapshot_id='snapshot')))
        seal(source)
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/cert_r4_2_experiment_v1.json').read_text())
        for name in ('random_forest', 'isolation_forest'):
            config['baseline'][name]['n_estimators'] = 3
        config['lstm_autoencoder'].update(epochs=1, hidden_size=4)
        config_path = root / 'config.json'
        config_path.write_text(json.dumps(config))
        return source, config_path, frame

    def write_context(self, source, contexts):
        with gzip.open(source / 'day_context.jsonl.gz', 'wt') as stream:
            for context in contexts:
                stream.write(json.dumps(context) + '\n')

    def test_holdout_cannot_change_fit_and_train_history_is_causal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            path = source / 'day_context.jsonl.gz'
            state = fit_global(frame, path)
            first = history_values(frame, path, state)
            changed = frame.copy()
            heldout = changed['split'] != 'train'
            changed.loc[heldout, 'total_events'] = 999999
            changed['label'] = 1 - changed['label']
            self.assertEqual(state, fit_global(changed, path))
            changed = frame.copy()
            last_train = changed.index[changed['split'].eq('train')][-1]
            changed.loc[last_train, 'total_events'] = 999999
            second = history_values(changed, path, state)
            pd.testing.assert_frame_equal(first.iloc[:last_train], second.iloc[:last_train])
            self.assertTrue(first.loc[heldout, 'cold_start'].eq(1).all())
            self.assertTrue(first.loc[heldout, 'prior30_events_log1p'].eq(0).all())
            # Nonconsecutive active days retain calendar gaps, with no fabricated padding.
            self.assertAlmostEqual(first.iloc[-1]['days_since_previous_active_log1p'], np.log1p(2))
            prepared = prepare_cert(source, config, root)
            saved = json.loads((prepared / 'feature_state.json').read_text())
            reapplied = transform(frame, path, saved)
            train = pd.read_csv(prepared / 'train.csv').set_index('event_id')
            mask = frame['split'].eq('train')
            np.testing.assert_allclose(train.loc[frame.loc[mask, 'item_id'], saved['columns']].to_numpy(),
                                       reapplied.loc[mask].to_numpy(), rtol=1e-6, atol=1e-6)
            frame.loc[heldout, FEATURES] = frame.loc[heldout, FEATURES] * 2
            frame.to_csv(source / 'account_days.csv', index=False)
            seal(source)
            again = prepare_cert(source, config, root)
            self.assertEqual(saved, json.loads((again / 'feature_state.json').read_text()))

    def test_all_models_same_endpoint_cohort_and_strict_background_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            preparation = prepare_cert(source, config, root)
            test = preparation / 'test.csv'
            hidden = preparation / 'hidden.csv'
            test.rename(hidden)
            try:
                runs = [candidate(preparation, root, model) for model in
                        ('random_forest', 'isolation_forest', 'lstm_autoencoder')]
            finally:
                hidden.rename(test)
            train_labels = frame.set_index('item_id')['label']
            arrays = np.load(runs[-1] / 'train_inputs.npz')
            self.assertTrue((train_labels.loc[arrays['member_ids'].ravel()] == 0).all())
            evaluations = [final_evaluate(run, preparation, root) for run in runs]
            self.assertEqual(len(compare(evaluations)), 3)
            for run, evaluation in zip(runs, evaluations):
                self.assertFalse((run / 'metrics_test.json').exists())
                self.assertEqual(json.loads((evaluation / 'lineage.json').read_text())['target'], 'endpoint_account_day_observable')
                verify(run)
            predictions = pd.read_csv(evaluations[0] / 'predictions_test.csv')
            self.assertEqual(len(predictions), 24)  # 2 users * (16 active days - 4 warmup).
            report = json.loads((preparation / 'cohort_report.json').read_text())
            self.assertEqual(report['test']['excluded_warmup_days'], 8)

    def test_reject_group_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            frame['group'] = 'shared'
            frame.to_csv(source / 'account_days.csv', index=False)
            seal(source)
            with self.assertRaisesRegex(ValueError, 'split leakage'):
                prepare_cert(source, config, root)

    def test_diagnostics_preserve_training_and_do_not_require_test(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, _ = self.fixture(root)
            values = json.loads(config.read_text())
            values['lstm_autoencoder'].update(epochs=3, early_stopping_patience=3)
            config.write_text(json.dumps(values))
            preparation = prepare_cert(source, config, root)
            (preparation / 'test.csv').unlink()
            plain = candidate(preparation, root, 'lstm_autoencoder')
            diagnostic = candidate(preparation, root, 'lstm_autoencoder', diagnostics=True)
            for key, value in torch.load(plain / 'model.pt', weights_only=True).items():
                self.assertTrue(torch.equal(value, torch.load(diagnostic / 'model.pt', weights_only=True)[key]))
            pd.testing.assert_frame_equal(pd.read_csv(plain / 'predictions_validation.csv'),
                                          pd.read_csv(diagnostic / 'predictions_validation.csv'))
            a = json.loads((plain / 'history.json').read_text())
            b = json.loads((diagnostic / 'history.json').read_text())
            self.assertEqual(a['selected_epoch'], b['selected_epoch'])
            self.assertEqual(a['stopping_reason'], b['stopping_reason'])
            for row in b['epochs']:
                self.assertTrue(np.isfinite(row['train_optimization_mse']))
                for value in row['reconstruction'].values():
                    self.assertTrue(value is None or np.isfinite(value))
            self.assertEqual(b['reconstruction_cohorts']['validation_positive_endpoints'], 4)
            verify(diagnostic)

    def test_global_only_has_equal_history_availability_and_train_only_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            cfg = json.loads(config.read_text())
            cfg['features']['history_policy'] = GLOBAL_POLICY
            config.write_text(json.dumps(cfg))
            prepared = prepare_cert(source, config, root)
            state = json.loads((prepared / 'feature_state.json').read_text())
            original = history_values(frame, source / 'day_context.jsonl.gz', state)
            # A fixed global reference cannot change with the account's split role.
            roles = frame.copy()
            roles['split'] = 'validation'
            pd.testing.assert_frame_equal(original, history_values(roles, source / 'day_context.jsonl.gz', state))
            self.assertTrue(original['cold_start'].eq(1).all())
            self.assertTrue(original['history_missing'].eq(1).all())
            self.assertTrue(original['prior30_events_log1p'].eq(0).all())
            for split in ('train', 'validation', 'test'):
                values = pd.read_csv(prepared / f'{split}.csv')
                self.assertTrue(values[['cold_start', 'history_missing', 'prior30_events_log1p']].eq(0).all().all())
            frame.loc[frame.split.ne('train'), FEATURES] *= 3
            frame['label'] = 1 - frame['label']
            frame.to_csv(source / 'account_days.csv', index=False)
            seal(source)
            again = prepare_cert(source, config, root)
            self.assertEqual(state, json.loads((again / 'feature_state.json').read_text()))
