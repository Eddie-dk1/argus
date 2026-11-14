from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from datasets.cert_experiments import (CAUSAL_POLICY, GLOBAL_POLICY, fit_global,
                                      history_values, prepare_cert, prepare_cert_frame, transform)
from experiment import candidate, final_evaluate, score, verify
from experiment_tracking import file_sha256
from prepare_cert_development import PLAN_ID, audit_dependencies, run
import test_cert_experiments as fixtures


class CausalHistoryTests(unittest.TestCase):
    def fixture(self, root):
        source, config, frame = fixtures.CertExperimentTests().fixture(root)
        settings = json.loads(config.read_text())
        settings['features']['history_policy'] = CAUSAL_POLICY
        settings['config_version'] = 'fixture_causal_v3'
        config.write_text(json.dumps(settings))
        return source, config, frame

    def write_context(self, path, rows):
        with gzip.open(path, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    def test_history_is_role_invariant_and_does_not_mutate_global_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _, frame = self.fixture(Path(tmp))
            context = source / 'day_context.jsonl.gz'
            state = fit_global(frame, context, CAUSAL_POLICY)
            before = deepcopy(state)
            original = history_values(frame, context, state)
            for role in ('train', 'validation', 'assessment', 'test'):
                changed = frame.drop(columns=['label', 'group']).assign(split=role)
                pd.testing.assert_frame_equal(original, history_values(changed, context, state))
            self.assertEqual(state, before)
            first = frame.groupby('account_id').cumcount().eq(0)
            self.assertTrue(original.loc[first, 'cold_start'].eq(1).all())
            self.assertTrue(original.loc[~first, 'cold_start'].eq(0).all())
            self.assertTrue(original.loc[frame.groupby('account_id').cumcount().ge(5), 'history_missing'].eq(0).all())

    def test_future_observations_cannot_change_earlier_features_or_frozen_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            preparation = prepare_cert(source, config, root)
            state = json.loads((preparation / 'feature_state.json').read_text())
            original = transform(frame, source / 'day_context.jsonl.gz', state)
            with gzip.open(source / 'day_context.jsonl.gz', 'rt') as stream:
                contexts = [json.loads(line) for line in stream]
            changed = frame.copy()
            future_index = frame.index[frame.account_id.eq('validation0')][10]
            changed.loc[future_index, 'total_events'] = 99999
            contexts[future_index]['domains'] = ['future-only.invalid']
            future_context = root / 'changed_context.jsonl.gz'
            self.write_context(future_context, contexts)
            later = transform(changed, future_context, state)
            np.testing.assert_array_equal(original.iloc[:future_index], later.iloc[:future_index])
            self.assertFalse(np.array_equal(original.iloc[future_index], later.iloc[future_index]))
            # An actual frozen RF is sensitive to inputs, but a later observation
            # must not alter any of its earlier scores.
            trained = candidate(preparation, root, 'random_forest')
            model = joblib.load(trained / 'model.joblib')
            np.testing.assert_array_equal(score(model, original.iloc[:future_index].to_numpy(), 'random_forest'),
                                          score(model, later.iloc[:future_index].to_numpy(), 'random_forest'))

    def test_labels_and_attribution_do_not_change_features_or_fitted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            contract = json.loads((source / 'contract.json').read_text())
            settings = json.loads(config.read_text())
            first = prepare_cert_frame(frame, source / 'day_context.jsonl.gz', contract, settings, source, root)
            changed = frame.copy()
            changed['label'] = 1 - changed['label']
            changed['scenario'] = 'untrusted answer-key provenance'
            changed['incident_id'] = 'not a feature'
            changed['group'] = changed['group'] + ':renamed'
            second = prepare_cert_frame(changed, source / 'day_context.jsonl.gz', contract, settings, source, root)
            state = json.loads((first / 'feature_state.json').read_text())
            self.assertEqual(state, json.loads((second / 'feature_state.json').read_text()))
            for role in ('train', 'validation', 'test'):
                np.testing.assert_array_equal(pd.read_csv(first / (role + '.csv'))[state['columns']],
                                              pd.read_csv(second / (role + '.csv'))[state['columns']])
            with gzip.open(first / 'history_dependencies.jsonl.gz', 'rt') as stream:
                dependencies = {row['item_id']: row for row in map(json.loads, stream)}
            user = frame.loc[frame.account_id.eq('validation0')].reset_index(drop=True)
            positive = user.iloc[7]
            self.assertEqual(positive['label'], 1)
            self.assertIn(positive.item_id, dependencies[user.iloc[8].item_id]['rolling30_members'])

    def test_inner_fit_state_excludes_calibration_and_assessment_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, frame = self.fixture(root)
            frame['split'] = frame['split'].replace({'test': 'assessment'})
            settings = json.loads(config.read_text())
            contract = json.loads((source / 'contract.json').read_text())
            kwargs = dict(development={'source_population': 'fabricated CV roles'})
            first = prepare_cert_frame(frame, source / 'day_context.jsonl.gz', contract, settings, source, root, **kwargs)
            with gzip.open(source / 'day_context.jsonl.gz', 'rt') as stream:
                contexts = [json.loads(line) for line in stream]
            changed = frame.copy()
            heldout = changed.split.ne('train')
            changed.loc[heldout, 'total_events'] = 123456
            for include, context in zip(heldout, contexts, strict=True):
                if include:
                    context['domains'] = ['heldout-only.invalid']
            other_context = root / 'heldout_changed.jsonl.gz'
            self.write_context(other_context, contexts)
            second = prepare_cert_frame(changed, other_context, contract, settings, source, root, **kwargs)
            self.assertEqual(json.loads((first / 'feature_state.json').read_text()),
                             json.loads((second / 'feature_state.json').read_text()))
            pd.testing.assert_frame_equal(pd.read_csv(first / 'train.csv'), pd.read_csv(second / 'train.csv'))
            self.assertFalse((second / 'test.csv').exists())

    def test_calendar_boundaries_gaps_current_day_and_expanding_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, full = self.fixture(root)
            frame = full.loc[full.account_id.eq('train0')].iloc[:9].copy().reset_index(drop=True)
            dates = ['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04', '2010-01-05',
                     '2010-01-06', '2010-01-31', '2010-02-01', '2010-03-10']
            frame['date'] = dates
            frame['day_key'] = ['train0:' + date for date in dates]
            frame['item_id'] = 'mini:' + frame['day_key']
            frame['total_events'] = np.arange(1, 10)
            contexts = [dict(day_key=key, pcs=['P'], domains=['first.invalid' if index in (0, 8) else f'd{index}.invalid'],
                             recipients=[]) for index, key in enumerate(frame.day_key)]
            context = root / 'calendar.jsonl.gz'
            self.write_context(context, contexts)
            state = fit_global(frame, context, CAUSAL_POLICY)
            dependencies = root / 'dependencies.jsonl.gz'
            values = history_values(frame, context, state, dependencies)
            self.assertAlmostEqual(values.iloc[5].prior30_events_log1p, np.log1p(15))
            self.assertEqual(values.iloc[5].history_missing, 0)
            self.assertAlmostEqual(values.iloc[6].prior30_events_log1p, np.log1p(21))  # Jan 1 is included.
            self.assertEqual(values.iloc[6].prior7_events_log1p, 0)
            self.assertAlmostEqual(values.iloc[7].prior30_events_log1p, np.log1p(27))  # Jan 1 expired.
            self.assertAlmostEqual(values.iloc[8].days_since_previous_active_log1p, np.log1p(37))
            self.assertEqual(values.iloc[8].history_missing, 1)
            self.assertEqual(values.iloc[8].cold_start, 0)
            self.assertEqual(values.iloc[8].domain_novel_fraction, 0)  # Vocabulary outlives rolling profile.
            self.assertEqual(audit_dependencies(dependencies, frame)['checked_rows'], 9)
            with gzip.open(dependencies, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            rows[-1]['rolling30_members'] = [frame.iloc[-1].item_id]
            self.write_context(dependencies, rows)
            with self.assertRaisesRegex(ValueError, 'dependency mismatch'):
                audit_dependencies(dependencies, frame)

    def development_plan(self, root, source, causal_config):
        settings = json.loads(causal_config.read_text())
        settings['features']['history_policy'] = GLOBAL_POLICY
        settings['config_version'] = 'fixture_global_v2'
        global_config = root / 'global.json'
        global_config.write_text(json.dumps(settings))
        plan = dict(plan_id=PLAN_ID, daily_preparation_id=source.name,
                    daily_manifest_sha256=file_sha256(source / 'complete.json'), minimum_free_gib=0,
                    configs={key: dict(path=str(path), sha256=file_sha256(path))
                             for key, path in [('global_only_v2', global_config), ('causal_all_v3', causal_config)]})
        path = root / 'plan.json'
        path.write_text(json.dumps(plan))
        return path

    def test_development_preparation_audit_and_all_models_work_without_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, _ = self.fixture(root)
            plan = self.development_plan(root, source, config)
            parent_before = file_sha256(source / 'complete.json')
            with (patch('experiment.RandomForestClassifier.fit', side_effect=AssertionError('model fitting')),
                  patch('experiment.IsolationForest.fit', side_effect=AssertionError('model fitting')),
                  patch('torch.optim.Adam', side_effect=AssertionError('model fitting'))):
                output = run(source, root, plan)
            verify(output)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertFalse(summary['model_fitting'])
            self.assertFalse(summary['test_feature_building_or_scoring'])
            self.assertEqual(file_sha256(source / 'complete.json'), parent_before)
            self.assertEqual(summary['pair_audit']['feature_count'], 71)
            preparation = output / summary['preparations']['causal_all_v3']['id']
            self.assertFalse((preparation / 'test.csv').exists())
            self.assertEqual(json.loads((preparation / 'contract.json').read_text())['scope'], 'primary_development')
            for model in ('random_forest', 'isolation_forest', 'lstm_autoencoder'):
                trained = candidate(preparation, root, model)
                self.assertFalse(json.loads((trained / 'selection.json').read_text())['test_accessed'])
                with self.assertRaisesRegex(ValueError, 'no real test population'):
                    final_evaluate(trained, preparation, root)

    def test_preparation_rejects_changed_hash_or_additional_experiment_factor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, _ = self.fixture(root)
            path = self.development_plan(root, source, config)
            plan = json.loads(path.read_text())
            settings = json.loads(config.read_text())
            settings['baseline']['random_forest']['n_estimators'] += 1
            config.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run(source, root, path)
            plan['configs']['causal_all_v3']['sha256'] = file_sha256(config)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Only history policy'):
                run(source, root, path)
            self.assertEqual(list(root.glob('cert_history_development_*')), [])


if __name__ == '__main__':
    unittest.main()
