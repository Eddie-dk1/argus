from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from datasets.cert_experiments import CAUSAL_POLICY, fit_global, prepare_cert_frame, transform
from datasets.cert_file_usb import COLUMNS, DEFINITION, FROZEN_POLICY, RAW_COLUMNS, ROLLING_POLICY, values
from datasets.cert_file_usb_audit import audit_preparations
from datasets.cert_validation import development_folds
from experiment import verify
from experiment_tracking import file_sha256
from run_cert_file_usb import PLAN_ID, read_plan, run
from run_cert_group_cv import run as run_previous
import test_cert_group_cv as fixtures


class CertFileUSBTests(unittest.TestCase):
    def fixture(self, root):
        source, frame, context, contract, config = fixtures.CertGroupCVTests().fixture(root)
        # Fabricated measurements only; no publisher records in tests.
        index = pd.to_datetime(frame.date).dt.day.to_numpy()
        frame['file_events'] = index % 4 + frame.label * 25
        frame['connect_count'] = index % 3 + frame.label * 4
        frame['device_paired_seconds'] = frame.connect_count * 900
        frame['total_events'] += frame.file_events + frame.connect_count * 2
        frame.to_csv(source / 'account_days.csv', index=False)
        from experiment import seal
        seal(source)
        return source, frame, context, contract, config

    def trio(self, root):
        source, frame, context, contract, config = self.fixture(root)
        folded, provenance = next(development_folds(frame, n_splits=3, calibration_splits=3))
        paths = {}
        for key, policy in [('causal_all_v3', CAUSAL_POLICY), ('file_usb_rolling_v4', ROLLING_POLICY),
                            ('file_usb_frozen_v4', FROZEN_POLICY)]:
            setting = deepcopy(config)
            setting['features']['history_policy'] = policy
            paths[key] = prepare_cert_frame(folded, context, contract, setting, source, root, development=provenance)
        return source, folded, context, paths

    def test_equal_roles_label_independence_and_future_cannot_change_earlier_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, frame, _, _, _ = self.fixture(Path(tmp))
            context = source / 'day_context.jsonl.gz'
            for policy in (ROLLING_POLICY, FROZEN_POLICY):
                state = fit_global(frame, context, policy)
                before = deepcopy(state)
                expected = values(frame, state)
                for role in ('train', 'validation', 'assessment', 'test'):
                    changed = frame.drop(columns=['label', 'group']).assign(split=role)
                    pd.testing.assert_frame_equal(values(changed, state), expected)
                future = frame.index[frame.account_id.eq('A18')][-1]
                changed = frame.copy()
                changed.loc[future, list(RAW_COLUMNS)] = 1e9
                later = values(changed, state)
                np.testing.assert_array_equal(expected.iloc[:future], later.iloc[:future])
                self.assertFalse(np.array_equal(expected.iloc[future], later.iloc[future]))
                self.assertEqual(state, before)

    def test_sustained_shift_adapts_only_rolling_and_frozen_includes_positive_warmup(self):
        dates = pd.date_range('2010-01-01', periods=60).strftime('%Y-%m-%d')
        frame = pd.DataFrame(dict(account_id='account', item_id=['i' + str(i) for i in range(60)],
                                  date=dates, split='train', label=[1] * 60))
        for name in RAW_COLUMNS:
            frame[name] = [0] * 10 + [100] * 50
        from datasets.cert_file_usb import fit_global as fit_usb
        outputs = {}
        for policy in (ROLLING_POLICY, FROZEN_POLICY):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'deps.gz'
                state = dict(policy=policy, file_usb_profile=fit_usb(frame, policy))
                result = values(frame, state, path)
                outputs[policy] = result
                with gzip.open(path, 'rt') as stream:
                    dependencies = [json.loads(line) for line in stream]
                self.assertEqual(dependencies[5]['profile_members'], frame.item_id[:5].tolist())
                self.assertNotIn('i5', dependencies[5]['profile_members'])
                self.assertEqual(result.iloc[4].file_usb_profile_missing, 1)
                self.assertEqual(result.iloc[5].file_usb_profile_missing, 0)
                pd.testing.assert_frame_equal(result, values(frame.drop(columns='label'), state))
                gap = frame.copy()
                gap.loc[59, 'date'] = '2010-05-01'
                missing = values(gap, state).iloc[-1].file_usb_profile_missing
                self.assertEqual(missing, int(policy == ROLLING_POLICY))
        self.assertGreater(outputs[ROLLING_POLICY].iloc[10, 0], 4)
        self.assertEqual(outputs[ROLLING_POLICY].iloc[-1, 0], 0)
        self.assertGreater(outputs[FROZEN_POLICY].iloc[-1, 0], 4)
        # The fixed floor prevents zero-IQR baselines from saturating ordinary changes.
        self.assertAlmostEqual(outputs[FROZEN_POLICY].iloc[-1, 0], np.log1p(100))

    def test_preparation_numerical_audit_preserves_all_common_inputs_and_catches_bad_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folded, _, paths = self.trio(Path(tmp))
            audit = audit_preparations(paths, folded)
            self.assertEqual(len(audit), 2)
            for report in audit.values():
                self.assertEqual(report['feature_count'], 75)
                self.assertTrue(report['causal_memberships_and_numerical_deviations_verified'])
            path = paths['file_usb_frozen_v4'] / 'file_usb_dependencies.jsonl.gz'
            with gzip.open(path, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            rows[-1]['profile_members'] = [rows[-1]['item_id']]
            with gzip.open(path, 'wt') as stream:
                stream.writelines(json.dumps(row) + '\n' for row in rows)
            with self.assertRaisesRegex(ValueError, 'Noncausal or incorrect'):
                audit_preparations(paths, folded)

    def test_calendar_boundary_and_clip_are_fixed_without_current_day_membership(self):
        from datasets.cert_file_usb import fit_global as fit_usb
        frame = pd.DataFrame(dict(account_id='a', item_id=list('abcdefg'), split='train',
            date=['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04', '2010-01-05',
                  '2010-01-31', '2010-02-01']))
        for name in RAW_COLUMNS:
            frame[name] = [0] * 5 + [1, 1e100]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'deps.gz'
            state = dict(policy=ROLLING_POLICY, file_usb_profile=fit_usb(frame, ROLLING_POLICY))
            result = values(frame, state, path)
            with gzip.open(path, 'rt') as stream:
                deps = [json.loads(line) for line in stream]
            self.assertEqual(deps[5]['profile_members'], list('abcde'))
            self.assertEqual(deps[6]['profile_members'], list('bcdef'))
            self.assertEqual(result.iloc[5, 0], np.log1p(1))
            self.assertEqual(result.iloc[6, 0], 20)

    def test_heldout_does_not_fit_new_fallback_scaler_or_train_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, folded, context, paths = self.trio(root)
            contract = json.loads((source / 'contract.json').read_text())
            changed = folded.copy()
            changed.loc[changed.split.ne('train'), list(RAW_COLUMNS)] = 1e9
            changed['label'] = 1 - changed.label
            for key in ('file_usb_rolling_v4', 'file_usb_frozen_v4'):
                setting = json.loads((paths[key] / 'config.json').read_text())
                other = prepare_cert_frame(changed, context, contract, setting, source, root,
                                           development=json.loads((paths[key] / 'development.json').read_text()))
                first = json.loads((paths[key] / 'feature_state.json').read_text())
                self.assertEqual(first, json.loads((other / 'feature_state.json').read_text()))
                expected = pd.read_csv(paths[key] / 'train.csv').drop(columns='label')
                pd.testing.assert_frame_equal(expected, pd.read_csv(other / 'train.csv').drop(columns='label'))
                # Saved state replay reproduces the entire prepared matrix.
                replay = transform(folded, context, first)
                part = pd.read_csv(paths[key] / 'assessment.csv')
                lookup = pd.DataFrame(replay.to_numpy(), index=folded.item_id, columns=first['columns'])
                np.testing.assert_array_equal(part[first['columns']].to_numpy(np.float32),
                                              lookup.loc[part.event_id].to_numpy())

    def plan(self, root, source, config):
        old_path = fixtures.CertGroupCVTests().plan(root, source, config)
        previous = run_previous(source, root, old_path)
        old = json.loads(old_path.read_text())
        configs = {'causal_all_v3': old['configs']['causal_all_v3']}
        for key, policy in [('file_usb_rolling_v4', ROLLING_POLICY), ('file_usb_frozen_v4', FROZEN_POLICY)]:
            setting = deepcopy(config)
            setting['config_version'] = key
            setting['features']['history_policy'] = policy
            path = root / (key + '.json')
            path.write_text(json.dumps(setting))
            configs[key] = dict(path=str(path), sha256=file_sha256(path))
        plan = dict(plan_id=PLAN_ID, daily_preparation_id=source.name,
                    daily_manifest_sha256=file_sha256(source / 'complete.json'), previous_cv_id=previous.name,
                    previous_cv_manifest_sha256=file_sha256(previous / 'complete.json'), configs=configs,
                    models=old['models'], cv=old['cv'], feature_definition=DEFINITION, objective=old['objective'],
                    budget=dict(candidates=18, new_fits=12, reused_candidates=6, max_lstm_epochs=1,
                                cpu_threads=1, search_trials=0, minimum_free_gib=0))
        path = root / 'usb_plan.json'
        path.write_text(json.dumps(plan))
        return path, previous

    def test_bounded_runner_reuses_old_models_and_preserves_oof_population_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, _, config = self.fixture(root)
            plan_path, previous = self.plan(root, source, config)
            old_digest = file_sha256(previous / 'complete.json')
            output = run(source, root, plan_path)
            verify(output)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['completed_candidates'], 18)
            self.assertEqual(summary['new_fits'], 12)
            self.assertEqual(sum(row['reused'] for row in summary['results']), 6)
            self.assertEqual(file_sha256(previous / 'complete.json'), old_digest)
            for row in summary['results']:
                self.assertTrue(row['checks']['assessment_open_guard'])
                self.assertEqual(row['checks']['test_open_attempts'], [])
            for pooled in summary['pooled_oof_decisions']:
                self.assertEqual(pooled['metrics']['support']['total'], 144)
                self.assertIsNone(pooled['metrics']['pr_auc'])
            report = json.loads((output / 'reports/adaptation.json').read_text())
            self.assertTrue(report['deviations'])
            for model in ('random_forest', 'isolation_forest', 'lstm_autoencoder'):
                for policy in ('causal_all_v3', 'file_usb_rolling_v4', 'file_usb_frozen_v4'):
                    self.assertEqual(sum(row['days'] for row in report['predictions']
                                         if row['model'] == model and row['policy'] == policy), 144)
            plan = json.loads(plan_path.read_text())
            broken = deepcopy(plan)
            broken['feature_definition']['log1p_iqr_floor'] = 0.000001
            plan_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'Unsupported bounded'):
                read_plan(plan_path, source, root)
            location = Path(plan['configs']['file_usb_frozen_v4']['path'])
            setting = json.loads(location.read_text())
            setting['lstm_autoencoder']['hidden_size'] += 1
            location.write_text(json.dumps(setting))
            plan['configs']['file_usb_frozen_v4']['sha256'] = file_sha256(location)
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Only the declared'):
                read_plan(plan_path, source, root)


if __name__ == '__main__':
    unittest.main()
