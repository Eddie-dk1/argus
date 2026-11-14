import gzip
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import pandas as pd

from datasets.cert_days import Day, FEATURES, FEATURE_SET
from datasets.cert_experiments import GLOBAL_POLICY, prepare_cert_frame
from datasets.cert_validation import development_folds, write_train_context, assess_development
from experiment import seal, candidate, verify
from experiment_tracking import file_sha256


class CertValidationTests(unittest.TestCase):
    def fixture(self, root):
        rows, contexts = [], []
        for day_index in range(12):
            date = str((pd.Timestamp('2010-01-01') + pd.Timedelta(days=day_index)).date())
            for index in range(22):
                account = f'A{index:02}'
                split = 'train' if index < 18 else 'validation' if index < 20 else 'test'
                label = int(day_index == 7)
                day = Day()
                for event in range(1 + day_index % 3 + label * 4):
                    day.add('logon', (str(event), '', account, 'PC', 'Logon'), 28000 + event, label)
                key = account + ':' + date
                rows.append(dict(item_id='snapshot:' + key, day_key=key, account_id=account, date=date,
                                 split=split, group='paired' if index < 2 else account,
                                 label=label, **day.features(date)))
                contexts.append(dict(day_key=key, pcs=['PC'], domains=[account + '.example'], recipients=[]))
        frame = pd.DataFrame(rows)
        source = root / 'days'
        source.mkdir()
        with gzip.open(source / 'day_context.jsonl.gz', 'wt') as stream:
            stream.writelines(json.dumps(value) + '\n' for value in contexts)
        frame.to_csv(source / 'account_days.csv', index=False)
        contract = dict(snapshot_id='snapshot', feature_set=FEATURE_SET, feature_columns=FEATURES)
        (source / 'contract.json').write_text(json.dumps(contract))
        seal(source)
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/cert_r4_2_global_only_v2.json').read_text())
        config['lstm_autoencoder'].update(epochs=1, hidden_size=4)
        context = root / 'train_context.jsonl.gz'
        write_train_context(frame, source / 'day_context.jsonl.gz', context)
        return source, frame, context, contract, config

    def test_nested_group_folds_exclude_primary_holdouts_and_preserve_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, frame, _, _, _ = self.fixture(Path(tmp))
            folds = list(development_folds(frame, n_splits=3, calibration_splits=3))
            again = list(development_folds(frame.iloc[::-1].sort_index(), n_splits=3, calibration_splits=3))
            assessment_accounts = []
            allowed = set(frame.loc[frame.split.eq('train'), 'account_id'])
            for (part, provenance), (_, repeated) in zip(folds, again):
                self.assertEqual(provenance, repeated)
                self.assertEqual(set(part.account_id), allowed)
                self.assertEqual(part.groupby('group')['split'].nunique().max(), 1)
                self.assertEqual(part.groupby('account_id')['split'].nunique().max(), 1)
                assessment_accounts.extend(set(part.loc[part.split.eq('assessment'), 'account_id']))
            self.assertEqual(len(assessment_accounts), len(set(assessment_accounts)))
            self.assertEqual(set(assessment_accounts), allowed)

    def test_fold_fit_excludes_calibration_assessment_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frame, context, contract, config = self.fixture(root)
            part, provenance = next(development_folds(frame, n_splits=3, calibration_splits=3))
            prep = prepare_cert_frame(part, context, contract, config, source, root, development=provenance)
            state = json.loads((prep / 'feature_state.json').read_text())
            heldout = part.split.ne('train')
            modified = part.copy()
            modified.loc[heldout, FEATURES] *= 5
            modified.loc[heldout, 'label'] = 1 - modified.loc[heldout, 'label']
            modified_context = root / 'changed_context.jsonl.gz'
            with gzip.open(context, 'rt') as inp, gzip.open(modified_context, 'wt') as out:
                for keep, line in zip(heldout, inp, strict=True):
                    value = json.loads(line)
                    if keep:
                        value['domains'] = ['never-in-fit.example']
                    out.write(json.dumps(value) + '\n')
            changed = prepare_cert_frame(modified, modified_context, contract, config, source, root,
                                         development=provenance)
            self.assertEqual(state, json.loads((changed / 'feature_state.json').read_text()))
            pd.testing.assert_frame_equal(pd.read_csv(prep / 'train.csv'), pd.read_csv(changed / 'train.csv'))
            assessment = prep / 'assessment.csv'
            hidden = prep / 'hidden.csv'
            assessment.rename(hidden)
            try:
                run = candidate(prep, root, 'lstm_autoencoder')
            finally:
                hidden.rename(assessment)
            self.assertFalse((prep / 'test.csv').exists())
            before = file_sha256(run / 'complete.json')
            evaluated = assess_development(run, prep, root)
            self.assertEqual(before, file_sha256(run / 'complete.json'))
            lineage = json.loads((evaluated / 'lineage.json').read_text())
            self.assertFalse(lineage['refit_or_recalibration'])
            self.assertEqual(lineage['threshold'], json.loads((run / 'selection.json').read_text())['threshold'])
            verify(evaluated)
            # Exercise JSON-serialized paths and the isolated process guard too.
            config_path = root / 'worker_config.json'
            config_path.write_text(json.dumps(config))
            result_path = root / 'worker_result.json'
            job_path = root / 'worker_job.json'
            job_path.write_text(json.dumps(dict(job_id='fixture_cv', phase='cv', fold=1,
                policy='global_only_v2', model='lstm_autoencoder', seed=config['seed'],
                preparation=str(prep), config=str(config_path), output_root=str(root),
                result=str(result_path), reuse=str(run))))
            script = Path(__file__).resolve().parents[1] / 'src/run_cert_robustness.py'
            completed = subprocess.run([sys.executable, str(script), '--root', str(root),
                                        '--job', str(job_path)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            result = json.loads(result_path.read_text())
            self.assertEqual(result['checks']['test_open_attempts'], [])
            self.assertIn('assessment_metrics', result)
