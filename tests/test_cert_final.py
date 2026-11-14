import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from datasets.cert_reporting import evidence_index, error_report
from experiment import candidate, compare, final_evaluate, seal, verify
from experiment_tracking import file_sha256, environment_metadata


class CertFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prep = cls.root/'preparation'
        cls.prep.mkdir()
        project = Path(__file__).resolve().parents[1]
        config = json.loads((project/'configs/cert_r4_2_global_only_v2.json').read_text())
        config['features']['sequence_length'] = 2
        config['baseline']['random_forest']['n_estimators'] = 3
        config['baseline']['isolation_forest']['n_estimators'] = 3
        config['lstm_autoencoder'].update(epochs=1,hidden_size=4)
        metadata = []
        evidence = []
        for split in ('train','validation','test'):
            rows = []
            for user in range(2):
                account=f'{split}{user}'
                for day in range(10):
                    date=f'2010-01-{day+1:02}'
                    label=int(day>=8)
                    key=account+':'+date
                    item='mini:'+key
                    rows.append(dict(event_id=item,user_id=account,timestamp=date,label=label,
                                     f1=float(day+label*5),f2=float(user+day%3)))
                    metadata.append(dict(item_id=item,day_key=key,account_id=account,group=account,split=split,label=label))
                    if label:
                        evidence.append(dict(day_key=key,event_id=item,source='logon',
                            attributions=[dict(incident_id=account,incident_user=account,scenario=str(user+1))]))
            pd.DataFrame(rows).to_csv(cls.prep/(split+'.csv'),index=False)
        contract=dict(snapshot_id='mini',preparation_id=cls.prep.name,sequence_length=2,
                      tabular_columns=['f1','f2'],sequence_columns=['f1','f2'],
                      target='endpoint_account_day_observable',history_policy=config['features']['history_policy'],
                      normal_window_policy='all_members_background',scope='primary_account_holdout')
        (cls.prep/'contract.json').write_text(json.dumps(contract))
        (cls.prep/'config.json').write_text(json.dumps(config))
        (cls.prep/'feature_state.json').write_text(json.dumps({'fixture':True}))
        seal(cls.prep)
        cls.candidates=[candidate(cls.prep,cls.root,name) for name in ('random_forest','isolation_forest','lstm_autoencoder')]
        cls.days=cls.root/'days'
        cls.days.mkdir()
        pd.DataFrame(metadata).to_csv(cls.days/'account_days.csv',index=False)
        (cls.days/'positive_evidence.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in evidence))
        seal(cls.days)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_final_never_fits_or_reselects_and_keeps_candidate_immutable(self):
        for run in self.candidates:
            before=file_sha256(run/'complete.json')
            with (patch('experiment.RandomForestClassifier.fit',side_effect=AssertionError('fit')),
                 patch('experiment.IsolationForest.fit',side_effect=AssertionError('fit')),
                 patch('torch.optim.Adam',side_effect=AssertionError('optimizer')),
                 patch('experiment.select_best_threshold',side_effect=AssertionError('threshold'))):
                evaluation=final_evaluate(run,self.prep,self.root)
            self.assertEqual(before,file_sha256(run/'complete.json'))
            lineage=json.loads((evaluation/'lineage.json').read_text())
            self.assertFalse(lineage['fit_or_threshold_selection'])
            self.assertEqual(lineage['threshold'],json.loads((run/'selection.json').read_text())['threshold'])
            timing=json.loads((evaluation/'timing.json').read_text())['inference']
            self.assertEqual(len(timing['durations_seconds']),5)
            with np.load(evaluation/'predictions_test.npz',allow_pickle=False) as arrays:
                expected=(arrays['scores']>=lineage['threshold']).astype(int)
                np.testing.assert_array_equal(expected,arrays['predictions'])
            verify(evaluation)

    def test_cross_preparation_comparison_is_explicit_and_still_checks_labels(self):
        first=final_evaluate(self.candidates[0],self.prep,self.root)
        second=self.root/'different_preparation_evaluation'
        shutil.copytree(first,second)
        lineage=json.loads((second/'lineage.json').read_text())
        lineage['preparation_id']='alternative'
        (second/'lineage.json').write_text(json.dumps(lineage))
        seal(second)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            compare([first,second])
        self.assertEqual(len(compare([first,second],allow_different_preparations=True)),2)
        predictions=pd.read_csv(second/'predictions_test.csv')
        predictions.loc[0,'y_true']=1-int(predictions.loc[0,'y_true'])
        predictions.to_csv(second/'predictions_test.csv',index=False)
        seal(second)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            compare([first,second],allow_different_preparations=True)

    def test_reporting_deduplicates_attributions_and_refuses_target_mismatch(self):
        path=self.root/'overlapping_evidence.jsonl'
        path.write_text(json.dumps(dict(day_key='A:day',attributions=[
            dict(scenario='1',incident_id='one'),dict(scenario='1',incident_id='one'),
            dict(scenario='3',incident_id='three')]))+'\n')
        evidence=evidence_index(path)
        predictions=pd.DataFrame(dict(item_id=['p','b'],y_true=[1,0],prediction=[1,1],score=[.8,.9]))
        metadata=pd.DataFrame(dict(item_id=['p','b'],day_key=['A:day','B:day'],account_id=['A','B'],group=['A','B'],label=[1,0]))
        report=error_report(predictions,metadata,evidence)
        self.assertEqual(report['scenarios']['1']['positive_days'],1)
        self.assertEqual(report['scenarios']['1']['incidents'],1)
        self.assertEqual(report['scenarios']['3']['incidents'],1)
        self.assertIsNone(report['scenarios']['2']['recall'])
        self.assertEqual(report['positive_days_with_multiple_scenarios'],1)
        self.assertEqual(report['background']['false_positive_days'],1)
        with self.assertRaisesRegex(ValueError,'evidence'):
            error_report(predictions,metadata,{})

    def test_full_runner_publishes_exposure_and_all_frozen_fixture_results(self):
        project=Path(__file__).resolve().parents[1]
        series=self.root/'fixture_frozen_series'
        series.mkdir()
        (series/'source').mkdir()
        for name in ('features.py','utils.py','train_lstm_autoencoder.py','experiment.py'):
            shutil.copy2(project/'src'/name,series/'source'/name)
        (series/'environment.json').write_text(json.dumps(environment_metadata()))
        (series/'summary.json').write_text(json.dumps({'fixture':True}))
        (series/'plan.json').write_text(json.dumps({'fixture':True}))
        registry=[]
        for run in self.candidates:
            selection=json.loads((run/'selection.json').read_text())
            name=selection['model']
            registry.append(dict(policy='fixture',model=name,seed=42,candidate=str(run),
                candidate_manifest_sha256=file_sha256(run/'complete.json'),preparation=str(self.prep),
                preparation_manifest_sha256=file_sha256(self.prep/'complete.json'),
                checkpoint_sha256=file_sha256(run/('model.pt' if name=='lstm_autoencoder' else 'model.joblib')),
                config_sha256=file_sha256(run/'config.json'),feature_state_sha256=file_sha256(run/'feature_state.json'),
                threshold=selection['threshold'],selected_epoch=1 if name=='lstm_autoencoder' else None))
        (series/'frozen_candidates.json').write_text(json.dumps(dict(test_evaluated=False,candidates=registry)))
        seal(series)
        plan=dict(plan_id='fixture_final',series_id=series.name,series_manifest_sha256=file_sha256(series/'complete.json'),
            frozen_registry_sha256=file_sha256(series/'frozen_candidates.json'),daily_preparation_id=self.days.name,
            daily_manifest_sha256=file_sha256(self.days/'complete.json'),policies=['fixture'],
            models=['random_forest','isolation_forest','lstm_autoencoder'],seeds=[42],expected_candidates=3,
            snapshot_id='mini',target='endpoint_account_day_observable',expected_test_endpoints=18,
            expected_test_positive_days=4,minimum_free_gib=0)
        path=self.root/'final_plan.json'
        path.write_text(json.dumps(plan))
        output=self.root/'finals'
        output.mkdir()
        command=[sys.executable,str(project/'src/run_cert_final.py'),'--root',str(output),
                 '--series',str(series),'--days',str(self.days),'--plan',str(path)]
        completed=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        result=next(output.glob('cert_final_*'))
        verify(result)
        summary=json.loads((result/'summary.json').read_text())
        self.assertEqual(summary['evaluations'],3)
        self.assertTrue(summary['test_evaluated'])
        self.assertTrue(summary['no_fitting_or_recalibration'])
        self.assertIsNotNone(summary['test_exposure']['first_prepared_test_access_utc'])
