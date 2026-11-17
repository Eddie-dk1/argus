"""Exercise the S6 worker and three-fold barrier on tiny invented inputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import run_cert_s6 as run
from cert_s6_worker import group_weights
from experiment_tracking import file_sha256, payload_sha256


class DispatchFixture(unittest.TestCase):
    def test_single_use_study_and_barrier(self):
        with TemporaryDirectory() as tmp:
            self.exercise(Path(tmp))

    def exercise(self, project):
        root = project/'results/experiments/fixture/sealed'
        root.mkdir(parents=True)
        configs = project/'configs'
        configs.mkdir()
        processed = project/'processed'
        daily = processed/'daily'
        daily.mkdir(parents=True)
        source = project/'old'
        source.mkdir()
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(30,75))
        y = np.array([0,1]*15)
        ids = np.array([f'test:a:{i}' for i in range(30)])
        groups = np.array(['g']*30)
        np.savez(source/'train_inputs.npz',X=x,y=y,item_ids=ids)
        for role in ('validation','assessment'):
            np.savez(source/(role+'_inputs.npz'),X=x,y=y,item_ids=ids)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingClassifier(max_iter=1,min_samples_leaf=2).fit(x,y)
            scores = model.predict_proba(x)[:,1]
        joblib.dump(model,source/'model.joblib')
        run.write(source/'config.json',dict(baseline=dict(threshold_percentiles=[90])))
        profiles,_ = run.select_operating_points(scores,y)
        run.write(source/'operating_points.json',profiles)
        np.savez(source/'predictions_assessment.npz',scores=scores,y=y,item_ids=ids,
                 **{name:scores>=spec['threshold'] for name,spec in profiles.items()})
        run.write(source/'complete.json',dict(files={name:file_sha256(source/name) for name in
                  ('operating_points.json','predictions_assessment.npz')}))
        binding = {k+'_sha256':payload_sha256(v.tolist()) for k,v in
                   [('weights',group_weights(y,groups)),('fit_ids',ids),('groups',groups),('labels',y)]}
        for key,name in [('frozen','complete.json'),('model','model.joblib'),('config','config.json')]:
            binding[key+'_sha256'] = file_sha256(source/name)
        run.write(daily/'account_split.json',dict(accounts={'a':dict(group='g')}))
        run.write(daily/'complete.json',dict(files={'account_split.json':file_sha256(daily/'account_split.json')}))
        protocol = json.loads((run.PROJECT/'configs/cert_s6_personal_context_v2.json').read_text())
        protocol['fit_population']['reference']['path'] = 'configs/bindings.json'
        protocol['sources']['daily'] = dict(id='daily',manifest_sha256=file_sha256(daily/'complete.json'))
        protocol['models']['base']['max_iter'] = 1
        run.write(configs/'cert_s6_personal_context_v2.json',protocol)
        jobs,controls = [],{}
        # One complete variant across three rotations per fold, plus weight-feature ablation.
        for fold in (1,2,3):
            for rotation in (1,2,3,0):
                key = f'H{fold}_H0_inner{rotation}' if rotation else f'final{fold}'
                cache = root/key
                cache.mkdir()
                roles,files = {},{}
                for role in ('train','validation','assessment'):
                    name = role+'_personal.npy'
                    np.save(cache/name,rng.uniform(size=(len(y),27)))
                    files[name] = file_sha256(cache/name)
                    roles[role] = dict(input_sha256=file_sha256(source/(role+'_inputs.npz')))
                controls[key] = dict(roles=roles,files=files)
                variants = ('Q0','Q3') if rotation else ('inner_winner',)
                jobs.extend(dict(id=key+':'+v,control=key,variant=v) for v in variants)
        run.write(configs/'bindings.json',dict(controls={k:binding for k in controls}))
        runtime = dict(controls=controls,jobs=jobs,budget=protocol['budget'] | {'minimum_free_disk_gib':0})
        ledger = dict(status='pending',jobs={j['id']:'pending' for j in jobs},research_attempts=0)
        run.write(root/'ledger.json',ledger)
        lock = configs/'lock.json'
        run.write(lock,dict(initial_ledger_sha256=file_sha256(root/'ledger.json')))
        outer_calls = []
        class InlineProcess:
            def __init__(self, command, **kwargs):
                self.returncode = 0
                phase = command[-1]
                job, = [j for j in jobs if j['id']==command[-3]]
                if phase == 'assess':
                    outer_calls.append(len(run.read(root/'outer_barrier.json')))
                run.worker(root,processed,job,phase)
            def __enter__(self):
                return self
            def __exit__(self,*args):
                pass
            def poll(self):
                return self.returncode
        with (patch.object(run,'PROJECT',project),patch.object(run,'LOCK',lock),
              patch.object(run,'context',return_value=runtime),
              patch.object(run,'locations',return_value=(source,source,source,file_sha256(source/'complete.json'))),
              patch.object(run.subprocess,'Popen',InlineProcess),threadpool_limits(limits=1)):
            run.execute(root,processed)
            with self.assertRaises(ValueError):
                run.execute(root,processed)
        ledger = run.read(root/'ledger.json')
        self.assertEqual(ledger['status'],'complete')
        self.assertEqual(outer_calls,[3,3,3])
        self.assertEqual(ledger['research_attempts'],9+sum(v=='Q3' for v in ledger['choices'].values()))
        self.assertEqual(set(ledger['final_calibration_eligible']),set(ledger['choices']))
