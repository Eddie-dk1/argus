"""S6 v2 single-use dispatcher and read-only progress command."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version

if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'

import joblib
import numpy as np

from cert_operating_point import PRIMARY_PROFILE, select_operating_points
from cert_s6_worker import fit_args, matrix, select_variant, train_variant
from experiment_tracking import file_sha256
from utils import compute_classification_metrics, select_best_threshold

PROJECT = Path(__file__).resolve().parents[1]
LOCK = PROJECT / 'configs/cert_s6_runtime_v2.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def check(path, digest):
    if file_sha256(path) != digest:
        raise ValueError(f'Changed frozen input: {path.name}')
    return path


def context(root):
    lock = read(LOCK)
    if root.resolve() != (PROJECT/'results/experiments'/lock['study_id']/lock['directory']).resolve():
        raise ValueError('Wrong S6 study')
    runtime = read(check(root/'runtime.json', lock['runtime_sha256']))
    for name, digest in (runtime['source_hashes'] | lock['dispatch_sources']).items():
        check(PROJECT/name, digest)
    if runtime['python'] != platform.python_version() or any(
            version(p) != v for p,v in runtime['packages'].items()):
        raise ValueError('Python or package version differs from frozen runtime')
    return runtime


def locations(root, processed, key, protocol):
    study = PROJECT / 'results/experiments' / protocol['fit_population']['s4_study']
    if '_inner' in key:
        directory = study/'cycles/A'
        manifest = read(check(directory/'complete.json', protocol['fit_population']['cycle_a_manifest_sha256']))
        results = read(check(directory/'summary.json',manifest['files']['summary.json']))['results']
        row, = [r for r in results if r['job']['job_id'] == key]
        return Path(row['candidate']), Path(row['frozen']), Path(row['assessment']), row['assessment_manifest_sha256']
    spec = protocol['sources']['control_bindings']
    runs = read(check(PROJECT/spec['path'],spec['sha256']))['runs']
    row, = [r for r in runs if r['job_id'] == key and r['series']=='operating_points']
    return (*(processed/row[k]['path'] for k in ('trained','operating_bundle','assessment')),
            row['assessment']['manifest_sha256'])


def worker(root, processed, job, phase):
    runtime = context(root)
    ledger = read(root/'ledger.json')
    if ledger['jobs'][job['id']] != 'running' or ledger['current'] != [job['id'],phase]:
        raise ValueError('Slot not reserved')
    protocol = read(PROJECT/'configs/cert_s6_personal_context_v2.json')
    key = job['control']
    binding = read(PROJECT/protocol['fit_population']['reference']['path'])['controls'][key]
    variant = ledger['choices'][key] if job['variant']=='inner_winner' else job['variant']
    cache = root/key
    record = runtime['controls'][key]
    directory = root/'jobs'/job['id'].replace(':','_')
    candidate, frozen, assessment, assessment_sha = locations(root,processed,key,protocol)
    if phase != 'assess':
        directory.mkdir()
        if variant == 'Q0':
            manifest = read(check(frozen/'complete.json',binding['frozen_sha256']))
            model_path = check(frozen/'model.joblib',binding['model_sha256'])
            profiles = read(check(frozen/'operating_points.json',manifest['files']['operating_points.json']))
        else:
            spec = protocol['sources']['daily']
            daily = processed/spec['id']
            manifest = read(check(daily/'complete.json',spec['manifest_sha256']))
            accounts = read(check(daily/'account_split.json',manifest['files']['account_split.json']))['accounts']
            with inputs(candidate,record,'train') as z:
                groups = np.array([accounts[k.split(':')[1]]['group'] for k in z['item_ids']])
                args = fit_args(variant,z,groups,extra(cache,record,'train',variant),binding)
                model = train_variant(variant,args,protocol)
            model_path = directory/'model.joblib'
            joblib.dump(model, model_path)
            with inputs(candidate,record,'validation') as z:
                x = matrix(variant,z['X'],extra(cache,record,'validation',variant))
                scores = model.predict_proba(x)[:,1]
                y = z['y']
                profiles,_ = select_operating_points(scores,y)
                config = read(check(candidate/'config.json',binding['config_sha256']))
                threshold,_,metrics,_ = select_best_threshold(scores,y,config['baseline']['threshold_percentiles'])
                profiles['legacy_f1'] = dict(threshold=threshold,calibration_metrics=metrics)
                np.savez_compressed(directory/'calibration.npz',item_ids=z['item_ids'],y=y,scores=scores)
        write(directory/'frozen.json',dict(variant=variant,model_path=str(model_path),
              model_sha256=file_sha256(model_path),profiles=profiles))
    if phase != 'freeze':
        if phase == 'assess':
            barrier = read(check(root/'outer_barrier.json',ledger['outer_barrier_sha256']))
            check(directory/'frozen.json',barrier[job['id']])
        saved = read(directory/'frozen.json')
        model_path = check(Path(saved['model_path']),saved['model_sha256'])
        with inputs(assessment,record,'assessment') as z:
            y = z['y']
            if variant == 'Q0':
                manifest = read(check(assessment/'complete.json',assessment_sha))
                name = 'predictions_assessment.npz'
                with np.load(check(assessment/name,manifest['files'][name]),allow_pickle=False) as prior:
                    scores = prior['scores']
                    if not np.array_equal(prior['item_ids'],z['item_ids']) or not np.array_equal(prior['y'],y):
                        raise ValueError('Historical prediction rows differ')
                    for name,spec in saved['profiles'].items():
                        if not np.array_equal(scores >= spec['threshold'],prior[name]):
                            raise ValueError('Historical decisions differ')
                seconds = None  # Exact reuse has no new inference timing.
            else:
                model = joblib.load(model_path)
                x = matrix(variant,z['X'],extra(cache,record,'assessment',variant))
                start = time.monotonic()
                scores = model.predict_proba(x)[:,1]
                seconds = time.monotonic()-start
            np.savez_compressed(directory/'decisions.npz', item_ids=z['item_ids'],y=y,scores=scores)
            profiles = {}
            for name,spec in saved['profiles'].items():
                metrics = compute_classification_metrics(y,scores>=spec['threshold'],scores)
                tn,fp = metrics['confusion_matrix'][0]
                metrics['fp_per_1000_background'] = 1000*fp/(tn+fp)
                if not np.any(scores >= spec['threshold']):
                    metrics['precision'] = None
                profiles[name] = metrics
            write(directory/'result.json',dict(status='complete',variant=variant,
                  eligible_on_calibration=saved['profiles'][PRIMARY_PROFILE]['eligible_on_calibration'],
                  metrics=profiles[PRIMARY_PROFILE],profiles=profiles,inference_seconds=seconds))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    write(directory/(phase+'_cost.json'),dict(peak_rss_bytes=rss))
    if rss > 8 * 1024**3:
        raise MemoryError('Worker exceeded peak RSS limit')


def inputs(directory, record, role):
    return np.load(check(directory/(role+'_inputs.npz'),record['roles'][role]['input_sha256']),allow_pickle=False)


def extra(cache, record, role, variant):
    if variant in ('Q1','Q3'):
        name = role+'_personal.npy'
        return np.load(check(cache/name,record['files'][name]),allow_pickle=False,mmap_mode='r')
    return None


def execute(root, processed):
    runtime = context(root)
    lock = read(LOCK)
    ledger_path = check(root/'ledger.json',lock['initial_ledger_sha256'])
    ledger = read(ledger_path)
    jobs = runtime['jobs']
    budget = runtime['budget']
    start = time.monotonic()
    def resources():
        if time.monotonic()-start >= budget['wall_time_hours']*3600:
            raise TimeoutError('Wall time limit')
        if shutil.disk_usage(root).free < budget['minimum_free_disk_gib']*1024**3:
            raise OSError('Disk limit')
    resources()
    # Validate every dependency before spending the first research slot.
    for key, record in runtime['controls'].items():
        for name, digest in record['files'].items():
            check(root/key/name,digest)
    resources()
    (root/'jobs').mkdir()  # Exclusive claim. An interrupted study cannot restart.
    ledger.update(status='running',choices={},current=None)
    def dispatch(job, phase):
        resources()
        key = job['id']
        variant = ledger['choices'].get(job['control'],job['variant'])
        if phase != 'assess' and variant != 'Q0':
            ledger['research_attempts'] += 1
            if ledger['research_attempts'] > budget['max_new_research_fit_attempts']:
                raise ValueError('Fit budget exceeded')
        ledger['jobs'][key] = 'running'
        ledger['current'] = [key,phase]
        write(ledger_path,ledger)
        directory = root/'jobs'/key.replace(':','_')
        before = time.monotonic()
        with (root/'jobs'/(key.replace(':','_')+'_'+phase+'.log')).open('x') as log:
            command = [sys.executable,__file__,'_worker',str(root),'--processed',str(processed),
                       '--slot',key,'--phase',phase]
            with subprocess.Popen(command,stdout=log,stderr=log) as child:
                try:
                    while child.poll() is None:
                        resources()
                        usage = subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
                        if usage.stdout.strip() and int(usage.stdout)*1024 > budget['peak_worker_rss_stop_gib']*1024**3:
                            raise MemoryError('Worker RSS limit')
                        time.sleep(.5)
                finally:
                    if child.poll() is None:
                        child.kill()
        cost = directory/(phase+'_cost.json')
        if cost.exists() and read(cost)['peak_rss_bytes'] > budget['peak_worker_rss_stop_gib']*1024**3:
            raise MemoryError('Worker peak RSS limit')
        ledger.setdefault('seconds',{})[key+'_'+phase] = time.monotonic()-before
        ledger['jobs'][key] = ('failed' if child.returncode else 'frozen' if phase=='freeze'
                               else 'reused' if variant=='Q0' else 'complete')
        ledger['current'] = None
        write(ledger_path,ledger)
    try:
        inner = [j for j in jobs if j['variant']!='inner_winner']
        finals = [j for j in jobs if j['variant']=='inner_winner']
        for j in inner:
            dispatch(j,'inner')
        for fold,j in enumerate(sorted(finals,key=lambda j:j['control']),1):
            reports = {}
            for slot in inner:
                if not slot['control'].startswith(f'H{fold}_') or ledger['jobs'][slot['id']] not in ('complete','reused'):
                    continue
                rotation = int(slot['control'].split('inner')[1])
                path = root/'jobs'/slot['id'].replace(':','_')/'result.json'
                report = read(path)
                if report['metrics']['precision'] is not None:
                    reports.setdefault(slot['variant'],{})[rotation] = report
            ledger['choices'][j['control']] = select_variant(reports)
        write(ledger_path,ledger)
        if None in ledger['choices'].values():
            raise ValueError('A fold has no rankable variant')
        for j in finals:
            dispatch(j,'freeze')
        if not all(ledger['jobs'][j['id']]=='frozen' for j in finals):
            raise ValueError('Incomplete final freeze')
        barrier = {j['id']:file_sha256(root/'jobs'/j['id'].replace(':','_')/'frozen.json') for j in finals}
        write(root/'outer_barrier.json',barrier)
        ledger['outer_barrier_sha256'] = file_sha256(root/'outer_barrier.json')
        for j in finals:
            dispatch(j,'assess')
        eligibility = {}
        for job in finals:
            path = root/'jobs'/job['id'].replace(':','_')/'frozen.json'
            eligibility[job['control']] = read(path)['profiles'][PRIMARY_PROFILE]['eligible_on_calibration']
        ledger['final_calibration_eligible'] = eligibility
        ledger['any_eligible_operating_point'] = any(eligibility.values())
        ledger['status'] = 'incomplete' if 'failed' in ledger['jobs'].values() else 'complete'
    except BaseException as exc:
        ledger.update(status='incomplete',error=str(exc))
        if ledger['current']:
            ledger['jobs'][ledger['current'][0]] = 'interrupted'
        raise
    finally:
        ledger['current'] = None
        write(ledger_path,ledger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','execute','_worker'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--processed',type=Path)
    parser.add_argument('--slot')
    parser.add_argument('--phase',choices=['inner','freeze','assess'])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'status':
        ledger = read(root/'ledger.json')
        print(json.dumps(dict(status=ledger['status'],current=ledger.get('current'),
            research_attempts=ledger['research_attempts'],jobs=dict(Counter(ledger['jobs'].values())))))
    elif not args.processed:
        parser.error('--processed required for execution')
    elif args.action == 'execute':
        execute(root,args.processed.resolve())
    else:
        job, = [j for j in context(root)['jobs'] if j['id']==args.slot]
        worker(root,args.processed.resolve(),job,args.phase)
