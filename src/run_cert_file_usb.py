"""Bounded file/USB ablation; preserve and reuse the nine step-2 v3 candidates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from datasets.cert_experiments import CAUSAL_POLICY, prepare_cert_frame
from datasets.cert_file_usb import ROLLING_POLICY, FROZEN_POLICY, DEFINITION
from datasets.cert_file_usb_audit import adaptation_report, audit_preparations
from datasets.cert_validation import development_folds, write_train_context
from experiment import allocate, finish, verify
from experiment_tracking import environment_metadata, file_sha256
from prepare_cert_development import audit_dependencies
from run_cert_group_cv import MODELS, PROJECT, check_source, summarize
from utils import load_json, save_json

PLAN_ID = 'cert_r4_2_file_usb_ablation_v4'
POLICIES = {'causal_all_v3': CAUSAL_POLICY, 'file_usb_rolling_v4': ROLLING_POLICY,
            'file_usb_frozen_v4': FROZEN_POLICY}


def read_plan(path, days, root):
    plan = load_json(path)
    budget, cv = plan['budget'], plan['cv']
    if (plan['plan_id'] != PLAN_ID or set(plan['configs']) != set(POLICIES)
            or plan['feature_definition'] != DEFINITION
            or len(plan['models']) != len(MODELS) or set(plan['models']) != set(MODELS)
            or cv['outer_folds'] < 2 or cv['inner_calibration_folds'] < 2
            or plan['objective'] != 'inner_calibration_f1'
            or budget['candidates'] != len(POLICIES) * len(MODELS) * cv['outer_folds']
            or budget['new_fits'] != 2 * len(MODELS) * cv['outer_folds']
            or budget['reused_candidates'] != len(MODELS) * cv['outer_folds']
            or budget['cpu_threads'] != 1 or budget['search_trials'] != 0):
        raise ValueError('Unsupported bounded file/USB plan')
    if not root.is_dir() or shutil.disk_usage(root).free < budget['minimum_free_gib'] * 1024**3:
        raise ValueError('Existing artifact root with sufficient free space required')
    if days.name != plan['daily_preparation_id'] or file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']:
        raise ValueError('Different daily source')
    previous = root / plan['previous_cv_id']
    if file_sha256(previous / 'complete.json') != plan['previous_cv_manifest_sha256']:
        raise ValueError('Different step-2 CV series')
    verify(previous, ['summary.json', 'plan.json'])
    old_plan, old = load_json(previous / 'plan.json'), load_json(previous / 'summary.json')
    if old_plan['daily_manifest_sha256'] != plan['daily_manifest_sha256'] or any(
            old_plan['cv'][key] != value for key, value in cv.items()):
        raise ValueError('Previous CV used another source or split/model seeds')
    reused = {(row['fold'], row['model']): row for row in old['results'] if row['policy'] == 'causal_all_v3'}
    if (len(reused) != budget['reused_candidates'] or len([row for row in old['results'] if row['policy'] == 'causal_all_v3']) != len(reused)
            or set(reused) != {(fold, model) for fold in range(1, cv['outer_folds'] + 1) for model in MODELS}):
        raise ValueError('Incomplete or duplicate frozen baseline')
    configs, comparable = {}, []
    for key, policy in POLICIES.items():
        setting = plan['configs'][key]
        location = PROJECT / setting['path']
        if file_sha256(location) != setting['sha256']:
            raise ValueError('Configuration hash mismatch')
        config = load_json(location)
        if (config['features']['history_policy'] != policy or config['seed'] != cv['model_seed']
                or config['lstm_autoencoder']['epochs'] > budget['max_lstm_epochs']):
            raise ValueError('Different declared feature policy, seed or epoch budget')
        configs[key] = config
        normalized = json.loads(json.dumps(config))
        normalized.pop('config_version')
        normalized['features'].pop('history_policy')
        comparable.append(normalized)
    if any(config != comparable[0] for config in comparable[1:]):
        raise ValueError('Only the declared file/USB feature/reference factor may differ')
    for row in reused.values():
        candidate = Path(row['candidate'])
        verify(candidate, ['config.json'])
        # Candidates serialize the same settings with save_json's formatting;
        # their file hash need not equal the original config file's hash.
        if (file_sha256(candidate / 'config.json') != row['config_sha256']
                or load_json(candidate / 'config.json') != configs['causal_all_v3']):
            raise ValueError('Baseline candidates used a different configuration')
    return plan, configs, previous, reused


def run(days, root, plan_path):
    days, root, plan_path = Path(days).resolve(), Path(root).resolve(), Path(plan_path).resolve()
    plan, configs, previous, reused = read_plan(plan_path, days, root)
    verify(days)
    frame = pd.read_csv(days / 'account_days.csv')
    if (set(frame.split) != {'train', 'validation', 'test'} or frame.item_id.duplicated().any()
            or frame.groupby('account_id').split.nunique().max() != 1
            or frame.groupby('group').split.nunique().max() != 1):
        raise ValueError('Invalid primary identities or connected-group assignments')
    output = allocate(root, 'cert_file_usb')
    print('SERIES ' + str(output), flush=True)
    try:
        for directory in ('source', 'settings', 'folds', 'preparations', 'candidates', 'jobs', 'results', 'logs', 'reports'):
            (output / directory).mkdir()
        save_json(plan, output / 'plan.json')
        save_json(environment_metadata(), output / 'environment.json')
        hashes = {}
        for source in sorted((PROJECT / 'src').rglob('*.py')):
            relative = source.relative_to(PROJECT / 'src')
            destination = output / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[str(relative)] = file_sha256(source)
        save_json(hashes, output / 'execution_source_hashes.json')
        for policy, setting in plan['configs'].items():
            shutil.copy2(PROJECT / setting['path'], output / 'settings' / (policy + '.json'))
        context = output / 'primary_train_context.jsonl.gz'
        write_train_context(frame, days / 'day_context.jsonl.gz', context)
        primary = frame.loc[frame.split.eq('train')].reset_index(drop=True)
        del frame
        cv = plan['cv']
        folds = list(development_folds(primary, n_splits=cv['outer_folds'], calibration_splits=cv['inner_calibration_folds'],
                                      seed=cv['outer_seed'], calibration_seed=cv['inner_calibration_seed_base']))
        for _, provenance in folds:
            name = 'fold_' + str(provenance['fold']) + '.json'
            verify(previous, ['folds/' + name])
            if provenance != load_json(previous / 'folds' / name):
                raise ValueError('Step 3 must preserve exact step-2 fold assignments')
            save_json(provenance, output / 'folds' / name)
        jobs, preparations = [], []
        contract = load_json(days / 'contract.json')
        for folded, provenance in folds:
            fold = provenance['fold']
            baseline_paths = {row['preparation'] for (index, _), row in reused.items() if index == fold}
            if len(baseline_paths) != 1:
                raise ValueError('Different baseline preparations within fold')
            baseline = Path(next(iter(baseline_paths)))
            verify(baseline)
            trio = {'causal_all_v3': baseline}
            for policy in ('file_usb_rolling_v4', 'file_usb_frozen_v4'):
                check_source(output, hashes)
                trio[policy] = prepare_cert_frame(folded, context, contract, configs[policy], days,
                                                 output / 'preparations', development=provenance)
                if (trio[policy] / 'test.csv').exists():
                    raise ValueError('Real test must not be prepared')
            audit = dict(fold=fold, file_usb=audit_preparations(trio, folded), causal_history={
                policy: audit_dependencies(path / 'history_dependencies.jsonl.gz', folded)
                for policy, path in trio.items() if policy != 'causal_all_v3'})
            save_json(audit, output / 'folds' / f'preparation_audit_{fold}.json')
            for policy, path in trio.items():
                preparations.append(dict(fold=fold, policy=policy, preparation=str(path),
                                         manifest_sha256=file_sha256(path / 'complete.json')))
                for model in MODELS:
                    job_id = f'cv_{fold}_{policy}_{model}'
                    job = dict(job_id=job_id, phase='cv', fold=fold, policy=policy, model=model, seed=cv['model_seed'],
                               preparation=str(path), config=str(output / 'settings' / (policy + '.json')),
                               output_root=str(output / 'candidates'), result=str(output / 'results' / (job_id + '.json')))
                    if policy == 'causal_all_v3':
                        old = reused[(fold, model)]
                        if (file_sha256(Path(old['candidate']) / 'complete.json') != old['candidate_manifest_sha256']
                                or file_sha256(path / 'complete.json') != old['preparation_manifest_sha256']):
                            raise ValueError('Frozen baseline artifact changed')
                        job['reuse'] = old['candidate']
                    save_json(job, output / 'jobs' / (job_id + '.json'))
                    jobs.append(job)
            print(json.dumps(dict(fold=fold, preparations_audited=True)), flush=True)
        del folds, folded
        save_json(preparations, output / 'preparations.json')
        save_json(jobs, output / 'declared_jobs.json')
        if len(jobs) != plan['budget']['candidates']:
            raise ValueError('Different declared job count')
        results = []
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED=str(cv['model_seed']))
        for job in jobs:
            check_source(output, hashes)
            started = time.perf_counter()
            with (output / 'logs' / (job['job_id'] + '.log')).open('w') as stream:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--job',
                                          str(output / 'jobs' / (job['job_id'] + '.json'))],
                                         stdout=stream, stderr=subprocess.STDOUT, env=env)
            seconds = time.perf_counter() - started
            if process.returncode:
                save_json(dict(job=job, returncode=process.returncode, seconds=seconds), output / 'failure.json')
                raise RuntimeError('File/USB worker failed: ' + job['job_id'] + '; inspect its saved log')
            result = load_json(Path(job['result']))
            if job.get('reuse'):
                old = reused[(job['fold'], job['model'])]
                if any(result[key] != old[key] for key in ('assessment_metrics', 'validation_metrics', 'threshold', 'checkpoint_sha256')):
                    raise ValueError('Reused baseline results are not reproducible')
            results.append(result)
            save_json(dict(completed=len(results), total=len(jobs), last_job=job['job_id'], last_worker_seconds=seconds),
                      output / 'progress.json')
            print(json.dumps(dict(completed=len(results), total=len(jobs), job=job['job_id'], reused=bool(job.get('reuse')),
                                  seconds=round(seconds, 2), assessment_f1=result['assessment_metrics']['f1_score'])), flush=True)
        summary = summarize(results, plan, primary, days, output, policies=POLICIES, plan_id=PLAN_ID,
                            comparisons=[('causal_all_v3', 'file_usb_rolling_v4'),
                                         ('file_usb_rolling_v4', 'file_usb_frozen_v4'),
                                         ('causal_all_v3', 'file_usb_frozen_v4')])
        adaptation_report(results, primary, output)
        summary.update(new_fits=plan['budget']['new_fits'], reused_candidates=plan['budget']['reused_candidates'],
                       previous_cv_id=previous.name, previous_cv_manifest_sha256=plan['previous_cv_manifest_sha256'])
        check_source(output, hashes)
        if (file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']
                or file_sha256(previous / 'complete.json') != plan['previous_cv_manifest_sha256']):
            raise ValueError('A prerequisite manifest changed')
        save_json(summary, output / 'summary.json')
        finish(output)
    except BaseException as error:
        save_json(dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', reason=str(error)),
                  output / 'status.json')
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job:
        from run_cert_robustness import worker
        worker(args.job, guard_assessment=True)
    elif args.days is None or args.root is None or args.plan is None:
        parser.error('--days, --root and --plan are required')
    else:
        print(run(args.days, args.root, args.plan))


if __name__ == '__main__':
    main()
