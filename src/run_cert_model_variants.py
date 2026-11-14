"""Step 4: isolated endpoint-score and fixed boosting comparisons on reused CV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE
from datasets.cert_experiments import CAUSAL_POLICY, GLOBAL_POLICY
from datasets.cert_file_usb import ROLLING_POLICY, FROZEN_POLICY
from datasets.cert_validation import development_folds
from experiment import allocate, finish, verify
from experiment_tracking import environment_metadata, file_sha256
from run_cert_group_cv import PROJECT, METRICS, check_source, summarize
from utils import load_json, save_json

PLAN_ID = 'cert_r4_2_endpoint_and_boosting_v5'
POLICIES = {'global_only_v2': GLOBAL_POLICY, 'causal_all_v3': CAUSAL_POLICY,
            'file_usb_rolling_v4': ROLLING_POLICY, 'file_usb_frozen_v4': FROZEN_POLICY}
OLD_MODELS = ('random_forest', 'isolation_forest', 'lstm_autoencoder')
NEW_METHODS = ('lstm_endpoint', BOOST_MODEL)
METHODS = (*OLD_MODELS, *NEW_METHODS)


def read_plan(path, days, root):
    plan = load_json(path)
    cv, budget = plan['cv'], plan['budget']
    if (plan['plan_id'] != PLAN_ID or set(plan['configs']) != set(POLICIES)
            or plan['methods'] != list(METHODS) or plan['objective'] != 'inner_calibration_f1'
            or set(plan['sources']) != {'group_cv_v3', 'file_usb_v4'}
            or cv['outer_folds'] < 2 or cv['inner_calibration_folds'] < 2
            or budget['new_fits'] != len(POLICIES) * len(NEW_METHODS) * cv['outer_folds']
            or budget['reused_candidates'] != len(POLICIES) * len(OLD_MODELS) * cv['outer_folds']
            or budget['assessments'] != len(POLICIES) * len(METHODS) * cv['outer_folds']
            or budget['search_trials'] != 0 or budget['cpu_threads'] != 1):
        raise ValueError('Unsupported bounded step-4 plan')
    if not root.is_dir() or shutil.disk_usage(root).free < budget['minimum_free_gib'] * 1024**3:
        raise ValueError('Existing artifact root with sufficient free space required')
    if days.name != plan['daily_preparation_id'] or file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']:
        raise ValueError('Different daily source')
    boost = plan['boosting']
    if (set(boost) != {'loss', 'learning_rate', 'max_iter', 'max_leaf_nodes', 'max_depth',
                       'min_samples_leaf', 'l2_regularization', 'max_bins', 'early_stopping', 'class_weight'}
            or boost['early_stopping'] is not False or boost['class_weight'] != 'balanced'
            or boost['loss'] != 'log_loss' or not 1 <= boost['max_iter'] <= budget['max_boosting_iterations']):
        raise ValueError('Boosting must have one declared supervised configuration without random early stopping')
    sources, summaries = {}, {}
    for key, spec in plan['sources'].items():
        source = root / spec['id']
        if file_sha256(source / 'complete.json') != spec['manifest_sha256']:
            raise ValueError('Prerequisite series changed')
        verify(source, ['summary.json', 'plan.json'])
        old_plan = load_json(source / 'plan.json')
        if old_plan['daily_manifest_sha256'] != plan['daily_manifest_sha256'] or any(
                old_plan['cv'][key] != value for key, value in cv.items()):
            raise ValueError('Source used different population or CV/model seeds')
        sources[key], summaries[key] = source, load_json(source / 'summary.json')
    controls = {}
    for policy in POLICIES:
        source = summaries['group_cv_v3' if policy == 'global_only_v2' else 'file_usb_v4']
        rows = [row for row in source['results'] if row['policy'] == policy]
        for row in rows:
            key = row['fold'], policy, row['model']
            if key in controls:
                raise ValueError('Duplicate baseline result')
            controls[key] = row
    expected = {(fold, policy, model) for fold in range(1, cv['outer_folds'] + 1)
                for policy in POLICIES for model in OLD_MODELS}
    if set(controls) != expected:
        raise ValueError('Incomplete baseline comparison')
    configs = {}
    for policy, history in POLICIES.items():
        spec = plan['configs'][policy]
        if set(spec) != {'baseline', *NEW_METHODS}:
            raise ValueError('Both independent hypotheses and the baseline config are required')
        values = {}
        for method, setting in spec.items():
            location = PROJECT / setting['path']
            if file_sha256(location) != setting['sha256']:
                raise ValueError('Configuration hash mismatch')
            config = load_json(location)
            if (config['features']['history_policy'] != history or config['seed'] != cv['model_seed']
                    or config['lstm_autoencoder']['epochs'] > budget['max_lstm_epochs']):
                raise ValueError('Different history policy, model seed or LSTM epoch budget')
            values[method] = config
        baseline = json.loads(json.dumps(values['baseline']))
        for method in NEW_METHODS:
            changed = json.loads(json.dumps(values[method]))
            changed['config_version'] = baseline['config_version']
            if method == 'lstm_endpoint':
                if changed['lstm_autoencoder'].pop('score_policy', None) != ENDPOINT_SCORE:
                    raise ValueError('The LSTM hypothesis must change only the endpoint score')
            elif changed['baseline'].pop(BOOST_MODEL, None) != boost:
                raise ValueError('The boosting hypothesis differs from its fixed parameters')
            if changed != baseline:
                raise ValueError('An undeclared extra experiment factor changed')
        configs[policy] = values
        for (fold, key, model), row in controls.items():
            if key != policy:
                continue
            candidate = Path(row['candidate'])
            verify(candidate, ['config.json'])
            if (load_json(candidate / 'config.json') != baseline
                    or file_sha256(candidate / 'config.json') != row['config_sha256']):
                raise ValueError('Frozen baseline configuration differs')
    return plan, configs, sources, controls


def method_differences(results, folds):
    differences = []
    for policy in POLICIES:
        for before, after in [('lstm_autoencoder', 'lstm_endpoint'), ('random_forest', BOOST_MODEL)]:
            for fold in range(1, folds + 1):
                old, new = [next(row for row in results if row['policy'] == policy and row['fold'] == fold
                                 and row['model'] == method)['assessment_metrics'] for method in (before, after)]
                differences.append(dict(policy=policy, fold=fold, before=before, after=after,
                    metrics={metric: new[metric] - old[metric] if new[metric] is not None and old[metric] is not None
                             else None for metric in METRICS}))
    return differences


def run(days, root, plan_path):
    days, root, plan_path = Path(days).resolve(), Path(root).resolve(), Path(plan_path).resolve()
    plan, configs, sources, controls = read_plan(plan_path, days, root)
    verify(days)
    frame = pd.read_csv(days / 'account_days.csv')
    if (set(frame.split) != {'train', 'validation', 'test'} or frame.item_id.duplicated().any()
            or frame.groupby('account_id').split.nunique().max() != 1
            or frame.groupby('group').split.nunique().max() != 1):
        raise ValueError('Invalid original primary group assignments')
    primary = frame.loc[frame.split.eq('train')].reset_index(drop=True)
    del frame
    output = allocate(root, 'cert_model_variants')
    print('SERIES ' + str(output), flush=True)
    try:
        for directory in ('source', 'settings', 'folds', 'candidates', 'jobs', 'results', 'reports', 'logs'):
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
        for policy, settings in plan['configs'].items():
            for method, spec in settings.items():
                shutil.copy2(PROJECT / spec['path'], output / 'settings' / (policy + '_' + method + '.json'))
        cv = plan['cv']
        folds = list(development_folds(primary, n_splits=cv['outer_folds'], calibration_splits=cv['inner_calibration_folds'],
                                      seed=cv['outer_seed'], calibration_seed=cv['inner_calibration_seed_base']))
        for _, provenance in folds:
            name = 'fold_' + str(provenance['fold']) + '.json'
            for source in sources.values():
                verify(source, ['folds/' + name])
                if provenance != load_json(source / 'folds' / name):
                    raise ValueError('Fold assignments changed since previous increments')
            save_json(provenance, output / 'folds' / name)
        del folds
        jobs, results, preps = [], [], {}
        for fold in range(1, cv['outer_folds'] + 1):
            for policy in POLICIES:
                paths = {controls[(fold, policy, model)]['preparation'] for model in OLD_MODELS}
                if len(paths) != 1:
                    raise ValueError('Different source preparations within a policy/fold')
                prep = Path(next(iter(paths)))
                verify(prep)
                contract = load_json(prep / 'contract.json')
                if (contract['scope'] != 'development_group_cv' or contract['history_policy'] != POLICIES[policy]
                        or contract['daily_manifest_sha256'] != plan['daily_manifest_sha256']
                        or (prep / 'test.csv').exists()):
                    raise ValueError('Only the original development preparations may be reused')
                preps[f'{fold}_{policy}'] = dict(path=str(prep), manifest_sha256=file_sha256(prep / 'complete.json'))
                for model in OLD_MODELS:
                    original = controls[(fold, policy, model)]
                    for field, manifest in [('candidate', 'candidate_manifest_sha256'), ('assessment', 'assessment_manifest_sha256')]:
                        path = Path(original[field])
                        if file_sha256(path / 'complete.json') != original[manifest]:
                            raise ValueError('Frozen control artifact changed')
                        verify(path)
                    if file_sha256(prep / 'complete.json') != original['preparation_manifest_sha256']:
                        raise ValueError('Frozen control preparation changed')
                    result = dict(original, artifact_model=model, reused=True, reused_assessment=True)
                    results.append(result)
                for method in NEW_METHODS:
                    model = 'lstm_autoencoder' if method == 'lstm_endpoint' else BOOST_MODEL
                    job_id = f'cv_{fold}_{policy}_{method}'
                    job = dict(job_id=job_id, phase='cv', fold=fold, policy=policy, method=method,
                               model=model, seed=cv['model_seed'], preparation=str(prep),
                               config=str(output / 'settings' / (policy + '_' + method + '.json')),
                               output_root=str(output / 'candidates'), result=str(output / 'results' / (job_id + '.json')))
                    save_json(job, output / 'jobs' / (job_id + '.json'))
                    jobs.append(job)
        save_json(preps, output / 'preparations.json')
        save_json(results, output / 'reused_controls.json')
        save_json(jobs, output / 'declared_jobs.json')
        if len(jobs) != plan['budget']['new_fits'] or len(results) != plan['budget']['reused_candidates']:
            raise ValueError('Different declared candidate budget')
        print(json.dumps(dict(preparations_verified=len(preps), reused_controls=len(results), new_jobs=len(jobs))), flush=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED=str(cv['model_seed']))
        for index, job in enumerate(jobs, 1):
            check_source(output, hashes)
            started = time.perf_counter()
            with (output / 'logs' / (job['job_id'] + '.log')).open('w') as stream:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--job',
                                          str(output / 'jobs' / (job['job_id'] + '.json'))],
                                         stdout=stream, stderr=subprocess.STDOUT, env=env)
            seconds = time.perf_counter() - started
            if process.returncode:
                save_json(dict(job=job, returncode=process.returncode, seconds=seconds), output / 'failure.json')
                raise RuntimeError('Step-4 worker failed: ' + job['job_id'] + '; inspect its saved log')
            result = load_json(Path(job['result']))
            result.update(artifact_model=result['model'], model=job['method'])
            results.append(result)
            save_json(dict(completed_new_fits=index, total_new_fits=len(jobs), last_job=job['job_id'], seconds=seconds),
                      output / 'progress.json')
            print(json.dumps(dict(completed_new_fits=index, total_new_fits=len(jobs), job=job['job_id'],
                                  seconds=round(seconds, 2), assessment_f1=result['assessment_metrics']['f1_score'])), flush=True)
        summary = summarize(results, plan, primary, days, output, policies=POLICIES, models=METHODS,
                            plan_id=PLAN_ID, comparisons=[])
        summary.update(new_fits=len(jobs), reused_candidates=plan['budget']['reused_candidates'],
                       paired_method_differences=method_differences(results, cv['outer_folds']),
                       method_labels={'lstm_endpoint': 'same LSTM/loss; endpoint-only score with inner-calibrated epoch/threshold',
                                      BOOST_MODEL: 'supervised tabular histogram boosting; compare to supervised RF'},
                       source_series=plan['sources'])
        check_source(output, hashes)
        for key, source in sources.items():
            if file_sha256(source / 'complete.json') != plan['sources'][key]['manifest_sha256']:
                raise ValueError('A prerequisite series manifest changed')
        if file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']:
            raise ValueError('Daily manifest changed')
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
