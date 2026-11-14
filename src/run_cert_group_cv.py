"""Bounded paired v2/v3 group-CV for RF, IF and LSTM on primary train only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from datasets.cert_experiments import CAUSAL_POLICY, GLOBAL_POLICY, prepare_cert_frame
from datasets.cert_reporting import error_report, evidence_index
from datasets.cert_validation import development_folds, write_train_context
from experiment import allocate, finish, verify
from experiment_tracking import environment_metadata, file_sha256
from prepare_cert_development import audit_dependencies, audit_pair
from utils import compute_classification_metrics, load_json, save_json

PROJECT = Path(__file__).resolve().parents[1]
PLAN_ID = 'cert_r4_2_paired_group_cv_v3'
POLICIES = {'global_only_v2': GLOBAL_POLICY, 'causal_all_v3': CAUSAL_POLICY}
MODELS = ('random_forest', 'isolation_forest', 'lstm_autoencoder')
METRICS = ('precision', 'recall', 'f1_score', 'pr_auc', 'roc_auc', 'false_positive_rate')


def read_plan(path, days, root):
    plan = load_json(path)
    cv, budget = plan['cv'], plan['budget']
    if (plan['plan_id'] != PLAN_ID or set(plan['configs']) != set(POLICIES)
            or len(plan['models']) != len(MODELS) or set(plan['models']) != set(MODELS)
            or cv['outer_folds'] < 2 or cv['inner_calibration_folds'] < 2
            or plan['objective'] != 'inner_calibration_f1'
            or budget['candidates'] != len(POLICIES) * len(MODELS) * cv['outer_folds']
            or budget['cpu_threads'] != 1 or budget['search_trials'] != 0):
        raise ValueError('Unsupported bounded paired CV plan')
    if not root.is_dir() or shutil.disk_usage(root).free < budget['minimum_free_gib'] * 1024**3:
        raise ValueError('Existing artifact root with sufficient free space required')
    if days.name != plan['daily_preparation_id'] or file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']:
        raise ValueError('Different daily source')
    previous = root / plan['development_preparation_id']
    if file_sha256(previous / 'complete.json') != plan['development_preparation_manifest_sha256']:
        raise ValueError('Different prerequisite development preparation')
    verify(previous, ['summary.json'])
    if load_json(previous / 'summary.json')['daily_manifest_sha256'] != plan['daily_manifest_sha256']:
        raise ValueError('Prerequisite used another daily source')
    configs, comparable = {}, []
    for key, policy in POLICIES.items():
        setting = plan['configs'][key]
        location = PROJECT / setting['path']
        if file_sha256(location) != setting['sha256']:
            raise ValueError('Configuration hash mismatch')
        config = load_json(location)
        if (config['features']['history_policy'] != policy or config['seed'] != cv['model_seed']
                or config['lstm_autoencoder']['epochs'] > budget['max_lstm_epochs']):
            raise ValueError('Configuration exceeds declared CV settings')
        configs[key] = config
        normalized = json.loads(json.dumps(config))
        normalized.pop('config_version')
        normalized['features'].pop('history_policy')
        comparable.append(normalized)
    if comparable[0] != comparable[1]:
        raise ValueError('Only the history factor may differ')
    return plan, configs


def check_source(output, hashes):
    for relative, digest in hashes.items():
        if (file_sha256(PROJECT / 'src' / relative) != digest
                or file_sha256(output / 'source' / relative) != digest):
            raise ValueError('Source changed during frozen CV execution')


def expected_jobs(plan, policies=None, models=None):
    policies = POLICIES if policies is None else policies
    models = MODELS if models is None else models
    return {(fold, policy, model) for fold in range(1, plan['cv']['outer_folds'] + 1)
            for policy in policies for model in models}


def summarize(results, plan, primary, days, output, *, policies=None, plan_id=PLAN_ID, comparisons=None, models=None):
    policies = tuple(POLICIES if policies is None else policies)
    models = tuple(MODELS if models is None else models)
    comparisons = [(policies[0], policies[1])] if comparisons is None else comparisons
    identities = [(row['fold'], row['policy'], row['model']) for row in results]
    if len(identities) != len(set(identities)) or set(identities) != expected_jobs(plan, policies, models):
        raise ValueError('Missing or duplicate CV result; every declared run must be reported')
    length = load_json(Path(results[0]['preparation']) / 'contract.json')['sequence_length']
    ordered = primary.sort_values(['account_id', 'date', 'item_id'])
    endpoints = ordered.loc[ordered.groupby('account_id').cumcount() >= length - 1]
    expected = endpoints[['item_id', 'label']].sort_values('item_id').reset_index(drop=True)
    # Attributions enter only after all fitting/calibration/inference has ended.
    allowed_keys = set(primary['day_key'])
    evidence = {key: value for key, value in evidence_index(days / 'positive_evidence.jsonl').items()
                if key in allowed_keys}
    collected, validation_cohorts = {}, {}
    assessment_cohorts = {}
    for row in results:
        fold = row['fold']
        cohort = row['cohort_sha256']
        if fold in validation_cohorts and validation_cohorts[fold] != cohort:
            raise ValueError('Different inner calibration population')
        validation_cohorts[fold] = cohort
        assessment = Path(row['assessment'])
        verify(assessment)
        if file_sha256(assessment / 'complete.json') != row['assessment_manifest_sha256']:
            raise ValueError('Assessment manifest changed')
        with np.load(assessment / 'predictions_assessment.npz', allow_pickle=False) as arrays:
            ids, y, values, decisions = (arrays[key] for key in ('item_ids', 'y', 'scores', 'predictions'))
        if fold not in assessment_cohorts:
            roles = load_json(output / 'folds' / f'fold_{fold}.json')['account_roles']
            assessment_cohorts[fold] = endpoints.loc[endpoints.account_id.map(roles).eq('assessment')]
        expected_fold = assessment_cohorts[fold]
        if (not np.isfinite(values).all()
                or not np.array_equal(decisions, values >= row['threshold'])
                or not np.array_equal(ids, expected_fold.item_id)
                or not np.array_equal(y, expected_fold.label)):
            raise ValueError('Assessment predictions or target mismatch')
        actual = compute_classification_metrics(y, decisions, values)
        if actual != row['assessment_metrics']:
            raise ValueError('Assessment metrics do not match exact saved scores')
        predictions = pd.DataFrame(dict(item_id=ids, y_true=y, score=values, prediction=decisions))
        predictions['fold'] = fold
        collected.setdefault((row['policy'], row['model']), []).append(predictions)
        save_json(error_report(predictions, primary, evidence), output / 'reports' / (row['job_id'] + '.json'))
        if not (row['checks']['assessment_open_guard']
                and row['checks']['assessment_after_candidate_complete']
                and row['checks']['assessment_open_attempts_during_fit'] == []
                and row['checks']['test_open_attempts'] == []):
            raise ValueError('A worker did not enforce the evaluation boundaries')
    aggregates, pooled, differences = [], [], []
    for policy in policies:
        for model in models:
            subset = sorted((row for row in results if row['policy'] == policy and row['model'] == model),
                            key=lambda row: row['fold'])
            statistics = {}
            for metric in METRICS:
                values = [row['assessment_metrics'][metric] for row in subset]
                valid = np.asarray([value for value in values if value is not None], dtype=float)
                statistics[metric] = {'mean': float(valid.mean()) if len(valid) else None,
                    'sample_std': float(valid.std(ddof=1)) if len(valid) > 1 else None,
                    'min': float(valid.min()) if len(valid) else None,
                    'max': float(valid.max()) if len(valid) else None, 'defined_folds': len(valid)}
            aggregates.append(dict(policy=policy, model=model, folds=len(subset), metrics=statistics))
            predictions = pd.concat(collected[(policy, model)], ignore_index=True).sort_values('item_id').reset_index(drop=True)
            if (predictions.item_id.duplicated().any() or not np.array_equal(predictions.item_id, expected.item_id)
                    or not np.array_equal(predictions.y_true, expected.label)):
                raise ValueError('Every primary train endpoint must be assessed exactly once')
            predictions.to_csv(output / 'reports' / f'oof_{policy}_{model}.csv', index=False)
            report = error_report(predictions, primary, evidence)
            report.pop('scores')  # Different fitted models need not share a score scale.
            save_json(report, output / 'reports' / f'oof_errors_{policy}_{model}.json')
            pooled.append(dict(policy=policy, model=model,
                metrics=compute_classification_metrics(predictions.y_true, predictions.prediction),
                scenarios=report['scenarios'], background=report['background']))
    for left, right in comparisons:
        for model in models:
            for fold in range(1, plan['cv']['outer_folds'] + 1):
                old, new = [next(row for row in results if row['model'] == model and row['fold'] == fold and row['policy'] == policy)
                            for policy in (left, right)]
                delta = {metric: new['assessment_metrics'][metric] - old['assessment_metrics'][metric]
                         if new['assessment_metrics'][metric] is not None and old['assessment_metrics'][metric] is not None
                         else None for metric in METRICS}
                differences.append(dict(model=model, fold=fold, **{right + '_minus_' + left: delta}))
    return dict(plan_id=plan_id, completed_candidates=len(results), results=results,
                fold_macro_aggregates=aggregates, pooled_oof_decisions=pooled, paired_fold_differences=differences,
                population=dict(days=len(primary), accounts=int(primary.account_id.nunique()),
                                independent_groups=int(primary.group.nunique()), endpoints=len(expected),
                                positive_endpoints=int(expected.label.sum())),
                every_endpoint_assessed_once_per_policy_model=True,
                model_seeds=[plan['cv']['model_seed']], no_real_test_scoring=True,
                no_outer_threshold_or_epoch_selection=True,
                interpretation='reused development folds; dependent fold variation, not a confidence interval or independent confirmation',
                pooled_rank_metrics='undefined: fold scores come from different fitted models; use within-fold PR/ROC AUC')


def run(days, root, plan_path):
    days, root, plan_path = Path(days).resolve(), Path(root).resolve(), Path(plan_path).resolve()
    plan, configs = read_plan(plan_path, days, root)
    verify(days)
    frame = pd.read_csv(days / 'account_days.csv')
    if (set(frame['split']) != {'train', 'validation', 'test'}
            or frame.item_id.duplicated().any()
            or frame.groupby('account_id')['split'].nunique().max() != 1
            or frame.groupby('group')['split'].nunique().max() != 1):
        raise ValueError('Invalid original account/group assignments')
    output = allocate(root, 'cert_group_cv')
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
        folds = list(development_folds(primary, n_splits=cv['outer_folds'],
                     calibration_splits=cv['inner_calibration_folds'], seed=cv['outer_seed'],
                     calibration_seed=cv['inner_calibration_seed_base']))
        for _, provenance in folds:
            save_json(provenance, output / 'folds' / f"fold_{provenance['fold']}.json")
        jobs, preparations = [], []
        contract = load_json(days / 'contract.json')
        # Finish declaring and auditing all inputs before any model trial.
        for folded, provenance in folds:
            fold = provenance['fold']
            pair = {}
            for policy, config in configs.items():
                check_source(output, hashes)
                pair[policy] = prepare_cert_frame(folded, context, contract, config, days,
                                                 output / 'preparations', development=provenance)
                if (pair[policy] / 'test.csv').exists():
                    raise ValueError('Real test must not be prepared for CV')
            audit = dict(fold=fold, pair=audit_pair(pair, roles=('train', 'validation', 'assessment')),
                         causal_history=audit_dependencies(pair['causal_all_v3'] / 'history_dependencies.jsonl.gz', folded))
            save_json(audit, output / 'folds' / f'preparation_audit_{fold}.json')
            for policy, path in pair.items():
                preparations.append(dict(fold=fold, policy=policy, preparation=str(path),
                                         manifest_sha256=file_sha256(path / 'complete.json')))
                for model in MODELS:
                    job_id = f'cv_{fold}_{policy}_{model}'
                    job = dict(job_id=job_id, phase='cv', fold=fold, policy=policy, model=model,
                               seed=cv['model_seed'], preparation=str(path),
                               config=str(output / 'settings' / (policy + '.json')),
                               output_root=str(output / 'candidates'), result=str(output / 'results' / (job_id + '.json')))
                    save_json(job, output / 'jobs' / (job_id + '.json'))
                    jobs.append(job)
            print(json.dumps({'fold': fold, 'preparations_audited': True}), flush=True)
        del folds, folded
        save_json(preparations, output / 'preparations.json')
        save_json(jobs, output / 'declared_jobs.json')
        if len(jobs) != plan['budget']['candidates']:
            raise ValueError('Different number of declared candidates')
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
                raise RuntimeError('CV worker failed: ' + job['job_id'] + '; inspect its saved log')
            result = load_json(Path(job['result']))
            results.append(result)
            save_json(dict(completed=len(results), total=len(jobs), last_job=job['job_id'], last_worker_seconds=seconds),
                      output / 'progress.json')
            print(json.dumps(dict(completed=len(results), total=len(jobs), job=job['job_id'],
                                  seconds=round(seconds, 2), assessment_f1=result['assessment_metrics']['f1_score'])), flush=True)
        summary = summarize(results, plan, primary, days, output)
        check_source(output, hashes)
        if file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']:
            raise ValueError('Daily source manifest changed')
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
    if args.job is not None:
        from run_cert_robustness import worker
        worker(args.job, guard_assessment=True)
    elif args.days is None or args.root is None or args.plan is None:
        parser.error('--days, --root and --plan are required')
    else:
        print(run(args.days, args.root, args.plan))


if __name__ == '__main__':
    main()
