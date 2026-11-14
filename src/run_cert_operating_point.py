"""Step 5: frozen development shortlist, five seeds, calibration-only FP budgets."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

from cert_model_variants import BOOST_MODEL
from cert_operating_point import (OBJECTIVE, PRIMARY_PROFILE, PROFILES, check_operating_point,
                                 select_operating_points)
from datasets.cert_file_usb import FROZEN_POLICY, ROLLING_POLICY
from datasets.cert_reporting import error_report, evidence_index
from datasets.cert_validation import development_folds
from experiment import allocate, candidate, configured_score, finish, inputs, verify
from experiment_tracking import environment_metadata, file_sha256
from run_cert_group_cv import PROJECT, METRICS, check_source
from utils import (compute_classification_metrics, load_json, measure_inference, save_json,
                   select_best_threshold)


PLAN_ID = 'cert_r4_2_operating_point_stability_v6'
SHORTLIST = {
    'hgb_rolling': ('file_usb_rolling_v4', BOOST_MODEL, ROLLING_POLICY),
    'rf_rolling': ('file_usb_rolling_v4', 'random_forest', ROLLING_POLICY),
    'rf_frozen': ('file_usb_frozen_v4', 'random_forest', FROZEN_POLICY),
}
SEEDS = [7, 21, 42, 84, 168]


def read_plan(plan_path, root):
    plan = load_json(plan_path)
    budget, cv = plan['budget'], plan['cv']
    count = len(SHORTLIST) * len(SEEDS) * cv['outer_folds']
    if (plan['plan_id'] != PLAN_ID or set(plan['shortlist']) != set(SHORTLIST)
            or plan['model_seeds'] != SEEDS or plan['reference_seed'] != 42
            or plan['primary_candidate'] != 'hgb_rolling' or plan['primary_profile'] != PRIMARY_PROFILE
            or plan['profiles'] != PROFILES or plan['objective'] != OBJECTIVE
            or cv['outer_folds'] < 2 or cv['inner_calibration_folds'] < 2
            or budget['new_fits'] != count - len(SHORTLIST) * cv['outer_folds']
            or budget['reused_candidates'] != len(SHORTLIST) * cv['outer_folds']
            or budget['frozen_bundles'] != count or budget['decision_assessments'] != count * (1 + len(PROFILES))
            or budget['search_trials'] != 0 or budget['cpu_threads'] != 1):
        raise ValueError('Unsupported bounded step-5 plan')
    if not root.is_dir() or shutil.disk_usage(root).free < budget['minimum_free_gib'] * 1024**3:
        raise ValueError('Existing root and sufficient free space required')
    spec = plan['source']
    source = root / spec['id']
    if file_sha256(source / 'complete.json') != spec['manifest_sha256']:
        raise ValueError('Step-4 source manifest changed')
    verify(source, ['summary.json', 'plan.json'])
    old_plan = load_json(source / 'plan.json')
    if (old_plan['daily_manifest_sha256'] != plan['daily_manifest_sha256']
            or any(old_plan['cv'][key] != value for key, value in cv.items())):
        raise ValueError('Different population or development folds')
    rows, settings = {}, {}
    summary = load_json(source / 'summary.json')
    for name, (policy, model, history) in SHORTLIST.items():
        spec = plan['shortlist'][name]
        setting = PROJECT / spec['path']
        if file_sha256(setting) != spec['sha256']:
            raise ValueError('Shortlist configuration changed')
        config = load_json(setting)
        if config['features']['history_policy'] != history or config['seed'] != 42:
            raise ValueError('Different feature policy or reference seed')
        if model == BOOST_MODEL and config['baseline'][model]['early_stopping'] is not False:
            raise ValueError('Automatic random early stopping forbidden')
        settings[name] = config
        for fold in range(1, cv['outer_folds'] + 1):
            selected = [r for r in summary['results'] if (r['policy'], r['model'], r['fold']) == (policy, model, fold)]
            if len(selected) != 1:
                raise ValueError('Incomplete or duplicate seed-42 frozen control')
            row = selected[0]
            location, prep = Path(row['candidate']), Path(row['preparation'])
            if (file_sha256(location / 'complete.json') != row['candidate_manifest_sha256']
                    or file_sha256(prep / 'complete.json') != row['preparation_manifest_sha256']):
                raise ValueError('Frozen control changed')
            verify(location)
            verify(prep)
            contract = load_json(prep / 'contract.json')
            if (load_json(location / 'config.json') != config or contract['scope'] != 'development_group_cv'
                    or contract['history_policy'] != history or (prep / 'test.csv').exists()
                    or contract['daily_manifest_sha256'] != plan['daily_manifest_sha256']):
                raise ValueError('Shortlist must retain exact existing configuration/preparation')
            rows[name, fold] = row
    return plan, source, rows, settings


def freeze_operating_candidate(trained, preparation, root, profiles=None):
    """Read fit/calibration only; copy immutable inference bundle before assessment."""
    trained, preparation, root = Path(trained), Path(preparation), Path(root)
    verify(trained)
    verify(preparation, ['train.csv', 'validation.csv', 'feature_state.json', 'contract.json'])
    config, contract = load_json(trained / 'config.json'), load_json(trained / 'contract.json')
    original = load_json(trained / 'selection.json')
    model_name = original['model']
    if (model_name not in ('random_forest', BOOST_MODEL) or contract.get('scope') != 'development_group_cv'
            or contract != load_json(preparation / 'contract.json') or original['test_accessed'] is not False
            or file_sha256(preparation / 'complete.json') != original['preparation_manifest_sha256']
            or file_sha256(trained / 'feature_state.json') != file_sha256(preparation / 'feature_state.json')):
        raise ValueError('Matching development-only tabular candidate required')
    model = joblib.load(trained / 'model.joblib')
    validation, train = [pd.read_csv(preparation / (role + '.csv')) for role in ('validation', 'train')]
    if not set(validation.user_id).isdisjoint(set(train.user_id)):
        raise ValueError('Accounts cross fit/calibration boundary')
    x, y, ids, members = inputs(validation, contract, model_name)
    with np.load(trained / 'validation_inputs.npz', allow_pickle=False) as saved:
        if not all(np.array_equal(saved[k], v) for k, v in zip(('X', 'y', 'item_ids', 'member_ids'), (x, y, ids, members))):
            raise ValueError('Different saved calibration inputs or members')
    with np.load(trained / 'train_inputs.npz', allow_pickle=False) as saved:
        if not (np.array_equal(saved['X'], train[contract['tabular_columns']].to_numpy())
                and np.array_equal(saved['y'], train.label)
                and np.array_equal(saved['item_ids'], train.event_id)):
            raise ValueError('Tabular training inputs differ from all labeled inner fit')
    scores = configured_score(model, x, model_name, config)
    old_predictions = pd.read_csv(trained / 'predictions_validation.csv')
    threshold, _, metrics, _ = select_best_threshold(scores, y, config['baseline']['threshold_percentiles'])
    if (threshold != original['threshold'] or metrics != load_json(trained / 'metrics_validation.json')
            or not np.array_equal(old_predictions.item_id, ids)
            or not np.array_equal(old_predictions.y_true, y)
            or not np.allclose(old_predictions.score, scores, rtol=1e-12, atol=1e-12)
            or not np.array_equal(old_predictions.prediction, scores >= threshold)):
        raise ValueError('Original calibration/F1 threshold does not reproduce')
    choices, table = select_operating_points(scores, y, profiles)
    choices = {'legacy_f1': dict(threshold=threshold, objective='unchanged_inner_calibration_f1_percentiles',
                               calibration_metrics=metrics), **choices}
    path = allocate(root, 'cert_operating_candidate')
    try:
        for name in ('model.joblib', 'config.json', 'contract.json', 'feature_state.json', 'validation_inputs.npz'):
            shutil.copy2(trained / name, path / name)
        np.savez_compressed(path / 'predictions_calibration.npz', item_ids=ids, y=y, scores=scores,
                            **{name: (scores >= spec['threshold']).astype(np.int64) for name, spec in choices.items()})
        table.to_csv(path / 'threshold_candidates.csv.gz', index=False, compression='gzip')
        save_json(choices, path / 'operating_points.json')
        save_json(dict(model=model_name, seed=config['seed'], source_candidate=str(trained),
            source_candidate_manifest_sha256=file_sha256(trained / 'complete.json'),
            preparation=str(preparation), preparation_manifest_sha256=file_sha256(preparation / 'complete.json'),
            checkpoint_sha256=file_sha256(trained / 'model.joblib'),
            calibration_only=True, new_operating_profiles_assessed_before_freeze=False,
            source_candidate_may_have_prior_development_assessment=True,
            candidate_grid='all unique calibration scores plus finite reject-all',
            tie_breakers=['higher recall', 'lower FP count', 'higher threshold'], no_refit=True), path / 'lineage.json')
        finish(path)
    except BaseException as error:
        save_json(dict(status='failed', reason=str(error)), path / 'status.json')
        raise
    return path


def assess_operating_candidate(frozen, preparation, root):
    """Load one frozen model; apply all numeric thresholds without any selection."""
    frozen, preparation = Path(frozen), Path(preparation)
    verify(frozen)
    lineage, contract = load_json(frozen / 'lineage.json'), load_json(frozen / 'contract.json')
    if (file_sha256(preparation / 'complete.json') != lineage['preparation_manifest_sha256']
            or contract != load_json(preparation / 'contract.json') or contract['scope'] != 'development_group_cv'):
        raise ValueError('Different development preparation')
    verify(preparation, ['assessment.csv'])
    digest = file_sha256(frozen / 'complete.json')
    x, y, ids, members = inputs(pd.read_csv(preparation / 'assessment.csv'), contract, lineage['model'])
    model = joblib.load(frozen / 'model.joblib')
    config = load_json(frozen / 'config.json')
    scores, timing = measure_inference(lambda: configured_score(model, x, lineage['model'], config),
                                      record_count=len(y), warmup_runs=1, measured_runs=5)
    choices = load_json(frozen / 'operating_points.json')
    decisions = {name: (scores >= spec['threshold']).astype(np.int64) for name, spec in choices.items()}
    metrics = {name: compute_classification_metrics(y, decisions[name], scores) for name in choices}
    checks = {name: check_operating_point(metrics[name], spec) for name, spec in choices.items() if name != 'legacy_f1'}
    output = allocate(root, 'cert_operating_assessment')
    np.savez_compressed(output / 'assessment_inputs.npz', X=x, y=y, item_ids=ids, member_ids=members)
    np.savez_compressed(output / 'predictions_assessment.npz', item_ids=ids, y=y, scores=scores, **decisions)
    save_json(metrics, output / 'metrics_assessment.json')
    save_json(checks, output / 'operating_checks.json')
    save_json(timing, output / 'timing.json')
    save_json(dict(frozen_candidate=str(frozen), frozen_candidate_manifest_sha256=digest,
        preparation=str(preparation), preparation_manifest_sha256=lineage['preparation_manifest_sha256'],
        refit_or_recalibration=False, score_dtype=str(scores.dtype), scope='development_group_cv'), output / 'lineage.json')
    if file_sha256(frozen / 'complete.json') != digest:
        raise ValueError('Frozen candidate changed during assessment')
    finish(output)
    return output


def worker(job_path):
    allowed, blocked = False, []
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        name = Path(os.fsdecode(args[0])).name
        if (name in ('test.csv', 'test_inputs.npz', 'predictions_test.npz', 'predictions_test.csv', 'metrics_test.json')
                or (not allowed and name in ('assessment.csv', 'assessment_inputs.npz', 'predictions_assessment.npz',
                                              'predictions_assessment.csv', 'metrics_assessment.json'))):
            blocked.append(os.fsdecode(args[0]))
            raise RuntimeError('Test is closed; outer assessment opens only after working points are frozen')
    sys.addaudithook(audit)
    job = load_json(job_path)
    prep, root, settings = Path(job['preparation']), Path(job['output_root']), Path(job['config'])
    started = time.perf_counter()
    trained = Path(job['reuse']) if job.get('reuse') else candidate(prep, root, job['model'], settings)
    training_seconds = time.perf_counter() - started
    if load_json(trained / 'config.json') != load_json(settings):
        raise ValueError('Worker configuration differs from frozen settings')
    frozen = freeze_operating_candidate(trained, prep, root)
    frozen_digest = file_sha256(frozen / 'complete.json')
    allowed = True
    assessed = assess_operating_candidate(frozen, prep, root)
    if blocked or file_sha256(frozen / 'complete.json') != frozen_digest:
        raise ValueError('Evaluation boundary or frozen manifest changed')
    result = dict(job, candidate=str(trained), frozen_candidate=str(frozen), assessment=str(assessed),
        reused=bool(job.get('reuse')), candidate_manifest_sha256=file_sha256(trained / 'complete.json'),
        frozen_candidate_manifest_sha256=frozen_digest, assessment_manifest_sha256=file_sha256(assessed / 'complete.json'),
        preparation_manifest_sha256=file_sha256(prep / 'complete.json'),
        checkpoint_sha256=file_sha256(frozen / 'model.joblib'), config_sha256=file_sha256(settings),
        operating_points=load_json(frozen / 'operating_points.json'),
        assessment_metrics=load_json(assessed / 'metrics_assessment.json'), operating_checks=load_json(assessed / 'operating_checks.json'),
        timing=load_json(assessed / 'timing.json'), candidate_seconds_including_io=training_seconds,
        peak_rss_bytes_worker=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checks=dict(test_open_guard=True, assessment_after_operating_freeze=True, blocked_open_attempts=[]))
    save_json(result, Path(job['result']))
    print(json.dumps(dict(job=job['job_id'], complete=True)), flush=True)


def statistics(values):
    valid = np.asarray([x for x in values if x is not None], dtype=float)
    return dict(mean=float(valid.mean()) if len(valid) else None,
                sample_std=float(valid.std(ddof=1)) if len(valid) > 1 else None,
                min=float(valid.min()) if len(valid) else None, max=float(valid.max()) if len(valid) else None,
                defined=len(valid))


def summarize(results, plan, primary, days, output):
    expected_keys = {(name, seed, fold) for name in SHORTLIST for seed in SEEDS
                     for fold in range(1, plan['cv']['outer_folds'] + 1)}
    keys = [(r['name'], r['seed'], r['fold']) for r in results]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError('Missing or duplicate predeclared stability run')
    ordered = primary.sort_values(['account_id', 'date', 'item_id'])
    length = load_json(Path(results[0]['preparation']) / 'contract.json')['sequence_length']
    endpoints = ordered.loc[ordered.groupby('account_id').cumcount().ge(length - 1)]
    expected_all = endpoints[['item_id', 'label']].sort_values('item_id').reset_index(drop=True)
    allowed_days = set(primary.day_key)
    evidence = {k: v for k, v in evidence_index(days / 'positive_evidence.jsonl').items() if k in allowed_days}
    collected, diagnostics, score_identity = {}, [], []
    for row in results:
        assessment, frozen = Path(row['assessment']), Path(row['frozen_candidate'])
        verify(assessment)
        verify(frozen)
        if (file_sha256(assessment / 'complete.json') != row['assessment_manifest_sha256']
                or file_sha256(frozen / 'complete.json') != row['frozen_candidate_manifest_sha256']
                or row['checks']['blocked_open_attempts'] or not row['checks']['assessment_after_operating_freeze']):
            raise ValueError('Invalid frozen assessment or access guard')
        roles = load_json(output / 'folds' / f"fold_{row['fold']}.json")['account_roles']
        expected = endpoints.loc[endpoints.account_id.map(roles).eq('assessment')]
        with np.load(assessment / 'predictions_assessment.npz', allow_pickle=False) as a:
            ids, y, scores = a['item_ids'], a['y'], a['scores']
            if not (np.array_equal(ids, expected.item_id) and np.array_equal(y, expected.label) and np.isfinite(scores).all()):
                raise ValueError('Changed outer cohort or labels')
            for profile, spec in row['operating_points'].items():
                decisions = a[profile]
                actual = compute_classification_metrics(y, decisions, scores)
                if actual != row['assessment_metrics'][profile] or not np.array_equal(decisions, scores >= spec['threshold']):
                    raise ValueError('Different frozen decisions or metrics')
                predictions = pd.DataFrame(dict(item_id=ids, y_true=y, prediction=decisions, score=scores, fold=row['fold']))
                collected.setdefault((row['name'], row['seed'], profile), []).append(predictions)
                diagnostics.append(dict(name=row['name'], seed=row['seed'], fold=row['fold'], profile=profile,
                    threshold=spec['threshold'], calibration=spec.get('eligible_on_calibration'),
                    assessment=row['operating_checks'].get(profile), metrics=actual))
    pooled = []
    for (name, seed, profile), parts in collected.items():
        predictions = pd.concat(parts, ignore_index=True).sort_values('item_id').reset_index(drop=True)
        if (predictions.item_id.duplicated().any() or not np.array_equal(predictions.item_id, expected_all.item_id)
                or not np.array_equal(predictions.y_true, expected_all.label)):
            raise ValueError('OOF endpoints must each occur exactly once for every seed/config/profile')
        report = error_report(predictions, primary, evidence)
        report.pop('scores')
        save_json(report, output / 'reports' / f'{name}_{seed}_{profile}.json')
        predictions.drop(columns='score').to_csv(output / 'reports' / f'oof_{name}_{seed}_{profile}.csv.gz', index=False, compression='gzip')
        metrics = compute_classification_metrics(predictions.y_true, predictions.prediction)
        pooled.append(dict(name=name, seed=seed, profile=profile, metrics=metrics, scenarios=report['scenarios'],
                           checks=check_operating_point(metrics, PROFILES[profile]) if profile in PROFILES else None))
    aggregates, fold_variation = [], []
    for name in SHORTLIST:
        for profile in ('legacy_f1', *PROFILES):
            subset = [r for r in pooled if (r['name'], r['profile']) == (name, profile)]
            aggregates.append(dict(name=name, profile=profile, seeds=SEEDS,
                metrics={m: statistics([r['metrics'][m] for r in subset]) for m in METRICS},
                fp=statistics([r['metrics']['confusion_matrix'][0][1] for r in subset]),
                tp=statistics([r['metrics']['confusion_matrix'][1][1] for r in subset])))
            for seed in SEEDS:
                subset = [r for r in diagnostics if (r['name'], r['seed'], r['profile']) == (name, seed, profile)]
                fold_variation.append(dict(name=name, seed=seed, profile=profile,
                    metrics={m: statistics([r['metrics'][m] for r in subset]) for m in METRICS}))
        for fold in range(1, plan['cv']['outer_folds'] + 1):
            subset = [r for r in results if (r['name'], r['fold']) == (name, fold)]
            arrays = []
            for r in subset:
                with np.load(Path(r['assessment']) / 'predictions_assessment.npz', allow_pickle=False) as a:
                    arrays.append(a['scores'])
            score_identity.append(dict(name=name, fold=fold, all_seed_scores_exactly_equal=all(np.array_equal(arrays[0], a) for a in arrays[1:])))
    return dict(plan_id=PLAN_ID, results=results, per_fold_profiles=diagnostics, pooled_oof=pooled,
        pooled_metrics_across_model_seeds=aggregates, fold_variation_per_seed=fold_variation, score_identity_across_seeds=score_identity,
        population=dict(days=len(primary), accounts=int(primary.account_id.nunique()), independent_groups=int(primary.group.nunique()),
                        endpoints=len(endpoints), positive_endpoints=int(endpoints.label.sum())),
        interpretation='Repeated development folds; model seeds are not new datasets; neither fold nor seed SD is a confidence interval.',
        pooled_rank_metrics='undefined across fitted folds; PR/ROC AUC reported within folds only',
        independent_confirmation=dict(status='unavailable', reason='Previously used CERT test and reused development folds; no newly accepted independent CERT source.',
                                      new_test_scoring=False, confirmatory_claim=False),
        frozen_primary_candidate=plan['primary_candidate'], frozen_primary_profile=plan['primary_profile'], no_post_assessment_selection=True)


def run(days, root, plan_path):
    days, root, plan_path = Path(days).resolve(), Path(root).resolve(), Path(plan_path).resolve()
    plan, source, controls, settings = read_plan(plan_path, root)
    if days.name != plan['daily_preparation_id'] or file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']:
        raise ValueError('Different daily source')
    verify(days)
    frame = pd.read_csv(days / 'account_days.csv')
    if (set(frame.split) != {'train', 'validation', 'test'} or frame.item_id.duplicated().any()
            or frame.groupby('account_id').split.nunique().max() != 1 or frame.groupby('group').split.nunique().max() != 1):
        raise ValueError('Changed primary assignments')
    primary = frame.loc[frame.split.eq('train')].reset_index(drop=True)
    del frame
    output = allocate(root, 'cert_operating_point')
    print('SERIES ' + str(output), flush=True)
    try:
        for name in ('source', 'settings', 'folds', 'jobs', 'results', 'logs', 'candidates', 'reports'):
            (output / name).mkdir()
        save_json(plan, output / 'plan.json')
        save_json(environment_metadata(), output / 'environment.json')
        hashes = {}
        for path in sorted((PROJECT / 'src').rglob('*.py')):
            rel = path.relative_to(PROJECT / 'src')
            target = output / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(rel)] = file_sha256(path)
        save_json(hashes, output / 'execution_source_hashes.json')
        cv = plan['cv']
        for _, provenance in development_folds(primary, n_splits=cv['outer_folds'], calibration_splits=cv['inner_calibration_folds'],
                                                seed=cv['outer_seed'], calibration_seed=cv['inner_calibration_seed_base']):
            name = f"fold_{provenance['fold']}.json"
            verify(source, ['folds/' + name])
            if provenance != load_json(source / 'folds' / name):
                raise ValueError('Development folds changed')
            save_json(provenance, output / 'folds' / name)
        jobs = []
        for name, (policy, model, _) in SHORTLIST.items():
            for seed in SEEDS:
                config = deepcopy(settings[name])
                config['seed'] = seed
                setting = output / 'settings' / f'{name}_{seed}.json'
                save_json(config, setting)
                for fold in range(1, cv['outer_folds'] + 1):
                    original = controls[name, fold]
                    job_id = f'{name}_seed{seed}_fold{fold}'
                    job = dict(job_id=job_id, name=name, policy=policy, model=model, seed=seed, fold=fold,
                        preparation=original['preparation'], config=str(setting), output_root=str(output / 'candidates'),
                        reuse=original['candidate'] if seed == 42 else None,
                        result=str(output / 'results' / (job_id + '.json')))
                    save_json(job, output / 'jobs' / (job_id + '.json'))
                    jobs.append(job)
        save_json(jobs, output / 'declared_jobs.json')
        save_json(dict(primary_candidate=plan['primary_candidate'], primary_profile=plan['primary_profile'],
            reference_seed=42, seeds=SEEDS, shortlist=plan['shortlist'], profiles=PROFILES,
            accepted_before_new_fits=True, no_winner_selection_after_assessment=True,
            independent_confirmation='unavailable; development only'), output / 'frozen_shortlist.json')
        results = []
        for index, job in enumerate(jobs, 1):
            check_source(output, hashes)
            started = time.perf_counter()
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED=str(job['seed']))
            with (output / 'logs' / (job['job_id'] + '.log')).open('w') as stream:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--job',
                                          str(output / 'jobs' / (job['job_id'] + '.json'))],
                                         stdout=stream, stderr=subprocess.STDOUT, env=env)
            if process.returncode:
                save_json(dict(job=job, returncode=process.returncode), output / 'failure.json')
                raise RuntimeError('Step-5 worker failed: ' + job['job_id'])
            result = load_json(Path(job['result']))
            if job['seed'] == 42:
                old = controls[job['name'], job['fold']]
                if result['assessment_metrics']['legacy_f1'] != old['assessment_metrics']:
                    raise ValueError('Seed-42 legacy F1 control changed')
            results.append(result)
            save_json(dict(completed=index, total=len(jobs), last_job=job['job_id']), output / 'progress.json')
            print(json.dumps(dict(completed=index, total=len(jobs), job=job['job_id'], seconds=round(time.perf_counter() - started, 2))), flush=True)
        summary = summarize(results, plan, primary, days, output)
        save_json(summary, output / 'summary.json')
        save_json([dict(name=r['name'], seed=r['seed'], fold=r['fold'], candidate=r['frozen_candidate'],
                       manifest_sha256=r['frozen_candidate_manifest_sha256'], checkpoint_sha256=r['checkpoint_sha256'],
                       operating_points=r['operating_points']) for r in results], output / 'frozen_candidates.json')
        check_source(output, hashes)
        if (file_sha256(source / 'complete.json') != plan['source']['manifest_sha256']
                or file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']):
            raise ValueError('Prerequisite manifest changed')
        finish(output)
    except BaseException as error:
        save_json(dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', reason=str(error)), output / 'status.json')
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--days', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
    elif args.root and args.days and args.plan:
        print(run(args.days, args.root, args.plan))
    else:
        parser.error('--root, --days and --plan are required, or --job')


if __name__ == '__main__':
    main()
