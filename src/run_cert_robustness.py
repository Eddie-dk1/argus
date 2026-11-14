"""Run the bounded CERT history comparison, grouped CV and five-seed repeats.

No real final evaluation is performed. Every worker refuses to open test.csv.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def worker(job_path, *, guard_assessment=False):
    blocked = []
    blocked_assessment = []
    assessment_allowed = False
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(args[0])).name
            if name == 'test.csv':
                blocked.append(os.fsdecode(args[0]))
                raise RuntimeError('Real test is closed during robustness work')
            if guard_assessment and name == 'assessment.csv' and not assessment_allowed:
                blocked_assessment.append(os.fsdecode(args[0]))
                raise RuntimeError('Outer assessment is closed while fitting or calibrating')
    sys.addaudithook(audit)

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from experiment import candidate, configured_score, verify
    from experiment_tracking import file_sha256
    from datasets.cert_validation import assess_development
    from train_lstm_autoencoder import LSTMAutoencoder
    from utils import load_json, save_json, measure_inference, compute_classification_metrics
    torch.set_num_threads(1)
    job = load_json(job_path)
    prep, output = Path(job['preparation']), Path(job['output_root'])
    config = load_json(Path(job['config']))
    model_name = job['model']
    started = time.perf_counter()
    path = Path(job['reuse']) if job.get('reuse') else candidate(
        prep, output, model_name, Path(job['config']), diagnostics=model_name == 'lstm_autoencoder')
    candidate_seconds = time.perf_counter() - started
    verify(path)
    assert load_json(path / 'config.json') == config
    contract = load_json(path / 'contract.json')
    selection = load_json(path / 'selection.json')
    assert selection['test_accessed'] is False
    assert selection['preparation_manifest_sha256'] == file_sha256(prep / 'complete.json')
    assert file_sha256(path / 'feature_state.json') == file_sha256(prep / 'feature_state.json')
    validation = pd.read_csv(prep / 'validation.csv')
    train = pd.read_csv(prep / 'train.csv')
    assert set(train.user_id).isdisjoint(set(validation.user_id))
    expected = validation.loc[validation.groupby('user_id').cumcount() >= contract['sequence_length'] - 1]
    predictions = pd.read_csv(path / 'predictions_validation.csv')
    assert np.array_equal(predictions.item_id, expected.event_id)
    assert np.array_equal(predictions.y_true, expected.label)
    with np.load(path / 'validation_inputs.npz', allow_pickle=False) as arrays:
        x, y, ids, members = arrays['X'], arrays['y'], arrays['item_ids'], arrays['member_ids']
    assert np.isfinite(x).all() and np.array_equal(ids, predictions.item_id) and np.array_equal(y, predictions.y_true)
    expected_members = []
    length = contract['sequence_length']
    for _, group in validation.groupby('user_id', sort=False):
        values = group.event_id.to_numpy()
        expected_members.extend(values[i-length+1:i+1] for i in range(length-1, len(values)))
    assert np.array_equal(members, np.asarray(expected_members))
    if model_name == 'lstm_autoencoder':
        expected_x = validation.set_index('event_id').loc[members.ravel(), contract['sequence_columns']].to_numpy(np.float32).reshape(x.shape)
        model = LSTMAutoencoder(len(contract['sequence_columns']), config[model_name]['hidden_size'])
        model.load_state_dict(torch.load(path / 'model.pt', map_location='cpu', weights_only=True))
        history = load_json(path / 'history.json')
        best = max(history['epochs'], key=lambda row: (row['validation']['f1_score'], row['validation']['recall'], -row['validation']['false_positive_rate']))
        assert best['epoch'] == history['selected_epoch']
    else:
        expected_x = expected[contract['tabular_columns']].to_numpy()
        model = joblib.load(path / 'model.joblib')
        history = None
    assert np.array_equal(x, expected_x)
    del expected_x, expected_members
    legacy_object_ids = False
    with np.load(path / 'train_inputs.npz', allow_pickle=False) as arrays:
        tx, ty = arrays['X'], arrays['y']
        try:
            tids = arrays['item_ids']
        except ValueError:
            if not job.get('reuse'):
                raise
            # Only manifest-verified locally generated legacy bundles are reused.
            # This is the same trusted-artifact boundary as joblib model reload.
            with np.load(path / 'train_inputs.npz', allow_pickle=True) as legacy:
                tids = legacy['item_ids']
            assert tids.dtype.kind == 'O' and all(isinstance(item, str) for item in tids)
            legacy_object_ids = True
        assert np.isfinite(tx).all()
        if model_name == 'lstm_autoencoder':
            tmembers = arrays['member_ids']
            assert (train.set_index('event_id').label.loc[tmembers.ravel()] == 0).all()
            assert np.array_equal(tids, tmembers[:, -1]) and (ty == 0).all()
            count = sum(int((np.convolve(group.label.to_numpy(), np.ones(length, int), 'valid') == 0).sum())
                        for _, group in train.groupby('user_id') if len(group) >= length)
            assert count == len(tx)
            lookup = train.set_index('event_id')
            for start in range(0, len(tx), 1024):
                keys = tmembers[start:start+1024]
                actual = lookup.loc[keys.ravel(), contract['sequence_columns']].to_numpy(np.float32).reshape(len(keys), length, -1)
                assert np.array_equal(tx[start:start+len(keys)], actual)
        else:
            expected_train = train if model_name in ('random_forest', 'hist_gradient_boosting') else train.loc[train.label.eq(0)]
            assert np.array_equal(tids, expected_train.event_id)
            assert np.array_equal(ty, expected_train.label)
            assert np.array_equal(tx, expected_train[contract['tabular_columns']].to_numpy())
        train_count = len(tx)
    values, timing = measure_inference(lambda: configured_score(model, x, model_name, config), record_count=len(y), warmup_runs=1, measured_runs=5)
    assert np.allclose(values, predictions.score, rtol=1e-6 if history else 1e-12, atol=1e-7 if history else 1e-12)
    assert np.array_equal(values >= selection['threshold'], predictions.prediction)
    metrics = compute_classification_metrics(y, predictions.prediction, values)
    stored = load_json(path / 'metrics_validation.json')
    for key, value in stored.items():
        assert np.isclose(value, metrics[key], rtol=1e-6, atol=1e-7) if isinstance(value, float) else value == metrics[key]
    grid = load_json(path / 'threshold_search.json')
    best_threshold = max(grid, key=lambda row: (row['f1_score'], row['recall'], -row['false_positive_rate'], -row['threshold']))
    assert best_threshold['threshold'] == selection['threshold']
    model_file = path / ('model.pt' if history else 'model.joblib')
    result = {key: job[key] for key in ('job_id', 'policy', 'model', 'seed', 'phase')}
    result.update(candidate=str(path), preparation=str(prep), reused=bool(job.get('reuse')),
                  candidate_manifest_sha256=file_sha256(path / 'complete.json'),
                  preparation_manifest_sha256=file_sha256(prep / 'complete.json'),
                  checkpoint_sha256=file_sha256(model_file), config_sha256=file_sha256(path / 'config.json'),
                  feature_state_sha256=file_sha256(path / 'feature_state.json'), threshold=selection['threshold'],
                  selected_percentile=best_threshold['percentile'], selected_epoch=history['selected_epoch'] if history else None,
                  stopping_reason=history.get('stopping_reason') if history else None,
                  completed_epochs=len(history['epochs']) if history else None, train_rows_or_windows=train_count,
                  validation_metrics=stored, validation_inference=timing,
                  cohort_sha256=hashlib.sha256(json.dumps(list(zip(ids.tolist(), y.tolist()))).encode()).hexdigest(),
                  candidate_seconds_including_io_selection_and_diagnostics=candidate_seconds,
                  peak_rss_bytes_worker=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  legacy_object_id_array=legacy_object_ids,
                  checks={'hashes': True, 'exact_numerical_inputs': True, 'exact_window_members': True,
                          'all_training_members_background': True if model_name == 'lstm_autoencoder' else None,
                          'reload_scores_and_decisions': True, 'threshold_grid_verified': True,
                          'test_open_guard': True, 'test_open_attempts': blocked})
    if job['phase'] == 'cv':
        before = file_sha256(path / 'complete.json')
        # candidate() has completed and its checkpoint/threshold are verified.
        # Only now may this process open the outer labels and numerical inputs.
        assessment_allowed = True
        assessment = assess_development(path, prep, output)
        verify(assessment)
        assert file_sha256(path / 'complete.json') == before
        result.update(assessment=str(assessment), assessment_metrics=load_json(assessment / 'metrics_assessment.json'),
                      assessment_manifest_sha256=file_sha256(assessment / 'complete.json'), fold=job['fold'])
    assert blocked == []
    assert blocked_assessment == []
    result['checks'].update(assessment_open_guard=guard_assessment,
                           assessment_open_attempts_during_fit=blocked_assessment,
                           assessment_after_candidate_complete=job['phase'] == 'cv')
    save_json(result, Path(job['result']))
    print(json.dumps({'job': job['job_id'], 'candidate': path.name, 'validation_f1': stored['f1_score'],
                      'assessment_f1': result.get('assessment_metrics', {}).get('f1_score')}), flush=True)


def summarize(results, key):
    import numpy as np
    rows = []
    for policy in ('causal_train_v1', 'global_only_v2'):
        for model in sorted({r['model'] for r in results}):
            subset = [r for r in results if r['policy'] == policy and r['model'] == model]
            if not subset:
                continue
            stats = {}
            for metric in ('precision', 'recall', 'f1_score', 'pr_auc', 'roc_auc', 'false_positive_rate'):
                values = np.asarray([r[key][metric] for r in subset], float)
                stats[metric] = {'mean': float(values.mean()), 'sample_std': float(values.std(ddof=1)) if len(values) > 1 else None,
                                 'min': float(values.min()), 'max': float(values.max())}
            rows.append({'policy': policy, 'model': model, 'runs': len(subset), 'metrics': stats})
    return rows


def run(args):
    import numpy as np
    import pandas as pd
    from datasets.cert_experiments import prepare_cert, prepare_cert_frame
    from datasets.cert_validation import development_folds, write_train_context
    from experiment import allocate, finish, verify
    from experiment_tracking import file_sha256, environment_metadata
    from utils import load_json, save_json
    plan = load_json(args.plan)
    if plan['plan_id'] != 'cert_r4_2_history_robustness_v2':
        raise ValueError('Unsupported bounded robustness plan')
    if shutil.disk_usage(args.root).free < plan['budget']['minimum_free_gib'] * 1024**3:
        raise ValueError('Insufficient SSD space for the declared robustness budget')
    assert args.preparation.name == plan['primary_preparation_id']
    assert file_sha256(args.preparation / 'complete.json') == plan['primary_preparation_manifest_sha256']
    assert args.days.name == plan['daily_preparation_id']
    verify(args.days)
    verify(args.preparation, ['contract.json', 'config.json', 'feature_state.json', 'train.csv', 'validation.csv'])
    for settings in plan['configs'].values():
        assert file_sha256(PROJECT / settings['path']) == settings['sha256']
    root = allocate(args.root, 'cert_robustness')
    print('SERIES ' + str(root), flush=True)
    for directory in ('settings', 'jobs', 'preparations', 'candidates', 'results', 'logs', 'source'):
        (root / directory).mkdir()
    shutil.copy2(args.plan, root / 'plan.json')
    source_hashes = {}
    for source in (PROJECT / 'src').rglob('*.py'):
        relative = source.relative_to(PROJECT / 'src')
        target = root / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(relative)] = file_sha256(source)
    save_json(source_hashes, root / 'execution_source_hashes.json')
    save_json(environment_metadata(), root / 'environment.json')
    configs = {}
    for policy, settings in plan['configs'].items():
        configs[policy] = load_json(PROJECT / settings['path'])
        shutil.copy2(PROJECT / settings['path'], root / 'settings' / (policy + '.json'))
    results = []
    previous = args.reuse_series
    if previous is not None:
        if load_json(previous / 'plan.json') != plan:
            raise ValueError('Cannot reuse a different robustness plan')
        save_json({'previous_series': str(previous), 'previous_status': load_json(previous / 'status.json'),
                   'reason': 'storage-format verification correction; model/config/preparation values unchanged'},
                  root / 'continuation.json')
        shutil.copy2(previous / 'execution_source_hashes.json', root / 'previous_execution_source_hashes.json')

    def execute(job):
        for relative, digest in source_hashes.items():
            assert file_sha256(PROJECT / 'src' / relative) == digest, 'Source changed during frozen execution'
        job_path = root / 'jobs' / (job['job_id'] + '.json')
        job.update(output_root=str(root / 'candidates'), result=str(root / 'results' / (job['job_id'] + '.json')))
        save_json(job, job_path)
        started = time.perf_counter()
        with (root / 'logs' / (job['job_id'] + '.log')).open('w') as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(args.root),
                                       '--job', str(job_path)], stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))
        elapsed = time.perf_counter() - started
        if completed.returncode:
            save_json({'failed_job': job, 'wall_seconds': elapsed, 'returncode': completed.returncode}, root / 'failure.json')
            raise RuntimeError(f"Job failed: {job['job_id']}; inspect saved log")
        result = load_json(Path(job['result']))
        results.append(result)
        save_json({'completed': len(results), 'last_job': job['job_id'], 'last_worker_seconds': elapsed}, root / 'progress.json')
        print(json.dumps({'completed': len(results), 'job': job['job_id'], 'seconds': round(elapsed, 2),
                          'validation_f1': result['validation_metrics']['f1_score'],
                          'assessment_f1': result.get('assessment_metrics', {}).get('f1_score')}), flush=True)

    try:
        cv = plan['cv']
        previous_cv = sorted((previous / 'results').glob('cv_*.json')) if previous is not None else []
        if len(previous_cv) == cv['new_candidates']:
            for location in previous_cv:
                result = load_json(location)
                assert result['phase'] == 'cv' and result['seed'] == cv['model_seed']
                for field, digest in [('candidate', 'candidate_manifest_sha256'),
                                      ('preparation', 'preparation_manifest_sha256'),
                                      ('assessment', 'assessment_manifest_sha256')]:
                    artifact = Path(result[field])
                    verify(artifact)
                    assert file_sha256(artifact / 'complete.json') == result[digest]
                result['reused_verified_assessment'] = True
                save_json(result, root / 'results' / location.name)
                results.append(result)
            assert {(r['fold'], r['policy']) for r in results} == {
                (fold, policy) for fold in range(1, cv['outer_folds']+1) for policy in cv['policies']}
            print('REUSED 6 verified CV assessments', flush=True)
        else:
            # Outer assessment never enters candidate or preprocessing fitting.
            frame = pd.read_csv(args.days / 'account_days.csv')
            context = root / 'development_train_context.jsonl.gz'
            write_train_context(frame, args.days / 'day_context.jsonl.gz', context)
            contract = load_json(args.days / 'contract.json')
            for folded, provenance in development_folds(frame, n_splits=cv['outer_folds'],
                    calibration_splits=cv['inner_calibration_folds'], seed=cv['outer_seed'],
                    calibration_seed=cv['inner_calibration_seed_base']):
                fold = provenance['fold']
                for policy in cv['policies']:
                    prep = prepare_cert_frame(folded, context, contract, configs[policy], args.days,
                                              root / 'preparations', development=provenance)
                    execute({'job_id': f'cv_{fold}_{policy}', 'phase': 'cv', 'fold': fold,
                             'policy': policy, 'model': cv['model'], 'seed': cv['model_seed'],
                             'preparation': str(prep), 'config': str(root / 'settings' / (policy + '.json'))})
            del frame, folded
        previous_preparations = []
        if previous is not None:
            for location in (previous / 'preparations').glob('cert_preparation_*'):
                value = load_json(location / 'contract.json')
                if value.get('scope') == 'primary_account_holdout' and value['history_policy'] == configs['global_only_v2']['features']['history_policy']:
                    verify(location, ['contract.json', 'config.json', 'feature_state.json', 'train.csv', 'validation.csv'])
                    previous_preparations.append(location)
        if len(previous_preparations) > 1:
            raise ValueError('Ambiguous reusable primary preparation')
        new_prep = previous_preparations[0] if previous_preparations else prepare_cert(
            args.days, root / 'settings/global_only_v2.json', root / 'preparations')
        preparations = {'causal_train_v1': args.preparation, 'global_only_v2': new_prep}
        for policy, prep in preparations.items():
            for seed in plan['seeds']:
                config = dict(configs[policy], seed=seed)
                config_path = root / 'settings' / f'{policy}_{seed}.json'
                save_json(config, config_path)
                for model in plan['models']:
                    reuse = args.root / plan['reuse_seed42'][model] if policy == 'causal_train_v1' and seed == 42 else None
                    if reuse is None and previous is not None:
                        matches = []
                        for location in (previous / 'candidates').glob('candidate_*'):
                            if not (location / 'complete.json').exists():
                                continue
                            selection = load_json(location / 'selection.json')
                            if (selection['model'] == model and selection['preparation_manifest_sha256'] == file_sha256(prep / 'complete.json')
                                    and load_json(location / 'config.json') == config):
                                matches.append(location)
                        if len(matches) > 1:
                            raise ValueError('Ambiguous reusable candidate; no metric-based choice allowed')
                        reuse = matches[0] if matches else None
                    execute({'job_id': f'primary_{policy}_{seed}_{model}', 'phase': 'primary',
                             'policy': policy, 'model': model, 'seed': seed, 'preparation': str(prep),
                             'config': str(config_path), 'reuse': str(reuse) if reuse else None})
        primary = [r for r in results if r['phase'] == 'primary']
        cv_results = [r for r in results if r['phase'] == 'cv']
        assert len(primary) == plan['primary_runs'] and len(cv_results) == cv['new_candidates']
        assert len({r['cohort_sha256'] for r in primary}) == 1
        paired = []
        for model in plan['models']:
            for seed in plan['seeds']:
                pair = {r['policy']: r for r in primary if r['model'] == model and r['seed'] == seed}
                paired.append({'model': model, 'seed': seed, 'new_minus_old': {
                    metric: pair['global_only_v2']['validation_metrics'][metric] - pair['causal_train_v1']['validation_metrics'][metric]
                    for metric in ('f1_score', 'pr_auc', 'recall', 'false_positive_rate')}})
        summary = {'plan_id': plan['plan_id'], 'test_evaluated': False, 'primary_runs': len(primary),
                   'cv_runs': len(cv_results), 'primary_statistics': summarize(primary, 'validation_metrics'),
                   'cv_statistics': summarize(cv_results, 'assessment_metrics'),
                   'paired_seed_differences': paired, 'runs': results,
                   'limitations': ['Model seeds share one account split, not independent datasets.',
                                   'CV outer folds are dependent and scenario 3 has one primary-train group.',
                                   'Validation informed the original shift diagnosis; final remains untouched.']}
        save_json(summary, root / 'summary.json')
        frozen = [{key: r[key] for key in ('policy', 'model', 'seed', 'candidate', 'candidate_manifest_sha256',
                   'preparation', 'preparation_manifest_sha256', 'checkpoint_sha256', 'config_sha256',
                   'feature_state_sha256', 'threshold', 'selected_epoch')} for r in primary]
        save_json({'scope': 'all configurations and seeds retained; no best-seed selection',
                   'test_evaluated': False, 'candidates': frozen}, root / 'frozen_candidates.json')
        finish(root)
        verify(root, ['summary.json', 'frozen_candidates.json', 'plan.json', 'execution_source_hashes.json'])
        print('COMPLETE ' + str(root), flush=True)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'reason': str(error)}, root / 'status.json')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing SSD artifact root')
    parser.add_argument('--plan', type=Path, default=PROJECT / 'configs/cert_r4_2_robustness_v2.json')
    parser.add_argument('--days', type=Path)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--reuse-series', type=Path, help='Reuse completed checksum-verified artifacts of the same plan in a new series')
    args = parser.parse_args()
    if args.job:
        worker(args.job)
    elif args.days is None or args.preparation is None:
        parser.error('--days and --preparation are required')
    else:
        run(args)


if __name__ == '__main__':
    main()
