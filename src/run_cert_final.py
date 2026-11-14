"""Final test assessment of every frozen CERT candidate; no fitting or selection."""
import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


def prohibit_fit(*args, **kwargs):
    raise RuntimeError('Fitting or threshold selection is forbidden during frozen final evaluation')


def check_inference_source(series):
    from experiment_tracking import file_sha256
    checked = {}
    for name in ('features.py', 'utils.py', 'train_lstm_autoencoder.py'):
        actual, frozen = PROJECT / 'src' / name, series / 'source' / name
        if file_sha256(actual) != file_sha256(frozen):
            raise ValueError(f'Frozen inference implementation changed: {name}')
        checked[name] = file_sha256(actual)
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in ('inputs', 'score')}
    current = functions(PROJECT / 'src/experiment.py')
    if current != functions(series / 'source/experiment.py') or set(current) != {'inputs', 'score'}:
        raise ValueError('Frozen model-input builder or score function changed')
    checked['experiment.inputs/score AST'] = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    return checked


def preflight(series, days, plan):
    from experiment import verify
    from experiment_tracking import file_sha256, environment_metadata
    from utils import load_json
    if (series.name != plan['series_id'] or days.name != plan['daily_preparation_id']
            or file_sha256(series / 'complete.json') != plan['series_manifest_sha256']
            or file_sha256(series / 'frozen_candidates.json') != plan['frozen_registry_sha256']
            or file_sha256(days / 'complete.json') != plan['daily_manifest_sha256']):
        raise ValueError('Final plan and frozen source identities differ')
    verify(series, ['frozen_candidates.json', 'summary.json', 'plan.json', 'environment.json'])
    # Parent integrity can be checked without opening either prepared test.csv.
    verify(days)
    source_checks = check_inference_source(series)
    previous_environment = load_json(series / 'environment.json')['dependencies']
    now_environment = environment_metadata()['dependencies']
    for dependency in ('numpy', 'pandas', 'scikit-learn', 'torch'):
        if previous_environment[dependency] != now_environment[dependency]:
            raise ValueError(f'Dependency changed since freezing: {dependency}')
    registry = load_json(series / 'frozen_candidates.json')
    if registry['test_evaluated']:
        raise ValueError('The input registry is not a pre-final freeze')
    rows = registry['candidates']
    expected = set(itertools.product(plan['policies'], plan['models'], plan['seeds']))
    actual = [(row['policy'], row['model'], row['seed']) for row in rows]
    if len(rows) != plan['expected_candidates'] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Frozen candidate set differs from the declared complete comparison')
    preparations = set()
    for index, row in enumerate(rows, 1):
        candidate, preparation = Path(row['candidate']), Path(row['preparation'])
        verify(candidate)
        if preparation not in preparations:
            verify(preparation, ['contract.json', 'config.json', 'feature_state.json'])
            preparations.add(preparation)
        selection = load_json(candidate / 'selection.json')
        contract = load_json(candidate / 'contract.json')
        config = load_json(candidate / 'config.json')
        model_file = candidate / ('model.pt' if row['model'] == 'lstm_autoencoder' else 'model.joblib')
        checks = [(candidate/'complete.json', row['candidate_manifest_sha256']),
                  (preparation/'complete.json', row['preparation_manifest_sha256']),
                  (model_file, row['checkpoint_sha256']), (candidate/'config.json', row['config_sha256']),
                  (candidate/'feature_state.json', row['feature_state_sha256'])]
        if any(file_sha256(path) != digest for path, digest in checks):
            raise ValueError('Frozen candidate file binding differs')
        if (selection['threshold'] != row['threshold'] or selection['model'] != row['model']
                or selection['preparation_manifest_sha256'] != row['preparation_manifest_sha256']
                or config['seed'] != row['seed'] or selection['test_accessed'] is not False
                or contract['snapshot_id'] != plan['snapshot_id'] or contract.get('target') != plan['target']
                or contract.get('scope', 'primary_account_holdout') != 'primary_account_holdout'
                or contract != load_json(preparation/'contract.json')):
            raise ValueError('Frozen candidate selection or target differs')
        if row['model'] == 'lstm_autoencoder' and load_json(candidate/'history.json')['selected_epoch'] != row['selected_epoch']:
            raise ValueError('Frozen LSTM epoch differs')
        if index % 5 == 0:
            print(json.dumps({'preflight_candidates_checked':index,'test_not_opened_by_preflight':True}),flush=True)
    return rows, source_checks


def aggregate(results, policies, models):
    import numpy as np
    summary = []
    for policy, model in itertools.product(policies, models):
        subset = [row for row in results if row['policy']==policy and row['model']==model]
        metrics = {}
        for name in ('precision','recall','f1_score','pr_auc','roc_auc','false_positive_rate'):
            values = np.asarray([row['metrics_test'][name] for row in subset],float)
            metrics[name] = {'mean':float(values.mean()),'sample_std':float(values.std(ddof=1)) if len(values)>1 else None,
                             'min':float(values.min()),'max':float(values.max())}
        scenarios = {}
        for scenario in ('1','2','3'):
            reports = [row['error_report']['scenarios'][scenario] for row in subset]
            recalls = [r['recall'] for r in reports if r['recall'] is not None]
            scenarios[scenario] = {'positive_days':reports[0]['positive_days'],
                'independent_groups':reports[0]['independent_groups'],
                'mean_recall':float(np.mean(recalls)) if recalls else None,
                'min_recall':float(np.min(recalls)) if recalls else None,
                'max_recall':float(np.max(recalls)) if recalls else None,
                'mean_false_negative_days':float(np.mean([r['false_negative_days'] for r in reports]))}
        summary.append({'policy':policy,'model':model,'runs':len(subset),'metrics':metrics,'scenarios':scenarios,
                        'median_inference_seconds':float(np.median([r['timing']['inference']['median_seconds'] for r in subset]))})
    return summary


def run(args):
    import numpy as np
    import pandas as pd
    import torch
    from datasets.cert_reporting import evidence_index, error_report
    from experiment import allocate, final_evaluate, compare, finish, verify
    from experiment_tracking import file_sha256, environment_metadata
    from utils import load_json, save_json, compute_classification_metrics
    torch.set_num_threads(1)
    plan = load_json(args.plan)
    if shutil.disk_usage(args.root).free < plan['minimum_free_gib']*1024**3:
        raise ValueError('Insufficient SSD space for final artifacts')
    rows, inference_checks = preflight(args.series, args.days, plan)
    root = allocate(args.root, 'cert_final')
    print('SERIES '+str(root),flush=True)
    (root/'evaluations').mkdir()
    (root/'reports').mkdir()
    shutil.copy2(args.plan,root/'plan.json')
    shutil.copy2(args.series/'frozen_candidates.json',root/'frozen_candidates.json')
    save_json(inference_checks,root/'frozen_inference_checks.json')
    save_json(environment_metadata(),root/'environment.json')
    source_hashes = {}
    for source in (PROJECT/'src').rglob('*.py'):
        relative = source.relative_to(PROJECT/'src')
        target = root/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        source_hashes[str(relative)] = file_sha256(source)
    save_json(source_hashes,root/'source_hashes.json')
    test_paths = {str((Path(row['preparation'])/'test.csv').resolve()) for row in rows}
    exposure = {}
    def audit(event, values):
        if event=='open' and isinstance(values[0],(str,bytes,Path)):
            import os
            value = str(Path(os.fsdecode(values[0])).resolve())
            if value in test_paths and exposure.get('first_prepared_test_access_utc') is None:
                exposure.update(first_prepared_test_access_utc=datetime.now(timezone.utc).isoformat(),
                    scope='frozen final checksum verification and evaluation',plan_id=plan['plan_id'],
                    future_use='used test; no untouched-holdout claim for new configurations')
                save_json(exposure,root/'test_exposure.json')
    sys.addaudithook(audit)
    results = []
    try:
        exposure.update(final_phase_started_utc=datetime.now(timezone.utc).isoformat(),
                        first_prepared_test_access_utc=None,plan_id=plan['plan_id'],
                        future_use='used test; no untouched-holdout claim for new configurations')
        save_json(exposure,root/'test_exposure.json')
        metadata = pd.read_csv(args.days/'account_days.csv',usecols=['item_id','day_key','account_id','group','split','label'])
        metadata = metadata.loc[metadata.split.eq('test')]
        evidence = evidence_index(args.days/'positive_evidence.jsonl')
        reference = None
        for index,row in enumerate(rows,1):
            for relative,digest in source_hashes.items():
                if file_sha256(PROJECT/'src'/relative)!=digest:
                    raise ValueError('Source changed during frozen final execution')
            candidate,preparation = Path(row['candidate']),Path(row['preparation'])
            gc.collect()
            started = time.perf_counter()
            # A scientific guard against accidentally entering a training path.
            with ExitStack() as stack:
                for target in ('experiment.RandomForestClassifier.fit','experiment.IsolationForest.fit',
                               'experiment.select_best_threshold','torch.optim.Adam','datasets.cert_experiments.fit_global'):
                    stack.enter_context(patch(target,side_effect=prohibit_fit))
                evaluation = final_evaluate(candidate,preparation,root/'evaluations')
            elapsed = time.perf_counter()-started
            verify(evaluation)
            lineage = load_json(evaluation/'lineage.json')
            if lineage['threshold']!=row['threshold'] or lineage['fit_or_threshold_selection']:
                raise ValueError('Final changed the frozen selection')
            if file_sha256(candidate/'complete.json')!=row['candidate_manifest_sha256']:
                raise ValueError('Final changed its source candidate')
            predictions = pd.read_csv(evaluation/'predictions_test.csv',float_precision='round_trip')
            with np.load(evaluation/'predictions_test.npz',allow_pickle=False) as outputs:
                if (not np.array_equal(outputs['item_ids'],predictions.item_id)
                        or not np.array_equal(outputs['y'],predictions.y_true)
                        or not np.array_equal(outputs['predictions'],predictions.prediction)
                        or not np.allclose(outputs['scores'],predictions.score,rtol=1e-6,atol=1e-7)):
                    raise ValueError('Exact final outputs and CSV view differ')
                predictions['score']=outputs['scores']
            cohort = predictions[['item_id','y_true']].reset_index(drop=True)
            if (len(cohort)!=plan['expected_test_endpoints'] or cohort.item_id.duplicated().any()
                    or int(cohort.y_true.sum())!=plan['expected_test_positive_days']
                    or (reference is not None and not cohort.equals(reference))):
                raise ValueError('Final cohort does not match the declared aligned population')
            reference = cohort
            with np.load(evaluation/'test_inputs.npz',allow_pickle=False) as arrays:
                if (not np.isfinite(arrays['X']).all() or not np.array_equal(arrays['item_ids'],predictions.item_id)
                        or not np.array_equal(arrays['y'],predictions.y_true)
                        or not np.array_equal(arrays['member_ids'][:,-1],predictions.item_id)):
                    raise ValueError('Saved final input identity or values differ')
            if not np.array_equal((predictions.score.to_numpy()>=row['threshold']).astype(int),predictions.prediction):
                raise ValueError('Final decisions do not use the frozen threshold')
            computed = compute_classification_metrics(predictions.y_true,predictions.prediction,predictions.score)
            stored = load_json(evaluation/'metrics_test.json')
            for key,value in stored.items():
                valid = np.isclose(value,computed[key],rtol=1e-6,atol=1e-7) if isinstance(value,float) else value==computed[key]
                if not valid:
                    raise ValueError('Saved final metrics do not reproduce')
            diagnostics = error_report(predictions,metadata,evidence)
            report = {key:row[key] for key in ('policy','model','seed','candidate','threshold','selected_epoch')}
            report.update(evaluation=str(evaluation),metrics_test=stored,error_report=diagnostics,
                          timing=load_json(evaluation/'timing.json'),evaluation_seconds_with_io=elapsed,
                          evaluation_manifest_sha256=file_sha256(evaluation/'complete.json'),
                          checkpoint_sha256=row['checkpoint_sha256'],refit_guard=True)
            name=f"{row['policy']}_{row['model']}_{row['seed']}.json"
            save_json(report,root/'reports'/name)
            results.append(report)
            save_json({'completed':index,'expected':len(rows),'last_report':name},root/'progress.json')
            print(json.dumps({'completed':index,'policy':row['policy'],'model':row['model'],'seed':row['seed'],
                              'test_f1':stored['f1_score'],'test_precision':stored['precision'],
                              'test_recall':stored['recall'],'seconds':round(elapsed,2)}),flush=True)
        compared = compare([Path(r['evaluation']) for r in results],allow_different_preparations=True)
        summary = {'plan_id':plan['plan_id'],'test_evaluated':True,'test_exposure':exposure,
                   'frozen_registry_sha256':plan['frozen_registry_sha256'],'evaluations':len(results),
                   'statistics':aggregate(results,plan['policies'],plan['models']),'runs':results,
                   'comparison':compared,'no_fitting_or_recalibration':True,
                   'source_series':str(args.series),'test_status':'used; final outcomes cannot guide a new confirmatory result on the same test'}
        save_json(summary,root/'summary.json')
        finish(root)
        verify(root,['summary.json','test_exposure.json','frozen_candidates.json','plan.json','frozen_inference_checks.json'])
        print('COMPLETE '+str(root),flush=True)
        return root
    except BaseException as error:
        save_json({'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                   'reason':str(error),'test_accessed':bool(exposure)},root/'status.json')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--series',type=Path,required=True)
    parser.add_argument('--days',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=PROJECT/'configs/cert_r4_2_final_v1.json')
    run(parser.parse_args())


if __name__=='__main__':
    main()
