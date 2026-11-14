"""S3 part 2: exact saved-input replay and descriptive development diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from itertools import groupby
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

# Match the original single-thread workers before importing numerical libraries.
if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import pandas as pd

from cert_operating_point import check_operating_point
from datasets.cert_reporting import evidence_index
from experiment import allocate, finish, verify
from experiment_tracking import environment_metadata, file_sha256, payload_sha256
from frozen_diagnostics import (METRICS, cohort_diagnostics, history_flags, observed_history,
                                statistics, variation_reports)
from frozen_diagnostic_report import write_report
from inference import PreparedBatch, batch_from_frame, load_frozen_run
from utils import compute_classification_metrics, load_json, save_json


PLAN_ID = 's3_frozen_replay_diagnostics_v1'
PROJECT = Path(__file__).resolve().parents[1]
TEST_FILES = {'test.csv', 'test_inputs.npz', 'predictions_test.npz', 'predictions_test.csv', 'metrics_test.json'}


def exact(actual, expected, description, *, dtype=False):
    if (not np.array_equal(actual, expected) or np.shape(actual) != np.shape(expected)
            or dtype and np.asarray(actual).dtype != np.asarray(expected).dtype):
        raise ValueError(f'Exact replay mismatch: {description}')


def checked_artifact(path, digest, names=None):
    path = Path(path).resolve()
    if file_sha256(path / 'complete.json') != digest:
        raise ValueError(f'Artifact manifest changed: {path}')
    return verify(path, names)


def validate_coverage(rows, expected, key):
    actual = [key(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Missing or duplicate declared fold/seed/model; incomplete diagnostics prohibited')


def read_catalog(model_variants, operating_points, plan):
    if (plan['plan_id'] != PLAN_ID or plan['budget'] != dict(research_fits=0, threshold_selections=0, search_trials=0)
            or plan['replay']['model_variants'] != 'all' or plan['replay']['operating_profiles'] != 'all_saved'
            or plan['diagnosis']['pooled_rank_metrics'] is not False
            or plan['diagnosis']['independent_confirmation'] is not False):
        raise ValueError('Unsupported frozen diagnosis protocol')
    catalog, series = [], {}
    current = environment_metadata()
    for name, location in (('model_variants', model_variants), ('operating_points', operating_points)):
        location = Path(location).resolve()
        spec = plan['sources'][name]
        if location.name != spec['id']:
            raise ValueError('Diagnostic source ID differs from declared plan')
        checked_artifact(location, spec['manifest_sha256'], ['summary.json', 'plan.json', 'environment.json'])
        summary, original = load_json(location / 'summary.json'), load_json(location / 'plan.json')
        old_environment = load_json(location / 'environment.json')
        if (old_environment['dependencies'] != current['dependencies']
                or old_environment['machine'] != current['machine']
                or original['budget']['cpu_threads'] != plan['replay']['cpu_threads']):
            raise ValueError('Replay requires original dependency versions, CPU architecture and thread budget')
        rows = summary['results']
        if len(rows) != spec['expected_runs']:
            raise ValueError('Missing declared source results')
        folds = range(1, original['cv']['outer_folds'] + 1)
        if name == 'model_variants':
            expected = {(fold, policy, method, original['cv']['model_seed']) for fold in folds
                        for policy in original['configs'] for method in original['methods']}
            key = lambda r: (r['fold'], r['policy'], r['model'], r['seed'])
        else:
            expected = {(fold, variant, seed) for fold in folds for variant in original['shortlist']
                        for seed in original['model_seeds']}
            key = lambda r: (r['fold'], r['name'], r['seed'])
            if original['reference_seed'] != plan['replay']['operating_reference_seed']:
                raise ValueError('Reference replay seed differs')
        validate_coverage(rows, expected, key)
        for fold in folds:
            verify(location, [f'folds/fold_{fold}.json'])
        series[name] = dict(path=location, plan=original, summary=summary)
        catalog.extend(dict(series=name, source=location, row=row) for row in rows)
    left, right = series['model_variants'], series['operating_points']
    if (right['plan']['source']['manifest_sha256'] != plan['sources']['model_variants']['manifest_sha256']
            or left['plan']['daily_manifest_sha256'] != right['plan']['daily_manifest_sha256']
            or left['summary']['population'] != right['summary']['population']):
        raise ValueError('Source series have incompatible populations or lineage')
    for fold in range(1, left['plan']['cv']['outer_folds'] + 1):
        if load_json(left['path'] / f'folds/fold_{fold}.json') != load_json(right['path'] / f'folds/fold_{fold}.json'):
            raise ValueError('Source development folds differ')
    return catalog, series


def read_primary_metadata(days, digest):
    """Read co-located reporting columns, retaining original primary train only."""
    days = Path(days)
    checked_artifact(days, digest, ['account_days.csv', 'positive_evidence.jsonl', 'contract.json'])
    columns = ['item_id', 'day_key', 'account_id', 'date', 'split', 'group', 'label']
    parts = [chunk.loc[chunk.split.eq('train')].copy() for chunk in
             pd.read_csv(days / 'account_days.csv', usecols=columns, chunksize=50000)]
    primary = pd.concat(parts, ignore_index=True)
    if (primary.empty or primary.item_id.isna().any() or primary.item_id.duplicated().any()
            or primary.groupby('account_id')['group'].nunique().max() != 1):
        raise ValueError('Invalid primary training metadata')
    allowed = set(primary.day_key)
    evidence = {k: v for k, v in evidence_index(days / 'positive_evidence.jsonl').items() if k in allowed}
    return primary, evidence


class PreparationContext:
    """One fold/policy at a time, retaining CPU memory bounds and builder identity."""
    def __init__(self, path, digest, primary, folds, *, inverse_atol=1e-5):
        self.path = Path(path)
        self.files = checked_artifact(self.path, digest, ['contract.json', 'feature_state.json',
            'development.json', 'validation.csv', 'assessment.csv'])
        self.contract = load_json(self.path / 'contract.json')
        self.state = load_json(self.path / 'feature_state.json')
        self.digest = digest
        if (self.contract.get('scope') != 'development_group_cv'
                or load_json(self.path / 'development.json') != folds):
            raise ValueError('Diagnosis requires the exact original development fold')
        self.frames, self.history, self.flags, self.batches = {}, {}, {}, {}
        self.expected_endpoints = {}
        roles = folds['account_roles']
        for role, stored in (('calibration', 'validation'), ('assessment', 'assessment')):
            frame = pd.read_csv(self.path / f'{stored}.csv')  # Preserve original parser/dtype.
            expected = primary.loc[primary.account_id.map(roles).eq(stored)].sort_values(['account_id', 'date', 'item_id'])
            exact(frame.event_id.to_numpy(str), expected.item_id.to_numpy(str), f'{role} preparation rows/order')
            exact(frame.user_id.to_numpy(str), expected.account_id.to_numpy(str), f'{role} accounts')
            exact(frame.label.to_numpy(), expected.label.to_numpy(), f'{role} labels')
            exact(pd.to_datetime(frame.timestamp).to_numpy(), pd.to_datetime(expected.date).to_numpy(), f'{role} timestamps')
            if not frame['user_id'].map(roles).eq(stored).all():
                raise ValueError('Account crosses original fold role')
            self.frames[role] = frame
            self.history[role] = observed_history(frame)
            self.flags[role] = history_flags(frame, self.state, inverse_atol=inverse_atol)
            eligible = expected.groupby('account_id').cumcount().ge(self.contract['sequence_length'] - 1)
            self.expected_endpoints[role] = expected.loc[eligible]

    def batch(self, runtime, role):
        if (runtime.spec.preparation_id != self.path.name or runtime.spec.preparation_manifest_sha256 != self.digest
                or runtime.spec.feature_state_sha256 != self.files['feature_state.json']
                or load_json(runtime.artifact_path / 'contract.json') != self.contract):
            raise ValueError('Runtime and reporting preparation differ')
        key = (role, runtime.spec.model == 'lstm_autoencoder')
        if key not in self.batches:
            self.batches[key] = batch_from_frame(self.frames[role], contract=self.contract, model=runtime.spec.model,
                preparation_manifest_sha256=self.digest, feature_state_sha256=self.files['feature_state.json'])
        built = self.batches[key]
        exact(built.item_ids, self.expected_endpoints[role].item_id.to_numpy(str), f'{role} endpoint order')
        return built


def bound_inputs(artifact, filename, files, runtime, context, role):
    if filename not in files:
        raise ValueError(f'Missing bound numerical inputs: {filename}')
    with np.load(Path(artifact) / filename, allow_pickle=False) as arrays:
        x, y, ids, members = (arrays[k] for k in ('X', 'y', 'item_ids', 'member_ids'))
    built = context.batch(runtime, role)
    for key, actual, expected in (('X', x, built.X), ('IDs', ids, built.item_ids), ('members', members, built.member_ids)):
        exact(actual, expected, f'{role} {key}', dtype=key == 'X')
    exact(y, context.expected_endpoints[role].label.to_numpy(), f'{role} endpoint labels')
    batch = PreparedBatch(X=x, feature_columns=built.feature_columns, item_ids=ids, member_ids=members,
        preparation_id=built.preparation_id, preparation_manifest_sha256=built.preparation_manifest_sha256,
        feature_state_sha256=built.feature_state_sha256)
    runtime.validate_batch(batch)
    return batch, y


def process_entry(entry, context, primary, evidence, output, *, reference_seed=42):
    """Replay requested models only; diagnose every saved model/seed/profile."""
    row, series = entry['row'], entry['series']
    if (row['preparation_manifest_sha256'] != context.digest
            or Path(row['preparation']).resolve() != context.path.resolve()):
        raise ValueError('Catalog and reporting preparation differ')
    operating = series == 'operating_points'
    artifact = Path(row['frozen_candidate'] if operating else row['candidate'])
    artifact_digest = row['frozen_candidate_manifest_sha256'] if operating else row['candidate_manifest_sha256']
    runtime = load_frozen_run(artifact, profile='legacy_f1', expected_manifest_sha256=artifact_digest)
    model = 'lstm_autoencoder' if row['model'] == 'lstm_endpoint' else row['model']
    if (runtime.spec.model != model or runtime.spec.scope != 'development_group_cv'
            or load_json(artifact / 'config.json')['seed'] != row['seed']):
        raise ValueError('Runtime model/seed/scope differs from source catalog')
    if operating:
        choices = load_json(artifact / 'operating_points.json')
        if choices != row['operating_points']:
            raise ValueError('Operating choices changed since the saved assessment')
    else:
        choices = {'legacy_f1': dict(threshold=runtime.spec.threshold,
                                   calibration_metrics=load_json(artifact / 'metrics_validation.json'))}
        if runtime.spec.threshold != row['threshold']:
            raise ValueError('Candidate threshold differs from saved summary')
    assessment = Path(row['assessment'])
    assessment_files = checked_artifact(assessment, row['assessment_manifest_sha256'])
    lineage = load_json(assessment / 'lineage.json')
    key = 'frozen_candidate_manifest_sha256' if operating else 'candidate_manifest_sha256'
    if (lineage[key] != artifact_digest or lineage['preparation_manifest_sha256'] != context.digest
            or lineage.get('scope') != 'development_group_cv' or lineage['refit_or_recalibration'] is not False):
        raise ValueError('Assessment lineage does not bind frozen model and preparation')
    if not operating and (lineage['candidate_id'] != artifact.name
            or lineage['preparation_id'] != context.path.name or lineage['model'] != model
            or lineage['threshold'] != runtime.spec.threshold
            or lineage['target'] != context.contract['target']):
        raise ValueError('Assessment model/target/threshold lineage differs')
    calibration, yc = bound_inputs(artifact, 'validation_inputs.npz', runtime.artifact_hashes, runtime, context, 'calibration')
    outer, yo = bound_inputs(assessment, 'assessment_inputs.npz', assessment_files, runtime, context, 'assessment')
    with np.load(assessment / 'predictions_assessment.npz', allow_pickle=False) as arrays:
        stored = {k: arrays[k] for k in arrays.files}
    exact(stored['item_ids'], outer.item_ids, 'assessment prediction IDs')
    exact(stored['y'], yo, 'assessment prediction labels')
    if str(stored['scores'].dtype) != lineage['score_dtype']:
        raise ValueError('Saved score dtype differs from lineage')
    if operating:
        with np.load(artifact / 'predictions_calibration.npz', allow_pickle=False) as arrays:
            saved_calibration = {k: arrays[k] for k in arrays.files}
    else:
        view = pd.read_csv(artifact / 'predictions_validation.csv', float_precision='round_trip')
        saved_calibration = dict(item_ids=view.item_id.to_numpy(str), y=view.y_true.to_numpy(),
            scores=view.score.to_numpy(dtype=stored['scores'].dtype), predictions=view.prediction.to_numpy())
    exact(saved_calibration['item_ids'], calibration.item_ids, 'calibration prediction IDs')
    exact(saved_calibration['y'], yc, 'calibration prediction labels')
    replay = not operating or row['seed'] == reference_seed
    scores = runtime.score(outer) if replay else stored['scores']
    exact(scores, stored['scores'], 'assessment scores', dtype=True)
    if replay:
        (output / 'replays').mkdir(exist_ok=True)
    replay_values = dict(item_ids=outer.item_ids, member_ids=outer.member_ids, y=yo, scores=scores)
    records = []
    identity = f"{series}_{row['job_id']}"
    for profile, choice in choices.items():
        selected = runtime if profile == 'legacy_f1' else load_frozen_run(artifact, profile=profile,
                                                                        expected_manifest_sha256=artifact_digest)
        decision_key = profile if operating else 'predictions'
        predicted = selected.predict(outer) if replay else stored[decision_key]
        exact(predicted, stored[decision_key], f'assessment {profile} decisions', dtype=True)
        exact(predicted, scores >= choice['threshold'], f'assessment {profile} threshold rule')
        cal_predicted = saved_calibration[decision_key]
        exact(cal_predicted, saved_calibration['scores'] >= choice['threshold'], f'calibration {profile} threshold rule')
        metrics = compute_classification_metrics(yo, predicted, scores)
        cal_metrics = compute_classification_metrics(yc, cal_predicted, saved_calibration['scores'])
        expected_metrics = row['assessment_metrics'][profile] if operating else row['assessment_metrics']
        file_metrics = load_json(assessment / 'metrics_assessment.json')
        if (metrics != expected_metrics or metrics != (file_metrics[profile] if operating else file_metrics)
                or cal_metrics != choice['calibration_metrics']
                or not operating and cal_metrics != row['validation_metrics']):
            raise ValueError('Saved calibration/assessment metrics do not reproduce')
        if operating and profile != 'legacy_f1':
            checks = {role: check_operating_point(m, choice) for role, m in (
                ('calibration', cal_metrics), ('assessment', metrics))}
            if (checks['assessment'] != row['operating_checks'][profile]
                    or checks['calibration']['both_met'] != choice['eligible_on_calibration']):
                raise ValueError('Saved operating feasibility/checks changed')
        else:
            checks = dict(calibration=None, assessment=None)
        record = dict(id=f'{identity}_{profile}', series=series, method=row.get('name', row['model']),
            feature_policy=selected.spec.feature_policy, policy=row['policy'], seed=row['seed'], fold=row['fold'],
            profile=profile, threshold=choice['threshold'], model=asdict(selected.spec),
            evaluation_source=dict(candidate=str(artifact), candidate_manifest_sha256=artifact_digest,
                assessment=str(assessment), assessment_manifest_sha256=row['assessment_manifest_sha256'],
                preparation=str(context.path), preparation_manifest_sha256=context.digest,
                calibration_scores='exact NPZ' if operating else 'legacy CSV restored to saved score dtype',
                assessment_inputs_sha256=assessment_files['assessment_inputs.npz'],
                feature_order_sha256=payload_sha256(list(selected.spec.feature_columns))),
            evidence_level='reused development; preservation/diagnosis; no independent confirmation',
            replay=dict(performed=replay, scores_exact=True if replay else None,
                decisions_exact=True if replay else None, numerical_inputs_exact=True, item_order_exact=True,
                feature_order_exact=True, ordered_window_members_exact=True, cpu_threads=1,
                dtype=str(scores.dtype), lstm_batch_size=selected.spec.batch_size),
            operating_checks=checks,
            saved_metric_values=dict(calibration=cal_metrics, assessment=metrics))
        for role, batch, y, value, decisions in (
            ('calibration', calibration, yc, saved_calibration['scores'], cal_predicted),
            ('assessment', outer, yo, scores, predicted)):
            record[role] = cohort_diagnostics(batch.item_ids, y, value, decisions, primary, evidence,
                history=context.history[role], flags=context.flags[role])
        records.append(record)
        replay_values[decision_key] = predicted
    if replay:
        np.savez_compressed(output / 'replays' / f'{identity}.npz', **replay_values)
    return records


def pooled_reports(records, primary, evidence, contexts, source_series):
    groups = defaultdict(list)
    for record in records:
        groups[(record['series'], record['method'], record['policy'], record['seed'], record['profile'])].append(record)
    pooled = []
    for (series, method, policy, seed, profile), rows in sorted(groups.items()):
        parts, histories, flag_parts = [], [], defaultdict(list)
        for row in sorted(rows, key=lambda r: r['fold']):
            source = Path(row['evaluation_source']['assessment'])
            with np.load(source / 'predictions_assessment.npz', allow_pickle=False) as arrays:
                parts.append(pd.DataFrame(dict(item_id=arrays['item_ids'], y=arrays['y'],
                    prediction=arrays[profile if series == 'operating_points' else 'predictions'])))
            context_path = contexts[row['evaluation_source']['preparation']]
            with np.load(context_path, allow_pickle=False) as arrays:
                histories.append(pd.Series(arrays['prior30'], index=arrays['item_ids']))
                for name in ('history_missing', 'cold_start', 'file_usb_profile_missing'):
                    if name in arrays.files:
                        flag_parts[name].append(pd.Series(arrays[name], index=arrays['item_ids']))
        frame = pd.concat(parts, ignore_index=True).sort_values('item_id').reset_index(drop=True)
        ordered = primary.sort_values(['account_id', 'date', 'item_id'])
        length = rows[0]['model']['sequence_length']
        expected = ordered.loc[ordered.groupby('account_id').cumcount().ge(length - 1)].sort_values('item_id')
        exact(frame.item_id.to_numpy(str), expected.item_id.to_numpy(str), 'pooled OOF cohort, exactly once')
        exact(frame.y.to_numpy(), expected.label.to_numpy(), 'pooled OOF target')
        history = pd.concat(histories)
        if history.index.duplicated().any():
            raise ValueError('Endpoint crosses outer folds')
        flags = {k: pd.concat(v) for k, v in flag_parts.items()}
        report = cohort_diagnostics(frame.item_id.to_numpy(str), frame.y.to_numpy(), None,
            frame.prediction.to_numpy(), primary, evidence, history=history, flags=flags)
        original = source_series[series]['summary']
        candidates = original['pooled_oof'] if series == 'operating_points' else original['pooled_oof_decisions']
        expected_row = next(r for r in candidates if (r['name'], r['seed'], r['profile']) == (method, seed, profile)) if series == 'operating_points' else next(
            r for r in candidates if (r['policy'], r['model']) == (policy, method))
        metrics = compute_classification_metrics(frame.y, frame.prediction)
        if metrics != expected_row['metrics'] or report['scenarios'] != expected_row['scenarios']:
            raise ValueError('Pooled saved decisions/scenario support changed')
        report.update(series=series, method=method, policy=policy, seed=seed, profile=profile,
                      ranks_not_pooled=True, folds=[r['fold'] for r in rows])
        pooled.append(report)
    return pooled


def score_seed_identity(catalog):
    groups = defaultdict(list)
    for entry in catalog:
        row = entry['row']
        if entry['series'] == 'operating_points':
            groups[(row['name'], row['fold'])].append(row)
    result = []
    for (method, fold), rows in sorted(groups.items()):
        arrays = []
        for row in sorted(rows, key=lambda r: r['seed']):
            with np.load(Path(row['assessment']) / 'predictions_assessment.npz', allow_pickle=False) as saved:
                arrays.append(saved['scores'])
        result.append(dict(method=method, fold=fold, seeds=sorted(r['seed'] for r in rows),
            all_seed_scores_exactly_equal=all(np.array_equal(arrays[0], a) for a in arrays[1:]),
            unique_score_arrays=len({(str(a.dtype), payload_sha256(a.shape), a.tobytes()) for a in arrays}),
            repeated_seeds_do_not_add_independent_groups=True))
    return result


def no_training():
    stack = ExitStack()
    for target in ('sklearn.ensemble.RandomForestClassifier.fit', 'sklearn.ensemble.IsolationForest.fit',
                   'sklearn.ensemble.HistGradientBoostingClassifier.fit', 'utils.select_best_threshold',
                   'experiment.select_best_threshold', 'cert_operating_point.select_operating_points',
                   'features.fit_feature_state', 'datasets.cert_experiments.fit_global', 'torch.optim.Adam'):
        stack.enter_context(patch(target, side_effect=AssertionError('Fit/calibration forbidden during frozen diagnosis')))
    return stack


def run(model_variants, operating_points, root, config_path):
    plan = load_json(Path(config_path))
    catalog, series = read_catalog(model_variants, operating_points, plan)
    output = allocate(root, 'frozen_diagnostics')
    print('OUTPUT ' + str(output), flush=True)
    try:
        save_json(plan, output / 'plan.json')
        save_json(environment_metadata(), output / 'environment.json')
        for name in ('source', 'contexts', 'entries'):
            (output / name).mkdir()
        hashes = {}
        for path in sorted((PROJECT / 'src').rglob('*.py')):
            relative = path.relative_to(PROJECT / 'src')
            target = output / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = file_sha256(path)
        save_json(hashes, output / 'source_hashes.json')
        first = catalog[0]['row']
        contract = load_json(Path(first['preparation']) / 'contract.json')
        primary, evidence = read_primary_metadata(contract['daily_source'], series['model_variants']['plan']['daily_manifest_sha256'])
        actual_population = dict(days=len(primary), accounts=int(primary.account_id.nunique()),
            independent_groups=int(primary.group.nunique()), endpoints=series['model_variants']['summary']['population']['endpoints'],
            positive_endpoints=series['model_variants']['summary']['population']['positive_endpoints'])
        if actual_population != series['model_variants']['summary']['population']:
            raise ValueError('Reporting primary population differs from source series')
        records, contexts = [], {}
        ordered = sorted(catalog, key=lambda e: (e['row']['preparation'], e['series'], e['row']['job_id']))
        with no_training():
            for path, entries in groupby(ordered, key=lambda e: e['row']['preparation']):
                entries = list(entries)
                row, source = entries[0]['row'], entries[0]['source']
                folds = load_json(source / f"folds/fold_{row['fold']}.json")
                context = PreparationContext(path, row['preparation_manifest_sha256'], primary, folds,
                    inverse_atol=plan['diagnosis']['binary_feature_inverse_atol'])
                for entry in entries:
                    if entry['row']['fold'] != row['fold']:
                        raise ValueError('Preparation reused across incompatible folds')
                    result = process_entry(entry, context, primary, evidence, output,
                        reference_seed=plan['replay']['operating_reference_seed'])
                    records.extend(result)
                    save_json(result, output / 'entries' / f"{entry['series']}_{entry['row']['job_id']}.json")
                    print(json.dumps(dict(completed=len({r['evaluation_source']['candidate'] for r in records}),
                        decision_profiles=len(records), job=entry['row']['job_id'], replay=result[0]['replay']['performed'])), flush=True)
                ids = context.expected_endpoints['assessment'].item_id.to_numpy(str)
                values = dict(item_ids=ids, prior30=context.history['assessment'].loc[ids].to_numpy(np.int64))
                values.update({k: v.loc[ids].to_numpy(np.int64) for k, v in context.flags['assessment'].items() if v is not None})
                target = output / 'contexts' / f'{context.path.name}.npz'
                np.savez_compressed(target, **values)
                contexts[str(context.path)] = target
                del context
        records.sort(key=lambda r: (r['series'], r['method'], r['policy'], r['seed'], r['fold'], r['profile']))
        pooled = pooled_reports(records, primary, evidence, contexts, series)
        summary = dict(schema_version='argus.frozen_diagnostics.v1', plan_id=PLAN_ID,
            quality_contribution='preservation and diagnosis', independent_confirmation=False,
            no_research_fit_or_recalibration=True, original_test_rescored=False,
            population=actual_population, records=records, pooled_oof=pooled,
            variation=variation_reports(records), score_identity_across_model_seeds=score_seed_identity(catalog),
            coverage=dict(source_models=len(catalog), decision_profiles=len(records), role_reports=2 * len(records),
                replay_models=len({r['evaluation_source']['candidate'] for r in records if r['replay']['performed']}),
                replay_decision_profiles=sum(r['replay']['performed'] for r in records),
                all_declared_folds_and_seeds_present=True),
            source_series={k: dict(path=str(v['path']), manifest_sha256=plan['sources'][k]['manifest_sha256']) for k, v in series.items()},
            missing_data_policy='Missing mandatory runs/artifacts abort; absent optional features and undefined metrics are explicit null with reasons.',
            limitations=['Repeated development folds; no independent quality confirmation.',
                'Seeds reuse the same groups/endpoints; identical HGB scores are not independent observations.',
                'Pooled cross-fold PR/ROC AUC and score distributions are undefined and omitted.',
                'Co-located daily metadata is filtered to primary train; no primary test inference.',
                'History/error associations and score distributions do not establish causal mechanisms.'])
        write_report(summary, output)
        save_json(summary, output / 'summary.json')
        write_tables(summary, output)
        finish(output)
        return output
    except BaseException as error:
        save_json(dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', reason=str(error)), output / 'status.json')
        raise


def write_tables(summary, output):
    rows = []
    for record in summary['records']:
        for role in ('calibration', 'assessment'):
            data = record[role]
            rows.append({**{k: record[k] for k in ('id', 'series', 'method', 'policy', 'seed', 'fold', 'profile', 'threshold')},
                'role': role, **{m: data['metrics'][m] for m in (*METRICS, 'fp', 'fn', 'tp', 'tn', 'fp_per_1000_background')},
                **data['metrics']['support'], **data['population'],
                **(record['operating_checks'][role] or {})})
    pd.DataFrame(rows).to_csv(output / 'cohort_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-variants', type=Path, required=True)
    parser.add_argument('--operating-points', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/s3_frozen_diagnostics_v1.json')
    args = parser.parse_args()
    protected = tuple({args.model_variants.resolve().parent, args.operating_points.resolve().parent})
    if any(args.root.resolve().is_relative_to(p) for p in protected):
        raise ValueError('Use an existing report root outside the read-only source artifact tree')
    blocked = []
    def audit(event, arguments):
        if event != 'open' or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        mode, flags = arguments[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if path.name in TEST_FILES or writing and any(path.is_relative_to(p) for p in protected):
            blocked.append(str(path))
            raise RuntimeError('Frozen diagnosis cannot open test artifacts or overwrite source series')
    sys.addaudithook(audit)
    output = run(args.model_variants, args.operating_points, args.root, args.config)
    save_json(dict(real_test_guard=True, protected_source_series=True, blocked_attempts=blocked), output / 'access_guard.json')
    if blocked:
        raise RuntimeError('Blocked access attempts must be investigated')
    finish(output)
    print(output)


if __name__ == '__main__':
    main()
