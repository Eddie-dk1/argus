"""Raw synthetic bytes_sent patches with frozen transforms and a full-split oracle."""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_tracking import file_sha256
from features import apply_feature_state
from frozen_runtime import new_result, save_batch, source_binding
from inference import FrozenRun, PreparedBatch, _sealed_files, batch_from_frame
from preprocessing import chronological_split
from utils import load_json, save_json


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / 'configs/s3_synthetic_scenarios_v1.json'
# Declared before any replay; never inferred from observed discrepancies.
NOOP_TOLERANCES = {'float32': dict(rtol=1e-6, atol=1e-7),
                   'float64': dict(rtol=1e-12, atol=1e-12)}
POSITIONS = ('endpoint', 'earlier')
OPERATIONS = ('noop', 'double', 'zero', 'set')
REPORTING_COLUMNS = ('label', 'anomaly_type')


def numeric_check(original: np.ndarray, rebuilt: np.ndarray, name: str) -> dict:
    if (original.shape != rebuilt.shape or original.dtype != rebuilt.dtype
            or str(original.dtype) not in NOOP_TOLERANCES
            or not np.isfinite(original).all() or not np.isfinite(rebuilt).all()):
        raise ValueError(f'{name}: incompatible numeric arrays')
    tolerance = NOOP_TOLERANCES[str(original.dtype)]
    if not np.allclose(original, rebuilt, **tolerance):
        raise ValueError(f'{name}: no-op exceeds predeclared {original.dtype} tolerances')
    return dict(dtype=str(original.dtype), **tolerance, passed=True,
        exact=bool(np.array_equal(original, rebuilt)),
        max_absolute_error=float(np.max(np.abs(original - rebuilt))) if original.size else 0.0)


def require_same_membership(base: PreparedBatch, variant: PreparedBatch) -> None:
    if (base.feature_columns != variant.feature_columns or base.X.dtype != variant.X.dtype
            or base.X.shape != variant.X.shape
            or not np.array_equal(base.item_ids, variant.item_ids)
            or not np.array_equal(base.member_ids, variant.member_ids)):
        raise ValueError('Scenario changed schema, item order or window membership')


def prepared_csv_boundary(frame: pd.DataFrame) -> pd.DataFrame:
    """Match preprocessing.py's CSV and experiment.prepare's copied split CSV.

    The builder emits float32; preparation reads that CSV, writes its isolated
    split, and candidate reads it again as float64. Do not cast float32 straight
    to float64 and silently replace the frozen tabular input representation.
    """
    for _ in range(2):
        frame = pd.read_csv(StringIO(frame.to_csv(index=False)))
    return frame


def rebuild_split(raw: pd.DataFrame, state: dict, contract: dict, split: str) -> pd.DataFrame:
    clean = raw.drop(columns=list(REPORTING_COLUMNS), errors='ignore').copy()
    clean['label'] = 0  # Legacy sort only; true labels cannot control transforms.
    processed, tabular, sequence = apply_feature_state(clean, state)
    if tabular != contract['tabular_columns'] or sequence != contract['sequence_columns']:
        raise ValueError('Original builder returned a different feature schema')
    processed['split'] = split
    return prepared_csv_boundary(processed)


@dataclass(frozen=True)
class ScenarioSource:
    run: FrozenRun
    preparation: Path
    files: dict[str, str]
    split: str
    contract: dict
    state: dict
    raw: pd.DataFrame
    prepared: pd.DataFrame

    def batch(self, frame: pd.DataFrame) -> PreparedBatch:
        batch = batch_from_frame(frame, contract=self.contract, model=self.run.spec.model,
            preparation_manifest_sha256=self.run.spec.preparation_manifest_sha256,
            feature_state_sha256=self.run.spec.feature_state_sha256)
        self.run.validate_batch(batch)
        return batch


def load_scenario_source(run: FrozenRun, preparation: Path, split: str) -> ScenarioSource:
    if run.spec.scenario != 'synthetic':
        raise ValueError('Raw what-if supports synthetic only; CERT requires a separate reaggregation contract')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown synthetic split')
    preparation = Path(preparation).expanduser().resolve()
    files = _sealed_files(preparation)
    spec = run.spec
    if (file_sha256(preparation / 'complete.json') != spec.preparation_manifest_sha256
            or preparation.name != spec.preparation_id
            or files.get('feature_state.json') != spec.feature_state_sha256):
        raise ValueError('Scenario preparation/state differs from frozen model')
    contract, state, config = [load_json(preparation / name) for name in
                                ('contract.json', 'feature_state.json', 'config.json')]
    if contract != load_json(run.artifact_path / 'contract.json'):
        raise ValueError('Scenario preparation contract differs from frozen model')
    run_config = load_json(run.artifact_path / 'config.json')
    if any(config[key] != run_config[key] for key in ('generator', 'split', 'features', 'scenario')):
        raise ValueError('Scenario preparation configuration differs')
    raw_name = 'synthetic/raw/synthetic_logs.csv'
    if files.get(raw_name) != spec.snapshot_id or f'{split}.csv' not in files:
        raise ValueError('Scenario needs original sealed synthetic raw and selected split')
    raw = pd.read_csv(preparation / raw_name)
    raw['event_id'] = [f'{spec.snapshot_id}:{i}' for i in range(len(raw))]
    fractions = config['split']
    raw = chronological_split(raw, train_fraction=fractions['train_fraction'],
        validation_fraction=fractions['validation_fraction'], test_fraction=fractions['test_fraction'])[split]
    prepared = pd.read_csv(preparation / f'{split}.csv')
    if (not np.array_equal(raw.event_id, prepared.event_id)
            or not np.array_equal(raw.user_id, prepared.user_id)
            or not np.array_equal(pd.to_datetime(raw.timestamp), pd.to_datetime(prepared.timestamp))
            or 'split' not in prepared or not (prepared['split'] == split).all()):
        raise ValueError('Raw event identity/actor/time/order differs from prepared split')
    if (not np.isfinite(raw[['bytes_sent', 'bytes_received']].to_numpy(dtype=float)).all()
            or (raw[['bytes_sent', 'bytes_received']] < 0).any().any()):
        raise ValueError('Synthetic source bytes must be finite and nonnegative')
    return ScenarioSource(run, preparation, files, split, contract, state, raw, prepared)


def select_window(raw: pd.DataFrame, *, sequence_length: int, rolling_window: int,
                  item_id: str | None = None) -> dict:
    """Selection can see only IDs, actor, time and positive raw volume."""
    clean = raw.loc[:, ['event_id', 'user_id', 'timestamp', 'bytes_sent']].copy()
    clean['timestamp'] = pd.to_datetime(clean['timestamp'], errors='raise')
    if (sequence_length < 2 or rolling_window < 1 or clean.event_id.duplicated().any()
            or clean.event_id.isna().any()):
        raise ValueError('Scenario needs unique event IDs, a window of at least two and prior history')
    clean = clean.sort_values(['user_id', 'timestamp', 'event_id'], kind='stable')
    for user, frame in clean.groupby('user_id', sort=True):
        frame = frame.reset_index(drop=True)
        # Full prior history at the earliest member; one future observation at endpoint.
        for endpoint in range(rolling_window + sequence_length - 1, len(frame) - 1):
            first = endpoint - sequence_length + 1
            if item_id is not None and str(frame.iloc[endpoint].event_id) != item_id:
                continue
            if frame.iloc[[first, endpoint]].bytes_sent.le(0).any():
                continue
            members = frame.iloc[first:endpoint + 1].event_id.astype(str).tolist()
            return dict(item_id=members[-1], member_ids=members, user_id=str(user),
                timestamp=frame.iloc[endpoint].timestamp.isoformat(),
                endpoint_event_id=members[-1], earlier_event_id=members[0],
                earliest_member_prior_events=first, endpoint_prior_events=endpoint,
                successor_event_id=str(frame.iloc[endpoint + 1].event_id),
                selection_policy='first eligible (user_id, timestamp, event_id); no labels or scores')
    raise ValueError('No eligible window with positive volume, full prior history and a successor in this split')


def patch_value(original: float, operation: str, value: Real | None) -> float:
    if operation not in OPERATIONS:
        raise ValueError('Unsupported patch operation')
    if operation != 'set' and value is not None:
        raise ValueError('An explicit value is allowed only for operation=set')
    result = {'noop': original, 'double': original * 2, 'zero': 0.0}.get(operation, value)
    if isinstance(result, (bool, np.bool_)) or not isinstance(result, Real):
        raise ValueError('bytes_sent must be a finite nonnegative number')
    try:
        result = float(result)
    except (OverflowError, ValueError) as error:
        raise ValueError('bytes_sent must be a finite nonnegative number') from error
    if not np.isfinite(result) or result < 0:
        raise ValueError('bytes_sent must be a finite nonnegative number')
    return result


def recompute_user(source: ScenarioSource, variant_raw: pd.DataFrame, user: str) -> pd.DataFrame:
    """All observations of one actor, including earlier context and later dependents."""
    affected = variant_raw.loc[variant_raw.user_id.astype(str) == user]
    rebuilt = rebuild_split(affected, source.state, source.contract, source.split).set_index('event_id')
    result = source.prepared.copy().set_index('event_id')
    # Provenance labels in the base frame never enter the frozen builder.
    columns = [c for c in rebuilt.columns if c not in REPORTING_COLUMNS]
    result.loc[rebuilt.index, columns] = rebuilt[columns]
    return result.reset_index()


def changed_features(base: pd.DataFrame, variant: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not np.array_equal(base.event_id, variant.event_id):
        raise ValueError('Prepared event order changed')
    a, b = base[columns].to_numpy(), variant[columns].to_numpy()
    rows, features = np.nonzero(a != b)
    return pd.DataFrame(dict(event_id=base.event_id.to_numpy()[rows],
        feature=np.asarray(columns)[features], base_value=a[rows, features],
        variant_value=b[rows, features], delta=b[rows, features] - a[rows, features]))


def run_scenario(run: FrozenRun, preparation: Path, root: Path, *, split: str = 'validation',
                 position: str = 'endpoint', operation: str = 'noop', field: str = 'bytes_sent',
                 value: Real | None = None, item_id: str | None = None) -> Path:
    if field != 'bytes_sent':
        raise ValueError('Only the raw field bytes_sent is supported')
    if position not in POSITIONS:
        raise ValueError('Position must be endpoint or earlier')
    protocol = load_json(PROTOCOL_PATH)
    if protocol['no_op_tolerances'] != NOOP_TOLERANCES:
        raise ValueError('Scenario protocol tolerances differ from the declared implementation')
    source = load_scenario_source(run, preparation, split)
    selected = select_window(source.raw, sequence_length=run.spec.sequence_length,
        rolling_window=int(source.state['rolling_window_events']), item_id=item_id)
    event_id = selected[position + '_event_id']
    original = float(source.raw.set_index('event_id').loc[event_id, field])
    replacement = patch_value(original, operation, value)
    variant_raw = source.raw.copy()
    variant_raw[field] = variant_raw[field].astype(float)
    variant_raw.loc[variant_raw.event_id == event_id, field] = replacement
    pd.testing.assert_frame_equal(source.raw.drop(columns=field), variant_raw.drop(columns=field))
    unchanged_raw = source.raw.event_id != event_id
    if not np.array_equal(source.raw.loc[unchanged_raw, field], variant_raw.loc[unchanged_raw, field]):
        raise ValueError('Patch changed more than one raw event')
    with new_result(root, 'synthetic_scenario', [run.artifact_path, source.preparation]) as output:
        scenario_id = output.name
        save_json(protocol, output / 'protocol.json')
        save_json(dict(path=str(PROTOCOL_PATH), sha256=file_sha256(PROTOCOL_PATH)), output / 'protocol_source.json')
        original_batch = source.batch(source.prepared)
        base_rebuilt = rebuild_split(source.raw, source.state, source.contract, split)
        rebuilt_batch = source.batch(base_rebuilt)
        require_same_membership(original_batch, rebuilt_batch)
        baseline_scores, baseline_decisions = run.score(original_batch), run.predict(original_batch)
        rebuilt_scores, rebuilt_decisions = run.score(rebuilt_batch), run.predict(rebuilt_batch)
        checks = dict(base_raw_inputs=numeric_check(original_batch.X, rebuilt_batch.X, 'Base raw inputs'),
            base_raw_features=numeric_check(source.prepared[source.contract['tabular_columns']].to_numpy(),
                base_rebuilt[source.contract['tabular_columns']].to_numpy(), 'Base raw features'),
            base_raw_scores=numeric_check(baseline_scores, rebuilt_scores, 'Base raw scores'))
        if not np.array_equal(baseline_decisions, rebuilt_decisions):
            raise ValueError('Base raw no-op changed decisions')
        variant = recompute_user(source, variant_raw, selected['user_id'])
        # A separate original-builder invocation sees every user, not optimized output.
        oracle = rebuild_split(variant_raw, source.state, source.contract, split)
        batch, oracle_batch = source.batch(variant), source.batch(oracle)
        require_same_membership(original_batch, batch)
        require_same_membership(batch, oracle_batch)
        if (not np.array_equal(batch.X, oracle_batch.X)
                or not np.array_equal(variant[source.contract['tabular_columns']].to_numpy(),
                                      oracle[source.contract['tabular_columns']].to_numpy())):
            raise ValueError('Per-user recomputation differs from full-split oracle')
        scores, decisions = run.score(batch), run.predict(batch)
        oracle_scores, oracle_decisions = run.score(oracle_batch), run.predict(oracle_batch)
        if (not np.array_equal(scores, oracle_scores) or not np.array_equal(decisions, oracle_decisions)):
            raise ValueError('Per-user inference differs from full-split oracle')
        other_rows = source.prepared.user_id.astype(str) != selected['user_id']
        columns = source.contract['tabular_columns']
        if not np.array_equal(source.prepared.loc[other_rows, columns], variant.loc[other_rows, columns]):
            raise ValueError('Scenario changed features of another user')
        users = source.prepared.set_index('event_id').loc[batch.item_ids].user_id.astype(str).to_numpy()
        other_items = users != selected['user_id']
        if (not np.array_equal(original_batch.X[other_items], batch.X[other_items])
                or not np.array_equal(baseline_scores[other_items], scores[other_items])
                or not np.array_equal(baseline_decisions[other_items], decisions[other_items])):
            raise ValueError('Scenario changed inputs or inference for another user')
        is_noop = replacement == original
        if is_noop:
            checks['variant_inputs'] = numeric_check(original_batch.X, batch.X, 'No-op inputs')
            checks['variant_scores'] = numeric_check(baseline_scores, scores, 'No-op scores')
            if not np.array_equal(baseline_decisions, decisions):
                raise ValueError('No-op changed decisions')
        checks.update(base_raw_decisions_exact=True, full_split_oracle_exact=True,
            membership_and_actor_time_order_preserved=True, other_users_unchanged=True)
        patch = dict(schema='argus.raw_patch.v1', scenario_id=scenario_id, event_id=event_id,
            field=field, operation=operation, position=position, original_value=original,
            replacement_value=replacement, is_noop=is_noop)
        save_json(patch, output / 'patch.json')
        for name, frame in (('raw_base', source.raw), ('raw_variant', variant_raw),
                            ('prepared_base', source.prepared), ('prepared_variant', variant)):
            frame.drop(columns=list(REPORTING_COLUMNS), errors='ignore').to_csv(output / f'{name}.csv', index=False)
        provenance = source.raw[['event_id', 'label', 'anomaly_type']].rename(
            columns={'label': 'original_label', 'anomaly_type': 'original_anomaly_type'})
        provenance.to_csv(output / 'base_label_provenance.csv', index=False)
        for name, current, values, predicted in (
                ('base', original_batch, baseline_scores, baseline_decisions),
                ('base_raw_recomputed', rebuilt_batch, rebuilt_scores, rebuilt_decisions),
                ('variant', batch, scores, decisions), ('oracle', oracle_batch, oracle_scores, oracle_decisions)):
            save_batch(output / f'{name}_inputs.npz', current)
            np.savez_compressed(output / f'{name}_predictions.npz', item_ids=current.item_ids,
                                scores=values, decisions=predicted)
        changes = changed_features(source.prepared, variant, columns)
        changes.to_csv(output / 'changed_features.csv', index=False)
        changed_inputs = np.any(original_batch.X != batch.X, axis=tuple(range(1, batch.X.ndim)))
        changed_scores, changed_decisions = scores != baseline_scores, decisions != baseline_decisions
        pd.DataFrame(dict(item_id=batch.item_ids, user_id=users, inputs_changed=changed_inputs,
            base_score=baseline_scores, variant_score=scores, score_delta=scores - baseline_scores,
            base_decision=baseline_decisions, variant_decision=decisions)).to_csv(output / 'responses.csv', index=False)
        selected_index = int(np.flatnonzero(batch.item_ids == selected['item_id'])[0])
        lineage = dict(schema='argus.synthetic_scenario.v1', scenario_id=scenario_id, model=run.metadata,
            preparation=source_binding(source.preparation, source.files), split=split, selected=selected,
            base_event_identity='original raw snapshot SHA256 : logical CSV row; unchanged inside variant',
            variant_ground_truth='unknown', original_label_location='base_label_provenance.csv',
            include_in_benchmark_metrics=False, fit_or_threshold_selection=False,
            quality_contribution='verified response to a raw patch; not detection improvement or real-world causality')
        save_json(lineage, output / 'lineage.json')
        summary = dict(scenario_id=scenario_id, model=run.spec.model, score_policy=run.spec.score_policy,
            split=split, patch=patch, selected=selected, checks=checks, items=len(batch.X),
            changed_feature_values=len(changes), changed_feature_names=sorted(changes.feature.unique().tolist()),
            changed_input_items=int(changed_inputs.sum()), changed_score_items=int(changed_scores.sum()),
            changed_decision_items=int(changed_decisions.sum()),
            selected_endpoint=dict(base_score=float(baseline_scores[selected_index]),
                variant_score=float(scores[selected_index]), base_decision=int(baseline_decisions[selected_index]),
                variant_decision=int(decisions[selected_index])),
            variant_ground_truth='unknown', include_in_benchmark_metrics=False)
        save_json(summary, output / 'summary.json')
        (output / 'report.md').write_text(
            f'# Synthetic scenario: {position} / {operation}\n\n'
            f'Field `{field}`: {original:g} → {replacement:g}. Frozen model `{run.spec.model}`, '
            f'profile `{run.spec.profile}`, threshold {run.spec.threshold:.17g}.\n\n'
            f'Full-split oracle: exact match. Other users: unchanged. '
            f'Changed inputs/scores/decisions: {changed_inputs.sum()}/{changed_scores.sum()}/{changed_decisions.sum()} '
            f'of {len(scores)} items.\n\n'
            'Variant ground truth is unknown; base labels are provenance only. '
            'Scenario responses do not enter benchmark metrics or establish detection improvement.\n', encoding='utf-8')
    return output
