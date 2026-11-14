"""Verified artifact inspection/replay and isolated outputs for the S3 CLI."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiment import allocate, finish
from experiment_tracking import environment_metadata, file_sha256
from frozen_diagnostics import descriptive_metrics
from inference import FrozenRun, PreparedBatch, _sealed_files, batch_from_frame
from utils import load_json, save_json


@contextmanager
def new_result(root: Path, kind: str, sources: list[Path]):
    """Never write inside an input artifact; retain failures as incomplete runs."""
    root = Path(root).expanduser().resolve()
    sources = sorted({Path(p).resolve() for p in sources})
    if any(root.is_relative_to(source) for source in sources):
        raise ValueError('Output root must be outside every source artifact')
    initial = {str(p): file_sha256(p / 'complete.json') for p in sources}
    output = allocate(root, kind)
    try:
        save_json(environment_metadata(), output / 'environment.json')
        code = output / 'runtime_source'
        code.mkdir()
        source_names = ('argus.py', 'inference.py', 'frozen_runtime.py', 'synthetic_scenarios.py',
                        'features.py', 'preprocessing.py', 'experiment.py', 'cert_model_variants.py',
                        'train_lstm_autoencoder.py', 'utils.py', 'frozen_diagnostics.py')
        hashes = {}
        for name in source_names:
            path = Path(__file__).resolve().parent / name
            shutil.copy2(path, code / name)
            hashes[name] = file_sha256(path)
        save_json(hashes, output / 'runtime_source_hashes.json')
        yield output
        for source in sources:
            if file_sha256(source / 'complete.json') != initial[str(source)]:
                raise ValueError('Source manifest changed during frozen operation')
            _sealed_files(source)
        save_json(dict(manifest_sha256=initial, all_source_files_unchanged=True),
                  output / 'source_preservation.json')
        finish(output)
    except BaseException as error:
        save_json(dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       reason=str(error)), output / 'status.json')
        raise


def save_batch(path: Path, batch: PreparedBatch) -> None:
    # Labels deliberately live outside every persisted inference batch.
    np.savez_compressed(path, X=batch.X, item_ids=batch.item_ids, member_ids=batch.member_ids)


def source_binding(path: Path, files: dict[str, str]) -> dict:
    return dict(path=str(path), manifest_sha256=file_sha256(path / 'complete.json'), files=files)


@dataclass(frozen=True)
class SavedInputs:
    path: Path
    role: str
    files: dict[str, str]
    batch: PreparedBatch
    labels: np.ndarray
    scores: np.ndarray
    decisions: np.ndarray
    score_storage: str


def load_saved_inputs(run: FrozenRun, source: Path | None = None,
                      role: str | None = None, expected_manifest: str | None = None) -> SavedInputs:
    """Bind only existing calibration/assessment arrays to their frozen model.

    This command does not construct a new final evaluation or open raw CERT.
    A separately sealed assessment must name this exact candidate manifest.
    """
    source = Path(source or run.artifact_path).expanduser().resolve()
    if run.spec.scenario == 'cert_r4_2':
        # Reject a mistakenly supplied final/preparation directory before the
        # full checksum verifier could open its closed test members.
        manifest = load_json(source / 'complete.json')
        closed = {'test.csv', 'test_inputs.npz', 'predictions_test.npz',
                  'predictions_test.csv', 'metrics_test.json'}
        if any(Path(name).name in closed for name in manifest.get('files', {})):
            raise ValueError('CERT final/test artifacts are outside this saved-input CLI')
    files = _sealed_files(source)
    if expected_manifest is not None and file_sha256(source / 'complete.json') != expected_manifest:
        raise ValueError('Saved-input manifest identity differs')
    operating = 'operating_points.json' in run.artifact_hashes
    role = role or ('calibration' if operating else 'validation')
    if role not in ('validation', 'calibration', 'assessment'):
        raise ValueError('Replay supports saved validation/calibration/assessment only')
    if role == 'assessment':
        if 'lineage.json' not in files:
            raise ValueError('Assessment requires sealed candidate lineage')
        lineage = load_json(source / 'lineage.json')
        key = 'frozen_candidate_manifest_sha256' if operating else 'candidate_manifest_sha256'
        if (lineage.get(key) != run.artifact_hashes['complete.json']
                or lineage.get('preparation_manifest_sha256') != run.spec.preparation_manifest_sha256
                or lineage.get('refit_or_recalibration') is not False):
            raise ValueError('Assessment lineage differs from frozen model/preparation')
        if not operating and (lineage.get('model') != run.spec.model
                              or lineage.get('threshold') != run.spec.threshold):
            raise ValueError('Assessment model/threshold differs')
        input_name, prediction_name = 'assessment_inputs.npz', 'predictions_assessment.npz'
    else:
        if file_sha256(source / 'complete.json') != run.artifact_hashes['complete.json']:
            raise ValueError('Calibration/validation inputs must belong to the frozen model artifact')
        if (role == 'calibration') != operating:
            raise ValueError('Use calibration for an operating bundle, validation for an isolated candidate')
        input_name = 'validation_inputs.npz'
        prediction_name = 'predictions_calibration.npz' if operating else 'predictions_validation.csv'
    if input_name not in files or prediction_name not in files:
        raise ValueError('Saved inputs or reference predictions missing from seal')
    with np.load(source / input_name, allow_pickle=False) as arrays:
        batch = PreparedBatch(X=arrays['X'], feature_columns=run.spec.feature_columns,
            item_ids=arrays['item_ids'], member_ids=arrays['member_ids'],
            preparation_id=run.spec.preparation_id,
            preparation_manifest_sha256=run.spec.preparation_manifest_sha256,
            feature_state_sha256=run.spec.feature_state_sha256)
        labels = arrays['y']
    run.validate_batch(batch)
    if labels.shape != (len(batch.X),) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Saved reporting labels must be binary and aligned')
    if prediction_name.endswith('.npz'):
        with np.load(source / prediction_name, allow_pickle=False) as reference:
            ids, y = reference['item_ids'], reference['y']
            scores = reference['scores']
            decisions = reference[run.spec.profile if operating else 'predictions']
        storage = 'exact_npz'
    else:
        reference = pd.read_csv(source / prediction_name, float_precision='round_trip')
        ids, y = reference.item_id.to_numpy(dtype=str), reference.y_true.to_numpy()
        dtype = np.float32 if run.spec.model == 'lstm_autoencoder' else np.float64
        scores = reference.score.to_numpy(dtype=dtype)
        decisions = reference.prediction.to_numpy()
        storage = 'legacy_csv_round_trip_restored_original_score_dtype'
    if (not np.array_equal(ids, batch.item_ids) or not np.array_equal(y, labels)
            or scores.shape != labels.shape or scores.dtype not in (np.float32, np.float64)
            or not np.isfinite(scores).all() or decisions.shape != labels.shape
            or not np.isin(decisions, [0, 1]).all()
            or not np.array_equal(decisions, scores >= run.spec.threshold)):
        raise ValueError('Saved predictions/labels/threshold are inconsistent with inputs')
    if role == 'validation':
        if 'validation.csv' not in files:
            raise ValueError('Candidate validation records missing from seal')
        built = batch_from_frame(pd.read_csv(source / 'validation.csv'),
            contract=load_json(run.artifact_path / 'contract.json'), model=run.spec.model,
            preparation_manifest_sha256=run.spec.preparation_manifest_sha256,
            feature_state_sha256=run.spec.feature_state_sha256)
        for name in ('X', 'item_ids', 'member_ids'):
            if not np.array_equal(getattr(batch, name), getattr(built, name)):
                raise ValueError('Saved validation inputs or membership differ from original builder')
    return SavedInputs(source, role, files, batch, labels, scores, decisions, storage)


def inspect_run(run: FrozenRun, root: Path, *, source: Path | None = None,
                role: str | None = None, expected_source_manifest: str | None = None) -> Path:
    saved = load_saved_inputs(run, source, role, expected_source_manifest) if source is not None else None
    sources = [run.artifact_path] + ([saved.path] if saved is not None else [])
    with new_result(root, 'frozen_inspect', sources) as output:
        data = dict(schema='argus.frozen_inspect.v1', model=run.metadata,
            capabilities=dict(inspect=True, replay_saved_inputs=True,
                raw_scenario_fields=['bytes_sent'] if run.spec.scenario == 'synthetic' else [],
                cert_raw_what_if=False),
            quality_contribution='artifact inspection; no new quality estimate',
            fit_or_threshold_selection=False)
        if saved is not None:
            data['prepared_inputs'] = dict(role=saved.role, source=source_binding(saved.path, saved.files),
                shape=list(saved.batch.X.shape), dtype=str(saved.batch.X.dtype), items=len(saved.batch.X),
                feature_columns=list(saved.batch.feature_columns),
                first_item_id=str(saved.batch.item_ids[0]) if len(saved.batch.X) else None,
                last_item_id=str(saved.batch.item_ids[-1]) if len(saved.batch.X) else None)
        save_json(data, output / 'inspection.json')
    return output


def replay_run(run: FrozenRun, root: Path, *, source: Path | None = None,
               role: str | None = None, expected_source_manifest: str | None = None) -> Path:
    saved = load_saved_inputs(run, source, role, expected_source_manifest)
    with new_result(root, 'frozen_replay', [run.artifact_path, saved.path]) as output:
        scores, decisions = run.score(saved.batch), run.predict(saved.batch)
        if scores.dtype != saved.scores.dtype or not np.array_equal(scores, saved.scores):
            raise ValueError('Replay scores differ from saved reference (exact comparison)')
        if not np.array_equal(decisions, saved.decisions):
            raise ValueError('Replay decisions differ from saved reference')
        save_batch(output / 'inputs.npz', saved.batch)
        np.savez_compressed(output / 'predictions.npz', item_ids=saved.batch.item_ids,
                            scores=scores, decisions=decisions)
        np.savez_compressed(output / 'reporting_labels.npz', item_ids=saved.batch.item_ids, y=saved.labels)
        save_json(descriptive_metrics(saved.labels, decisions, scores), output / 'metrics.json')
        save_json(dict(schema='argus.frozen_replay.v1', model=run.metadata,
            source=source_binding(saved.path, saved.files), role=saved.role,
            score_storage=saved.score_storage, items=len(scores), scores_exact=True, decisions_exact=True,
            fit_or_threshold_selection=False, independent_quality_confirmation=False), output / 'replay.json')
    return output
