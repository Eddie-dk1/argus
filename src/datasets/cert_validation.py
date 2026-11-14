"""Grouped, nested-calibration development checks using primary CERT train only."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def development_folds(frame, *, n_splits=3, calibration_splits=5, seed=142, calibration_seed=242):
    """Keep each incident-connected group intact at both splitting levels.

    Assessment labels organize the split, but never choose an epoch/threshold.
    The validation role is an inner calibration partition of outer train.
    """
    frame = frame.loc[frame['split'].eq('train')].reset_index(drop=True)
    if (frame.empty or frame['item_id'].duplicated().any()
            or frame.groupby('account_id')['group'].nunique().max() != 1):
        raise ValueError('Invalid primary training population for CV')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for index, (fit, assessment) in enumerate(splitter.split(np.zeros(len(frame)), frame.label, frame.group), 1):
        inner = frame.iloc[fit]
        calibration = StratifiedGroupKFold(n_splits=calibration_splits, shuffle=True,
                                          random_state=calibration_seed + index)
        inner_fit, inner_validation = next(calibration.split(np.zeros(len(inner)), inner.label, inner.group))
        roles = np.full(len(frame), 'assessment', dtype=object)
        roles[fit[inner_fit]] = 'train'
        roles[fit[inner_validation]] = 'validation'
        folded = frame.copy()
        folded['split'] = roles
        if folded.groupby('group')['split'].nunique().max() != 1:
            raise ValueError('Group crossed a development boundary')
        if any(folded.loc[folded.split.eq(role), 'label'].nunique() != 2
               for role in ('train', 'validation', 'assessment')):
            raise ValueError('A development partition lacks class support')
        yield folded, {'fold': index, 'source_population': 'primary train only',
                       'outer_seed': seed, 'calibration_seed': calibration_seed + index,
                       'outer_folds': n_splits, 'calibration_folds': calibration_splits,
                       'account_roles': dict(zip(folded.account_id, folded.split)),
                       'group_roles': dict(zip(folded.group, folded.split)),
                       'roles': {'train': 'fit features and model', 'validation': 'choose epoch and threshold',
                                 'assessment': 'frozen outer-fold evaluation only'}}


def write_train_context(frame, source_path, destination):
    """Filter contexts before CV; preserve date/account order and exact identity."""
    from datasets.cert_experiments import read_context
    with gzip.open(destination, 'wt', compresslevel=1) as stream:
        for keep, context in zip(frame['split'].eq('train'), read_context(source_path, frame), strict=True):
            if keep:
                stream.write(json.dumps(context) + '\n')


def assess_development(candidate_path, preparation, root):
    """Evaluate an immutable checkpoint on assessment, without calibration/fitting."""
    from experiment import allocate, finish, inputs, configured_score, verify
    from experiment_tracking import file_sha256
    from utils import load_json, save_json, compute_classification_metrics
    candidate_path, preparation = Path(candidate_path), Path(preparation)
    verify(candidate_path)
    verify(preparation, ['contract.json', 'assessment.csv', 'development.json'])
    contract = load_json(preparation / 'contract.json')
    selection = load_json(candidate_path / 'selection.json')
    name = selection['model']
    if (contract.get('scope') != 'development_group_cv'
            or name not in ('random_forest', 'isolation_forest', 'lstm_autoencoder', 'hist_gradient_boosting')
            or selection['preparation_manifest_sha256'] != file_sha256(preparation / 'complete.json')
            or load_json(candidate_path / 'contract.json') != contract):
        raise ValueError('Assessment requires matching development-only artifacts')
    if not np.isfinite(selection['threshold']):
        raise ValueError('Invalid frozen threshold')
    frozen_manifest = file_sha256(candidate_path / 'complete.json')
    config = load_json(candidate_path / 'config.json')
    path = allocate(root, 'development_assessment')
    try:
        x, y, ids, members = inputs(pd.read_csv(preparation / 'assessment.csv'), contract, name)
        if name == 'lstm_autoencoder':
            import torch
            from train_lstm_autoencoder import LSTMAutoencoder
            model = LSTMAutoencoder(len(contract['sequence_columns']), config[name]['hidden_size'])
            model.load_state_dict(torch.load(candidate_path / 'model.pt', map_location='cpu', weights_only=True))
        else:
            import joblib
            model = joblib.load(candidate_path / 'model.joblib')
        values = configured_score(model, x, name, config)
        predicted = (values >= selection['threshold']).astype(int)
        save_json(compute_classification_metrics(y, predicted, values), path / 'metrics_assessment.json')
        lineage = {'candidate_id': candidate_path.name, 'preparation_id': preparation.name,
                   'candidate_manifest_sha256': frozen_manifest,
                   'preparation_manifest_sha256': file_sha256(preparation / 'complete.json'),
                   'threshold': selection['threshold'], 'scope': 'development_group_cv',
                   'model': name, 'target': contract.get('target'), 'score_dtype': str(values.dtype),
                   'refit_or_recalibration': False}
        if 'score_policy' in selection:
            lineage['score_policy'] = selection['score_policy']
        save_json(lineage, path / 'lineage.json')
        pd.DataFrame({'item_id': ids, 'y_true': y, 'score': values, 'prediction': predicted}).to_csv(
            path / 'predictions_assessment.csv', index=False)
        np.savez_compressed(path / 'assessment_inputs.npz', X=x, y=y, item_ids=ids, member_ids=members)
        np.savez_compressed(path / 'predictions_assessment.npz', item_ids=ids, y=y,
                            scores=values, predictions=predicted)
        if file_sha256(candidate_path / 'complete.json') != frozen_manifest:
            raise ValueError('Candidate changed during frozen assessment')
        finish(path)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': str(error)}, path / 'status.json')
        raise
    return path
