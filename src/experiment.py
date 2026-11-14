"""Isolated synthetic/CERT preparation, validation-only candidates and frozen final evaluation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from features import build_sequences
from utils import load_json, save_json, compute_classification_metrics, select_best_threshold, set_seed, measure_inference
from experiment_tracking import file_sha256, environment_metadata


def seal(root):
    files = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'complete.json'}
    save_json({'files': files}, root / 'complete.json')


def verify(root, names=None):
    manifest = load_json(root / 'complete.json')['files']
    selected = manifest if names is None else names
    for name in selected:
        if name not in manifest or file_sha256(root / name) != manifest[name]:
            raise ValueError(f'Artifact changed or missing: {name}')
    return manifest


def allocate(root, kind):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Artifact root must already exist (no silent missing-volume fallback)')
    path = root / (kind + '_' + uuid.uuid4().hex)
    path.mkdir()
    save_json({'status': 'running'}, path / 'status.json')
    return path


def finish(path):
    save_json({'status': 'complete'}, path / 'status.json')
    seal(path)


def prepare(config_path, root):
    path = allocate(root, 'preparation')
    config = load_json(config_path)
    save_json(config, path / 'config.json')
    env = dict(os.environ, ARGUS_DATA_ROOT=str(path))
    source = Path(__file__).resolve().parent
    try:
        for script in ('generate_data.py', 'preprocessing.py'):
            subprocess.run([sys.executable, str(source / script), '--config', str(path / 'config.json')],
                           env=env, check=True, capture_output=True, text=True)
        processed = path / 'synthetic/processed'
        state = load_json(processed / 'feature_state.json')
        feature_config = load_json(processed / 'feature_config.json')
        raw_hash = file_sha256(path / 'synthetic/raw/synthetic_logs.csv')
        save_json({'snapshot_id': raw_hash, 'preparation_id': path.name,
                   'sequence_length': feature_config['sequence_length'],
                   'tabular_columns': feature_config['feature_columns'],
                   'sequence_columns': feature_config['feature_columns_lstm']}, path / 'contract.json')
        save_json(state, path / 'feature_state.json')
        save_json(environment_metadata(), path / 'environment.json')
        for split in ('train', 'validation', 'test'):
            frame = pd.read_csv(processed / f'events_{split}.csv')
            frame.to_csv(path / f'{split}.csv', index=False)
        finish(path)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': str(error)}, path / 'status.json')
        raise
    return path


def inputs(frame, contract, model):
    if "event_id" not in frame or frame["event_id"].duplicated().any():
        raise ValueError("Preparation requires stable unique event IDs")
    seq = build_sequences(frame, contract['sequence_columns'], contract['sequence_length'])
    if not len(seq['y']):
        raise ValueError('Empty endpoint cohort')
    ids = seq['item_ids'].astype(str)
    endpoints = frame.set_index('event_id').loc[ids]
    x = seq['X'] if model == 'lstm_autoencoder' else endpoints[contract['tabular_columns']].to_numpy()
    return x, seq['y'], ids, seq['member_ids']


def score(model, x, name):
    if name == 'random_forest':
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    if name == 'isolation_forest':
        return -model.score_samples(x)
    import torch
    from train_lstm_autoencoder import score_reconstruction_errors
    return score_reconstruction_errors(model, x, device=torch.device('cpu'), batch_size=128)


def configured_score(model, x, name, config):
    """Named extensions; absent policy delegates to the exact frozen scorer."""
    from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE, WINDOW_SCORE, endpoint_errors
    if name == BOOST_MODEL:
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    if name == 'lstm_autoencoder':
        policy = config[name].get('score_policy', WINDOW_SCORE)
        if policy == ENDPOINT_SCORE:
            return endpoint_errors(model, x)
        if policy != WINDOW_SCORE:
            raise ValueError('Unsupported LSTM score policy')
    return score(model, x, name)


def candidate(preparation, root, name, config_path=None, *, diagnostics=False):
    from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE, WINDOW_SCORE
    if diagnostics and name != 'lstm_autoencoder':
        raise ValueError('Reconstruction diagnostics require LSTM')
    preparation = Path(preparation).resolve()
    # Crucially do not open, hash or deserialize test artifacts in this process.
    names = ['contract.json', 'feature_state.json', 'config.json', 'train.csv', 'validation.csv']
    verify(preparation, names)
    contract = load_json(preparation / 'contract.json')
    config = load_json(config_path or preparation / 'config.json')
    original = load_json(preparation / 'config.json')
    for key in ('generator', 'split', 'features', 'scenario'):
        if config[key] != original[key]:
            raise ValueError('Candidate cannot change preparation configuration')
    path = allocate(root, 'candidate')
    try:
        set_seed(config['seed'])
        train = pd.read_csv(preparation / 'train.csv')
        validation = pd.read_csv(preparation / 'validation.csv')
        x, y, ids, members = inputs(validation, contract, name)
        if name == 'lstm_autoencoder':
            import torch
            from train_lstm_autoencoder import LSTMAutoencoder
            cfg = config[name]
            if cfg.get('score_policy', WINDOW_SCORE) not in (WINDOW_SCORE, ENDPOINT_SCORE):
                raise ValueError('Unsupported LSTM score policy')
            if cfg.get('selection_metric', 'f1') != 'f1':
                raise ValueError('Isolated protocol supports validation F1 only')
            if min(cfg['epochs'], cfg['batch_size'], cfg['hidden_size'], cfg['early_stopping_patience']) <= 0:
                raise ValueError('Training parameters must be positive')
            sequences = build_sequences(train, contract['sequence_columns'], contract['sequence_length'])
            # Preserve v0.2 endpoint-normal training policy explicitly.
            normal_mask = sequences['y'] == 0
            if contract.get('normal_window_policy') == 'all_members_background':
                member_labels = train.set_index('event_id')['label'].loc[sequences['member_ids'].ravel()].to_numpy()
                normal_mask = (member_labels.reshape(sequences['member_ids'].shape) == 0).all(axis=1)
            xt = sequences['X'][normal_mask]
            np.savez_compressed(path / 'train_inputs.npz', X=xt,
                                item_ids=sequences['item_ids'][normal_mask],
                                member_ids=sequences['member_ids'][normal_mask],
                                y=sequences['y'][normal_mask])
            if not len(xt):
                raise ValueError('No endpoint-normal training windows')
            model = LSTMAutoencoder(x.shape[2], cfg['hidden_size'])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xt)),
                                                batch_size=cfg['batch_size'], shuffle=True)
            validation_background = None
            if diagnostics:
                validation_labels = validation.set_index('event_id')['label'].loc[members.ravel()].to_numpy()
                validation_background = (validation_labels.reshape(members.shape) == 0).all(axis=1)
            best = None
            history = []
            stale = 0
            for epoch in range(cfg['epochs']):
                started = time.perf_counter()
                train_loss_sum = 0.0
                model.train()
                for (batch,) in loader:
                    optimizer.zero_grad()
                    loss = ((model(batch) - batch) ** 2).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite reconstruction loss')
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.detach()) * len(batch)
                train_seconds = time.perf_counter() - started
                started = time.perf_counter()
                values = configured_score(model, x, name, config)
                threshold, percentile, metrics, grid = select_best_threshold(values, y, cfg['threshold_percentiles'])
                selection_seconds = time.perf_counter() - started
                key = (metrics['f1_score'], metrics['recall'], -metrics['false_positive_rate'])
                row = {'epoch': epoch + 1, 'validation': metrics,
                       'threshold': threshold, 'percentile': percentile,
                       'train_optimization_mse': train_loss_sum / len(xt),
                       'training_seconds': train_seconds, 'validation_selection_seconds': selection_seconds}
                if diagnostics:
                    started = time.perf_counter()
                    # Even a non-shuffled DataLoader consumes a base seed. Preserve
                    # RNG so extra diagnostic passes cannot change training order.
                    with torch.random.fork_rng(devices=[]):
                        train_errors = configured_score(model, xt, name, config)
                    def mean_errors(mask):
                        return float(values[mask].mean()) if mask.any() else None
                    row['reconstruction'] = {
                        'train_fit_mse': float(train_errors.mean()),
                        'validation_all_members_background_mse': mean_errors(validation_background),
                        'validation_background_endpoint_mse': mean_errors(y == 0),
                        'validation_positive_endpoint_mse': mean_errors(y == 1),
                    }
                    row['diagnostics_seconds'] = time.perf_counter() - started
                    save_json(row, path / 'progress.json')
                    print(json.dumps({'candidate': path.name, **row}), file=sys.stderr, flush=True)
                history.append(row)
                if best is None or key > best[0]:
                    best = (key, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch + 1)
                    stale = 0
                else:
                    stale += 1
                if stale >= cfg['early_stopping_patience']:
                    break
            if best is None:
                raise ValueError('No training epochs')
            model.load_state_dict(best[1])
            torch.save(model.state_dict(), path / 'model.pt')
            save_json({'epochs': history, 'selected_epoch': best[2],
                       'stopping_reason': 'patience' if stale >= cfg['early_stopping_patience'] else 'max_epochs',
                       'diagnostics_enabled': diagnostics,
                       'reconstruction_cohorts': {
                           'train_fit_windows': len(xt),
                           'validation_all_background_windows': int(validation_background.sum()) if diagnostics else None,
                           'validation_background_endpoints': int((y == 0).sum()),
                           'validation_positive_endpoints': int((y == 1).sum())}}, path / 'history.json')
            grid_values = cfg['threshold_percentiles']
        else:
            cfg = config['baseline'][name]
            if name == BOOST_MODEL:
                from sklearn.ensemble import HistGradientBoostingClassifier
                if cfg.get('early_stopping') is not False:
                    raise ValueError('Grouped boosting forbids an automatic random validation split')
                model = HistGradientBoostingClassifier(**cfg, random_state=config['seed'])
            else:
                cls = RandomForestClassifier if name == 'random_forest' else IsolationForest
                model = cls(**cfg, random_state=config['seed'], n_jobs=1)
            xt = train[contract['tabular_columns']].to_numpy()
            yt = train['label'].to_numpy()
            if name in ('random_forest', BOOST_MODEL):
                model.fit(xt, yt)
            else:
                model.fit(xt[yt == 0])
            mask = np.ones(len(yt), dtype=bool) if name in ('random_forest', BOOST_MODEL) else yt == 0
            np.savez_compressed(path / 'train_inputs.npz', X=xt[mask], y=yt[mask],
                                item_ids=train['event_id'].to_numpy(dtype=str)[mask])
            joblib.dump(model, path / 'model.joblib')
            grid_values = config['baseline']['threshold_percentiles']
        values = configured_score(model, x, name, config)
        threshold, _, metrics, grid = select_best_threshold(values, y, grid_values)
        save_json(config, path / 'config.json')
        save_json(contract, path / 'contract.json')
        shutil.copy2(preparation / 'feature_state.json', path / 'feature_state.json')
        save_json(environment_metadata(), path / 'environment.json')
        selection = {'model': name, 'threshold': threshold, 'preparation_id': preparation.name,
                   'preparation_manifest_sha256': file_sha256(preparation / 'complete.json'),
                   'training_policy': contract.get('normal_window_policy', 'endpoint-normal') if name == 'lstm_autoencoder' else 'supervised-all / IF-background-only',
                   'selection_metric': 'validation_f1', 'test_accessed': False}
        if name == 'lstm_autoencoder' and 'score_policy' in config[name]:
            selection['score_policy'] = config[name]['score_policy']
        if name == BOOST_MODEL:
            selection.update(score_policy='positive_class_probability_v1', training_policy='supervised_all_inner_fit_days')
        save_json(selection, path / 'selection.json')
        save_json(metrics, path / 'metrics_validation.json')
        save_json(grid, path / 'threshold_search.json')
        shutil.copy2(preparation / 'train.csv', path / 'train.csv')
        shutil.copy2(preparation / 'validation.csv', path / 'validation.csv')
        np.savez_compressed(path / 'validation_inputs.npz', X=x, y=y, item_ids=ids, member_ids=members)
        pd.DataFrame({'item_id': ids, 'y_true': y, 'score': values,
                      'prediction': (values >= threshold).astype(int)}).to_csv(path / 'predictions_validation.csv', index=False)
        finish(path)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': str(error)}, path / 'status.json')
        raise
    return path


def final_evaluate(candidate_path, preparation, root):
    candidate_path, preparation = Path(candidate_path).resolve(), Path(preparation).resolve()
    verify(candidate_path)
    selection = load_json(candidate_path / 'selection.json')
    if file_sha256(preparation / 'complete.json') != selection['preparation_manifest_sha256']:
        raise ValueError('Different preparation')
    name = selection['model']
    contract = load_json(candidate_path / 'contract.json')
    config = load_json(candidate_path / 'config.json')
    if contract.get('scope') == 'development_group_cv':
        raise ValueError('Development CV requires assessment, not real final evaluation')
    if contract.get('scope') == 'primary_development':
        raise ValueError('Primary development preparation has no real test population')
    verify(preparation, ['contract.json', 'test.csv'])
    if contract != load_json(preparation / 'contract.json'):
        raise ValueError('Candidate and preparation contracts differ')
    path = allocate(root, 'evaluation')
    try:
        started = time.perf_counter()
        x, y, ids, members = inputs(pd.read_csv(preparation / 'test.csv'), contract, name)
        input_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if name == 'lstm_autoencoder':
            import torch
            from train_lstm_autoencoder import LSTMAutoencoder
            model = LSTMAutoencoder(len(contract['sequence_columns']), config[name]['hidden_size'])
            model.load_state_dict(torch.load(candidate_path / 'model.pt', map_location='cpu', weights_only=True))
        else:
            model = joblib.load(candidate_path / 'model.joblib')
        load_seconds = time.perf_counter() - started
        timing_config = config.get('timing', {})
        values, timing = measure_inference(lambda: configured_score(model, x, name, config), record_count=len(y),
            warmup_runs=timing_config.get('warmup_runs', 1), measured_runs=timing_config.get('measured_runs', 5))
        save_json({'prepared_input_load_and_window_seconds': input_seconds,
                   'model_load_seconds': load_seconds, 'inference': timing,
                   'device': 'cpu', 'batch_size': 128 if name == 'lstm_autoencoder' else len(y),
                   'scope': 'prepared full test endpoint cohort; preprocessing fit and training excluded'}, path / 'timing.json')
        save_json(environment_metadata(), path / 'environment.json')
        predictions = (values >= selection['threshold']).astype(int)
        save_json(compute_classification_metrics(y, predictions, values), path / 'metrics_test.json')
        save_json({'candidate_id': candidate_path.name, 'preparation_id': preparation.name,
                   'candidate_manifest_sha256': file_sha256(candidate_path / 'complete.json'),
                   'preparation_manifest_sha256': selection['preparation_manifest_sha256'],
                   'snapshot_id': contract['snapshot_id'], 'threshold': selection['threshold'],
                   'model': name, 'seed': config['seed'], 'history_policy': contract.get('history_policy'),
                   'score_dtype': str(values.dtype),
                   'fit_or_threshold_selection': False,
                   'target': contract.get('target', 'endpoint_event')}, path / 'lineage.json')
        np.savez_compressed(path / 'test_inputs.npz', X=x, y=y, item_ids=ids, member_ids=members)
        pd.DataFrame({'item_id': ids, 'y_true': y, 'score': values,
                      'prediction': predictions}).to_csv(path / 'predictions_test.csv', index=False)
        # Preserve exact float32/float64 scores at threshold ties; CSV is a view.
        np.savez_compressed(path / 'predictions_test.npz', item_ids=ids, y=y,
                            scores=values, predictions=predictions)
        finish(path)
    except BaseException as error:
        save_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': str(error)}, path / 'status.json')
        raise
    return path


def compare(evaluations, *, allow_different_preparations=False):
    reference = None
    reference_lineage = None
    rows = []
    for location in evaluations:
        location = Path(location)
        verify(location)
        lineage = load_json(location / 'lineage.json')
        cohort = pd.read_csv(location / 'predictions_test.csv')[['item_id', 'y_true']].sort_values('item_id').reset_index(drop=True)
        if cohort['item_id'].duplicated().any():
            raise ValueError('Duplicate evaluation items')
        identity = (lineage['snapshot_id'], lineage['target'])
        if not allow_different_preparations:
            identity = (lineage['preparation_id'], *identity)
        if reference is not None and (identity != reference_lineage or not cohort.equals(reference)):
            raise ValueError('Incompatible evaluation populations or targets')
        reference, reference_lineage = cohort, identity
        rows.append({'evaluation': location.name, 'candidate': lineage['candidate_id'],
                     'preparation': lineage['preparation_id'], 'history_policy': lineage.get('history_policy'),
                     'model': lineage.get('model'), 'seed': lineage.get('seed'),
                     **load_json(location / 'metrics_test.json')})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing artifact root, e.g. on SSD')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('prepare-cert'); p.add_argument('--days', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p = sub.add_parser('candidate'); p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--model', choices=['random_forest', 'isolation_forest', 'lstm_autoencoder', 'hist_gradient_boosting'], required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--diagnostics', action='store_true', help='LSTM reconstruction curves; preserves training RNG')
    p = sub.add_parser('final'); p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--preparation', type=Path, required=True)
    p = sub.add_parser('compare'); p.add_argument('evaluations', type=Path, nargs='+')
    p.add_argument('--allow-different-preparations', action='store_true',
                   help='Explicit comparison of preparations; snapshot/target/exact items and labels must still match')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.config, args.root)
    elif args.command == 'prepare-cert':
        from datasets.cert_experiments import prepare_cert
        result = prepare_cert(args.days, args.config, args.root)
    elif args.command == 'candidate': result = candidate(args.preparation, args.root, args.model, args.config,
                                                        diagnostics=args.diagnostics)
    elif args.command == 'final': result = final_evaluate(args.candidate, args.preparation, args.root)
    else: result = json.dumps(compare(args.evaluations, allow_different_preparations=args.allow_different_preparations), indent=2)
    print(result)


if __name__ == '__main__':
    main()
