"""Read-only CPU inference for sealed isolated candidates and operating bundles.

Only locally trusted joblib/PyTorch artifacts are supported. Hashes bind saved
files; they are not signatures. This module never fits or selects a threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier

from cert_model_variants import BOOST_MODEL, ENDPOINT_SCORE, WINDOW_SCORE
from experiment import configured_score, verify
from experiment_tracking import file_sha256
from features import (BINARY_AND_CYCLICAL_FEATURES, CATEGORICAL_COLUMNS,
                      CONTINUOUS_FEATURES, LSTM_FEATURE_COLUMNS, build_sequences,
                      safe_category_token)
from utils import load_json


SCORE_POLICIES = {
    'random_forest': 'positive_class_probability_v1',
    'isolation_forest': 'negative_score_samples_v1',
    BOOST_MODEL: 'positive_class_probability_v1',
    'lstm_autoencoder': WINDOW_SCORE,
}
SCORE_DIRECTION = 'higher_is_more_anomalous'
DECISION_OPERATOR = '>='
INFERENCE_BATCH_SIZE = 128  # Both existing LSTM scorers use this, independently of fit batch size.
CERT_SEQUENCE_POLICY = 'last N active account-days; no padding; calendar gaps retained as feature'


def _digest(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'Invalid {name}: expected a SHA-256 digest')
    return value


def _columns(value: Any, name: str) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(c, str) or not c for c in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'Invalid ordered {name}')
    return tuple(value)


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError('A finite numeric frozen threshold is required')
    return float(value)


def _sealed_files(path: Path) -> dict[str, str]:
    """Validate names before using the existing full-bundle checksum verifier."""
    manifest = load_json(path / 'complete.json')
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError('Expected a sealed isolated artifact (complete.json files map)')
    for name, digest in files.items():
        relative = Path(name)
        if (not name or relative.is_absolute() or '..' in relative.parts
                or not (path / relative).resolve().is_relative_to(path)):
            raise ValueError(f'Invalid artifact member: {name}')
        _digest(digest, f'artifact checksum for {name}')
    verify(path)
    if 'status.json' not in files or load_json(path / 'status.json').get('status') != 'complete':
        raise ValueError('Frozen artifact status must be complete')
    return files


def _feature_policy(contract: dict, config: dict, state: dict) -> str:
    """Check saved schema/state agreement without recomputing or fitting inputs."""
    tabular = _columns(contract.get('tabular_columns'), 'tabular columns')
    sequence = _columns(contract.get('sequence_columns'), 'sequence columns')
    provenance = {'label', 'y', 'y_true', 'target', 'answer_key', 'event_id', 'item_id',
                  'user_id', 'account_id', 'timestamp', 'split', 'group', 'source_file'}
    if provenance.intersection((*tabular, *sequence)):
        raise ValueError('Labels and identifier provenance cannot be inference features')
    length = _positive_integer(contract.get('sequence_length'), 'sequence_length')
    if config['features'].get('sequence_length') != length:
        raise ValueError('Configuration and contract sequence lengths differ')
    scenario = contract.get('scenario', config.get('scenario'))
    if scenario != config.get('scenario'):
        raise ValueError('Configuration and contract scenarios differ')
    if scenario == 'cert_r4_2':
        from datasets.cert_experiments import POLICIES
        policy = contract.get('history_policy')
        if policy not in POLICIES:
            raise ValueError('Unsupported CERT history policy')
        if state.get('policy') != policy or config['features'].get('history_policy') != policy:
            raise ValueError('Feature state/configuration history policy differs from contract')
        if tuple(state.get('columns', ())) != tabular or tabular != sequence:
            raise ValueError('Feature state and ordered columns differ')
        mean, scale = np.asarray(state.get('mean'), dtype=float), np.asarray(state.get('scale'), dtype=float)
        if (mean.shape != (len(tabular),) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Invalid frozen feature scaler')
        if contract.get('normal_window_policy') != 'all_members_background':
            raise ValueError('Unsupported CERT normal window policy')
        if contract.get('sequence_policy') != CERT_SEQUENCE_POLICY:
            raise ValueError('Unsupported CERT sequence policy')
        return policy
    if scenario != 'synthetic':
        raise ValueError(f'Unsupported prepared scenario: {scenario}')
    if (contract.get('history_policy') is not None or state.get('policy') is not None
            or config['features'].get('history_policy') is not None
            or contract.get('normal_window_policy', 'endpoint-normal') != 'endpoint-normal'
            or state.get('profile_fit_policy') != 'normal-train-events-only'):
        raise ValueError('Unsupported synthetic feature/history policy')
    if (state.get('continuous_features') != CONTINUOUS_FEATURES
            or state.get('binary_and_cyclical_features') != BINARY_AND_CYCLICAL_FEATURES
            or state.get('categorical_columns') != CATEGORICAL_COLUMNS
            or state.get('rolling_window_events') != config['features'].get('rolling_window_events')):
        raise ValueError('Synthetic feature state differs from supported builder')
    expected = [f'feat_{c}' for c in (*CONTINUOUS_FEATURES, *BINARY_AND_CYCLICAL_FEATURES)]
    for c in CONTINUOUS_FEATURES:
        if (not np.isfinite(state['continuous_mean'][c]) or not np.isfinite(state['continuous_scale'][c])
                or state['continuous_scale'][c] <= 0):
            raise ValueError('Invalid frozen feature scaler')
    for c in CATEGORICAL_COLUMNS:
        categories = _columns(state['categories'][c], f'{c} categories')
        expected.extend(f'feat_{c}__{safe_category_token(v)}' for v in categories)
        expected.append(f'feat_{c}__unknown')
    expected_sequence = [c for c in LSTM_FEATURE_COLUMNS if c in expected]
    expected_sequence.extend(c for c in expected if c.startswith(
        ('feat_event_type__', 'feat_login_status__', 'feat_protocol__')))
    if tabular != tuple(expected) or sequence != tuple(dict.fromkeys(expected_sequence)):
        raise ValueError('Synthetic feature state and ordered columns differ')
    return 'synthetic_train_state_v1'


@dataclass(frozen=True)
class PreparedBatch:
    """Numeric inputs and their provenance; no labels or threshold overrides.

    Arrays retain their original dtype and order. The caller owns them and must
    not mutate them during inference. Membership may overlap between windows.
    """

    X: np.ndarray
    feature_columns: tuple[str, ...]
    item_ids: np.ndarray
    member_ids: np.ndarray
    preparation_id: str
    preparation_manifest_sha256: str
    feature_state_sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'feature_columns', _columns(self.feature_columns, 'batch feature columns'))


@dataclass(frozen=True)
class InferenceSpec:
    model: str
    scenario: str
    snapshot_id: str
    preparation_id: str
    preparation_manifest_sha256: str
    feature_state_sha256: str
    feature_columns: tuple[str, ...]
    sequence_length: int
    feature_policy: str
    score_policy: str
    score_direction: str
    decision_operator: str
    threshold: float
    profile: str
    available_profiles: tuple[str, ...]
    training_policy: str
    scope: str
    device: str = 'cpu'
    batch_size: int | None = None


@dataclass(frozen=True)
class FrozenRun:
    spec: InferenceSpec
    artifact_path: Path
    artifact_hashes: Mapping[str, str]
    _model: Any = field(repr=False, compare=False)
    _config: dict = field(repr=False, compare=False)

    @property
    def metadata(self) -> dict:
        return {**asdict(self.spec), 'artifact_path': str(self.artifact_path),
                'artifact_hashes': dict(self.artifact_hashes),
                'semantics_source': 'isolated_candidate_v1 / operating_bundle_v6; explicit declarations checked',
                'fit_or_threshold_selection': False}

    def validate_batch(self, batch: PreparedBatch) -> None:
        if not isinstance(batch, PreparedBatch):
            raise TypeError('Inference requires a PreparedBatch without labels')
        spec, x = self.spec, batch.X
        if (batch.preparation_id != spec.preparation_id
                or batch.preparation_manifest_sha256 != spec.preparation_manifest_sha256
                or batch.feature_state_sha256 != spec.feature_state_sha256):
            raise ValueError('Batch preparation or frozen feature state identity differs')
        if batch.feature_columns != spec.feature_columns:
            raise ValueError('Batch feature order differs from frozen schema')
        ndim = 3 if spec.model == 'lstm_autoencoder' else 2
        if (not isinstance(x, np.ndarray) or x.ndim != ndim or x.dtype not in (np.float32, np.float64)
                or x.shape[-1] != len(spec.feature_columns) or not np.isfinite(x).all()):
            raise ValueError('Batch requires finite float32/float64 inputs with the frozen shape')
        if ndim == 3 and (x.shape[1] != spec.sequence_length or x.dtype != np.float32):
            raise ValueError('LSTM requires the frozen sequence length and float32 inputs')
        ids, members = batch.item_ids, batch.member_ids
        if (not isinstance(ids, np.ndarray) or ids.ndim != 1 or ids.dtype.kind != 'U'
                or ids.shape != (len(x),) or (ids == '').any() or len(np.unique(ids)) != len(ids)):
            raise ValueError('Item IDs must be unique nonempty Unicode strings aligned with inputs')
        if (not isinstance(members, np.ndarray) or members.dtype.kind != 'U'
                or members.shape != (len(x), spec.sequence_length) or (members == '').any()):
            raise ValueError('Window membership must match frozen length and item order')
        if not np.array_equal(ids, members[:, -1]):
            raise ValueError('Window endpoint membership differs from item IDs')
        ordered = np.sort(members, axis=1)
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError('Repeated member inside a window')

    def score(self, batch: PreparedBatch) -> np.ndarray:
        self.validate_batch(batch)
        if not len(batch.X):
            return np.empty(0, dtype=np.float32 if self.spec.model == 'lstm_autoencoder' else np.float64)
        values = configured_score(self._model, batch.X, self.spec.model, self._config)
        if values.shape != (len(batch.X),) or not np.isfinite(values).all():
            raise ValueError('Frozen scorer returned invalid scores')
        return values

    def predict(self, batch: PreparedBatch) -> np.ndarray:
        # Every supported saved runner uses >=, including threshold ties.
        return (self.score(batch) >= self.spec.threshold).astype(np.int64)


@dataclass(frozen=True)
class FrozenLSTMRun(FrozenRun):
    def reconstruct(self, batch: PreparedBatch) -> np.ndarray:
        """Full-window reconstruction, even when the selected score is endpoint-only."""
        self.validate_batch(batch)
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        if not len(batch.X):
            return np.empty_like(batch.X)
        self._model.eval()
        loader = DataLoader(TensorDataset(torch.from_numpy(batch.X).float()),
                            batch_size=INFERENCE_BATCH_SIZE, shuffle=False)
        result = []
        with torch.no_grad():
            for (values,) in loader:
                result.append(self._model(values).cpu().numpy())
        reconstruction = np.concatenate(result)
        if reconstruction.shape != batch.X.shape or not np.isfinite(reconstruction).all():
            raise ValueError('Frozen model returned invalid reconstruction')
        return reconstruction


def _load_frozen_run(path: Path, profile: str | None, expected_manifest_sha256: str | None) -> FrozenRun:
    digest = file_sha256(path / 'complete.json')
    if expected_manifest_sha256 is not None and digest != _digest(expected_manifest_sha256, 'expected manifest'):
        raise ValueError('Frozen artifact manifest identity differs')
    files = _sealed_files(path)
    required = {'config.json', 'contract.json', 'feature_state.json'}
    operating = 'operating_points.json' in files
    if operating == ('selection.json' in files):
        raise ValueError('Expected exactly one candidate selection or operating-point bundle')
    required.add('lineage.json' if operating else 'selection.json')
    if not required.issubset(files):
        raise ValueError(f'Missing frozen artifacts: {sorted(required - files.keys())}')
    config, contract, state = [load_json(path / n) for n in ('config.json', 'contract.json', 'feature_state.json')]
    feature_policy = _feature_policy(contract, config, state)
    source = load_json(path / ('lineage.json' if operating else 'selection.json'))
    name = source['model']
    if name not in SCORE_POLICIES:
        raise ValueError(f'Unsupported frozen model: {name}')
    if operating:
        choices = load_json(path / 'operating_points.json')
        if not isinstance(choices, dict) or not choices:
            raise ValueError('Operating bundle has no profiles')
        for key, choice in choices.items():
            if not isinstance(choice, dict) or not key:
                raise ValueError('Invalid operating profile')
            _threshold(choice.get('threshold'))
        if profile is None:
            raise ValueError(f'Explicit operating profile required; available: {list(choices)}')
        if profile not in choices:
            raise ValueError(f'Unknown operating profile {profile!r}; available: {list(choices)}')
        if source.get('no_refit') is not True or source.get('calibration_only') is not True:
            raise ValueError('Unsupported operating bundle freeze policy')
        preparation_id = Path(source['preparation']).name
        _digest(source.get('source_candidate_manifest_sha256'), 'source candidate manifest')
    else:
        choices = {'legacy_f1': source}
        if profile not in (None, 'legacy_f1'):
            raise ValueError('Isolated candidate has only the legacy_f1 profile')
        profile = 'legacy_f1'
        preparation_id = source['preparation_id']
        if source.get('test_accessed') is not False:
            raise ValueError('Unsupported candidate selection policy')
    if preparation_id != contract.get('preparation_id') or not preparation_id:
        raise ValueError('Frozen candidate and contract preparation identities differ')
    preparation_digest = _digest(source['preparation_manifest_sha256'], 'preparation manifest')
    threshold = _threshold(choices[profile].get('threshold'))
    model_config = config[name] if name == 'lstm_autoencoder' else config['baseline'][name]
    policy = model_config.get('score_policy', SCORE_POLICIES[name])
    allowed = (WINDOW_SCORE, ENDPOINT_SCORE) if name == 'lstm_autoencoder' else (SCORE_POLICIES[name],)
    if policy not in allowed:
        raise ValueError(f'Unsupported {name} score policy: {policy}')
    for record in (contract, config, model_config, source, *choices.values()):
        for key, expected in (('score_policy', policy), ('score_direction', SCORE_DIRECTION),
                              ('decision_operator', DECISION_OPERATOR), ('decision', 'score >= threshold')):
            if key in record and record[key] != expected:
                raise ValueError(f'Unsupported or inconsistent frozen {key}: {record[key]}')
    if 'threshold' in contract and _threshold(contract['threshold']) != threshold:
        raise ValueError('Contract and selected frozen thresholds differ')
    model_file = 'model.pt' if name == 'lstm_autoencoder' else 'model.joblib'
    if model_file not in files:
        raise ValueError(f'Missing frozen weights: {model_file}')
    if operating and source.get('checkpoint_sha256') != files[model_file]:
        raise ValueError('Frozen operating checkpoint identity differs')
    columns = tuple(contract['sequence_columns'] if name == 'lstm_autoencoder' else contract['tabular_columns'])
    if name == 'lstm_autoencoder':
        import torch
        from train_lstm_autoencoder import LSTMAutoencoder
        hidden = _positive_integer(config[name]['hidden_size'], 'hidden_size')
        model = LSTMAutoencoder(len(columns), hidden)
        model.load_state_dict(torch.load(path / model_file, map_location='cpu', weights_only=True), strict=True)
        model.eval()
        runtime_class = FrozenLSTMRun
    else:
        try:
            model = joblib.load(path / model_file)
        except Exception as error:
            raise ValueError(f'Cannot deserialize frozen tabular weights: {error}') from error
        expected_class = {'random_forest': RandomForestClassifier, 'isolation_forest': IsolationForest,
                          BOOST_MODEL: HistGradientBoostingClassifier}[name]
        if not isinstance(model, expected_class) or getattr(model, 'n_features_in_', None) != len(columns):
            raise ValueError('Serialized model family or feature dimension differs from contract')
        if name != 'isolation_forest' and not np.array_equal(model.classes_, [0, 1]):
            raise ValueError('Frozen classifier requires background=0 and positive=1 classes')
        runtime_class = FrozenRun
    training_policy = ('background_only_all_members' if contract.get('normal_window_policy') == 'all_members_background'
                       else 'background_only_endpoints') if name == 'lstm_autoencoder' else (
                       'background_only_rows' if name == 'isolation_forest' else 'supervised_all_train_rows')
    spec = InferenceSpec(model=name, scenario=config['scenario'], snapshot_id=contract['snapshot_id'],
        preparation_id=preparation_id, preparation_manifest_sha256=preparation_digest,
        feature_state_sha256=files['feature_state.json'], feature_columns=columns,
        sequence_length=contract['sequence_length'], feature_policy=feature_policy, score_policy=policy,
        score_direction=SCORE_DIRECTION, decision_operator=DECISION_OPERATOR, threshold=threshold,
        profile=profile, available_profiles=tuple(choices), training_policy=training_policy,
        scope=contract.get('scope', 'synthetic_chronological_holdout'),
        batch_size=INFERENCE_BATCH_SIZE if name == 'lstm_autoencoder' else None)
    hashes = MappingProxyType({'complete.json': digest, **files})
    return runtime_class(spec=spec, artifact_path=path, artifact_hashes=hashes, _model=model, _config=config)


def load_frozen_run(path: str | Path, *, profile: str | None = None,
                    expected_manifest_sha256: str | None = None) -> FrozenRun:
    """Load a sealed candidate without opening its original preparation/source.

    Operating bundles require an explicit profile, including legacy_f1. Legacy
    candidates have one saved threshold and may omit profile. Missing semantics
    are resolved only by their known historical formats, never by score quality.
    """
    path = Path(path).expanduser().resolve()
    try:
        return _load_frozen_run(path, profile, expected_manifest_sha256)
    except (KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError(f'Invalid frozen inference artifact {path}: {error}') from error


def batch_from_frame(frame: pd.DataFrame, *, contract: dict, model: str,
                     preparation_manifest_sha256: str, feature_state_sha256: str) -> PreparedBatch:
    """Use the existing window builder on one already prepared split.

    Identity hashes describe the producer's preparation/state. Only numeric
    features, event_id, user_id and timestamp enter the builder. Any reporting
    labels are discarded; zero placeholders satisfy its legacy return schema.
    This helper does not transform raw data or verify a split artifact's seal.
    """
    if model not in SCORE_POLICIES:
        raise ValueError(f'Unsupported frozen model: {model}')
    if 'split' in frame and frame['split'].nunique(dropna=False) != 1:
        raise ValueError('Batch builder requires one split')
    sequence = _columns(contract['sequence_columns'], 'sequence columns')
    tabular = _columns(contract['tabular_columns'], 'tabular columns')
    columns = list(dict.fromkeys(['event_id', 'user_id', 'timestamp', *sequence, *tabular]))
    clean = frame.loc[:, columns].copy()
    clean['label'] = 0
    windows = build_sequences(clean, list(sequence), contract['sequence_length'])
    ids, members = windows['item_ids'], windows['member_ids']
    x = windows['X'] if model == 'lstm_autoencoder' else clean.set_index('event_id').loc[ids, list(tabular)].to_numpy()
    return PreparedBatch(X=x, feature_columns=sequence if model == 'lstm_autoencoder' else tabular,
        item_ids=ids, member_ids=members, preparation_id=contract['preparation_id'],
        preparation_manifest_sha256=preparation_manifest_sha256, feature_state_sha256=feature_state_sha256)
