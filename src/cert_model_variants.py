"""Explicit CERT score/model variants; legacy inference stays unchanged."""
import numpy as np

WINDOW_SCORE = 'window_mean_mse_v1'
ENDPOINT_SCORE = 'endpoint_mean_mse_v2'
BOOST_MODEL = 'hist_gradient_boosting'


def endpoint_errors(model, sequences, *, batch_size=128):
    """Reconstruct all inputs, score the last day only; this is not forecasting."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if sequences.ndim != 3 or min(sequences.shape[1:]) < 1 or batch_size < 1:
        raise ValueError('Endpoint score requires nonempty time/feature dimensions')
    if not np.isfinite(sequences).all():
        raise ValueError('Non-finite endpoint inputs')
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(sequences).float()), batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstruction = model(batch)
            errors.append(((reconstruction[:, -1, :] - batch[:, -1, :]) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errors).astype(np.float32) if errors else np.empty(0, dtype=np.float32)
