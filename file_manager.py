from pathlib import Path
import json
import numpy as np
import torch


def _as_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    return np.asarray(x, dtype=np.float64)


def save_C1_seed(filename, theta_amp, theta_phase, *, eM, q, lam=0.0, m_over_e=0.0, k=1, loss=None, diagnostics=None):
    """
    Save a C1 solution for reuse as a continuation seed.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    theta_amp = _as_numpy(theta_amp)
    theta_phase = _as_numpy(theta_phase)

    metadata = {
        "eM": float(eM),
        "q": float(q),
        "lam": float(lam),
        "m_over_e": float(m_over_e),
        "k": int(k),
        "p_amp": int((theta_amp.size - 1) // 3),
        "p_phase": int((theta_phase.size - 2) // 3),
        "loss": None if loss is None else float(
            loss.detach().cpu() if torch.is_tensor(loss) else loss
        ),
        "diagnostics": diagnostics or {},
    }

    np.savez_compressed(filename, theta_amp=theta_amp, theta_phase=theta_phase, metadata_json=np.asarray(json.dumps(metadata)))
    print(f"Saved seed to {filename}")