from pathlib import Path
from dataclasses import dataclass, field
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
    
def load_C1_seed(filename, *, device="cpu", dtype=torch.float64, trainable=True):
    """
    Load a saved C1 seed.

    Returns
    -------
    theta_amp, theta_phase, metadata
    """
    with np.load(filename, allow_pickle=False) as data:
        theta_amp_np = data["theta_amp"].copy()
        theta_phase_np = data["theta_phase"].copy()
        metadata = json.loads(data["metadata_json"].item())

    theta_amp = torch.tensor(theta_amp_np, dtype=dtype, device=device)
    theta_phase = torch.tensor(theta_phase_np, dtype=dtype, device=device)

    if trainable:
        theta_amp = torch.nn.Parameter(theta_amp)
        theta_phase = torch.nn.Parameter(theta_phase)

    return theta_amp, theta_phase, metadata

@dataclass
class C1Result:
    theta_amp: np.ndarray
    theta_phase: np.ndarray
    metadata: dict
    path: Path | None = None
    validation: dict | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def load(cls, filename):
        filename = Path(filename)

        with np.load(filename, allow_pickle=False) as data:
            return cls(
                theta_amp=data["theta_amp"].copy(),
                theta_phase=data["theta_phase"].copy(),
                metadata=json.loads(data["metadata_json"].item()),
                path=filename,
            )

    @property
    def diagnostics(self):
        return self.metadata.get("diagnostics", {})

    @property
    def eM(self):
        return self.metadata["eM"]

    @property
    def q(self):
        return self.metadata["q"]

    @property
    def p_amp(self):
        return (len(self.theta_amp) - 1) // 3

    @property
    def p_phase(self):
        return (len(self.theta_phase) - 2) // 3

    def to_torch(self, device="cpu", trainable=False):
        theta_amp = torch.tensor(
            self.theta_amp,
            dtype=torch.float64,
            device=device,
        )
        theta_phase = torch.tensor(
            self.theta_phase,
            dtype=torch.float64,
            device=device,
        )

        if trainable:
            theta_amp = torch.nn.Parameter(theta_amp)
            theta_phase = torch.nn.Parameter(theta_phase)

        return theta_amp, theta_phase

    def row(self):
        return {
            "file": None if self.path is None else self.path.name,
            "eM": self.eM,
            "q": self.q,
            "p_amp": self.p_amp,
            "p_phase": self.p_phase,
            **self.diagnostics,
        }
        
    def evaluate(
        self,
        rho_drho_func,
        omega_domega_func,
        rplus_and_Qtarget_func,
        *,
        force=False,
        **validator_kwargs,
    ):
        """
        Run and cache validate_C1.

        The returned dictionary contains V, Phi, xi, r_U, Q,
        diagnostics, and the other validation profiles.
        """
        if self.validation is None or force:
            import validator

            self.validation = validator.validate_C1(
                theta_amp=self.theta_amp,
                theta_phase=self.theta_phase,
                eM=self.metadata["eM"],
                q=self.metadata["q"],
                lam=self.metadata.get("lam", 0.0),
                m_over_e=self.metadata.get("m_over_e", 0.0),
                k=self.metadata.get("k", 1),
                rho_drho_func=rho_drho_func,
                omega_domega_func=omega_domega_func,
                rplus_and_Qtarget_func=rplus_and_Qtarget_func,
                **validator_kwargs,
            )

        return self.validation
        