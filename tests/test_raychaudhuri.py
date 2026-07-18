import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import solver


DTYPE = torch.float64


def assert_tensors_close(test_case, actual, expected, *, rtol=1e-11, atol=1e-12):
    difference = torch.max(torch.abs(actual - expected)).item()
    test_case.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=atol), msg=f"maximum absolute difference: {difference:.16e}")


def smooth_focusing(V):
    return 0.25 + 0.4 * V + 0.3 * V.square() + 0.1 * torch.sin(3.0 * V).square()


def load_seed(name):
    with np.load(ROOT / "seeds" / name, allow_pickle=False) as data:
        theta_amp = torch.tensor(data["theta_amp"], dtype=DTYPE)
        theta_phase = torch.tensor(data["theta_phase"], dtype=DTYPE)
        metadata = json.loads(data["metadata_json"].item())

    return theta_amp, theta_phase, metadata


def full_loss_and_gradients(recurrence, *, n_grid=1001):
    theta_amp_seed, theta_phase_seed, metadata = load_seed("C1_seed_50_p4.npz")
    theta_amp = theta_amp_seed.clone().requires_grad_(True)
    theta_phase = theta_phase_seed.clone().requires_grad_(True)
    production_recurrence = solver.raychaudhuri_rk4_torch

    try:
        solver.raychaudhuri_rk4_torch = recurrence
        loss = solver.loss_C1(theta_amp, theta_phase, metadata["eM"], q=metadata["q"], lam=metadata.get("lam", 0.0), m_over_e=metadata.get("m_over_e", 0.0), rho_drho_func=solver.tanh_ansatz, omega_domega_func=solver.omega_domega, n_grid=n_grid)
        gradients = torch.autograd.grad(loss, (theta_amp, theta_phase))
    finally:
        solver.raychaudhuri_rk4_torch = production_recurrence

    return loss.detach(), tuple(gradient.detach() for gradient in gradients)


def one_adam_step(recurrence, *, n_grid=1001, lr=1e-3):
    theta_amp_seed, theta_phase_seed, metadata = load_seed("C1_seed_50_p4.npz")
    theta_amp = torch.nn.Parameter(theta_amp_seed.clone())
    theta_phase = torch.nn.Parameter(theta_phase_seed.clone())
    optimizer = torch.optim.Adam((theta_amp, theta_phase), lr=lr)
    production_recurrence = solver.raychaudhuri_rk4_torch

    try:
        solver.raychaudhuri_rk4_torch = recurrence
        optimizer.zero_grad(set_to_none=True)
        loss = solver.loss_C1(theta_amp, theta_phase, metadata["eM"], q=metadata["q"], lam=metadata.get("lam", 0.0), m_over_e=metadata.get("m_over_e", 0.0), rho_drho_func=solver.tanh_ansatz, omega_domega_func=solver.omega_domega, n_grid=n_grid)
        loss.backward()
        optimizer.step()
    finally:
        solver.raychaudhuri_rk4_torch = production_recurrence

    return theta_amp.detach(), theta_phase.detach()


class RaychaudhuriTests(unittest.TestCase):
    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            solver._TransferRecurrence.apply(torch.zeros(4, 2, dtype=DTYPE))

        with self.assertRaisesRegex(ValueError, "at least two"):
            solver.raychaudhuri_rk4_torch(torch.zeros(1, dtype=DTYPE), torch.zeros(1, dtype=DTYPE), torch.zeros(0, dtype=DTYPE))

        V = torch.tensor([0.0, 0.7, 0.6, 1.0], dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            solver.raychaudhuri_rk4_torch(V, torch.zeros_like(V), torch.zeros_like(V[:-1]))

    def test_endpoint_data(self):
        V = torch.linspace(0.0, 1.0, 129, dtype=DTYPE)
        V_mid = 0.5 * (V[:-1] + V[1:])
        xi, xip = solver.raychaudhuri_rk4_torch(V, smooth_focusing(V), smooth_focusing(V_mid))
        self.assertEqual(xi[-1].item(), 1.0)
        self.assertEqual(xip[-1].item(), 0.0)

    def test_zero_focusing_uniform_and_nonuniform(self):
        grids = (torch.linspace(0.0, 1.0, 101, dtype=DTYPE), torch.tensor([0.0, 0.01, 0.07, 0.21, 0.58, 0.83, 1.0], dtype=DTYPE))

        for V in grids:
            with self.subTest(points=V.numel()):
                xi, xip = solver.raychaudhuri_rk4_torch(V, torch.zeros_like(V), torch.zeros_like(V[:-1]))
                assert_tensors_close(self, xi, torch.ones_like(V), rtol=0.0, atol=0.0)
                assert_tensors_close(self, xip, torch.zeros_like(V), rtol=0.0, atol=0.0)

    def test_legacy_forward_equivalence(self):
        grids = [torch.linspace(0.0, 1.0, points, dtype=DTYPE) for points in (5, 37, 1001)]
        grids.append(torch.tensor([0.0, 0.013, 0.09, 0.31, 0.55, 0.91, 1.0], dtype=DTYPE))

        for V in grids:
            V_mid = 0.5 * (V[:-1] + V[1:])
            choices = ((smooth_focusing(V), smooth_focusing(V_mid)), (0.1 + torch.exp(V), 0.1 + torch.exp(V_mid)))

            for E_grid, E_mid in choices:
                with self.subTest(points=V.numel(), choice=float(E_grid[0])):
                    legacy = solver._raychaudhuri_rk4_legacy(V, E_grid, E_mid)
                    transfer = solver.raychaudhuri_rk4_torch(V, E_grid, E_mid)
                    assert_tensors_close(self, transfer[0], legacy[0])
                    assert_tensors_close(self, transfer[1], legacy[1])

    def test_custom_recurrence_gradcheck(self):
        generator = torch.Generator().manual_seed(1729)
        M = (torch.eye(2, dtype=DTYPE).repeat(9, 1, 1) + 0.02 * torch.randn(9, 2, 2, dtype=DTYPE, generator=generator)).requires_grad_(True)
        passed = torch.autograd.gradcheck(solver._TransferRecurrence.apply, (M,), eps=1e-6, atol=1e-7, rtol=1e-5)
        self.assertTrue(passed)

    def test_gradients_with_respect_to_focusing(self):
        V = torch.linspace(0.0, 1.0, 31, dtype=DTYPE).pow(1.4)
        V_mid = 0.5 * (V[:-1] + V[1:])
        E_grid_values = smooth_focusing(V)
        E_mid_values = smooth_focusing(V_mid)
        weights_xi = torch.linspace(0.2, 1.3, V.numel(), dtype=DTYPE)
        weights_xip = torch.linspace(1.1, 0.4, V.numel(), dtype=DTYPE)

        def gradients(recurrence):
            E_grid = E_grid_values.clone().requires_grad_(True)
            E_mid = E_mid_values.clone().requires_grad_(True)
            xi, xip = recurrence(V, E_grid, E_mid)
            loss = torch.sum(weights_xi * xi.square() + weights_xip * xip.square())
            return torch.autograd.grad(loss, (E_grid, E_mid))

        legacy_gradients = gradients(solver._raychaudhuri_rk4_legacy)
        transfer_gradients = gradients(solver.raychaudhuri_rk4_torch)
        assert_tensors_close(self, transfer_gradients[0], legacy_gradients[0], rtol=2e-10, atol=2e-12)
        assert_tensors_close(self, transfer_gradients[1], legacy_gradients[1], rtol=2e-10, atol=2e-12)

    def test_full_C1_loss_and_parameter_gradients(self):
        production = solver.raychaudhuri_rk4_torch
        legacy_loss, legacy_gradients = full_loss_and_gradients(solver._raychaudhuri_rk4_legacy)
        transfer_loss, transfer_gradients = full_loss_and_gradients(production)
        assert_tensors_close(self, transfer_loss, legacy_loss, rtol=2e-10, atol=1e-15)
        assert_tensors_close(self, transfer_gradients[0], legacy_gradients[0], rtol=2e-9, atol=2e-12)
        assert_tensors_close(self, transfer_gradients[1], legacy_gradients[1], rtol=2e-9, atol=2e-12)

    def test_one_adam_step(self):
        production = solver.raychaudhuri_rk4_torch
        legacy_parameters = one_adam_step(solver._raychaudhuri_rk4_legacy)
        transfer_parameters = one_adam_step(production)
        assert_tensors_close(self, transfer_parameters[0], legacy_parameters[0], rtol=1e-9, atol=1e-10)
        assert_tensors_close(self, transfer_parameters[1], legacy_parameters[1], rtol=1e-9, atol=1e-10)

    def test_scipy_validator_baseline(self):
        import validator

        theta_amp, theta_phase, metadata = load_seed("C1_seed_35_p4.npz")
        validation = validator.validate_C1(theta_amp, theta_phase, eM=metadata["eM"], q=metadata["q"], lam=metadata.get("lam", 0.0), m_over_e=metadata.get("m_over_e", 0.0), k=metadata.get("k", 1), rho_drho_func=solver.tanh_ansatz, omega_domega_func=solver.omega_domega, rplus_and_Qtarget_func=solver.rplus_and_Qtarget, n_grid=1001, n_profile=1001, match_tol=1e-6)
        diagnostics = validation["diagnostics"]
        baseline = {"charge_residual": -7.25686433167283e-09, "Re_Phi_U_1": -4.641718685249874e-07, "Im_Phi_U_1": 3.8580070665927835e-08, "residual_norm": 4.6582894656997696e-07, "inf_xi": 0.005365295213026399, "sup_rU": -0.07606843615650094, "phase_winding": 1.6604841254726408}

        for key, expected in baseline.items():
            with self.subTest(diagnostic=key):
                self.assertAlmostEqual(diagnostics[key], expected, delta=1e-11)


if __name__ == "__main__":
    unittest.main()
