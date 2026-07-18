import argparse
import gc
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import solver


DTYPE = torch.float64


def load_seed():
    with np.load(ROOT / "seeds" / "C1_seed_50_p4.npz", allow_pickle=False) as data:
        theta_amp = torch.tensor(data["theta_amp"], dtype=DTYPE)
        theta_phase = torch.tensor(data["theta_phase"], dtype=DTYPE)
        metadata = json.loads(data["metadata_json"].item())

    return theta_amp, theta_phase, metadata


def expand_tanh_parameters(theta, width):
    old_width = (theta.numel() - 1) // 3
    if width < old_width:
        raise ValueError(f"width must be at least {old_width}.")

    a0 = theta[:1]
    a = theta[1:old_width + 1]
    w = theta[old_width + 1:2 * old_width + 1]
    b = theta[2 * old_width + 1:]
    extra = width - old_width

    if extra == 0:
        return theta.clone()

    extra_a = torch.zeros(extra, dtype=theta.dtype)
    extra_w = torch.linspace(3.0, 15.0, extra, dtype=theta.dtype)
    centres = torch.linspace(0.1, 0.9, extra, dtype=theta.dtype)
    extra_b = -extra_w * centres
    return torch.cat((a0, a, extra_a, w, extra_w, b, extra_b))


def parameters_for_width(width):
    theta_amp, theta_phase, metadata = load_seed()
    expanded_amp = expand_tanh_parameters(theta_amp, width)
    expanded_phase = torch.cat((theta_phase[:1], expand_tanh_parameters(theta_phase[1:], width)))
    return expanded_amp, expanded_phase, metadata


@contextmanager
def use_recurrence(recurrence):
    production_recurrence = solver.raychaudhuri_rk4_torch
    solver.raychaudhuri_rk4_torch = recurrence

    try:
        yield
    finally:
        solver.raychaudhuri_rk4_torch = production_recurrence


def forward_backward(theta_amp, theta_phase, metadata, recurrence, n_grid):
    theta_amp.grad = None
    theta_phase.grad = None

    with use_recurrence(recurrence):
        start = time.perf_counter()
        loss = solver.loss_C1(theta_amp, theta_phase, metadata["eM"], q=metadata["q"], lam=metadata.get("lam", 0.0), m_over_e=metadata.get("m_over_e", 0.0), rho_drho_func=solver.tanh_ansatz, omega_domega_func=solver.omega_domega, n_grid=n_grid)
        loss.backward()
        elapsed = time.perf_counter() - start

    return loss.detach().item(), elapsed


def benchmark_case(width, n_grid, warmup, repeats):
    theta_amp, theta_phase, metadata = parameters_for_width(width)
    implementations = (("legacy", solver._raychaudhuri_rk4_legacy), ("eager-transfer", solver._raychaudhuri_rk4_transfer_eager), ("custom-adjoint", solver.raychaudhuri_rk4_torch))
    measurements = {}
    losses = {}

    for name, recurrence in implementations:
        theta_amp_parameter = torch.nn.Parameter(theta_amp.clone())
        theta_phase_parameter = torch.nn.Parameter(theta_phase.clone())

        for _ in range(warmup):
            forward_backward(theta_amp_parameter, theta_phase_parameter, metadata, recurrence, n_grid)
            gc.collect()

        times = []
        for _ in range(repeats):
            loss, elapsed = forward_backward(theta_amp_parameter, theta_phase_parameter, metadata, recurrence, n_grid)
            times.append(elapsed)
            gc.collect()

        losses[name] = loss
        measurements[name] = {"minimum": min(times), "median": statistics.median(times)}

    reference_loss = losses["legacy"]
    for name, loss in losses.items():
        if not np.isclose(loss, reference_loss, rtol=2e-10, atol=1e-15):
            raise RuntimeError(f"Loss mismatch for {name}: legacy={reference_loss:.16e}, {name}={loss:.16e}")

    return losses, measurements


def profile_node_counts(width, n_grid):
    theta_amp_value, theta_phase_value, metadata = parameters_for_width(width)
    implementations = (("legacy", solver._raychaudhuri_rk4_legacy), ("custom-adjoint", solver.raychaudhuri_rk4_torch))
    keys = ("MulBackward0", "AddBackward0", "SelectBackward0", "aten::select", "_TransferRecurrenceBackward")
    summaries = {}

    for name, recurrence in implementations:
        theta_amp = torch.nn.Parameter(theta_amp_value.clone())
        theta_phase = torch.nn.Parameter(theta_phase_value.clone())
        theta_amp.grad = None
        theta_phase.grad = None

        with use_recurrence(recurrence), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            loss = solver.loss_C1(theta_amp, theta_phase, metadata["eM"], q=metadata["q"], lam=metadata.get("lam", 0.0), m_over_e=metadata.get("m_over_e", 0.0), rho_drho_func=solver.tanh_ansatz, omega_domega_func=solver.omega_domega, n_grid=n_grid)
            loss.backward()

        events = {event.key: event.count for event in profiler.key_averages()}
        summaries[name] = {key: events.get(key, 0) for key in keys}

    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark full C1 forward/backward time for Raychaudhuri recurrences.")
    parser.add_argument("--grids", nargs="+", type=int, default=(1001, 4001))
    parser.add_argument("--widths", nargs="+", type=int, default=(4, 32))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Also compare selected autograd-node counts outside the wall-clock benchmark.")
    args = parser.parse_args()

    if args.warmup < 2:
        parser.error("--warmup must be at least 2.")
    if args.repeats < 3:
        parser.error("--repeats must be at least 3.")

    return args


def main():
    args = parse_args()
    print(f"PyTorch {torch.__version__}; device=cpu; dtype=float64; threads={torch.get_num_threads()}; warmup={args.warmup}; repeats={args.repeats}")
    print(" width  n_grid  implementation           loss       min (s)    median (s)")
    print("------  ------  ----------------  -----------  ------------  ------------")

    for width in args.widths:
        for n_grid in args.grids:
            losses, measurements = benchmark_case(width, n_grid, args.warmup, args.repeats)

            for name in ("legacy", "eager-transfer", "custom-adjoint"):
                values = measurements[name]
                print(f"{width:6d}  {n_grid:6d}  {name:16s}  {losses[name]:11.4e}  {values['minimum']:12.6f}  {values['median']:12.6f}")

            speedup = measurements["legacy"]["median"] / measurements["custom-adjoint"]["median"]
            print(f"        legacy/custom median speedup: {speedup:.3f}x")

    if args.profile:
        width = args.widths[0]
        n_grid = args.grids[0]
        print(f"\nOptional profiler node counts (width={width}, n_grid={n_grid})")
        summaries = profile_node_counts(width, n_grid)
        print(" implementation    MulBackward0  AddBackward0  SelectBackward0  aten::select  custom backward")
        for name, counts in summaries.items():
            print(f" {name:16s} {counts['MulBackward0']:12d} {counts['AddBackward0']:13d} {counts['SelectBackward0']:16d} {counts['aten::select']:13d} {counts['_TransferRecurrenceBackward']:16d}")


if __name__ == "__main__":
    main()
