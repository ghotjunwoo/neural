import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.integrate import solve_ivp, simpson, cumulative_simpson
from scipy.interpolate import CubicSpline, PchipInterpolator


# ============================================================
# Helpers
# ============================================================

def _to_numpy_1d(x):
    """Convert a NumPy array or Torch tensor to a float64 1D array."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    return np.asarray(x, dtype=np.float64).reshape(-1)


def _call_ansatz(func, V, theta, k, backend, device="cpu"):
    """
    Evaluate one (value, derivative) Ansatz call.

    Supports either:
        backend="torch"
        backend="numpy"
    """
    if backend == "torch":
        V_t = torch.as_tensor(
            V,
            dtype=torch.float64,
            device=device,
        )

        theta_t = torch.as_tensor(
            theta,
            dtype=torch.float64,
            device=device,
        )

        with torch.no_grad():
            try:
                value, derivative = func(
                    V_t,
                    theta_t,
                    k=k,
                )
            except TypeError:
                value, derivative = func(
                    V_t,
                    theta_t,
                )

        return (
            _to_numpy_1d(value),
            _to_numpy_1d(derivative),
        )

    if backend == "numpy":
        try:
            value, derivative = func(
                V,
                theta,
                k=k,
            )
        except TypeError:
            value, derivative = func(
                V,
                theta,
            )

        return (
            _to_numpy_1d(value),
            _to_numpy_1d(derivative),
        )

    raise ValueError(
        "backend must be either 'torch' or 'numpy'."
    )


def evaluate_ansatz_chunked(
    func,
    V,
    theta,
    k,
    *,
    backend="torch",
    chunk_size=2048,
    device="cpu",
):
    """
    Evaluate an Ansatz in chunks.

    Chunking avoids constructing an enormous
    len(V) x p intermediate tensor for large p.
    """
    V = _to_numpy_1d(V)
    theta = _to_numpy_1d(theta)

    values = []
    derivatives = []

    for start in range(0, V.size, chunk_size):
        stop = min(start + chunk_size, V.size)

        value, derivative = _call_ansatz(
            func=func,
            V=V[start:stop],
            theta=theta,
            k=k,
            backend=backend,
            device=device,
        )

        values.append(value)
        derivatives.append(derivative)

    return (
        np.concatenate(values),
        np.concatenate(derivatives),
    )


def cumulative_simpson_zero(y, x):
    """Cumulative Simpson integral with value zero at x[0]."""
    return cumulative_simpson(
        y,
        x=x,
        initial=0.0,
    )


# ============================================================
# Accurate C1 validator
# ============================================================

def validate_C1_scipy(
    theta_amp,
    theta_phase,
    *,
    eM,
    q,
    rho_drho_func,
    omega_domega_func,
    rplus_and_Qtarget_func,
    lam=0.0,
    m_over_e=0.0,
    k=1,
    n_grid=10001,
    n_profile=None,
    rtol=1e-11,
    atol=1e-13,
    max_step=2e-3,
    interpolation="cubic",
    ansatz_backend="torch",
    ansatz_device="cpu",
    chunk_size=2048,
    match_tol=1e-7,
    xi_margin=0.0,
    rU_margin=0.0,
    denominator_tol=1e-13,
):
    """
    Strict SciPy validation of a candidate C1 gluing solution.

    Parameters
    ----------
    theta_amp, theta_phase
        NumPy arrays or Torch tensors.

    eM
        Fixed dimensionless coupling used during optimization.

    q, lam, m_over_e
        Physical parameters.

    rho_drho_func, omega_domega_func
        The same Ansatz functions used during training.

    rplus_and_Qtarget_func
        Usually your existing rplus_and_Qtarget function.

    n_grid
        Number of output/quadrature points. Use an odd number.

    n_profile
        Number of points used to precompute the Raychaudhuri
        coefficient E(V). Defaults to n_grid.

    interpolation
        "cubic" or "pchip".

    Returns
    -------
    result : dict
        Contains all arrays, residuals, diagnostics, and validity flags.
    """
    theta_amp = _to_numpy_1d(theta_amp)
    theta_phase = _to_numpy_1d(theta_phase)

    if n_grid < 3 or n_grid % 2 == 0:
        raise ValueError(
            "n_grid must be an odd integer at least 3."
        )

    if n_profile is None:
        n_profile = n_grid

    if n_profile < 3:
        raise ValueError("n_profile must be at least 3.")

    V = np.linspace(0.0, 1.0, n_grid)
    V_profile = np.linspace(0.0, 1.0, n_profile)

    # --------------------------------------------------------
    # Evaluate the scalar-field Ansätze
    # --------------------------------------------------------

    rho, drho = evaluate_ansatz_chunked(
        rho_drho_func,
        V,
        theta_amp,
        k,
        backend=ansatz_backend,
        chunk_size=chunk_size,
        device=ansatz_device,
    )

    omega, domega = evaluate_ansatz_chunked(
        omega_domega_func,
        V,
        theta_phase,
        k,
        backend=ansatz_backend,
        chunk_size=chunk_size,
        device=ansatz_device,
    )

    if n_profile == n_grid:
        rho_profile = rho
        drho_profile = drho
        domega_profile = domega
    else:
        rho_profile, drho_profile = evaluate_ansatz_chunked(
            rho_drho_func,
            V_profile,
            theta_amp,
            k,
            backend=ansatz_backend,
            chunk_size=chunk_size,
            device=ansatz_device,
        )

        _, domega_profile = evaluate_ansatz_chunked(
            omega_domega_func,
            V_profile,
            theta_phase,
            k,
            backend=ansatz_backend,
            chunk_size=chunk_size,
            device=ansatz_device,
        )

    focusing_profile = (
        drho_profile**2
        + rho_profile**2 * domega_profile**2
    )

    if not np.all(np.isfinite(focusing_profile)):
        raise FloatingPointError(
            "The focusing coefficient contains non-finite values."
        )

    # --------------------------------------------------------
    # Interpolate E(V) for the adaptive ODE solver
    # --------------------------------------------------------

    if interpolation == "cubic":
        focusing_interp = CubicSpline(
            V_profile,
            focusing_profile,
            extrapolate=False,
        )
    elif interpolation == "pchip":
        focusing_interp = PchipInterpolator(
            V_profile,
            focusing_profile,
            extrapolate=False,
        )
    else:
        raise ValueError(
            "interpolation must be 'cubic' or 'pchip'."
        )

    profile_midpoints = 0.5 * (
        V_profile[:-1] + V_profile[1:]
    )

    interpolated_min = float(
        np.min(focusing_interp(profile_midpoints))
    )

    if interpolated_min < -1e-10:
        warnings.warn(
            "The interpolated focusing coefficient becomes "
            f"negative: min={interpolated_min:.3e}. "
            "Increase n_profile or use interpolation='pchip'.",
            RuntimeWarning,
        )

    # --------------------------------------------------------
    # Adaptive Raychaudhuri solve
    #
    # xi'  = xip
    # xip' = -E(V) xi
    #
    # xi(1)=1, xip(1)=0
    # --------------------------------------------------------

    def raychaudhuri_rhs(v, y):
        xi, xip = y
        E = float(focusing_interp(v))

        return np.array([
            xip,
            -E * xi,
        ])

    solve_kwargs = {
        "fun": raychaudhuri_rhs,
        "t_span": (1.0, 0.0),
        "y0": np.array([1.0, 0.0]),
        "method": "DOP853",
        "rtol": rtol,
        "atol": atol,
        "dense_output": True,
    }

    if max_step is not None:
        solve_kwargs["max_step"] = max_step

    sol = solve_ivp(**solve_kwargs)

    if not sol.success:
        raise RuntimeError(
            "Raychaudhuri solve failed: "
            + sol.message
        )

    xi, xip = sol.sol(V)

    if not np.all(np.isfinite(xi)):
        raise FloatingPointError(
            "The Raychaudhuri solution contains non-finite xi."
        )

    if not np.all(np.isfinite(xip)):
        raise FloatingPointError(
            "The Raychaudhuri solution contains non-finite xi'."
        )

    # --------------------------------------------------------
    # Charge transport
    # --------------------------------------------------------

    r_plus, Qtarget = rplus_and_Qtarget_func(
        q,
        lam,
    )

    r_plus = float(r_plus)
    Qtarget = float(Qtarget)
    e = float(eM)
    m = float(m_over_e * e)

    charge_density = (
        xi**2
        * rho**2
        * domega
    )

    cumulative_I = cumulative_simpson_zero(
        charge_density,
        V,
    )

    I_theta = float(cumulative_I[-1])

    Q = (
        e
        * r_plus**2
        * cumulative_I
    )

    charge_residual = float(
        Q[-1] - Qtarget
    )

    # --------------------------------------------------------
    # Geometry and r_U transport
    # --------------------------------------------------------

    r = r_plus * xi
    r_V = r_plus * xip

    if np.min(np.abs(r)) <= denominator_tol:
        raise FloatingPointError(
            "r becomes too close to zero during validation."
        )

    if abs(r_V[0]) <= denominator_tol:
        raise FloatingPointError(
            "r_V(0) is too close to zero to determine r_U(0)."
        )

    rU0 = (
        lam * r[0]**2 / 3.0 - 1.0
    ) / (4.0 * r_V[0])

    rU_density = (
        -0.25
        + Q**2 / (4.0 * r**2)
        + lam * r**2 / 4.0
        + m**2 * r**2 * rho**2 / 4.0
    )

    r_times_rU = (
        r[0] * rU0
        + cumulative_simpson_zero(
            rU_density,
            V,
        )
    )

    r_U = r_times_rU / r

    # --------------------------------------------------------
    # Gauge potential
    # --------------------------------------------------------

    AU_density = -Q / (2.0 * r**2)

    A_U = cumulative_simpson_zero(
        AU_density,
        V,
    )

    # --------------------------------------------------------
    # Scalar wave transport and C1 endpoint residual
    # --------------------------------------------------------

    phase = np.exp(-1j * omega)

    Phi = rho * phase

    Phi_V = (
        drho
        - 1j * rho * domega
    ) * phase

    PhiU_density = (
        -r_U * Phi_V
        + 1j * e * Q * Phi / (4.0 * r)
        - 1j * e * A_U * r_V * Phi
        - 1j * e * A_U * r * Phi_V
        - 0.25 * m**2 * r * Phi
    )

    Phi_U_1 = (
        simpson(
            PhiU_density,
            x=V,
        )
        / r[-1]
    )

    re_Phi_U_1 = float(Phi_U_1.real)
    im_Phi_U_1 = float(Phi_U_1.imag)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    focusing_density = (
        drho**2
        + rho**2 * domega**2
    )

    residual_vector = np.array([
        charge_residual,
        re_Phi_U_1,
        im_Phi_U_1,
    ])

    residual_norm = float(
        np.linalg.norm(residual_vector)
    )

    max_residual = float(
        np.max(np.abs(residual_vector))
    )

    inf_xi = float(np.min(xi))
    sup_rU = float(np.max(r_U))

    finite = bool(
        np.all(np.isfinite(residual_vector))
        and np.all(np.isfinite(r_U))
        and np.all(np.isfinite(Q))
        and np.all(np.isfinite(A_U))
    )

    matching_ok = bool(
        max_residual <= match_tol
    )

    xi_ok = bool(
        inf_xi > xi_margin
    )

    rU_ok = bool(
        sup_rU < -rU_margin
    )

    valid = bool(
        finite
        and matching_ok
        and xi_ok
        and rU_ok
    )

    diagnostics = {
        "charge_residual": charge_residual,
        "Re_Phi_U_1": re_Phi_U_1,
        "Im_Phi_U_1": im_Phi_U_1,
        "residual_norm": residual_norm,
        "max_residual": max_residual,
        "inf_xi": inf_xi,
        "sup_rU": sup_rU,
        "I_theta": I_theta,
        "phase_winding": float(
            omega[-1] - omega[0]
        ),
        "min_domega": float(np.min(domega)),
        "max_domega": float(np.max(domega)),
        "min_r": float(np.min(r)),
        "rV_0": float(r_V[0]),
        "rU_0": float(rU0),
        "interpolated_min_focusing": interpolated_min,
        "ode_nfev": int(sol.nfev),
        "ode_njev": int(sol.njev),
        "ode_nlu": int(sol.nlu),
        "ode_message": sol.message,
        "matching_ok": matching_ok,
        "xi_ok": xi_ok,
        "rU_ok": rU_ok,
        "finite": finite,
        "valid": valid,
    }

    return {
        "V": V,
        "rho": rho,
        "drho": drho,
        "omega": omega,
        "domega": domega,
        "Phi": Phi,
        "Phi_V": Phi_V,
        "xi": xi,
        "xip": xip,
        "r": r,
        "r_V": r_V,
        "r_U": r_U,
        "Q": Q,
        "A_U": A_U,
        "charge_density": charge_density,
        "focusing_density": focusing_density,
        "rU_density": rU_density,
        "AU_density": AU_density,
        "PhiU_density": PhiU_density,
        "Phi_U_1": Phi_U_1,
        "r_plus": r_plus,
        "Qtarget": Qtarget,
        "e": e,
        "m": m,
        "diagnostics": diagnostics,
        "settings": {
            "eM": eM,
            "q": q,
            "lam": lam,
            "m_over_e": m_over_e,
            "k": k,
            "n_grid": n_grid,
            "n_profile": n_profile,
            "rtol": rtol,
            "atol": atol,
            "max_step": max_step,
            "interpolation": interpolation,
            "match_tol": match_tol,
            "xi_margin": xi_margin,
            "rU_margin": rU_margin,
        },
    }

def print_C1_validation(result):
    d = result["diagnostics"]
    s = result["settings"]

    status = "VALID" if d["valid"] else "INVALID"

    print("=" * 62)
    print(f"C1 SCIPY VALIDATION: {status}")
    print("=" * 62)

    print("\nMatching residuals")
    print(
        f"  Q(1)-Q_target       = "
        f"{d['charge_residual']:+.12e}"
    )
    print(
        f"  Re Phi_U(1)         = "
        f"{d['Re_Phi_U_1']:+.12e}"
    )
    print(
        f"  Im Phi_U(1)         = "
        f"{d['Im_Phi_U_1']:+.12e}"
    )
    print(
        f"  residual L2 norm    = "
        f"{d['residual_norm']:.12e}"
    )
    print(
        f"  largest residual    = "
        f"{d['max_residual']:.12e}"
    )

    print("\nAdmissibility")
    print(
        f"  inf xi              = "
        f"{d['inf_xi']:+.12e}"
    )
    print(
        f"  sup r_U             = "
        f"{d['sup_rU']:+.12e}"
    )
    print(
        f"  xi condition        = {d['xi_ok']}"
    )
    print(
        f"  r_U condition       = {d['rU_ok']}"
    )

    print("\nPhase")
    print(
        f"  omega(1)-omega(0)   = "
        f"{d['phase_winding']:+.12e}"
    )
    print(
        f"  min omega'          = "
        f"{d['min_domega']:+.12e}"
    )
    print(
        f"  max omega'          = "
        f"{d['max_domega']:+.12e}"
    )

    print("\nNumerics")
    print(
        f"  grid points         = {s['n_grid']}"
    )
    print(
        f"  profile points      = {s['n_profile']}"
    )
    print(
        f"  DOP853 evaluations  = {d['ode_nfev']}"
    )
    print(
        f"  interpolated min E  = "
        f"{d['interpolated_min_focusing']:+.12e}"
    )
    print(
        f"  matching tolerance  = "
        f"{s['match_tol']:.3e}"
    )
    print("=" * 62)
    
def plot_C1_profiles(
    result,
    *,
    figsize=(15, 8),
    save=None,
    show=True,
):
    V = result["V"]
    d = result["diagnostics"]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
        constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(V, result["rho"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Amplitude $\rho$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[0, 1]
    ax.plot(V, result["omega"], label=r"$\omega$")
    ax.plot(V, V, linestyle="--", label=r"$V$")
    ax.set_title(
        rf"Phase, winding={d['phase_winding']:.6g}"
    )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\omega$")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(V, result["domega"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(
        rf"$\omega'$: "
        rf"[{d['min_domega']:.3g}, "
        rf"{d['max_domega']:.3g}]"
    )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\omega'$")

    ax = axes[1, 0]
    ax.plot(V, result["xi"], label=r"$\xi$")
    ax.plot(V, result["xip"], label=r"$\xi'$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(
        rf"Raychaudhuri, "
        rf"$\inf\xi={d['inf_xi']:.3e}$"
    )
    ax.set_xlabel(r"$V$")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(V, result["Q"], label=r"$Q(V)$")
    ax.axhline(
        result["Qtarget"],
        linestyle="--",
        label=r"$Q_{\rm target}$",
    )
    ax.set_title(
        rf"Charge residual="
        rf"{d['charge_residual']:+.3e}"
    )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$Q$")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(V, result["r_U"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(
        rf"$r_U$, "
        rf"$\sup r_U={d['sup_rU']:.3e}$"
    )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$r_U$")

    if save is not None:
        fig.savefig(
            save,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes

def plot_C1_diagnostics(
    result,
    *,
    figsize=(15, 8),
    save=None,
    show=True,
):
    V = result["V"]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
        constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(V, result["drho"], label=r"$\rho'$")
    ax.plot(V, result["domega"], label=r"$\omega'$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Field derivatives")
    ax.set_xlabel(r"$V$")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        V,
        result["charge_density"],
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(
        r"Charge density "
        r"$\xi^2\rho^2\omega'$"
    )
    ax.set_xlabel(r"$V$")

    ax = axes[0, 2]
    ax.plot(
        V,
        result["focusing_density"],
    )
    ax.set_title(
        r"Focusing "
        r"$\rho'^2+\rho^2\omega'^2$"
    )
    ax.set_xlabel(r"$V$")

    ax = axes[1, 0]
    ax.plot(V, result["A_U"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Gauge potential $A_U$")
    ax.set_xlabel(r"$V$")

    ax = axes[1, 1]
    ax.plot(
        V,
        result["Phi"].real,
        label=r"$\Re\Phi$",
    )
    ax.plot(
        V,
        result["Phi"].imag,
        label=r"$\Im\Phi$",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Complex scalar field $\Phi$")
    ax.set_xlabel(r"$V$")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(
        V,
        result["PhiU_density"].real,
        label="real",
    )
    ax.plot(
        V,
        result["PhiU_density"].imag,
        label="imaginary",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(
        r"Integrand determining $\Phi_U(1)$"
    )
    ax.set_xlabel(r"$V$")
    ax.legend()

    if save is not None:
        fig.savefig(
            save,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes

def plot_C1_residuals(
    result,
    *,
    figsize=(7, 5),
    save=None,
    show=True,
):
    diagnostics = result["diagnostics"]
    tolerance = result["settings"]["match_tol"]

    labels = [
        r"$|Q(1)-Q_{\rm target}|$",
        r"$|\Re\Phi_U(1)|$",
        r"$|\Im\Phi_U(1)|$",
    ]

    values = np.abs([
        diagnostics["charge_residual"],
        diagnostics["Re_Phi_U_1"],
        diagnostics["Im_Phi_U_1"],
    ])

    # Avoid problems displaying an exact zero on a logarithmic axis.
    values_for_plot = np.maximum(
        values,
        np.finfo(float).tiny,
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
    )

    ax.bar(labels, values_for_plot)
    ax.axhline(
        tolerance,
        linestyle="--",
        label=f"tolerance = {tolerance:.1e}",
    )

    ax.set_yscale("log")
    ax.set_ylabel("absolute residual")
    ax.set_title("C1 matching residuals")
    ax.legend()

    if save is not None:
        fig.savefig(
            save,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, ax

def C1_grid_convergence(
    theta_amp,
    theta_phase,
    *,
    grids=(2501, 5001, 10001),
    profile_factor=1,
    **validator_kwargs,
):
    """
    Run the strict validator at several resolutions.

    profile_factor=1:
        n_profile = n_grid

    profile_factor=2:
        n_profile = 2*n_grid - 1
    """
    results = []

    print(
        " n_grid    charge residual"
        "       Re Phi_U"
        "       Im Phi_U"
        "       inf xi"
        "       sup r_U"
    )

    for n_grid in grids:
        n_profile = (
            profile_factor * (n_grid - 1) + 1
        )

        result = validate_C1_scipy(
            theta_amp=theta_amp,
            theta_phase=theta_phase,
            n_grid=n_grid,
            n_profile=n_profile,
            **validator_kwargs,
        )

        d = result["diagnostics"]
        results.append(result)

        print(
            f"{n_grid:7d}  "
            f"{d['charge_residual']:+.5e}  "
            f"{d['Re_Phi_U_1']:+.5e}  "
            f"{d['Im_Phi_U_1']:+.5e}  "
            f"{d['inf_xi']:+.5e}  "
            f"{d['sup_rU']:+.5e}"
        )

    return results
