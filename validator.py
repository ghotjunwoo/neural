import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.lines import Line2D
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
        V_t = torch.as_tensor(V, dtype=torch.float64, device=device)
        theta_t = torch.as_tensor(theta, dtype=torch.float64, device=device)

        with torch.no_grad():
            try:
                value, derivative = func(V_t, theta_t, k=k)
            except TypeError:
                value, derivative = func(V_t, theta_t)

        return (_to_numpy_1d(value), _to_numpy_1d(derivative))

    if backend == "numpy":
        try:
            value, derivative = func(V, theta, k=k)
        except TypeError:
            value, derivative = func(V, theta)

        return (_to_numpy_1d(value), _to_numpy_1d(derivative))

    raise ValueError("backend must be either 'torch' or 'numpy'.")


def evaluate_ansatz_chunked(func, V, theta, k, *, backend="torch", chunk_size=2048, device="cpu"):
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

        value, derivative = _call_ansatz(func=func, V=V[start:stop], theta=theta, k=k, backend=backend, device=device)

        values.append(value)
        derivatives.append(derivative)

    return np.concatenate(values), np.concatenate(derivatives)


def cumulative_simpson_zero(y, x):
    """Cumulative Simpson integral with value zero at x[0]."""
    return cumulative_simpson(y, x=x, initial=0.0)


# ============================================================
# Accurate C1 validator
# ============================================================

def validate_C1(theta_amp, theta_phase, *, eM, q, rho_drho_func, omega_domega_func, rplus_and_Qtarget_func, lam=0.0, m_over_e=0.0, k=1, n_grid=10001, n_profile=None, rtol=1e-11, atol=1e-13, max_step=2e-3, interpolation="cubic", ansatz_backend="torch", ansatz_device="cpu", chunk_size=2048, match_tol=1e-7, xi_margin=0.0, rU_margin=0.0, denominator_tol=1e-13):
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
        raise ValueError("n_grid must be an odd integer at least 3.")

    if n_profile is None:
        n_profile = n_grid

    if n_profile < 3:
        raise ValueError("n_profile must be at least 3.")

    V = np.linspace(0.0, 1.0, n_grid)
    V_profile = np.linspace(0.0, 1.0, n_profile)

    # --------------------------------------------------------
    # Evaluate the scalar-field Ansätze
    # --------------------------------------------------------

    rho, drho = evaluate_ansatz_chunked(rho_drho_func, V, theta_amp, k, backend=ansatz_backend, chunk_size=chunk_size, device=ansatz_device)

    omega, domega = evaluate_ansatz_chunked(omega_domega_func, V, theta_phase, k, backend=ansatz_backend, chunk_size=chunk_size, device=ansatz_device)

    if n_profile == n_grid:
        rho_profile = rho
        drho_profile = drho
        domega_profile = domega
    else:
        rho_profile, drho_profile = evaluate_ansatz_chunked(rho_drho_func, V_profile, theta_amp, k, backend=ansatz_backend, chunk_size=chunk_size, device=ansatz_device)

        _, domega_profile = evaluate_ansatz_chunked(omega_domega_func, V_profile, theta_phase, k, backend=ansatz_backend, chunk_size=chunk_size, device=ansatz_device)

    focusing_profile = drho_profile**2 + rho_profile**2 * domega_profile**2

    if not np.all(np.isfinite(focusing_profile)):
        raise FloatingPointError("The focusing coefficient contains non-finite values.")

    # --------------------------------------------------------
    # Interpolate E(V) for the adaptive ODE solver
    # --------------------------------------------------------

    if interpolation == "cubic":
        focusing_interp = CubicSpline(V_profile, focusing_profile, extrapolate=False)
    elif interpolation == "pchip":
        focusing_interp = PchipInterpolator(V_profile, focusing_profile, extrapolate=False)
    else:
        raise ValueError("interpolation must be 'cubic' or 'pchip'.")

    profile_midpoints = 0.5 * (V_profile[:-1] + V_profile[1:])

    interpolated_min = float(np.min(focusing_interp(profile_midpoints)))

    if interpolated_min < -1e-10:
        warnings.warn("The interpolated focusing coefficient becomes " f"negative: min={interpolated_min:.3e}. " "Increase n_profile or use interpolation='pchip'.", RuntimeWarning)

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
        raise RuntimeError("Raychaudhuri solve failed: " + sol.message)

    xi, xip = sol.sol(V)

    if not np.all(np.isfinite(xi)):
        raise FloatingPointError("The Raychaudhuri solution contains non-finite xi.")

    if not np.all(np.isfinite(xip)):
        raise FloatingPointError("The Raychaudhuri solution contains non-finite xi'.")

    # --------------------------------------------------------
    # Charge transport
    # --------------------------------------------------------

    r_plus, Qtarget = rplus_and_Qtarget_func(q, lam)

    r_plus = float(r_plus)
    Qtarget = float(Qtarget)
    e = float(eM)
    m = float(m_over_e * e)

    charge_density = xi**2 * rho**2 * domega

    cumulative_I = cumulative_simpson_zero(charge_density, V)

    I_theta = float(cumulative_I[-1])

    Q = e * r_plus**2 * cumulative_I

    charge_residual = float(Q[-1] - Qtarget)

    # --------------------------------------------------------
    # Geometry and r_U transport
    # --------------------------------------------------------

    r = r_plus * xi
    r_V = r_plus * xip

    if np.min(np.abs(r)) <= denominator_tol:
        raise FloatingPointError("r becomes too close to zero during validation.")

    if abs(r_V[0]) <= denominator_tol:
        raise FloatingPointError("r_V(0) is too close to zero to determine r_U(0).")

    rU0 = (lam * r[0]**2 / 3.0 - 1.0) / (4.0 * r_V[0])

    rU_density = -0.25 + Q**2 / (4.0 * r**2) + lam * r**2 / 4.0 + m**2 * r**2 * rho**2 / 4.0

    r_times_rU = r[0] * rU0 + cumulative_simpson_zero(rU_density, V)

    r_U = r_times_rU / r

    # --------------------------------------------------------
    # Gauge potential
    # --------------------------------------------------------

    AU_density = -Q / (2.0 * r**2)

    A_U = cumulative_simpson_zero(AU_density, V)

    # --------------------------------------------------------
    # Scalar wave transport and C1 endpoint residual
    # --------------------------------------------------------

    phase = np.exp(-1j * omega)

    Phi = rho * phase

    Phi_V = (drho - 1j * rho * domega) * phase

    PhiU_density = -r_U * Phi_V + 1j * e * Q * Phi / (4.0 * r) - 1j * e * A_U * r_V * Phi - 1j * e * A_U * r * Phi_V - 0.25 * m**2 * r * Phi

    Phi_U_1 = simpson(PhiU_density, x=V) / r[-1]

    re_Phi_U_1 = float(Phi_U_1.real)
    im_Phi_U_1 = float(Phi_U_1.imag)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    focusing_density = drho**2 + rho**2 * domega**2

    residual_vector = np.array([
        charge_residual,
        re_Phi_U_1,
        im_Phi_U_1,
    ])

    residual_norm = float(np.linalg.norm(residual_vector))

    max_residual = float(np.max(np.abs(residual_vector)))

    inf_xi = float(np.min(xi))
    sup_rU = float(np.max(r_U))

    finite = bool(np.all(np.isfinite(residual_vector)) and np.all(np.isfinite(r_U)) and np.all(np.isfinite(Q)) and np.all(np.isfinite(A_U)))

    matching_ok = bool(max_residual <= match_tol)

    xi_ok = bool(inf_xi > xi_margin)

    rU_ok = bool(sup_rU < -rU_margin)

    valid = bool(finite and matching_ok and xi_ok and rU_ok)

    diagnostics = {
        "charge_residual": charge_residual,
        "Re_Phi_U_1": re_Phi_U_1,
        "Im_Phi_U_1": im_Phi_U_1,
        "residual_norm": residual_norm,
        "max_residual": max_residual,
        "inf_xi": inf_xi,
        "sup_rU": sup_rU,
        "I_theta": I_theta,
        "phase_winding": float(omega[-1] - omega[0]),
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
    print(f"  Q(1)-Q_target       = {d['charge_residual']:+.12e}")
    print(f"  Re Phi_U(1)         = {d['Re_Phi_U_1']:+.12e}")
    print(f"  Im Phi_U(1)         = {d['Im_Phi_U_1']:+.12e}")
    print(f"  residual L2 norm    = {d['residual_norm']:.12e}")
    print(f"  largest residual    = {d['max_residual']:.12e}")

    print("\nAdmissibility")
    print(f"  inf xi              = {d['inf_xi']:+.12e}")
    print(f"  sup r_U             = {d['sup_rU']:+.12e}")
    print(f"  xi condition        = {d['xi_ok']}")
    print(f"  r_U condition       = {d['rU_ok']}")

    print("\nPhase")
    print(f"  omega(1)-omega(0)   = {d['phase_winding']:+.12e}")
    print(f"  min omega'          = {d['min_domega']:+.12e}")
    print(f"  max omega'          = {d['max_domega']:+.12e}")

    print("\nNumerics")
    print(f"  grid points         = {s['n_grid']}")
    print(f"  profile points      = {s['n_profile']}")
    print(f"  DOP853 evaluations  = {d['ode_nfev']}")
    print(f"  interpolated min E  = {d['interpolated_min_focusing']:+.12e}")
    print(f"  matching tolerance  = {s['match_tol']:.3e}")
    print("=" * 62)

def _style_C1_dashboard_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.margins(x=0)

def plot_C1_dashboard(result, *, figsize=(15, 8), save=None, show=True):
    """Compact 2x3 dashboard for inspecting one validated C1 solution."""
    V = result["V"]
    d = result["diagnostics"]
    s = result["settings"]
    status = "VALID" if d["valid"] else "INVALID"
    status_color = "#059669" if d["valid"] else "#dc2626"
    blue, amber, violet = "#2563eb", "#d97706", "#7c3aed"

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(rf"C1 validation dashboard  ·  $eM={s['eM']:g}$  ·  {status}", fontsize=16, fontweight="semibold", color=status_color)

    ax = axes[0, 0]
    ax.plot(V, result["rho"], color=blue, linewidth=2.2, label=r"$\rho$")
    ax.plot(V, result["Phi"].real, color=amber, linewidth=1.5, linestyle="--", label=r"$\Re\Phi$")
    ax.plot(V, result["Phi"].imag, color=violet, linewidth=1.5, linestyle=":", label=r"$\Im\Phi$")
    ax.axhline(0.0, color="#64748b", linewidth=0.8)
    ax.set_title("Scalar field", loc="left", fontweight="semibold")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel("field value")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(V, result["omega"], color=blue, linewidth=2.0, label=r"$\omega$")
    ax.plot(V, result["domega"], color=amber, linewidth=1.6, linestyle="--", label=r"$\omega'$")
    ax.plot(V, V, color="#64748b", linewidth=1.0, linestyle=":", label=r"$V$")
    ax.set_title(rf"Phase  ·  $\Delta\omega={d['phase_winding']:.4g}$", loc="left", fontweight="semibold")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel("phase / derivative")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 2]
    ax.plot(V, result["xi"], color=blue, linewidth=2.0, label=r"$\xi$")
    ax.plot(V, result["xip"], color=violet, linewidth=1.6, linestyle="--", label=r"$\xi'$")
    ax.axhline(s["xi_margin"], color=status_color if not d["xi_ok"] else "#64748b", linewidth=1.0, linestyle=":")
    ax.set_title(rf"Raychaudhuri  ·  $\min\xi={d['inf_xi']:.2e}$", loc="left", fontweight="semibold")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel("geometry")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(V, result["Q"], color=blue, linewidth=2.2, label=r"$Q(V)$")
    ax.axhline(result["Qtarget"], color=amber, linewidth=1.4, linestyle="--", label=r"$Q_{\rm target}$")
    ax.set_title(rf"Charge  ·  $\Delta Q={d['charge_residual']:+.2e}$", loc="left", fontweight="semibold")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$Q$")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    rU_limit = -s["rU_margin"]
    ax.plot(V, result["r_U"], color=violet, linewidth=2.2)
    ax.axhline(rU_limit, color=status_color if not d["rU_ok"] else "#64748b", linewidth=1.0, linestyle=":", label="admissibility limit")
    ax.set_title(rf"Ingoing expansion  ·  $\max r_U={d['sup_rU']:.2e}$", loc="left", fontweight="semibold")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$r_U$")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 2]
    residual_labels = [r"$\Delta Q$", r"$\Re\Phi_U$", r"$\Im\Phi_U$"]
    residual_values = np.abs([d["charge_residual"], d["Re_Phi_U_1"], d["Im_Phi_U_1"]])
    residual_plot_values = np.maximum(residual_values, np.finfo(float).tiny)
    residual_colors = ["#059669" if value <= s["match_tol"] else "#dc2626" for value in residual_values]
    bars = ax.bar(residual_labels, residual_plot_values, color=residual_colors, width=0.62)
    ax.axhline(s["match_tol"], color=amber, linewidth=1.4, linestyle="--", label=rf"tol. $={s['match_tol']:.1e}$")
    for bar, value in zip(bars, residual_values):
        ax.annotate(f"{value:.1e}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#475569")
    ax.set_yscale("log")
    ax.set_title(rf"Matching  ·  $\|R\|_2={d['residual_norm']:.2e}$", loc="left", fontweight="semibold")
    ax.set_ylabel("absolute residual")
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.flat:
        _style_C1_dashboard_axis(ax)

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

    if show:
        plt.show()

    return fig, axes
    
def plot_C1_profiles(result, *, figsize=(15, 8), save=None, show=True):
    V = result["V"]
    d = result["diagnostics"]

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(V, result["rho"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Amplitude $\rho$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\rho$")

    ax = axes[0, 1]
    ax.plot(V, result["omega"], label=r"$\omega$")
    ax.plot(V, V, linestyle="--", label=r"$V$")
    ax.set_title(rf"Phase, winding={d['phase_winding']:.6g}")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\omega$")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(V, result["domega"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(rf"$\omega'$: [{d['min_domega']:.3g}, {d['max_domega']:.3g}]")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\omega'$")

    ax = axes[1, 0]
    ax.plot(V, result["xi"], label=r"$\xi$")
    ax.plot(V, result["xip"], label=r"$\xi'$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(rf"Raychaudhuri, $\inf\xi={d['inf_xi']:.3e}$")
    ax.set_xlabel(r"$V$")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(V, result["Q"], label=r"$Q(V)$")
    ax.axhline(result["Qtarget"], linestyle="--", label=r"$Q_{\rm target}$")
    ax.set_title(rf"Charge residual={d['charge_residual']:+.3e}")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$Q$")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(V, result["r_U"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(rf"$r_U$, $\sup r_U={d['sup_rU']:.3e}$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$r_U$")

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes

def plot_C1_diagnostics(result, *, figsize=(15, 8), save=None, show=True):
    V = result["V"]

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(V, result["drho"], label=r"$\rho'$")
    ax.plot(V, result["domega"], label=r"$\omega'$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Field derivatives")
    ax.set_xlabel(r"$V$")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(V, result["charge_density"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Charge density $\xi^2\rho^2\omega'$")
    ax.set_xlabel(r"$V$")

    ax = axes[0, 2]
    ax.plot(V, result["focusing_density"])
    ax.set_title(r"Focusing $\rho'^2+\rho^2\omega'^2$")
    ax.set_xlabel(r"$V$")

    ax = axes[1, 0]
    ax.plot(V, result["A_U"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Gauge potential $A_U$")
    ax.set_xlabel(r"$V$")

    ax = axes[1, 1]
    ax.plot(V, result["Phi"].real, label=r"$\Re\Phi$")
    ax.plot(V, result["Phi"].imag, label=r"$\Im\Phi$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Complex scalar field $\Phi$")
    ax.set_xlabel(r"$V$")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(V, result["PhiU_density"].real, label="real")
    ax.plot(V, result["PhiU_density"].imag, label="imaginary")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(r"Integrand determining $\Phi_U(1)$")
    ax.set_xlabel(r"$V$")
    ax.legend()

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes

def plot_C1_residuals(result, *, figsize=(7, 5), save=None, show=True):
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
    values_for_plot = np.maximum(values, np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.bar(labels, values_for_plot)
    ax.axhline(tolerance, linestyle="--", label=f"tolerance = {tolerance:.1e}")

    ax.set_yscale("log")
    ax.set_ylabel("absolute residual")
    ax.set_title("C1 matching residuals")
    ax.legend()

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

def _C1_comparison_records(results):
    records = []

    for item in results:
        if isinstance(item, dict):
            validation = item
            eM = validation.get("settings", {}).get("eM", validation.get("e"))
        else:
            validation = getattr(item, "validation", None)
            eM = getattr(item, "eM", None)

        if validation is None:
            raise ValueError("Every result must be evaluated before plotting comparisons.")

        if eM is None:
            raise ValueError("Each validation result must contain an eM value.")

        records.append((float(eM), validation))

    if not records:
        raise ValueError("At least one validation result is required.")

    return sorted(records, key=lambda record: record[0])

def _C1_panel_values(result, key, component):
    values = result[key]
    return getattr(values, component) if component is not None else values

def _C1_comparison_legends(ax, records, colors, series, extra_styles=()):
    eM_handles = [Line2D([0], [0], color=color, linewidth=2, label=rf"$eM={eM:g}$") for (eM, _), color in zip(records, colors)]
    style_specs = [(label, linestyle) for _, _, label, linestyle in series] + list(extra_styles)

    if len(style_specs) > 1:
        style_handles = [Line2D([0], [0], color="black", linestyle=linestyle, label=label) for label, linestyle in style_specs]
        style_legend = ax.legend(handles=style_handles, loc="best", title="quantity")
        ax.add_artist(style_legend)

    ax.legend(handles=eM_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title=r"coupling")

def plot_C1_comparison_suite(results, *, panels=None, figsize=(9, 5), cmap="viridis", residual_tolerance=1e-6, show=True):
    """
    Plot each validator panel separately with all eM values overlaid.

    Parameters
    ----------
    results
        Evaluated C1Result objects or validation dictionaries.

    panels
        Optional iterable selecting panel names. By default every
        profile, diagnostic, and residual panel is generated.

    Returns
    -------
    figures : dict
        Maps each panel name to its (figure, axis) pair.
    """
    records = _C1_comparison_records(results)
    colors = plt.get_cmap(cmap)(np.linspace(0.08, 0.92, len(records)))

    panel_specs = {
        "rho": (r"Amplitude $\rho$", r"$\rho$", (("rho", None, r"$\rho$", "-"),), True),
        "omega": (r"Phase $\omega$", r"$\omega$", (("omega", None, r"$\omega$", "-"),), False),
        "domega": (r"Phase derivative $\omega'$", r"$\omega'$", (("domega", None, r"$\omega'$", "-"),), True),
        "raychaudhuri": (r"Raychaudhuri variables", "value", (("xi", None, r"$\xi$", "-"), ("xip", None, r"$\xi'$", "--")), True),
        "charge": (r"Charge transport", r"$Q$", (("Q", None, r"$Q(V)$", "-"),), False),
        "r_U": (r"Ingoing expansion $r_U$", r"$r_U$", (("r_U", None, r"$r_U$", "-"),), True),
        "field_derivatives": (r"Field derivatives", "derivative", (("drho", None, r"$\rho'$", "-"), ("domega", None, r"$\omega'$", "--")), True),
        "charge_density": (r"Charge density $\xi^2\rho^2\omega'$", "charge density", (("charge_density", None, r"$\xi^2\rho^2\omega'$", "-"),), True),
        "focusing_density": (r"Focusing $\rho'^2+\rho^2\omega'^2$", r"$E(V)$", (("focusing_density", None, r"$E(V)$", "-"),), False),
        "A_U": (r"Gauge potential $A_U$", r"$A_U$", (("A_U", None, r"$A_U$", "-"),), True),
        "Phi": (r"Complex scalar field $\Phi$", r"$\Phi$", (("Phi", "real", r"$\Re\Phi$", "-"), ("Phi", "imag", r"$\Im\Phi$", "--")), True),
        "PhiU_density": (r"Integrand determining $\Phi_U(1)$", "integrand", (("PhiU_density", "real", r"real", "-"), ("PhiU_density", "imag", r"imaginary", "--")), True),
    }

    available_panels = tuple(panel_specs) + ("residuals",)
    selected_panels = available_panels if panels is None else tuple(panels)
    unknown_panels = set(selected_panels) - set(available_panels)

    if unknown_panels:
        raise ValueError(f"Unknown comparison panels: {sorted(unknown_panels)}")

    figures = {}

    for panel in selected_panels:
        if panel == "residuals":
            eM_values = np.asarray([eM for eM, _ in records])
            residual_specs = (("charge_residual", r"$|Q(1)-Q_{\rm target}|$", "o"), ("Re_Phi_U_1", r"$|\Re\Phi_U(1)|$", "s"), ("Im_Phi_U_1", r"$|\Im\Phi_U(1)|$", "^"))
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

            for key, label, marker in residual_specs:
                values = np.asarray([abs(result["diagnostics"][key]) for _, result in records])
                ax.plot(eM_values, np.maximum(values, np.finfo(float).tiny), marker=marker, label=label)

            ax.axhline(residual_tolerance, color="black", linestyle=":", label=rf"tolerance $={residual_tolerance:.1e}$")

            ax.set_yscale("log")
            ax.set_xlabel(r"$eM$")
            ax.set_ylabel("absolute residual")
            ax.set_title("C1 matching residuals across coupling")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            figures[panel] = (fig, ax)

            if show:
                plt.show()

            continue

        title, ylabel, series, zero_line = panel_specs[panel]
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for (_, result), color in zip(records, colors):
            for key, component, _, linestyle in series:
                ax.plot(result["V"], _C1_panel_values(result, key, component), color=color, linestyle=linestyle)

        extra_styles = []

        if panel == "omega":
            reference_V = records[0][1]["V"]
            ax.plot(reference_V, reference_V, color="black", linestyle=":")
            extra_styles.append((r"$V$", ":"))

        if panel == "charge":
            targets = np.asarray([result["Qtarget"] for _, result in records])
            if np.allclose(targets, targets[0]):
                ax.axhline(targets[0], color="black", linestyle=":")
                extra_styles.append((r"$Q_{\rm target}$", ":"))
            else:
                for (_, result), color in zip(records, colors):
                    ax.axhline(result["Qtarget"], color=color, linestyle=":")
                extra_styles.append((r"$Q_{\rm target}$", ":"))

        if zero_line:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

        ax.set_xlabel(r"$V$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _C1_comparison_legends(ax, records, colors, series, extra_styles)
        figures[panel] = (fig, ax)

        if show:
            plt.show()

    return figures

def C1_grid_convergence(theta_amp, theta_phase, *, grids=(2501, 5001, 10001), profile_factor=1, **validator_kwargs):
    """
    Run the strict validator at several resolutions.

    profile_factor=1:
        n_profile = n_grid

    profile_factor=2:
        n_profile = 2*n_grid - 1
    """
    results = []

    print(" n_grid    charge residual       Re Phi_U       Im Phi_U       inf xi       sup r_U")

    for n_grid in grids:
        n_profile = profile_factor * (n_grid - 1) + 1

        result = validate_C1(theta_amp=theta_amp, theta_phase=theta_phase, n_grid=n_grid, n_profile=n_profile, **validator_kwargs)

        d = result["diagnostics"]
        results.append(result)

        print(f"{n_grid:7d}  {d['charge_residual']:+.5e}  {d['Re_Phi_U_1']:+.5e}  {d['Im_Phi_U_1']:+.5e}  {d['inf_xi']:+.5e}  {d['sup_rU']:+.5e}")

    return results
