import numpy as np
# import scipy.optimize as opt
from scipy.integrate import solve_ivp, simpson, cumulative_simpson
import torch
from bh_horizons import *

def xi_integral(theta_amp, theta_phase, k, rho_drho_func, omega_domega_func, rtol=1e-8, atol=1e-10):

    V_grid = np.linspace(0, 1, 5001)
    _, domega_grid = omega_domega_func(V_grid, theta_phase)

    def domega_fast(V):
        return np.interp(np.asarray(V), V_grid, domega_grid)

    def f(V, y):
        xi, xip = y
        rho, drho = rho_drho_func(V, theta_amp, k=k)
        domega = domega_fast(V)

        return [xip, -xi * (drho**2 + (domega * rho)**2)]

    sol = solve_ivp(f, t_span=(1.0, 0.0), y0=[1.0, 0.0], t_eval=V_grid[::-1],method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Raychaudhuri solve failed: {sol.message}")
    if sol.y.shape[1] != len(V_grid):
        raise RuntimeError("Incomplete Raychaudhuri solution.")

    xi_vals = sol.y[0][::-1]
    xip_vals = sol.y[1][::-1]

    rho_vals = rho_drho_func(V_grid, theta_amp, k=k)[0]
    domega_vals = domega_fast(V_grid)

    I = simpson(xi_vals**2 * rho_vals**2 * domega_vals, x=V_grid)

    return xi_vals, xip_vals, I

def solve_Phi_wave(theta_1, theta_2, r, r_U, r_V, Q, A_U, eM, m, rho_drho_func, omega_domega_func, k=1):
    
    V_grid = np.linspace(0.0, 1.0, len(r))
    rho, drho = rho_drho_func(V_grid, theta_1, k=k)
    omega, domega = omega_domega_func(V_grid, theta_2)

    phase = np.exp(-1j * omega)

    Phi = rho * phase
    Phi_V = (drho - 1j * rho * domega) * phase

    integrand = (
        -r_U * Phi_V
        + 1j * eM * Q * Phi / (4.0 * r)
        - 1j * eM * A_U * r_V * Phi
        - 1j * eM * A_U * r * Phi_V
        - 0.25 * m**2 * r * Phi
    )

    Phi_U_1 = simpson(integrand, x=V_grid) / r[-1]

    return Phi_U_1.real, Phi_U_1.imag

def AU_values(V_grid, r_vals, Q_vals):
    integrand = -Q_vals / (2 * r_vals**2)
    AU_vals = cumulative_simpson(integrand, x=V_grid,initial=0.0)
    return AU_vals

def var_values(theta_1, theta_2, k, xi_vals, xip_vals, q, lam, rho_drho_func, omega_domega_func, m_over_e=0.0, eM=None):
    V_grid = np.linspace(0.0, 1.0, len(xi_vals))
    r_plus, Qtarget = rplus_and_Qtarget(q, lam)

    r_vals = r_plus * xi_vals
    rp_vals = r_plus * xip_vals

    rho_vals, _ = rho_drho_func(V_grid, theta_1, k=k)
    _, domega_vals = omega_domega_func(V_grid, theta_2)

    integrand_I = xi_vals**2 * rho_vals**2 * domega_vals
    cum_I = cumulative_simpson(integrand_I, x=V_grid, initial=0.0)
    I_a = cum_I[-1]

    e = Qtarget / (r_plus**2 * I_a) if eM is None else eM
    m = m_over_e * e

    if eM is None:
        Q_vals = Qtarget * cum_I / I_a
    else:
        Q_vals = e * r_plus**2 * cum_I

    charge_residual = Q_vals[-1] - Qtarget

    rU0 = (lam * r_vals[0]**2 / 3.0 - 1.0) / (4.0 * rp_vals[0])
    
    integrand_rU = (
        -0.25
        + Q_vals**2 / (4.0 * r_vals**2)
        + lam * r_vals**2 / 4.0
        + m**2 * r_vals**2 * rho_vals**2 / 4.0
    )

    r_rU_vals = (r_vals[0] * rU0 + cumulative_simpson(integrand_rU, x=V_grid, initial=0.0))
    rU_vals = r_rU_vals / r_vals

    return {
        "sup_rU": np.max(rU_vals),
        "rU_vals": rU_vals,
        "Q_vals": Q_vals,
        "r_plus": r_plus,
        "Qtarget": Qtarget,
        "I_theta": I_a,
        "e": e,
        "m": m,
        "m_over_e": m_over_e,
        "charge_residual": charge_residual,
    }


def penalty_xi(xi_vals, delta=1e-3):
    violation = max(delta - np.min(xi_vals), 0.0)
    return violation**2


def penalty_rU(rU_vals, delta=1e-3):
    violation = max(np.max(rU_vals) + delta, 0.0)
    return violation**2

def finite_difference_gradient(func, theta, eps=1e-4):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        e_i = np.zeros_like(theta)
        e_i[i] = 1.0
        term = func(theta + eps*e_i) - func(theta - eps*e_i)
        grad[i] = term / (2 * eps)
    return grad

def loss_C1(theta_1, theta_2, eM, q, rho_drho_func, omega_domega_func, lam=0.0, m_over_e=0.0, w_Q=1.0, w_1=1.0, w_penalty=1000.0, delta_pen=1e-4):
    xi_vals, xip_vals, _ = xi_integral(theta_1, theta_2, k=1, rho_drho_func=rho_drho_func, omega_domega_func=omega_domega_func)

    if not np.all(np.isfinite(xi_vals)):
        return 1e12

    out = var_values(theta_1=theta_1, theta_2=theta_2, k=1, xi_vals=xi_vals, xip_vals=xip_vals, q=q, lam=lam, m_over_e=m_over_e, eM=eM, rho_drho_func=rho_drho_func, omega_domega_func=omega_domega_func)

    V = np.linspace(0.0, 1.0, len(xi_vals))

    r = out["r_plus"] * xi_vals
    r_V = out["r_plus"] * xip_vals
    r_U = out["rU_vals"]
    Q = out["Q_vals"]
    A_U = AU_values(V, r, Q)

    re_phi_u, im_phi_u = solve_Phi_wave(theta_1=theta_1, theta_2=theta_2, r=r, r_U=r_U, r_V=r_V, Q=Q, A_U=A_U, eM=eM, m=out["m"], rho_drho_func=rho_drho_func, omega_domega_func=omega_domega_func)

    penalty = w_penalty * (penalty_xi(xi_vals, delta=delta_pen) + penalty_rU(r_U, delta=delta_pen))

    return w_Q * out["charge_residual"]**2 + w_1 * (re_phi_u**2 + im_phi_u**2) + penalty

def fd_adam(x0, loss_fn, lr=1e-3, eps=1e-4, max_iter=100, print_every=20):
    theta = torch.tensor(np.asarray(x0, dtype=float), dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    best_x = np.asarray(x0, dtype=float).copy()
    best_loss = loss_fn(best_x)

    for iteration in range(max_iter):
        x = theta.detach().cpu().numpy().copy()
        loss = best_loss if iteration == 0 else loss_fn(x)

        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_x = x.copy()

        if best_loss < 1e-12:
            print("Matching tolerance reached.")
            break

        grad = finite_difference_gradient(
            loss_fn,
            x,
            eps=eps,
        )

        if np.linalg.norm(grad) < 1e-8:
            print("Gradient tolerance reached.")
            break
        
        if not np.all(np.isfinite(grad)):
            print("Non-finite gradient.")
            break

        optimizer.zero_grad(set_to_none=True)

        theta.grad = torch.from_numpy(grad).to(
            dtype=theta.dtype,
            device=theta.device,
        )

        optimizer.step()

        if iteration % print_every == 0:
            print(
                f"step={iteration:4d}, "
                f"loss={loss:.6e}, "
                f"best={best_loss:.6e}, "
                f"|g|={np.linalg.norm(grad):.6e}"
            )

    return best_x, best_loss
