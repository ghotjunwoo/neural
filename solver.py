import numpy as np
import torch
from bh_horizons import *

def cumulative_integral(y, x):
    cumulative = torch.cumulative_trapezoid(y, x=x, dim=0)
    return torch.cat([torch.zeros(1, dtype=y.dtype, device=y.device), cumulative])

def solve_Phi_wave(V, rho, drho, omega, domega, r, r_U, r_V, Q, A_U, eM, m):
    """
    Compute Phi_U at V=1.
    All input arrays have shape (n_grid,).
    """

    phase = torch.exp(-1j * omega)

    Phi = rho * phase
    Phi_V = (drho - 1j * rho * domega) * phase

    integrand = (
        -(r_U + 1j * eM * A_U * r) * Phi_V
        + 1j * eM * (Q / (4.0 * r) - A_U * r_V) * Phi
        - 0.25 * m**2 * r * Phi
    )

    Phi_U_1 = torch.trapezoid(integrand, x=V, dim=0) / r[-1]

    return Phi_U_1.real, Phi_U_1.imag

def AU_values(V_grid, r_vals, Q_vals):
    integrand = -Q_vals / (2 * r_vals**2)
    AU_vals = cumulative_integral(integrand, x=V_grid)
    return AU_vals

def transport_values(state, q, lam, m_over_e=0.0, eM=None):
    """
    Construct Q, r, r_V, and r_U from the differentiable
    state returned by xi_integral.

    All returned arrays remain connected to autograd.
    """
    V = state["V"]
    xi = state["xi"]
    xip = state["xip"]
    rho = state["rho"]
    domega = state["domega"]

    dtype = V.dtype
    device = V.device

    r_plus_value, Qtarget_value = rplus_and_Qtarget(q, lam)

    r_plus = torch.as_tensor(r_plus_value, dtype=dtype, device=device)
    Qtarget = torch.as_tensor(Qtarget_value, dtype=dtype, device=device)

    r = r_plus * xi
    r_V = r_plus * xip

    charge_density = xi**2 * rho**2 * domega
    cum_I = cumulative_integral(charge_density, V)
    I_theta = cum_I[-1]

    if eM is None:
        # Infer e so that Q(1) = Qtarget exactly.
        e = Qtarget / (r_plus**2 * I_theta)
        Q = Qtarget * cum_I / I_theta
    else:
        e = torch.as_tensor(eM, dtype=dtype, device=device)
        Q = e * r_plus**2 * cum_I

    m = m_over_e * e
    charge_residual = Q[-1] - Qtarget

    rU0 = (lam * r[0]**2 / 3.0 - 1.0) / (4.0 * r_V[0])

    rU_integrand = (
        -0.25
        + Q**2 / (4.0 * r**2)
        + lam * r**2 / 4.0
        + m**2 * r**2 * rho**2 / 4.0
    )

    r_rU = r[0] * rU0 + cumulative_integral(rU_integrand, V)
    r_U = r_rU / r

    return {
        "V": V,
        "r": r,
        "r_V": r_V,
        "r_U": r_U,
        "Q": Q,
        "r_plus": r_plus,
        "Qtarget": Qtarget,
        "I_theta": I_theta,
        "e": e,
        "m": m,
        "m_over_e": m_over_e,
        "charge_residual": charge_residual,
        "sup_rU": torch.max(r_U),
    }

def penalty_xi(xi_vals, delta=1e-3):
    violation = torch.relu(delta - xi_vals[0])
    return violation**2


def penalty_rU(rU_vals, delta=1e-3):
    violation = torch.relu(delta + rU_vals)
    return torch.mean(violation**2)

def unpack_theta(theta):
    p = (theta.numel() - 1) // 3
    a0 = theta[0]
    a = theta[1:p+1]
    w = theta[p+1:2*p+1]
    b = theta[2*p+1:]
    return a0, a, w, b

def tanh_ansatz(V, theta, k):
    """Neural network Ansatz for rho and its derivative."""
    a0, a, w, b = unpack_theta(theta)
    
    # Shape: (n_grid, p)
    z = (V[:, None] * w[None, :] + b[None, :])
    tanh_z = torch.tanh(z)

    # Shape: (n_grid,)
    network = a0 + torch.sum(a[None, :] * tanh_z, dim=1)
    # dN/dV
    dnetwork = torch.sum(a[None, :] * w[None, :] * (1.0 - tanh_z.square()), dim=1)

    n = k + 1
    P = V**n * (1.0 - V)**n
    dP = (n * V**(n - 1) * (1.0 - V)**(n - 1) * (1.0 - 2.0 * V))

    x = P * network
    dx = dP * network + P * dnetwork

    return x, dx

def omega_domega(V, theta, k):
    log_kappa = theta[0]
    kappa = torch.exp(log_kappa)
    theta = theta[1:]
    x, dx = tanh_ansatz(V, theta, k=0)
    return kappa * V + x, kappa + dx
    
def raychaudhuri_rk4(V, E_grid, E_mid):
    if V.ndim != 1:
        raise ValueError("V must be one-dimensional.")

    if E_grid.shape != V.shape:
        raise ValueError("E_grid must have the same shape as V.")

    if E_mid.shape != V[:-1].shape:
        raise ValueError("E_mid must have length len(V)-1.")

    # Data at V=1.
    xi = torch.ones((), dtype=V.dtype, device=V.device)
    xip = torch.zeros((), dtype=V.dtype,device=V.device,
    )

    # Stored initially in descending-V order.
    xi_reverse = [xi]
    xip_reverse = [xip]
    n_grid = V.numel()

    for j in range(n_grid - 1, 0, -1):
        h = V[j - 1] - V[j]
        
        E_right = E_grid[j]
        E_left = E_grid[j - 1]
        E_half = E_mid[j - 1]
        
        # Stage 1: at V[j].
        k1_xi = xip
        k1_xip = -E_right * xi

        # Stage 2: at the midpoint.
        xi_2 = xi + 0.5 * h * k1_xi
        xip_2 = xip + 0.5 * h * k1_xip

        k2_xi = xip_2
        k2_xip = -E_half * xi_2

        # Stage 3: again at the midpoint.
        xi_3 = xi + 0.5 * h * k2_xi
        xip_3 = xip + 0.5 * h * k2_xip

        k3_xi = xip_3
        k3_xip = -E_half * xi_3

        # Stage 4: at V[j-1].
        xi_4 = xi + h * k3_xi
        xip_4 = xip + h * k3_xip

        k4_xi = xip_4
        k4_xip = -E_left * xi_4

        # RK4 update.
        xi = xi + h * (k1_xi + 2.0 * k2_xi + 2.0 * k3_xi + k4_xi) / 6.0
        xip = xip + h * (k1_xip + 2.0 * k2_xip + 2.0 * k3_xip + k4_xip) / 6.0

        xi_reverse.append(xi)
        xip_reverse.append(xip)

    # Reverse back into increasing-V order.
    xi_values = torch.stack(xi_reverse[::-1])
    xip_values = torch.stack(xip_reverse[::-1])

    return xi_values, xip_values

def xi_integral(theta_amp, theta_phase, k, rho_drho_func, omega_domega_func, n_grid=1001):

    if theta_amp.dtype != theta_phase.dtype:
        raise ValueError("Amplitude and phase parameters must have the same dtype.")

    if theta_amp.device != theta_phase.device:
        raise ValueError("Amplitude and phase parameters must be on the same device.")
    
    dtype = theta_amp.dtype
    device = theta_amp.device
        
    V = torch.linspace(0.0, 1.0, n_grid, dtype=dtype, device=device)
    V_mid = 0.5 * (V[:-1] + V[1:])
    
    rho, drho = rho_drho_func(V, theta_amp, k=k)
    rho_mid, drho_mid = rho_drho_func(V_mid, theta_amp, k=k)
    omega, domega = omega_domega_func(V, theta_phase, k=k)
    _, domega_mid = omega_domega_func(V_mid, theta_phase, k=k)
    
    E_grid = drho**2 + (domega * rho)**2
    E_mid = drho_mid**2 + (domega_mid * rho_mid)**2
    
    xi, xip = raychaudhuri_rk4(V, E_grid, E_mid)

    integrand = xi**2 * rho**2 * domega
    I = torch.trapezoid(integrand, x=V, dim=0)

    return {
        "V": V,
        "rho": rho,
        "drho": drho,
        "omega": omega,
        "domega": domega,
        "xi": xi,
        "xip": xip,
        "I": I,
    }

def loss_C1(theta_1, theta_2, eM, q, rho_drho_func, omega_domega_func, lam=0.0, m_over_e=0.0, w_Q=1.0, w_1=1.0, w_penalty=1000.0, delta_pen=1e-4, n_grid=1001):
    state = xi_integral(theta_1, theta_2, k=1, rho_drho_func=rho_drho_func, omega_domega_func=omega_domega_func, n_grid=n_grid)
    out = transport_values(state, q=q, lam=lam, m_over_e=m_over_e, eM=eM)
    
    rho = state["rho"]
    drho = state["drho"]
    omega = state["omega"]
    domega = state["domega"]

    V = out["V"]
    r = out["r"]
    r_U = out["r_U"]
    r_V = out["r_V"]
    Q = out["Q"]
    
    A_U = AU_values(V, r, Q)

    re_phi_u, im_phi_u = solve_Phi_wave(V, rho, drho, omega, domega, r, r_U, r_V, Q, A_U, out["e"], out["m"])

    penalty = w_penalty * (penalty_xi(state["xi"], delta=delta_pen) + penalty_rU(r_U, delta=delta_pen))

    return w_Q * out["charge_residual"]**2 + w_1 * (re_phi_u**2 + im_phi_u**2) + penalty

import torch


def adam_optimize(parameters, loss_fn, *,
    lr=1e-3, betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0.0, amsgrad=False, max_iter=1000, loss_tol=1e-12, grad_tol=1e-8, min_delta=0.0, patience=None, max_grad_norm=None, print_every=20):
    """
    Minimize a differentiable scalar Torch loss using Adam.

    Parameters
    ----------
    parameters:
        Iterable of Torch tensors or nn.Parameters with
        requires_grad=True.

    loss_fn:
        Zero-argument function returning a scalar Torch tensor.
        It must remain connected to all trainable parameters.

    min_delta:
        Minimum loss decrease counted as an improvement.

    patience:
        Stop after this many consecutive non-improving iterations.
        None disables patience stopping.

    max_grad_norm:
        Optional gradient clipping threshold.
        None disables gradient clipping.

    Returns
    -------
    Dictionary containing:
        best_params, best_loss, history, optimizer, stop_reason.

    Notes
    -----
    The live parameters remain at the latest Adam iterate.
    best_params stores separate copies of the best iterate.
    """

    params = list(parameters)

    if not params:
        raise ValueError("At least one parameter tensor is required.")

    if any(not p.requires_grad for p in params):
        raise ValueError(
            "Every optimized parameter must have requires_grad=True."
        )

    devices = {p.device for p in params}
    dtypes = {p.dtype for p in params}

    if len(devices) != 1:
        raise ValueError("All parameters must be on the same device.")

    if len(dtypes) != 1:
        raise ValueError("All parameters must have the same dtype.")

    device = params[0].device
    dtype = params[0].dtype

    optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=adam_eps, weight_decay=weight_decay, amsgrad=amsgrad)

    best_loss = float("inf")
    best_params = [parameter.detach().clone() for parameter in params]

    history = []
    stale_steps = 0
    stop_reason = "maximum iterations reached"

    for step in range(max_iter):
        # Gradients accumulate unless explicitly cleared.
        optimizer.zero_grad(set_to_none=True)

        # One complete forward calculation.
        loss = loss_fn()

        if not torch.is_tensor(loss):
            raise TypeError(
                "loss_fn must return a Torch tensor, "
                f"not {type(loss).__name__}."
            )

        if loss.ndim != 0:
            raise ValueError(
                "loss_fn must return a scalar tensor; "
                f"received shape {tuple(loss.shape)}."
            )

        if not loss.requires_grad:
            raise RuntimeError(
                "The loss is disconnected from the parameters. "
                "Check for detach(), numpy(), item(), or NumPy/SciPy "
                "operations inside the forward calculation."
            )

        if not torch.isfinite(loss).item():
            stop_reason = "non-finite loss"
            print(f"Stopped at step {step}: non-finite loss.")
            break

        loss_value = loss.detach().item()

        # Store the current parameters before Adam changes them.
        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_params = [parameter.detach().clone() for parameter in params]
            stale_steps = 0
        else:
            stale_steps += 1

        if loss_value <= loss_tol:
            stop_reason = "loss tolerance reached"
            print(
                f"Stopped at step {step}: "
                f"loss tolerance reached ({loss_value:.6e})."
            )
            break

        if patience is not None and stale_steps >= patience:
            stop_reason = "patience exhausted"
            print(
                f"Stopped at step {step}: "
                f"no sufficient improvement for {patience} steps."
            )
            break

        # One reverse pass calculates every parameter derivative.
        loss.backward()

        missing_gradients = [index for index, parameter in enumerate(params) if parameter.grad is None]

        if missing_gradients:
            raise RuntimeError(
                "Some parameters are disconnected from the loss. "
                f"Missing gradient indices: {missing_gradients}"
            )

        if max_grad_norm is None:
            grad_norm_squared = torch.zeros((), dtype=dtype, device=device)

            for parameter in params:
                grad_norm_squared = (grad_norm_squared + parameter.grad.detach().square().sum())

            grad_norm = torch.sqrt(grad_norm_squared)

        else:
            # Returns the norm before clipping.
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm, error_if_nonfinite=False)

        if not torch.isfinite(grad_norm).item():
            stop_reason = "non-finite gradient"
            print(f"Stopped at step {step}: non-finite gradient.")
            break

        grad_value = grad_norm.detach().item()

        history.append({
            "step": step,
            "loss": loss_value,
            "best_loss": best_loss,
            "grad_norm": grad_value,
        })

        if step % print_every == 0:
            print(
                f"step={step:5d}, "
                f"loss={loss_value:.6e}, "
                f"best={best_loss:.6e}, "
                f"|g|={grad_value:.6e}"
            )

        if grad_value <= grad_tol:
            stop_reason = "gradient tolerance reached"
            print(
                f"Stopped at step {step}: "
                f"gradient tolerance reached ({grad_value:.6e})."
            )
            break

        optimizer.step()

    else:
        # The last Adam update has not yet had its loss evaluated.
        # Evaluate it once without constructing another graph.
        with torch.no_grad():
            final_loss = loss_fn()

        if torch.isfinite(final_loss).item():
            final_loss_value = final_loss.item()

            if final_loss_value < best_loss - min_delta:
                best_loss = final_loss_value
                best_params = [parameter.detach().clone() for parameter in params]

    return {
        "best_params": best_params,
        "best_loss": best_loss,
        "history": history,
        "optimizer": optimizer,
        "stop_reason": stop_reason,
    }

def evaluate_C1_torch(
    theta_1,
    theta_2,
    *,
    eM,
    q,
    rho_drho_func,
    omega_domega_func,
    lam=0.0,
    m_over_e=0.0,
    n_grid=4001,
):
    """
    Evaluate the individual C1 residuals using the Torch
    training discretisation, without constructing gradients.
    """

    with torch.no_grad():
        state = xi_integral(
            theta_amp=theta_1,
            theta_phase=theta_2,
            k=1,
            rho_drho_func=rho_drho_func,
            omega_domega_func=omega_domega_func,
            n_grid=n_grid,
        )

        transport = transport_values(
            state,
            q=q,
            lam=lam,
            m_over_e=m_over_e,
            eM=eM,
        )

        V = state["V"]
        r = transport["r"]
        r_V = transport["r_V"]
        r_U = transport["r_U"]
        Q = transport["Q"]

        A_U = AU_values(V, r, Q)
        re_phi_u, im_phi_u = solve_Phi_wave(
            V=V,
            rho=state["rho"],
            drho=state["drho"],
            omega=state["omega"],
            domega=state["domega"],
            r=r,
            r_U=r_U,
            r_V=r_V,
            Q=Q,
            A_U=A_U,
            eM=transport["e"],
            m=transport["m"],
        )

        charge_residual = transport["charge_residual"]

        matching_loss = (
            charge_residual**2
            + re_phi_u**2
            + im_phi_u**2
        )

        return {
            "n_grid": n_grid,
            "charge_residual": charge_residual.item(),
            "Re_Phi_U_1": re_phi_u.item(),
            "Im_Phi_U_1": im_phi_u.item(),
            "matching_loss": matching_loss.item(),
            "residual_norm": torch.sqrt(
                matching_loss
            ).item(),
            "inf_xi": torch.min(
                state["xi"]
            ).item(),
            "sup_rU": torch.max(
                r_U
            ).item(),
            "phase_winding": (
                state["omega"][-1]
                - state["omega"][0]
            ).item(),
            "min_domega": torch.min(
                state["domega"]
            ).item(),
            "max_domega": torch.max(
                state["domega"]
            ).item(),
        }

# def make_theta_phase_init(
#     theta_amp_init,
#     p_phase=None,
#     perturbation=0.0,
#     seed=0,
# ):
#     """
#     Initialize the direct phase Ansatz

#         omega(V) = exp(log_kappa) * V + x(V)
    
#     at, or very close to, fixed phase omega(V)=V.

#     Parameter ordering:
#         [log_kappa, a0, a, w, b]
#     """
#     theta_amp_init = np.asarray(
#         theta_amp_init,
#         dtype=np.float64,
#     )

#     if (theta_amp_init.size - 1) % 3 != 0:
#         raise ValueError(
#             "theta_amp_init must have length 1 + 3*p."
#         )

#     p_amp = (theta_amp_init.size - 1) // 3
#     p = p_amp if p_phase is None else int(p_phase)

#     if p < 1:
#         raise ValueError("p_phase must be positive.")

#     # Tanh transition centres distributed across the interval.
#     centres = np.linspace(0.15, 0.85, p)

#     # Moderate slopes: neither almost linear nor excessively sharp.
#     widths = np.linspace(4.0, 12.0, p)

#     # w_j V + b_j = 0 at V = centre_j.
#     biases = -widths * centres

#     rng = np.random.default_rng(seed)

#     # Zero gives exactly omega=V.
#     # A tiny value such as 1e-4 gives a small symmetry-breaking seed.
#     a0 = perturbation * rng.normal()
#     a = perturbation * rng.normal(size=p)

#     theta_phase_init = np.concatenate([
#         np.array([0.0]),  # log_kappa=0, hence kappa=1
#         np.array([a0]),
#         a,
#         widths,
#         biases,
#     ])

#     assert theta_phase_init.size == 3 * p + 2

#     return theta_phase_init