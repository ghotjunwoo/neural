from scipy.optimize import brentq
import numpy as np

def degen_roots(lam):
    """
    Returns the extremal roots of the RN-(A)dS polynomial for given Lambda.
    """
    coeffs = [2*lam/3, 0.0, -1.0, 1.0]
    roots = np.roots(coeffs)

    real_pos = sorted([
        r.real for r in roots
        if abs(r.imag) < 1e-10 and r.real > 0
    ])

    return real_pos

def Q2_degenerate(lam, r):
    '''
    Returns Q^2 for a degenerate horizon at radius r for given Lambda.
    '''
    return r - lam * r**4 / 3

def Qmax_RNdS(lam):
    """
    M=1. Returns Qmax for RN-(A)dS.
    For lam > 0, choose the smaller positive extremal root.
    """
    
    # if lam = 0
    if abs(lam) < 1e-14:
        return 1.0

    # otherwise
    real_pos = degen_roots(lam)

    if len(real_pos) == 0:
        raise ValueError(f"No positive extremal root for Lambda={lam}")

    # For positive Lambda, the smaller positive root is the cold/extremal BH root.
    # For negative Lambda, there is only one positive root.
    r_ext = real_pos[0]
    Q2 = Q2_degenerate(lam, r_ext)
    if Q2 <= 0:
        raise ValueError(f"Qmax^2 <= 0 for Lambda={lam}")

    return np.sqrt(Q2) 

def positive_horizon_roots(lam, Q):
    """
    Positive roots of f(r)=0 for M=1:
        f = 1 - 2/r + Q^2/r^2 - Lambda r^2/3.
    We solve r^2 f(r)=0.
    """
    if abs(lam) < 1e-14:
        roots = np.roots([1.0, -2.0, Q**2])
    else:
        roots = np.roots([-lam/3, 0.0, 1.0, -2.0, Q**2])

    real_pos = sorted([
        r.real for r in roots
        if abs(r.imag) < 1e-8 and r.real > 1e-9
    ])

    return real_pos


def rplus_and_Qtarget(q, lam):
    Qmax = Qmax_RNdS(lam)
    Qtarget = q * Qmax

    roots = positive_horizon_roots(lam, Qtarget)

    if len(roots) == 0:
        raise ValueError(f"No positive horizon roots for q={q}, Lambda={lam}")

    if lam > 0:
        # RN-dS: roots are roughly inner, event, cosmological.
        # If q=0, there may be only event and cosmological.
        if len(roots) >= 3:
            r_plus = roots[-2]
        elif len(roots) == 2:
            r_plus = roots[0]
        else:
            raise ValueError(f"Only one positive root for q={q}, Lambda={lam}")
    else:
        # RN or RN-AdS: event horizon is largest positive root.
        r_plus = roots[-1]

    return r_plus, Qtarget

def Q_nariai(lam):
    """
    Larger positive double-root branch:
        r_+ = r_c.

    This gives the endpoint where the black-hole and cosmological
    horizons coincide.
    """
    roots = degen_roots(lam)
    r_nar = roots[-1]
    return np.sqrt(max(0.0, Q2_degenerate(lam, r_nar)))


def find_lambda_max_q(q):
    """
    M=1. Returns maximum positive Lambda M^2 for fixed

        q = Q / Q_max(Lambda).

    This is the point where r_+ = r_c.
    """

    if abs(q) < 1e-14:
        return 1.0 / 9.0

    if abs(q - 1.0) < 1e-14:
        return 2.0 / 9.0

    def F(lam):
        return q * Qmax_RNdS(lam) - Q_nariai(lam)

    return brentq(
        F,
        1.0 / 9.0,
        2.0 / 9.0,
        xtol=1e-13,
        rtol=1e-13,
    )
    