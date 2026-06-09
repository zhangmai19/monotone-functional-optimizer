r"""Virtual surplus Φ(θ, φ, d) — the integrand of the 2D objective.

From the paper, the per-type virtual surplus is:

  Φ = [V − E[I]]·f·m − V_θ·(1−F)·m − V_φ·f·(1−M)

For the canonical case (f=m=1, 1−F=2−θ, 1−M=2−φ):

  Φ(θ,φ,d) = V − E[I] − V_θ·(2−θ) − V_φ·(2−φ)

Saved in closed form by grouping terms in e^{−φd/θ} and e^{−d/θ}.
"""

import numpy as np
from .primitives import (
    L_BAR, V, expected_indemnity, V_theta, V_phi,
    hazard_theta, hazard_phi,
)


def Phi(theta, phi, d):
    """Virtual surplus for a single type (θ, φ) with deductible d.

    Parameters
    ----------
    theta, phi : float  — type coordinates ∈ [1, 2].
    d : float           — deductible ∈ [0, L_bar].

    Returns
    -------
    float
    """
    # All terms evaluated directly from primitives
    v = V(theta, phi, d)
    ei = expected_indemnity(theta, d)
    vt = V_theta(theta, phi, d)
    vp = V_phi(theta, phi, d)
    h_t = hazard_theta(theta)
    h_p = hazard_phi(phi)

    return v - ei - vt * h_t - vp * h_p


def Phi_d_closed(theta, phi, d):
    """Closed-form Φ = e^{−φd/θ}·C_d − θ·e^{−d/θ} + Φ_const.

    Return (Phi, coeff_d, exp_term, linear_term, const) for diagnostics.

    C_d = 2(θ−1)/φ + θ(2−φ)/φ² + d·[(2−φ)/φ − (2−θ)/θ]

    The d-independent part (involving L_bar only) is the constant.
    """
    a = np.exp(-phi * d / theta)        # e^{-φd/θ}
    b = np.exp(-d / theta)               # e^{-d/θ}

    # C_d = coefficient of e^{-φd/θ} in the d-dependent part
    c_d = (2.0 * (theta - 1.0) / phi
           + theta * (2.0 - phi) / (phi * phi)
           + d * ((2.0 - phi) / phi - (2.0 - theta) / theta))

    # Constant involves L_bar terms
    a_bar = np.exp(-phi * L_BAR / theta)
    c_bar = (2.0 * (theta - 1.0) / phi
             + theta * (2.0 - phi) / (phi * phi)
             + L_BAR * ((2.0 - phi) / phi - (2.0 - theta) / theta))
    b_bar = np.exp(-L_BAR / theta)

    const = -a_bar * c_bar + theta * b_bar

    return a * c_d - theta * b + const, c_d, a, -theta * b, const


def dPhi_dd(theta, phi, d):
    """∂Φ/∂d — analytical derivative for use in pointwise optimization.

    Φ = e^{−φd/θ}·C_d − θ·e^{−d/θ} + const  (d appears only in first two terms)

    C_d = C₀ + d·C_lin
    where C₀ = 2(θ−1)/φ + θ(2−φ)/φ²
          C_lin = (2−φ)/φ − (2−θ)/θ

    ∂Φ/∂d = e^{−φd/θ}·[−(φ/θ)·C_d + C_lin] + e^{−d/θ}
    """
    a = np.exp(-phi * d / theta)
    b = np.exp(-d / theta)

    # C_0 and C_lin for C_d = C_0 + d * C_lin
    c0 = 2.0 * (theta - 1.0) / phi + theta * (2.0 - phi) / (phi * phi)
    c_lin = (2.0 - phi) / phi - (2.0 - theta) / theta
    cd = c0 + d * c_lin

    return a * (-(phi / theta) * cd + c_lin) + b
