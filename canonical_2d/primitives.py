"""Fundamental functions for the 2D optimal insurance problem.

All formulas are closed-form for the canonical parameterization:
  H_θ(l) = 1 - exp(-l/θ)          (exponential loss CDF)
  g_φ(p)  = p^φ                     (power distortion)
  θ ∈ [1,2], φ ∈ [1,2], L_bar = 10
  f(θ)=1, m(φ)=1  (uniform densities)
"""

import numpy as np


# ── Global parameter ──
L_BAR = 10.0


# ══════════════════════════════════════════════════════════════
# Expected indemnity (stop-loss with deductible d)
# ══════════════════════════════════════════════════════════════

def expected_indemnity(theta, d):
    """E[I] = ∫_d^{L_bar} e^{-l/θ} dl = θ(e^{-d/θ} - e^{-L_bar/θ})."""
    return theta * (np.exp(-d / theta) - np.exp(-L_BAR / theta))


# ══════════════════════════════════════════════════════════════
# Willingness-to-pay  V(θ, φ, d)
# ══════════════════════════════════════════════════════════════

def V(theta, phi, d):
    """V = ∫_d^{L_bar} e^{-φl/θ} dl = (θ/φ)(e^{-φd/θ} - e^{-φL_bar/θ})."""
    a = np.exp(-phi * d / theta)
    b = np.exp(-phi * L_BAR / theta)
    return (theta / phi) * (a - b)


def V_theta(theta, phi, d):
    """∂V/∂θ = e^{-φd/θ}(1/φ + d/θ) - e^{-φL_bar/θ}(1/φ + L_bar/θ).

    Derived analytically from the closed form of V.
    """
    a = np.exp(-phi * d / theta)
    b = np.exp(-phi * L_BAR / theta)
    return a * (1.0 / phi + d / theta) - b * (1.0 / phi + L_BAR / theta)


def V_phi(theta, phi, d):
    """∂V/∂φ = -e^{-φd/θ}(θ/φ² + d/φ) + e^{-φL_bar/θ}(θ/φ² + L_bar/φ).

    Derived analytically from the closed form of V.
    Note: V_phi ≤ 0 (more distortion → lower willingness-to-pay).
    """
    a = np.exp(-phi * d / theta)
    b = np.exp(-phi * L_BAR / theta)
    return -a * (theta / (phi * phi) + d / phi) + b * (theta / (phi * phi) + L_BAR / phi)


# ══════════════════════════════════════════════════════════════
# Hazard factors for the canonical uniform case
# ══════════════════════════════════════════════════════════════

def hazard_theta(theta):
    """1 − F(θ) = 2 − θ  for uniform on [1, 2]."""
    return 2.0 - theta


def hazard_phi(phi):
    """1 − M(φ) = 2 − φ  for uniform on [1, 2]."""
    return 2.0 - phi
