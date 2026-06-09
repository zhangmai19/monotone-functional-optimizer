"""2D solvers for the optimal stop-loss insurance problem.

Approaches:
  1. solve_pointwise() — per-type unconstrained max_d Φ(θ,φ,d).
  2. solve_2d_slsqp() — full constrained optimization via SLSQP.
     Reliable for grids up to about 25×25 (≈625 vars, ≈1250 constraints).
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .primitives import L_BAR
from .virtual_surplus import Phi


# ══════════════════════════════════════════════════════════════
# Pointwise unconstrained maximizer
# ══════════════════════════════════════════════════════════════

def solve_pointwise(N_theta=50, N_phi=50):
    """Find argmax_d Φ(θ, φ, d) independently at each grid node.

    Returns
    -------
    theta_vals, phi_vals : ndarray (1-D)
    D_pw : ndarray (N_θ+1, N_φ+1) — pointwise maximizers.
    Phi_pw : ndarray — Φ at the maximizer.
    """
    theta_vals = np.linspace(1.0, 2.0, N_theta + 1)
    phi_vals = np.linspace(1.0, 2.0, N_phi + 1)
    D_pw = np.zeros((N_theta + 1, N_phi + 1))
    Phi_pw = np.zeros((N_theta + 1, N_phi + 1))

    for i, theta in enumerate(theta_vals):
        for j, phi in enumerate(phi_vals):
            res = minimize_scalar(
                lambda d: -Phi(theta, phi, d),
                bounds=(0.0, L_BAR),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 200},
            )
            D_pw[i, j] = res.x
            Phi_pw[i, j] = -res.fun

    return theta_vals, phi_vals, D_pw, Phi_pw


# ══════════════════════════════════════════════════════════════
# Full constrained 2D solve (SLSQP)
# ══════════════════════════════════════════════════════════════

def solve_2d(N_theta=20, N_phi=20, maxiter=5000, verbose=True):
    """Solve the full 2D monotone-constrained problem via SLSQP.

    Variables: D[i,j] ∈ [0, L_bar], flattened to a vector.
    Constraints: D[i+1,j] ≤ D[i,j] and D[i,j+1] ≤ D[i,j] (linear inequalities).

    Recommended grid: N_θ = N_φ ≤ 25 for reasonable solve time.

    Parameters
    ----------
    N_theta, N_phi : int
        Grid resolution.
    maxiter : int
        Maximum SLSQP iterations.
    verbose : bool

    Returns
    -------
    result : dict
        D_opt, theta_vals, phi_vals, objective, success, nit, pw_D, pw_Phi
    """
    theta_vals = np.linspace(1.0, 2.0, N_theta + 1)
    phi_vals = np.linspace(1.0, 2.0, N_phi + 1)

    n_vars = (N_theta + 1) * (N_phi + 1)

    def idx(i, j):
        return i * (N_phi + 1) + j

    # Objective: -Σ Φ(D_ij)  (minimize for scipy)
    def objective(x):
        total = 0.0
        for i in range(N_theta + 1):
            for j in range(N_phi + 1):
                total += Phi(theta_vals[i], phi_vals[j], x[idx(i, j)])
        return -total

    # Initial guess from pointwise maximizer
    _, _, D_pw, Phi_pw = solve_pointwise(N_theta, N_phi)
    x0 = D_pw.flatten()

    # Bounds [0, L_bar] for all variables
    bounds = [(0.0, L_BAR) for _ in range(n_vars)]

    # Linear monotonicity constraints
    constraints = []
    # θ-constraints: D[i,j] ≥ D[i+1,j]
    for i in range(N_theta):
        for j in range(N_phi + 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: x[idx(i, j)] - x[idx(i + 1, j)],
            })
    # φ-constraints: D[i,j] ≥ D[i,j+1]
    for i in range(N_theta + 1):
        for j in range(N_phi):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: x[idx(i, j)] - x[idx(i, j + 1)],
            })

    if verbose:
        print(f"  SLSQP: {n_vars} variables, {len(constraints)} constraints")

    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-12},
    )

    D_opt = result.x.reshape(N_theta + 1, N_phi + 1)

    return {
        'D_opt': D_opt,
        'theta_vals': theta_vals,
        'phi_vals': phi_vals,
        'objective': -result.fun,
        'success': result.success,
        'nit': result.nit,
        'message': result.message,
        'pw_D': D_pw,
        'pw_Phi': Phi_pw,
    }
