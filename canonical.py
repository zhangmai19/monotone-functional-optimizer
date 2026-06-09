r"""Canonical problem: Optimal Stop-Loss Insurance under 2D Adverse Selection.

Implements the integrand P(t, h, h'), boundary term, and solver for:

  max  Π[h] = ∫₀¹ P(t, h(t), h'(t)) dt  +  boundary_term(h(0))
  s.t. 0 ≤ h(t) ≤ L_bar,  h'(t) ≤ 0

Reference
---------
Mai Zhang & Ka Chun Cheung, "Optimal Insurance: Adverse Selection for
2-dimension continuous types" — Sections 5–6.

Canonical Example
-----------------
  θ ∈ [1, 2],  φ ∈ [1, 2]           (uniform on each dimension)
  H_θ(l) = 1 − exp(−l/θ)            (exponential loss CDF)
  g_φ(p) = p^φ                       (power distortion)
  L_bar  = 10
  Diagonal: θ_t = 1+t, φ_t = 1+t
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize, bisect


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _clip_pos(x, eps=1e-12):
    """Clip to positive, guarding log(0)."""
    return np.maximum(np.abs(x) if np.isscalar(x) else x, eps)


# ═══════════════════════════════════════════════════════════════════════
# G(t) and its derivative decomposition
# ═══════════════════════════════════════════════════════════════════════

def compute_G_and_derivs(t, h):
    """Compute G(t) and the coefficients γ, δ in  d/dt[ln G] = γ + δ·h'.

    G(t) = (1 − e^{−h/(1+t)})^{1+t}

    Parameters
    ----------
    t : float ∈ [0, 1]
    h : float  (h(t), guarded ≥ ε)

    Returns
    -------
    G_val : float
    gamma : float   — h'-independent part of d/dt[ln G]
    delta : float   — coefficient of h' in d/dt[ln G]
    ln_G  : float   — ln(G) (useful downstream)
    """
    theta_t = 1.0 + t
    phi_t = 1.0 + t

    a_t = _clip_pos(1.0 - np.exp(-h / theta_t), 1e-14)

    G_val = a_t ** phi_t
    ln_G = phi_t * np.log(a_t)

    exp_t = np.exp(-h / theta_t)

    # d/dt [ln G] = ln(a_t) + (1+t)·(1/a_t)·da_t/dt
    # da_t/dt = ∂a/∂t + ∂a/∂h·h' = −h/(1+t)²·e^{−h/(1+t)} + h'/(1+t)·e^{−h/(1+t)}
    # →  d/dt[ln G]  = ln(a_t) + h'·e^{...} / a_t  −  h·e^{...} / (a_t·(1+t))
    #                 = γ            + δ·h'
    gamma = np.log(a_t) - h * exp_t / (a_t * theta_t)
    delta = exp_t / a_t

    return G_val, gamma, delta, ln_G


# ═══════════════════════════════════════════════════════════════════════
# Inner integrand terms at a single θ
# ═══════════════════════════════════════════════════════════════════════

def _inner_terms_at_theta(theta, t, h, hp, G_val, gamma, delta, ln_G):
    """Evaluate all θ-dependent terms of P at a single inner-quadrature node.

    Returns
    -------
    contrib_term1 : float     —  ∂F₁/∂t · ∫₀ʰ −H_θ(l) dl   (this θ's part of term 1)
    contrib_term3 : float     —  G·(2−θ)·∂F₁/∂θ            (this θ's part of term 3 coeff)
    """
    A = _clip_pos(1.0 - np.exp(-h / theta), 1e-14)
    ln_A = np.log(A)
    ln2_A = ln_A * ln_A
    exp_th = np.exp(-h / theta)
    inv_theta = 1.0 / theta

    # ── ∂F₁/∂t = α₁ + β₁·h' ──
    alpha_1 = gamma / ln_A
    # β₁ = (δ·ln_A  −  ln_G·exp(−h/θ) / (θ·A)) / ln²_A
    beta_1 = (delta * ln_A - ln_G * exp_th * inv_theta / A) / ln2_A

    # ── ∂F₁/∂θ = ln_G · h · exp(−h/θ) / (θ² · A · ln²_A) ──
    dF1_dtheta = ln_G * h * exp_th / (theta * theta * A * ln2_A)

    # ── ∫₀ʰ −H_θ(l) dl = −h + θ·(1 − e^{−h/θ}) ──
    inner_integral = -h + theta * (1.0 - exp_th)

    contrib_term1 = (alpha_1 + beta_1 * hp) * inner_integral
    contrib_term3 = G_val * (2.0 - theta) * dF1_dtheta  # (2−θ) = 1−F(θ)

    return contrib_term1, contrib_term3


# ═══════════════════════════════════════════════════════════════════════
# P(t, h, h')  —  the full integrand
# ═══════════════════════════════════════════════════════════════════════

def make_canonical_P(inner_n=30):
    """Create the canonical integrand P(t, h, h').

    P includes an inner integral over θ ∈ [1, 2] computed with
    `inner_n`-point Gauss–Legendre quadrature.

    P(t, h, h')  =  term1  +  (term4_coeff − term3_coeff)·h'

    where term2 vanishes because F(θ̲) = F(1) = 0 in the canonical example.

    Parameters
    ----------
    inner_n : int
        Gauss–Legendre nodes for the inner θ-integral (≥ 30 recommended).

    Returns
    -------
    callable  P(t, h, hp) → float
    """
    xi, wi = leggauss(inner_n)
    # Map [-1, 1] → [1, 2]
    theta_nodes = 1.5 + 0.5 * xi
    theta_weights = 0.5 * wi

    def P(t, h, hp):
        # ── numeric guard ──
        h_safe = _clip_pos(h, 1e-8)

        # ── G(t) ──
        G_val, gamma, delta, ln_G = compute_G_and_derivs(t, h_safe)

        # ── inner θ-integral ──
        term1 = 0.0
        term3_coeff = 0.0

        for k in range(inner_n):
            c1, c3 = _inner_terms_at_theta(
                theta_nodes[k], t, h_safe, hp,
                G_val, gamma, delta, ln_G,
            )
            term1 += c1 * theta_weights[k]
            term3_coeff += c3 * theta_weights[k]

        # ── term 4: G(t)·(1 − M(F₁(θ̲,t)))·h' ──
        # M(φ) = φ − 1  ⇒  1 − M(x) = 2 − x
        A_lower = _clip_pos(1.0 - np.exp(-h_safe), 1e-14)
        ln_A_lower = np.log(A_lower)
        F1_lower = ln_G / ln_A_lower
        term4_coeff = G_val * (2.0 - F1_lower)

        return term1 + (term4_coeff - term3_coeff) * hp

    return P


# ═══════════════════════════════════════════════════════════════════════
# Boundary term
# ═══════════════════════════════════════════════════════════════════════

def canonical_boundary_term(h0):
    """Boundary contribution: (1 − M(φ̲))·∫₀^{h₀} g_φ̲(H_θ̲(l)) dl.

    For the canonical example (θ̲ = 1, φ̲ = 1, g₁(p) = p, H₁(l) = 1−e^{−l}):
        = h₀ − 1 + e^{−h₀}
    """
    h0_safe = _clip_pos(h0, 0.0)
    return h0_safe - 1.0 + np.exp(-h0_safe)


# ═══════════════════════════════════════════════════════════════════════
# Full objective  J(h) = −Π[h]   (for scipy minimize)
# ═══════════════════════════════════════════════════════════════════════

def make_canonical_objective(inner_n=30, outer_quad_order=5):
    """Create J(h) = −Π[h] for the canonical insurance problem.

    Π[h] = ∫₀¹ P(t, h, h') dt  +  boundary_term(h₀)

    Parameters
    ----------
    inner_n : int
        Gauss–Legendre nodes for the inner θ-integral.
    outer_quad_order : int
        Gauss–Legendre nodes per subinterval for the outer t-integral.

    Returns
    -------
    callable  J(h) → float  (negated total surplus; minimize this)
    """
    from quadrature import integrate_full

    P = make_canonical_P(inner_n=inner_n)

    def J(h):
        N = len(h) - 1
        x_nodes = np.linspace(0.0, 1.0, N + 1)
        integral = integrate_full(P, x_nodes, h, quad_order=outer_quad_order)
        boundary = canonical_boundary_term(h[0])
        return -(integral + boundary)

    return J


# ═══════════════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════════════

def solve_canonical(N=100, inner_n=30, outer_quad_order=5, L_bar=10.0,
                    method="SLSQP", maxiter=2000):
    """Solve the canonical optimal insurance problem.

    Parameters
    ----------
    N : int
        Number of subintervals (→ N+1 decision variables).
    inner_n : int
        Gauss–Legendre nodes for the inner θ-integral (≥ 30).
    outer_quad_order : int
        Gauss–Legendre nodes per subinterval for the outer integral.
    L_bar : float
        Upper bound on the deductible path.
    method : str
        scipy method: 'SLSQP' or 'trust-constr'.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Augmented with:
        - result.objective_value : float  (Π, the maximized surplus)
        - result.h_opt : ndarray
        - result.boundary : float
        - result.N, result.inner_n, result.L_bar
    """
    J = make_canonical_objective(inner_n=inner_n,
                                 outer_quad_order=outer_quad_order)

    # Initial guess: linear decreasing from L_bar to 0
    h0 = np.linspace(L_bar, 0.0, N + 1)

    # Bounds  [0, L_bar]
    bounds = [(0.0, L_bar) for _ in range(N + 1)]

    # Monotonicity  h_i ≥ h_{i+1}
    cons = []
    for i in range(N):
        cons.append({
            "type": "ineq",
            "fun": lambda h, i=i: h[i] - h[i + 1],
        })

    opts = {"maxiter": maxiter}
    if method == "SLSQP":
        opts["ftol"] = 1e-12

    result = minimize(
        J, h0,
        method=method,
        bounds=bounds,
        constraints=cons,
        options=opts,
    )

    # Augment
    result.objective_value = -result.fun  # Π
    result.h_opt = result.x
    result.boundary = canonical_boundary_term(result.x[0])
    result.N = N
    result.inner_n = inner_n
    result.L_bar = L_bar

    return result


# ═══════════════════════════════════════════════════════════════════════
# G(t) evaluator (for diagnostics / reconstruction)
# ═══════════════════════════════════════════════════════════════════════

def G_of_t(t, h_val):
    """G(t) for a given (t, h(t))."""
    theta_t = 1.0 + t
    phi_t = 1.0 + t
    a = _clip_pos(1.0 - np.exp(-h_val / theta_t), 1e-14)
    return a ** phi_t


# ═══════════════════════════════════════════════════════════════════════
# Deductible surface reconstruction  d(θ, φ)
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_surface(h_opt, n_theta=50, n_phi=50):
    """Reconstruct the deductible surface d(θ, φ) from the optimal path h*(t).

    For each (θ, φ) ∈ [1,2]×[1,2], find t* solving:
        F₁(θ, t*) = φ
    where  F₁(θ, t) = ln G(t) / ln(1 − e^{−h*(t)/θ}),
           G(t) = (1 − e^{−h*(t)/(1+t)})^{1+t}.

    Then  d(θ, φ) = h*(t*).

    Parameters
    ----------
    h_opt : ndarray of shape (N+1,)
        Optimal nodal values.
    n_theta, n_phi : int
        Grid resolution.

    Returns
    -------
    theta_grid : ndarray (n_theta, n_phi)
    phi_grid : ndarray (n_theta, n_phi)
    d_surface : ndarray (n_theta, n_phi)
    """
    N = len(h_opt) - 1
    t_nodes = np.linspace(0.0, 1.0, N + 1)
    # Pre-compute ln G at each t for fast lookup
    ln_G_at_t = np.array([
        np.log(_clip_pos(1.0 - np.exp(-h_opt[i] / (1.0 + t_nodes[i])), 1e-14))
        * (1.0 + t_nodes[i])
        for i in range(N + 1)
    ])

    theta_vals = np.linspace(1.0, 2.0, n_theta)
    phi_vals = np.linspace(1.0, 2.0, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing="ij")
    d_surface = np.full((n_theta, n_phi), np.nan)

    # Interpolators
    def ln_G_interp(t):
        return np.interp(t, t_nodes, ln_G_at_t)

    def h_interp(t):
        return np.interp(t, t_nodes, h_opt)

    def F1_of_t_theta(t, theta):
        ln_G = ln_G_interp(t)
        A = _clip_pos(1.0 - np.exp(-h_interp(t) / theta), 1e-14)
        return ln_G / np.log(A)

    for i, theta in enumerate(theta_vals):
        for j, phi in enumerate(phi_vals):
            # Special case: on diagonal, t = θ − 1 = φ − 1
            if abs(theta - phi) < 1e-3:
                t_diag = theta - 1.0
                if 0.0 <= t_diag <= 1.0:
                    d_surface[i, j] = h_interp(t_diag)
                continue

            # For off-diagonal, find t solving F₁(θ, t) = φ
            # F₁(θ, t) is monotone in t (typically decreasing? or increasing?)
            # Evaluate at endpoints
            f0 = F1_of_t_theta(0.0, theta)
            f1 = F1_of_t_theta(1.0, theta)

            # F₁ at t=0: h(0) is large, (1 − e^{−h(0)}) ≈ 1, ln ≈ 0, G → ...
            # Actually F₁(θ, 0) and F₁(θ, 1) are both in [1,2] typically
            if phi < min(f0, f1) or phi > max(f0, f1):
                # φ out of range — find nearest boundary t
                if abs(phi - f0) < abs(phi - f1):
                    d_surface[i, j] = h_opt[0]
                else:
                    d_surface[i, j] = h_opt[-1]
                continue

            try:
                t_star = bisect(
                    lambda t: F1_of_t_theta(t, theta) - phi,
                    0.0, 1.0,
                    xtol=1e-8, maxiter=100,
                )
                d_surface[i, j] = h_interp(t_star)
            except Exception:
                d_surface[i, j] = np.nan

    return theta_grid, phi_grid, d_surface
