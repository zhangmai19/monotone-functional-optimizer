"""Run all 2D diagnostics and save results.

Diagnostics (per the spec):
  1. Pointwise unconstrained maximizer — is it already monotone?
  2. Full constrained optimization via PAVA coordinate descent.
  3. Cross-section comparison: diagonal of 2D solution vs 1D h*(t).
  4. Profit value comparison: π*_2D vs π*_1D.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .solve_2d import solve_pointwise, solve_2d
from .virtual_surplus import Phi
from .primitives import L_BAR


# ══════════════════════════════════════════════════════════════
# Diagnostic 1: Pointwise maximizer
# ══════════════════════════════════════════════════════════════

def diagnostic_pointwise(N=50, save_fig=True):
    """Compute per-type unconstrained max, check monotonicity."""
    print(f"\n{'='*60}")
    print(f"[Diagnostic 1] Pointwise unconstrained max (N={N}×{N})")
    print(f"{'='*60}")

    theta_v, phi_v, D_pw, Phi_pw = solve_pointwise(N_theta=N, N_phi=N)

    # Check monotonicity
    diff_theta = np.diff(D_pw, axis=0)
    diff_phi = np.diff(D_pw, axis=1)

    vio_theta = np.sum(diff_theta > 1e-10)
    vio_phi = np.sum(diff_phi > 1e-10)
    max_vio_theta = diff_theta.max() if vio_theta > 0 else 0.0
    max_vio_phi = diff_phi.max() if vio_phi > 0 else 0.0

    print(f"  d range: [{D_pw.min():.4f}, {D_pw.max():.4f}]")
    print(f"  Monotone in θ (Δ≤0): violations={vio_theta}/{diff_theta.size}, "
          f"max={max_vio_theta:.2e}")
    print(f"  Monotone in φ (Δ≤0): violations={vio_phi}/{diff_phi.size}, "
          f"max={max_vio_phi:.2e}")

    already_monotone = (vio_theta == 0 and vio_phi == 0)
    print(f"  Already monotone? {'YES' if already_monotone else 'NO'}")

    # Save figure
    if save_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Surface
        im = axes[0].pcolormesh(phi_v, theta_v, D_pw, cmap='viridis',
                                 shading='auto')
        axes[0].set_xlabel(r'$\varphi$', fontsize=12)
        axes[0].set_ylabel(r'$\theta$', fontsize=12)
        axes[0].set_title('Pointwise Maximizer $d^{pw}(\\theta,\\varphi)$',
                          fontsize=13, fontweight='bold')
        # Invert y-axis so θ=2 is at top (matches surface convention)
        axes[0].invert_yaxis()
        plt.colorbar(im, ax=axes[0], label='d')

        # Diagonal cross-section
        diag_idx = np.arange(N + 1)
        diag_d = D_pw[diag_idx, diag_idx]
        t_vals = np.linspace(0, 1, N + 1)
        axes[1].plot(t_vals, diag_d, 'o-', color='steelblue', linewidth=1.5,
                     markersize=3, label='diagonal $d^{pw}$')
        axes[1].set_xlabel('t = θ−1 = φ−1', fontsize=12)
        axes[1].set_ylabel('d', fontsize=12)
        axes[1].set_title('Diagonal Cross-section', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle('Diagnostic 1 — Pointwise Unconstrained Maximizer',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()

        fname = 'figures/diag1_pointwise.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")

    return theta_v, phi_v, D_pw, Phi_pw, already_monotone


# ══════════════════════════════════════════════════════════════
# Diagnostic 2: Full constrained optimization
# ══════════════════════════════════════════════════════════════

def diagnostic_full_2d(N=50, save_fig=True):
    """Solve the full 2D constrained problem via PAVA coordinate descent."""
    print(f"\n{'='*60}")
    print(f"[Diagnostic 2] Full constrained 2D solve (N={N}×{N})")
    print(f"{'='*60}")

    result = solve_2d(N_theta=N, N_phi=N, max_iter=30, tol=1e-8, verbose=True)
    D_opt = result['D_opt']
    theta_v = result['theta_vals']
    phi_v = result['phi_vals']

    # Verify monotonicity (tolerance for numerical noise from minimize_scalar)
    mono_theta = np.all(np.diff(D_opt, axis=0) <= 1e-8)
    mono_phi = np.all(np.diff(D_opt, axis=1) <= 1e-8)
    print(f"  Monotone in θ: {mono_theta}")
    print(f"  Monotone in φ: {mono_phi}")
    print(f"  Objective ΣΦ = {result['objective']:.8f}")
    print(f"  D range: [{D_opt.min():.4f}, {D_opt.max():.4f}]")

    if save_fig:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

        # Constrained surface
        im0 = axes[0].pcolormesh(phi_v, theta_v, D_opt, cmap='viridis',
                                  shading='auto')
        axes[0].set_xlabel(r'$\varphi$', fontsize=12)
        axes[0].set_ylabel(r'$\theta$', fontsize=12)
        axes[0].set_title(r'Constrained $D^*(\theta,\varphi)$',
                          fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        plt.colorbar(im0, ax=axes[0], label='d')

        # Pointwise (reference)
        im1 = axes[1].pcolormesh(phi_v, theta_v, result['pw_D'], cmap='viridis',
                                  shading='auto')
        axes[1].set_xlabel(r'$\varphi$', fontsize=12)
        axes[1].set_ylabel(r'$\theta$', fontsize=12)
        axes[1].set_title('Pointwise $d^{pw}$ (unconstrained)',
                          fontsize=13, fontweight='bold')
        axes[1].invert_yaxis()
        plt.colorbar(im1, ax=axes[1], label='d')

        # Diagonal cross-section
        t_vals = np.linspace(0, 1, N + 1)
        diag_opt = D_opt[np.arange(N + 1), np.arange(N + 1)]
        diag_pw = result['pw_D'][np.arange(N + 1), np.arange(N + 1)]
        axes[2].plot(t_vals, diag_opt, '-', color='steelblue', linewidth=2,
                     label='constrained $D^*$')
        axes[2].plot(t_vals, diag_pw, '--', color='darkorange', linewidth=1.5,
                     alpha=0.7, label='pointwise $d^{pw}$')
        axes[2].set_xlabel('t = θ−1 = φ−1', fontsize=12)
        axes[2].set_ylabel('d', fontsize=12)
        axes[2].set_title('Diagonal Cross-section', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.suptitle('Diagnostic 2 — Full Constrained 2D Solution',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()

        fname = 'figures/diag2_full_2d.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")

    return result


# ══════════════════════════════════════════════════════════════
# Diagnostic 3: Cross-section comparison with 1D solution
# ══════════════════════════════════════════════════════════════

def diagnostic_compare_1d(result_2d, result_1d, save_fig=True):
    """Compare 2D diagonal with 1D h*(t)."""
    print(f"\n{'='*60}")
    print(f"[Diagnostic 3] Cross-section comparison: 2D diagonal vs 1D h*(t)")
    print(f"{'='*60}")

    D_opt = result_2d['D_opt']
    N2 = len(result_2d['theta_vals']) - 1
    t_2d = np.linspace(0, 1, N2 + 1)
    diag_2d = D_opt[np.arange(N2 + 1), np.arange(N2 + 1)]

    h_1d = result_1d
    t_1d = np.linspace(0, 1, len(h_1d) - 1) if len(
        np.atleast_1d(result_1d)) > 2 else np.linspace(0, 1, len(h_1d))
    # result_1d could be h_opt array directly or a dict
    if isinstance(result_1d, np.ndarray):
        h_1d = result_1d
        N1 = len(h_1d) - 1
        t_1d_nodes = np.linspace(0, 1, N1 + 1)
    else:
        # Assume dict with 'h_opt'
        h_1d = result_1d.get('h_opt', result_1d)
        if hasattr(h_1d, 'shape'):
            N1 = len(h_1d) - 1
            t_1d_nodes = np.linspace(0, 1, N1 + 1)
        else:
            raise TypeError("result_1d must be ndarray or dict with 'h_opt'")

    # Interpolate 1D to 2D grid
    h_interp = np.interp(t_2d, t_1d_nodes, h_1d)

    err_L2 = np.sqrt(np.trapezoid((diag_2d - h_interp)**2, t_2d))
    err_max = np.max(np.abs(diag_2d - h_interp))

    print(f"  L² error (2D diagonal vs 1D): {err_L2:.6f}")
    print(f"  Max error:                    {err_max:.6f}")

    if save_fig:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(t_1d_nodes[:len(h_1d)], h_1d, '-', color='crimson', linewidth=2,
                label=f'1D $h^*(t)$  (N={N1})')
        ax.plot(t_2d, diag_2d, 'o-', color='steelblue', linewidth=1.5,
                markersize=3, label=f'2D diagonal  (N={N2})')
        ax.set_xlabel('t', fontsize=12)
        ax.set_ylabel('d (deductible)', fontsize=12)
        ax.set_title('1D vs 2D — Diagonal Cross-section Comparison',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.text(0.02, 0.98,
                f'$L^2$ error = {err_L2:.4f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fname = 'figures/diag3_compare_1d.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")

    return err_L2, err_max


# ══════════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════════

def run_all(N_2d=50, load_1d=None):
    """Run all three diagnostics.

    Parameters
    ----------
    N_2d : int
        Grid resolution for the 2D problem.
    load_1d : str or None
        Path to 1D canonical_results.npz, or None to skip comparison.
    """
    # Diag 1
    theta_v, phi_v, D_pw, Phi_pw, monotone = diagnostic_pointwise(N=N_2d)

    # Diag 2
    result_2d = diagnostic_full_2d(N=N_2d)

    # Diag 3
    if load_1d is not None:
        data = np.load(load_1d)
        h_1d = data['h_opt']
        diagnostic_compare_1d(result_2d, h_1d)
    else:
        print("\n[Diagnostic 3] Skipped — no 1D result file provided")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Pointwise monotone?  {'YES' if monotone else 'NO'}")
    print(f"  2D constrained D range: [{result_2d['D_opt'].min():.4f}, "
          f"{result_2d['D_opt'].max():.4f}]")
    print(f"  2D objective ΣΦ = {result_2d['objective']:.8f}")

    # Profit comparison placeholder (needs proper scaling)
    d_theta = (2.0 - 1.0) / N_2d
    d_phi = (2.0 - 1.0) / N_2d
    pi_2d = result_2d['objective'] * d_theta * d_phi
    print(f"  π_2D (trapezoidal) ≈ {pi_2d:.6f}")

    return result_2d
