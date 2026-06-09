"""Run all 2D diagnostics and compare with 1D solution.

Usage:
    python -m canonical_2d.run_all_2d
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canonical_2d.solve_2d import solve_pointwise, solve_2d
from canonical_2d.virtual_surplus import Phi
from canonical_2d.primitives import L_BAR


def main(N_2d=30, save_fig=True):
    print(f"{'='*60}")
    print(f"2D DIAGNOSTICS — Optimal Stop-Loss Deductible Surface")
    print(f"Grid: {N_2d}×{N_2d}, types θ,φ ∈ [1,2]×[1,2], L_bar={L_BAR}")
    print(f"{'='*60}")

    # ── 1. Pointwise maximizer ──
    print(f"\n[1] Pointwise unconstrained max...")
    theta_v, phi_v, D_pw, Phi_pw = solve_pointwise(N_theta=N_2d, N_phi=N_2d)
    mono_pw_theta = np.all(np.diff(D_pw, axis=0) <= 0)
    mono_pw_phi = np.all(np.diff(D_pw, axis=1) <= 0)
    print(f"    d range: [{D_pw.min():.2f}, {D_pw.max():.2f}]")
    print(f"    Monotone in θ: {mono_pw_theta},  in φ: {mono_pw_phi}")

    # ── 2. Full constrained solve ──
    print(f"\n[2] Full constrained 2D solve (PAVA coordinate descent)...")
    result = solve_2d(N_theta=N_2d, N_phi=N_2d, max_iter=30, tol=1e-6,
                      verbose=True)
    D_opt = result['D_opt']
    mono_opt_theta = np.all(np.diff(D_opt, axis=0) <= 0)
    mono_opt_phi = np.all(np.diff(D_opt, axis=1) <= 0)
    print(f"    D range: [{D_opt.min():.2f}, {D_opt.max():.2f}]")
    print(f"    Monotone in θ: {mono_opt_theta},  in φ: {mono_opt_phi}")
    print(f"    Objective ΣΦ = {result['objective']:.6f}")

    # ── 3. Comparison with 1D ──
    print(f"\n[3] Diagonal cross-section...")
    diag_idx = np.arange(N_2d + 1)
    diag_2d = D_opt[diag_idx, diag_idx]
    t_2d = np.linspace(0, 1, N_2d + 1)

    # Load 1D solution if available
    try:
        data_1d = np.load('figures/canonical_results.npz')
        h_1d = data_1d['h_opt']
        N1 = len(h_1d) - 1
        t_1d = np.linspace(0, 1, N1 + 1)
        h_interp = np.interp(t_2d, t_1d, h_1d)
        err_L2 = np.sqrt(np.trapezoid((diag_2d - h_interp)**2, t_2d))
        print(f"    1D solution: h ∈ [{h_1d.min():.4f}, {h_1d.max():.4f}]")
        print(f"    2D diagonal: d ∈ [{diag_2d.min():.4f}, {diag_2d.max():.4f}]")
        print(f"    L² error (2D diagonal vs 1D): {err_L2:.4f}")
        has_1d = True
    except Exception as e:
        print(f"    1D solution not available: {e}")
        has_1d = False

    if not save_fig:
        return result

    print(f"\n[4] Generating plots...")

    # ── Figure: 3-panel diagnostic ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel A: Pointwise maximizer
    im0 = axes[0].pcolormesh(phi_v, theta_v, D_pw, cmap='RdYlBu_r',
                              shading='auto', vmin=0, vmax=L_BAR)
    axes[0].set_xlabel(r'$\varphi$', fontsize=12)
    axes[0].set_ylabel(r'$\theta$', fontsize=12)
    axes[0].set_title('Pointwise $\max_d \Phi$ (unconstrained)', fontsize=12,
                      fontweight='bold')
    axes[0].invert_yaxis()
    plt.colorbar(im0, ax=axes[0], label='$d^{pw}$')

    # Panel B: Constrained 2D solution
    im1 = axes[1].pcolormesh(phi_v, theta_v, D_opt, cmap='RdYlBu_r',
                              shading='auto', vmin=0, vmax=L_BAR)
    axes[1].set_xlabel(r'$\varphi$', fontsize=12)
    axes[1].set_ylabel(r'$\theta$', fontsize=12)
    axes[1].set_title('Constrained $D^*$ (monotone in both)', fontsize=12,
                      fontweight='bold')
    axes[1].invert_yaxis()
    plt.colorbar(im1, ax=axes[1], label='$D^*$')

    # Panel C: Cross-section comparison
    axes[2].plot(t_2d, diag_2d, 'o-', color='steelblue', linewidth=1.5,
                 markersize=3, label=f'2D diagonal (N={N_2d})')
    if has_1d:
        axes[2].plot(t_1d, h_1d, '-', color='crimson', linewidth=2,
                     label=f'1D $h^*(t)$ (N={N1})')
    axes[2].set_xlabel('$t = \\theta-1 = \\varphi-1$', fontsize=12)
    axes[2].set_ylabel('deductible', fontsize=12)
    axes[2].set_title('Diagonal Cross-section', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

    fig.suptitle('2D Direct Formulation — Optimal Stop-Loss Deductible Surface',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/2d_full_diagnostics.png', dpi=150, bbox_inches='tight')
    print("    Saved: figures/2d_full_diagnostics.png")

    # ── 3D surface ──
    fig2, ax3d = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
    T, P = np.meshgrid(theta_v, phi_v, indexing='ij')
    surf = ax3d.plot_surface(T, P, D_opt, cmap='viridis', edgecolor='none',
                              alpha=0.9, antialiased=True)
    ax3d.set_xlabel(r'$\theta$ (risk)', fontsize=11, labelpad=8)
    ax3d.set_ylabel(r'$\varphi$ (distortion)', fontsize=11, labelpad=8)
    ax3d.set_zlabel(r'$d(\theta,\varphi)$', fontsize=11, labelpad=8)
    ax3d.set_title('Optimal Deductible Surface — Direct 2D Formulation',
                   fontsize=13, fontweight='bold')
    fig2.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
    fig2.tight_layout()
    fig2.savefig('figures/2d_surface_3d.png', dpi=150, bbox_inches='tight')
    print("    Saved: figures/2d_surface_3d.png")

    # ── Save data ──
    np.savez('figures/canonical_2d_results.npz',
             theta=theta_v, phi=phi_v, D_opt=D_opt, D_pw=D_pw,
             diag_2d=diag_2d, t_2d=t_2d,
             objective=result['objective'], N=N_2d)
    print("    Saved: figures/canonical_2d_results.npz")

    return result


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(N_2d=N)
