r"""示例问题与实验.

每个实验定义一个适定的 (well-posed) Problem, 求解并绘图.
运行方式:
    python experiments.py          # 运行所有实验, 显示图形
    python experiments.py --save   # 保存图片到当前目录
"""

import sys

import numpy as np

from problem import Problem
from solver import solve
from plots import plot_both, plot_convergence

# ======================================================================
# 实验 1 — 单调 L² 逼近 (最小二乘)
# ======================================================================
#   min  ∫_0^1 [ f(x) − target(x) ]² dx
#   s.t. f'(x) ≤ 0
#
# 目标函数自然有下界 (≥ 0), 始终适定.
# target 有上升段时, 约束绑定 → 最优解是 target 的"单调包络".
# ======================================================================


def experiment_1_monotone_fit(N=60):
    """单调非增函数去逼近带上升段的目标."""
    print(f"[实验 1] 单调 L² 逼近  (N={N})")

    def target(x):
        # 前半段下降, 后半段上升 — 约束会在上升段绑定
        return np.where(x < 0.55,
                        1.0 - 0.8 * x,
                        0.56 + 0.8 * np.sin(6 * (x - 0.55)))

    def P(x, f, fp):
        return (f - target(x)) ** 2

    prob = Problem(P=P, a=0.0, b=1.0, N=N, maximize=False,
                   monotone="nonincreasing")
    res = solve(prob, quad_order=5)

    print(f"  成功: {res.success},  迭代: {res.nit}")
    print(f"  MSE 积分: {res.fun:.6f}")
    print(f"  f(0)={res.x[0]:.4f},  f(1)={res.x[-1]:.4f}")

    # 验证单调性
    h = res.x
    ok = all(h[i] >= h[i + 1] for i in range(len(h) - 1))
    print(f"  单调性满足: {ok}")

    fig = plot_both(prob, h, title="Exp 1 — monotone L² fit")
    # 叠加 target
    x_dense = np.linspace(0, 1, 400)
    fig.axes[0].plot(x_dense, target(x_dense), "--", color="gray",
                     linewidth=1.5, alpha=0.8, label="target")
    fig.axes[0].legend()
    return prob, res, fig


# ======================================================================
# 实验 2 — 最小作用量 (有势能)
# ======================================================================
#   min  ∫_0^1 [ ½·(f')² + V(f) ] dx
#   s.t. f'(x) ≤ 0
#
# V(f) = ½·(f − g(x))² — 跟踪一个目标函数 g(x).
# 无约束欧拉-拉格朗日:  f'' + (g − f) = 0.
# 自然边界: f'(0) = f'(1) = 0.
# 若 g 单调非增, 无约束解自动满足 f' ≤ 0.
# ======================================================================


def experiment_2_minimal_action(N=50):
    """最小作用量: ½ f'² + ½ (f − g)², g 单调下降."""
    print(f"[实验 2] 最小作用量  (N={N})")

    def g(x):
        # 单调非增的 target → 无约束解自动满足约束
        return np.exp(-1.5 * x) * np.cos(2.0 * x)

    def V(f, x):
        return 0.5 * (f - g(x)) ** 2

    def P(x, f, fp):
        return 0.5 * fp * fp + V(f, x)

    prob = Problem(P=P, a=0.0, b=1.0, N=N, maximize=False,
                   monotone="nonincreasing")
    res = solve(prob, quad_order=5)

    print(f"  成功: {res.success},  迭代: {res.nit}")
    print(f"  目标值: {res.fun:.6f}")
    h = res.x
    print(f"  f(0)={h[0]:.4f},  f(1)={h[-1]:.4f}")
    ok = all(h[i] >= h[i + 1] for i in range(len(h) - 1))
    print(f"  单调性满足: {ok}")

    fig = plot_both(prob, h, title="Exp 2 — min ∫(½ f'² + ½ (f−g)²)")
    x_dense = np.linspace(0, 1, 400)
    fig.axes[0].plot(x_dense, g(x_dense), "--", color="gray",
                     linewidth=1.5, alpha=0.8, label="g(x)")
    fig.axes[0].legend()
    return prob, res, fig


# ======================================================================
# 实验 3 — 加权面积 + 二次正则化
# ======================================================================
#   max  ∫_0^1 [ w(x)·f(x) − ½·β·f(x)² ] dx
#   s.t. f'(x) ≤ 0
#
# β·f²/2 项防止 f 无界发散. 最优解由 Euler-Lagrange 决定:
#   w − β·f + f'' = 0  (忽略约束时).
# 约束可能在部分区间绑定.
# ======================================================================


def experiment_3_weighted_regularized(N=60, beta=2.0):
    """加权面积 + 二次正则化."""
    print(f"[实验 3] 加权面积 + 正则化  (N={N}, β={beta})")

    def w(x):
        # 权重集中在中间偏左
        return np.exp(-25.0 * (x - 0.35) ** 2)

    def P(x, f, fp):
        return w(x) * f - 0.5 * beta * f * f

    prob = Problem(P=P, a=0.0, b=1.0, N=N, maximize=True,
                   monotone="nonincreasing")
    res = solve(prob, quad_order=5)

    print(f"  成功: {res.success},  迭代: {res.nit}")
    print(f"  目标值 (max ∫): {-res.fun:.6f}")
    h = res.x
    print(f"  f(0)={h[0]:.4f},  f(1)={h[-1]:.4f}")
    ok = all(h[i] >= h[i + 1] for i in range(len(h) - 1))
    print(f"  单调性满足: {ok}")

    fig = plot_both(prob, h, title="Exp 3 — max ∫(w·f − β·f²/2)")
    # 叠加权重
    x_dense = np.linspace(0, 1, 400)
    w_vals = w(x_dense)
    w_scaled = w_vals / w_vals.max() * h.max()
    fig.axes[0].plot(x_dense, w_scaled, "--", color="purple",
                     linewidth=1.2, alpha=0.7, label="w(x) [scaled]")
    fig.axes[0].legend()
    return prob, res, fig


# ======================================================================
# 实验 4 — 收敛性研究
# ======================================================================

def experiment_4_convergence():
    """收敛性: 用实验 2 的最小作用量问题, 研究 L² 误差 vs N."""
    print("[实验 4] 收敛性研究")

    def g(x):
        return np.exp(-1.5 * x) * np.cos(2.0 * x)

    def P(x, f, fp):
        return 0.5 * fp * fp + 0.5 * (f - g(x)) ** 2

    template = Problem(P=P, a=0.0, b=1.0, N=10, maximize=False,
                       monotone="nonincreasing")

    N_list = [8, 12, 16, 24, 32, 48, 64]
    try:
        fig, errors = plot_convergence(template, N_list,
                                       quad_order=5, ref_N=80)
        print(f"  Δx = {(template.b - template.a) / np.array(N_list)}")
        print(f"  L² 误差: {[f'{e:.2e}' for e in errors]}")
        return fig, errors
    except Exception as e:
        print(f"  ⚠ 收敛性研究遇到问题: {e}")
        return None, None


# ======================================================================
# 主入口
# ======================================================================

def main():
    save = "--save" in sys.argv or "-s" in sys.argv
    show = "--no-show" not in sys.argv

    experiments = [
        ("1_monotone_fit", experiment_1_monotone_fit),
        ("2_minimal_action", experiment_2_minimal_action),
        ("3_weighted_regularized", experiment_3_weighted_regularized),
        ("4_convergence", experiment_4_convergence),
    ]

    figs = {}
    for name, fn in experiments:
        print(f"\n{'=' * 60}")
        try:
            result = fn()
            if result is not None:
                # 取最后一个 matplotlib Figure
                if isinstance(result, tuple):
                    for item in reversed(result):
                        if hasattr(item, "savefig"):
                            figs[name] = item
                            break
        except Exception as e:
            import traceback
            print(f"  ❌ 失败: {e}")
            traceback.print_exc()

    if save:
        for name, fig in figs.items():
            fname = f"experiment_{name}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  已保存: {fname}")

    if show and figs:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
