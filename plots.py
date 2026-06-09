"""可视化: 绘制最优解 f(x) 与导数 f'(x)."""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 颜色 / 样式常量
# ---------------------------------------------------------------------------
COLOR_SOLUTION = "steelblue"
COLOR_NODES = "crimson"
COLOR_DERIV = "darkorange"
MARKER_SIZE = 6


# ---------------------------------------------------------------------------
# 单图函数
# ---------------------------------------------------------------------------

def plot_solution(problem, h_opt, ax=None, label=None,
                  show_nodes=True, fill=False, **kwargs):
    """绘制最优函数 f(x).

    节点间线性连接 (分段线性插值), 节点处可选标记.

    Parameters
    ----------
    problem : Problem
    h_opt : array-like
        最优节点值, 长度 N+1.
    ax : Axes or None
    label : str or None
    show_nodes : bool
        是否标出离散节点.
    fill : bool
        是否填充曲线下方区域.
    kwargs : 传给 ax.plot.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = problem.x_nodes
    h = np.asarray(h_opt)

    kw = {"color": COLOR_SOLUTION, "linewidth": 2, "zorder": 2}
    kw.update(kwargs)

    ax.plot(x, h, "-", label=label, **kw)

    if show_nodes:
        ax.scatter(x, h, color=COLOR_NODES, s=20, zorder=3,
                   clip_on=False)

    if fill:
        ax.fill_between(x, h, alpha=0.12, color=COLOR_SOLUTION)

    ax.set_xlabel("x")
    ax.set_ylabel("f (x)")
    if label:
        ax.legend()

    return ax


def plot_derivative(problem, h_opt, ax=None, label=None, **kwargs):
    """绘制 f'(x) — 分段常数.

    用 step 图表示每个子区间上的常数值.
    在每个子区间内 f' = (h_{i+1} - h_i) / dx.

    Parameters
    ----------
    problem : Problem
    h_opt : array-like
    ax : Axes or None
    label : str or None

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    h = np.asarray(h_opt)
    x = problem.x_nodes
    dx = problem.dx
    fp = np.diff(h) / dx

    kw = {"color": COLOR_DERIV, "linewidth": 2, "where": "post"}
    kw.update(kwargs)

    ax.step(x, np.append(fp, fp[-1]), label=label, **kw)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f '(x)")
    if label:
        ax.legend()

    return ax


# ---------------------------------------------------------------------------
# 组合图
# ---------------------------------------------------------------------------

def plot_both(problem, h_opt, title=None, fname=None):
    """并排显示 f(x) 和 f'(x).

    Parameters
    ----------
    problem : Problem
    h_opt : array-like
    title : str or None
    fname : str or None
        若提供则保存到文件.

    Returns
    -------
    fig
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_solution(problem, h_opt, ax=ax1)
    plot_derivative(problem, h_opt, ax=ax2)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    fig.tight_layout()

    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 收敛性分析
# ---------------------------------------------------------------------------

def plot_convergence(problem, N_list, quad_order=3, ref_N=400,
                     show_rate=True):
    """研究 N → ∞ 时的收敛性.

    以 N=ref_N 的解作为"真解"参考, 计算各 N 的 L² 误差.

    Parameters
    ----------
    problem : Problem
        模板问题 (其 N 参数会被覆盖).
    N_list : list of int
        要测试的 N 值.
    quad_order : int
    ref_N : int
        参考解的离散点数 (建议 ≤ 80, SLSQP 对大规模约束较慢).
    show_rate : bool
        是否标注收敛阶.

    Returns
    -------
    fig, errors : (Figure, list of float)
    """
    from solver import solve

    # ---- 参考解 ----
    ref_prob = type(problem)(
        problem.P, problem.a, problem.b, ref_N,
        maximize=problem.maximize, monotone=problem.monotone,
    )
    ref_res = solve(ref_prob, quad_order=quad_order)
    if not ref_res.success:
        raise RuntimeError(f"参考解 (N={ref_N}) 未收敛: {ref_res.message}")
    ref_h = ref_res.x

    # ---- 逐 N 求解 ----
    errors = []
    for N in N_list:
        prob = type(problem)(
            problem.P, problem.a, problem.b, N,
            maximize=problem.maximize, monotone=problem.monotone,
        )
        res = solve(prob, quad_order=quad_order)
        h_opt = res.x

        # 插值到参考网格
        h_interp = np.interp(ref_prob.x_nodes, prob.x_nodes, h_opt)
        err = np.sqrt(np.trapezoid((h_interp - ref_h) ** 2,
                                   ref_prob.x_nodes))
        errors.append(err)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    dx_list = [(problem.b - problem.a) / N for N in N_list]
    ax.loglog(dx_list, errors, "o-", color=COLOR_SOLUTION, linewidth=1.5,
              markersize=5)

    if show_rate and len(errors) >= 2:
        # 拟合收敛阶
        log_dx = np.log(dx_list)
        log_err = np.log(errors)
        slope, _ = np.polyfit(log_dx, log_err, 1)
        ax.plot(dx_list, np.exp(log_err[0]) * (np.array(dx_list) / dx_list[0]) ** slope,
                "--", color="gray", alpha=0.6,
                label=f"slope ≈ {slope:.2f}")
        ax.legend()

    ax.set_xlabel("Δx = (b−a)/N")
    ax.set_ylabel("L² error vs reference")
    ax.set_title(f"Convergence (ref N={ref_N})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, errors
