"""求解器: 组装约束与目标, 调用 scipy.optimize.minimize.

离散化方案 (分段线性有限元):
  - N+1 个等距节点 x_i = a + i·dx, 优化变量 h_i = f(x_i)
  - 子区间内 f 线性:  f'(x) = (h_{i+1} - h_i) / dx  (分段常数)
  - 单调约束退化为线性不等式: h_i ≥ h_{i+1} (非增) 或 h_i ≤ h_{i+1} (非减)
  - 积分用 Gauss-Legendre 求积逐区间计算
"""

import numpy as np
from scipy.optimize import minimize

from objective import make_objective


def solve(problem, quad_order=3, method="SLSQP", h0=None,
          lb=None, ub=None, options=None):
    """求解离散变分问题.

    Parameters
    ----------
    problem : Problem
    quad_order : int
        每个子区间的 Gauss-Legendre 求积点数 (默认 3).
    method : str
        scipy 优化方法. 'SLSQP' (默认) 和 'trust-constr' 均可处理约束.
    h0 : array-like or None
        初始猜测, 长度 N+1. None 则用线性下降 1 → 0.
    lb, ub : float or None
        每个变量的下/上界. None 表示无界 (实际设为 ±1e12).
    options : dict or None
        传递给 scipy.optimize.minimize 的 options.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        除标准字段外, 附加:
        - result.problem : Problem (方便后续绘图)
        - result.quad_order : int
    """
    # ---- 初始猜测 ----
    if h0 is None:
        # 默认: 从 1 线性下降到 0, 满足非增约束
        h0 = np.linspace(1.0, 0.0, problem.N + 1)
    else:
        h0 = np.asarray(h0, dtype=float)
        if len(h0) != problem.N + 1:
            raise ValueError(
                f"h0 长度应为 {problem.N + 1}, 得到 {len(h0)}"
            )

    # ---- 目标函数 ----
    J = make_objective(problem, quad_order)

    # ---- 单调性约束 (线性不等式) ----
    cons = []
    if problem.monotone == "nonincreasing":
        # h_i - h_{i+1} ≥ 0  ⟺  f 非增
        for i in range(problem.N):
            cons.append({
                "type": "ineq",
                "fun": lambda h, i=i: h[i] - h[i + 1],
            })
    else:  # nondecreasing
        # h_{i+1} - h_i ≥ 0  ⟺  f 非减
        for i in range(problem.N):
            cons.append({
                "type": "ineq",
                "fun": lambda h, i=i: h[i + 1] - h[i],
            })

    # ---- 变量边界 ----
    _lb = -1e12 if lb is None else lb
    _ub = 1e12 if ub is None else ub
    bounds = [(_lb, _ub) for _ in range(problem.N + 1)]

    # ---- 求解器选项 ----
    default_opts = {"maxiter": 2000, "ftol": 1e-12}
    if options:
        default_opts.update(options)

    # ---- 求解 ----
    result = minimize(
        J, h0,
        method=method,
        bounds=bounds,
        constraints=cons,
        options=default_opts,
    )

    # 附加元信息
    result.problem = problem
    result.quad_order = quad_order

    return result


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def solve_auto(problem, quad_order=3, methods=("SLSQP", "trust-constr"),
               h0=None, lb=None, ub=None, options=None):
    """依次尝试多种方法, 返回第一个成功的结果.

    某些问题用 SLSQP 可能不收敛, trust-constr 作为备选.
    """
    for method in methods:
        try:
            result = solve(
                problem, quad_order=quad_order, method=method,
                h0=h0, lb=lb, ub=ub, options=options,
            )
            if result.success:
                return result
        except Exception:
            continue
    # 全部失败, 返回最后一次结果
    return solve(
        problem, quad_order=quad_order, method=methods[-1],
        h0=h0, lb=lb, ub=ub, options=options,
    )
