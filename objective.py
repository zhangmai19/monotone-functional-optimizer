"""目标函数: 将连续变分积分离散化为 scipy 可优化的标量函数.

J(h) = ± Σ_i ∫_{x_i}^{x_{i+1}} P(x, f_h(x), f_h'(x)) dx

其中 f_h 是节点值 h 的分段线性插值; 符号由 maximize/minimize 决定
(scipy.optimize.minimize 只做最小化, 最大化时翻转符号).
"""

from quadrature import integrate_full


def make_objective(problem, quad_order=3):
    """构造目标函数 J(h) → float.

    Parameters
    ----------
    problem : Problem
    quad_order : int
        每个子区间的 Gauss-Legendre 点数 (默认 3).

    Returns
    -------
    callable
        J(h) 接受长度 N+1 的 1-D array, 返回标量.
        scipy 总是最小化, 所以 problem.maximize=True 时返回负积分.
    """
    P = problem.P
    x_nodes = problem.x_nodes
    sign = -1.0 if problem.maximize else 1.0

    def J(h):
        return sign * integrate_full(P, x_nodes, h, quad_order)

    return J
