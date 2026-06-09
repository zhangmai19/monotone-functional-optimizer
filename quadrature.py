"""数值求积: Gauss-Legendre 求积规则及逐区间积分."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_nodes_weights(k: int = 3):
    """k 点 Gauss-Legendre 求积节点与权重 (在 [-1, 1] 上).

    对次数 ≤ 2k-1 的多项式精确.

    Parameters
    ----------
    k : int
        求积点数. 常用: k=3 对 5 次多项式精确; k=5 对 9 次精确.

    Returns
    -------
    nodes : ndarray of shape (k,)
    weights : ndarray of shape (k,)
    """
    return leggauss(k)


def integrate_interval(P, x_left, x_right, f_left, f_right, quad_order=3):
    """在单个子区间上计算 ∫ P(x, f(x), fp) dx.

    假设 f 在 [x_left, x_right] 上为线性函数:
        f(x)  = f_left + fp * (x - x_left)
        fp     = (f_right - f_left) / (x_right - x_left)   (常数)

    Parameters
    ----------
    P : callable(x, f, fp) -> float
    x_left, x_right : float
    f_left, f_right : float
    quad_order : int
        Gauss-Legendre 点数 (默认 3).

    Returns
    -------
    float
    """
    dx = x_right - x_left
    if dx <= 0:
        return 0.0

    # 区间内 f' 为常数 (线性插值的导数)
    fp = (f_right - f_left) / dx

    # Gauss 节点与权重 (在 [-1, 1])
    xi, w = leggauss(quad_order)

    # 映射到 [x_left, x_right]
    half = 0.5 * dx
    mid = 0.5 * (x_left + x_right)
    nodes = mid + half * xi
    weights = half * w

    # 逐点累加
    total = 0.0
    for i in range(quad_order):
        f_val = f_left + fp * (nodes[i] - x_left)
        total += weights[i] * P(nodes[i], f_val, fp)

    return total


def integrate_full(P, x_nodes, h, quad_order=3):
    """在所有子区间上计算全积分 ∫_a^b P(x, f, f') dx.

    Parameters
    ----------
    P : callable
    x_nodes : ndarray of shape (N+1,)
        节点坐标 (等距).
    h : ndarray of shape (N+1,)
        节点函数值.
    quad_order : int

    Returns
    -------
    float
    """
    total = 0.0
    for i in range(len(x_nodes) - 1):
        total += integrate_interval(
            P, x_nodes[i], x_nodes[i + 1],
            h[i], h[i + 1], quad_order,
        )
    return total
