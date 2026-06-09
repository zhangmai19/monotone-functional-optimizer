"""问题定义: 封装变分问题的全部参数.

优化目标:  max (或 min) ∫_a^b P(x, f, f') dx
约束条件:  f'(x) ≤ 0 (非增) 或 f'(x) ≥ 0 (非减)
边界条件:  f(a), f(b) 自由 (自然边界)
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Problem:
    """变分问题的离散化描述.

    用 N+1 个等距节点的函数值 h_i = f(x_i) 表示 f,
    节点之间线性插值, f' 在每个子区间内为常数.

    Parameters
    ----------
    P : callable
        被积函数 P(x, f, fp), 三个参数均为标量 float, 返回标量.
    a, b : float
        区间端点, a < b.
    N : int
        子区间个数; 优化变量为 N+1 个节点值 h_0, ..., h_N.
    maximize : bool
        True → 最大化积分; False → 最小化.
    monotone : {'nonincreasing', 'nondecreasing'}
        'nonincreasing' → f' ≤ 0 (默认);
        'nondecreasing' → f' ≥ 0.
    """

    P: Callable[[float, float, float], float]
    a: float
    b: float
    N: int
    maximize: bool = True
    monotone: str = "nonincreasing"

    def __post_init__(self):
        if self.monotone not in ("nonincreasing", "nondecreasing"):
            raise ValueError(
                f"monotone 必须是 'nonincreasing' 或 'nondecreasing', "
                f"得到 '{self.monotone}'"
            )
        if self.a >= self.b:
            raise ValueError(f"需要 a < b, 得到 a={self.a}, b={self.b}")
        if self.N < 2:
            raise ValueError(f"N 至少为 2, 得到 N={self.N}")

        self.dx = (self.b - self.a) / self.N
        self.x_nodes = np.linspace(self.a, self.b, self.N + 1)

    @property
    def n_vars(self) -> int:
        """优化变量个数."""
        return self.N + 1
