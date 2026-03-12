import numpy as np
from scipy.optimize import minimize
from objective import objective


def solve(N=10, L=1.0):
    h0 = np.linspace(L, 0.0, N + 1)

    bounds = [(0.0, L) for _ in range(N + 1)]

    cons = []
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda h, i=i: h[i] - h[i + 1]
        })

    result = minimize(objective, h0, bounds=bounds, constraints=cons)
    return result


if __name__ == "__main__":
    res = solve()
    print(res)