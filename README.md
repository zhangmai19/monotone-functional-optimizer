# monotone-functional-optimizer

**Discrete optimization of variational functionals under monotonicity constraints.**

Given an integral functional

$$
J[f] = \int_a^b P\bigl(x,\; f(x),\; f'(x)\bigr)\; dx,
$$

this solver finds

$$
f^* = \underset{f\;:\; f' \le 0}{\operatorname{arg\,max}}\;\; J[f]
\qquad\text{or}\qquad
f^* = \underset{f\;:\; f' \le 0}{\operatorname{arg\,min}}\;\; J[f]
$$

with **free boundary conditions** at both endpoints — no prescribed values for $f(a)$ or $f(b)$.

The sign of the monotonicity constraint is configurable ($f' \le 0$ non-increasing / $f' \ge 0$ non-decreasing); the objective can be maximized or minimized.

---

## Method

### Discretization — piecewise-linear finite elements

The interval $[a, b]$ is divided into $N$ equal subintervals of length $\Delta x = (b-a)/N$. The unknown function $f$ is represented by its nodal values

$$
h_i = f(x_i), \qquad x_i = a + i\cdot\Delta x,\qquad i = 0, \dots, N.
$$

Between nodes, $f$ is linearly interpolated:

$$
f_h(x) = h_i + \frac{h_{i+1} - h_i}{\Delta x}\,(x - x_i),
\qquad x \in [x_i, x_{i+1}].
$$

Consequently, the derivative is **piecewise constant**:

$$
f_h'(x) = \frac{h_{i+1} - h_i}{\Delta x}
\quad\text{on}\quad (x_i, x_{i+1}).
$$

The monotonicity constraint $f' \le 0$ collapses to $N$ **linear inequalities**:

$$
h_i \ge h_{i+1}, \qquad i = 0, \dots, N-1.
$$

### Numerical quadrature — Gauss-Legendre

On each subinterval $[x_i, x_{i+1}]$, the integral

$$
\int_{x_i}^{x_{i+1}} P\bigl(x,\, f_h(x),\, f_h'(x)\bigr)\; dx
$$

is evaluated with a $k$-point Gauss-Legendre rule (default $k = 3$), which is exact for polynomials of degree $\le 2k-1$:

$$
\int_{x_i}^{x_{i+1}} g(x)\,dx
\;\approx\;
\frac{\Delta x}{2} \sum_{j=1}^{k} w_j\;
g\!\left(\frac{x_i + x_{i+1}}{2} + \frac{\Delta x}{2}\,\xi_j\right).
$$

### Optimization

The fully discrete problem

$$
\min_{h \in \mathbb R^{N+1}}\; \pm\sum_{i=0}^{N-1}
\int_{x_i}^{x_{i+1}} P\bigl(x,\, f_h(x),\, f_h'(x)\bigr)\; dx
\quad\text{s.t.}\quad
h_0 \ge h_1 \ge \dots \ge h_N
$$

is solved with **SLSQP** (Sequential Least Squares Programming) via `scipy.optimize.minimize`. The sign $\pm$ is $-$ for maximization, $+$ for minimization, since `scipy` always minimizes.

If SLSQP fails, `solve_auto` falls back to `trust-constr`.

---

## Installation

```bash
git clone git@github.com:zhangmai19/monotone-functional-optimizer.git
cd monotone-functional-optimizer
pip install -r requirements.txt
```

Core dependencies:

| package  | version  |
|----------|----------|
| `numpy`  | ≥ 2.0    |
| `scipy`  | ≥ 1.12   |
| `matplotlib` | ≥ 3.8 |

---

## Quick start

```python
from problem import Problem
from solver import solve
from plots import plot_both

# Define your integrand P(x, f, f')
def P(x, f, fp):
    return f - 0.5 * 0.3 * fp * fp

# Set up the problem
prob = Problem(
    P=P,
    a=0.0, b=1.0,       # interval
    N=50,                # number of subintervals → 51 variables
    maximize=True,       # maximize the integral
    monotone="nonincreasing",  # f' ≤ 0
)

# Solve
res = solve(prob)

# Inspect
print(res.success)       # True
print(res.x[0])           # f(a)
print(res.x[-1])          # f(b)

# Plot
plot_both(prob, res.x)
```

---

## Examples

### 1. Monotone $L^2$ fit

Find the non-increasing function that best approximates a target $g(x)$ in the least-squares sense:

$$
\min_{f' \le 0}\;
\int_0^1 \bigl[f(x) - g(x)\bigr]^2\; dx.
$$

**Target:** $g(x)$ rises on $[0.55, 1.0]$, so the constraint $f' \le 0$ binds — the optimal $f$ "flattens" through the rising part.

```python
def target(x):
    return np.where(x < 0.55,
                    1.0 - 0.8 * x,
                    0.56 + 0.8 * np.sin(6 * (x - 0.55)))

def P(x, f, fp):
    return (f - target(x)) ** 2

prob = Problem(P=P, a=0.0, b=1.0, N=80, maximize=False)
res = solve(prob)
# success=True, nit=43, J=0.0594
```

![Exp 1 — monotone L² fit](figures/exp1_monotone_fit.png)

---

### 2. Minimal action with potential

$$
\min_{f' \le 0}\;
\int_0^1 \left[\,\frac{1}{2}\,f'(x)^2 + \frac{1}{2}\,\bigl(f(x) - g(x)\bigr)^2\right]\; dx.
$$

The unconstrained Euler–Lagrange equation is

$$
f''(x) - f(x) = -g(x),
\qquad f'(0) = f'(1) = 0 \;\; \text{(natural boundary)}.
$$

When $g(x) = e^{-1.5x}\cos(2x)$ (already non-increasing), the constraint is **inactive** — the unconstrained minimizer already satisfies $f' \le 0$.

```python
def g(x):
    return np.exp(-1.5 * x) * np.cos(2.0 * x)

def P(x, f, fp):
    return 0.5 * fp**2 + 0.5 * (f - g(x))**2

prob = Problem(P=P, a=0.0, b=1.0, N=50, maximize=False)
res = solve(prob)
# success=True, nit=65, J=0.0502
```

![Exp 2 — minimal action](figures/exp2_minimal_action.png)

---

### 3. Weighted area with quadratic regularization

$$
\max_{f' \le 0}\;
\int_0^1 \Bigl[\,w(x)\,f(x) - \tfrac{\beta}{2}\,f(x)^2\Bigr]\; dx,
\qquad
w(x) = \exp\!\bigl(-25\,(x-0.35)^2\bigr).
$$

The $-\frac{\beta}{2}f^2$ term acts as a regularizer — without it the integral would be unbounded (push $f \to +\infty$). The optimum concentrates $f$ where the weight $w(x)$ is large, while respecting the monotonicity constraint.

```python
def w(x):
    return np.exp(-25 * (x - 0.35)**2)

def P(x, f, fp):
    return w(x) * f - 0.5 * 2.0 * f**2

prob = Problem(P=P, a=0.0, b=1.0, N=50, maximize=True)
res = solve(prob)
# success=True, nit=45, max ∫P = 0.0495
```

![Exp 3 — weighted area + regularization](figures/exp3_weighted_area.png)

---

## Convergence

The piecewise-linear discretization is $\mathcal O(\Delta x)$ accurate in $L^2$ norm:

| $N$ | $\Delta x$ | $L^2$ error | rate |
|-----|------------|-------------|------|
| 8   | 0.125     | $4.0\times 10^{-4}$ | — |
| 12  | 0.083     | $1.9\times 10^{-4}$ | 0.77 |
| 16  | 0.063     | $1.2\times 10^{-4}$ | 0.85 |
| 24  | 0.042     | $6.9\times 10^{-5}$ | 0.92 |
| 32  | 0.031     | $5.3\times 10^{-5}$ | 0.83 |
| 48  | 0.021     | $2.9\times 10^{-5}$ | 0.95 |

The empirically estimated convergence rate is $\approx 1.0$, matching the theoretical $\mathcal O(\Delta x)$ for linear finite elements.

![Exp 4 — convergence](figures/exp4_convergence.png)

---

## API reference

### `Problem`

```python
@dataclass
class Problem:
    P:        Callable[[float, float, float], float]  # integrand
    a:        float                                    # left endpoint
    b:        float                                    # right endpoint
    N:        int                                      # #subintervals
    maximize: bool = True                               # max / min
    monotone: str  = "nonincreasing"                    # or "nondecreasing"
```

Properties: `dx`, `x_nodes`, `n_vars`.

### `solve`

```python
def solve(problem, quad_order=3, method="SLSQP",
          h0=None, lb=None, ub=None, options=None) -> OptimizeResult
```

Returns `scipy.optimize.OptimizeResult`, augmented with `result.problem` and `result.quad_order`.

### `solve_auto`

```python
def solve_auto(problem, methods=("SLSQP", "trust-constr"), ...) -> OptimizeResult
```

Tries each method in sequence; returns the first successful result.

### `make_objective`

```python
def make_objective(problem, quad_order=3) -> Callable
```

Returns `J(h)` — the scalar-valued objective suitable for `scipy.optimize.minimize`.

### Plotting

| function | what it draws |
|----------|---------------|
| `plot_solution(problem, h_opt)` | $f(x)$ — nodes + piecewise-linear |
| `plot_derivative(problem, h_opt)` | $f'(x)$ — piecewise-constant step |
| `plot_both(problem, h_opt)` | side-by-side: $f$ and $f'$ |
| `plot_convergence(problem, N_list)` | $L^2$ error vs $\Delta x$ |

---

## File structure

```
├── problem.py       # Problem definition and parameters
├── quadrature.py    # Gauss-Legendre quadrature + per-interval integration
├── objective.py     # Scalar objective J(h) for scipy
├── solver.py        # Constraint assembly + SLSQP solver
├── plots.py         # Visualization utilities
├── experiments.py   # Example problems and convergence study
├── requirements.txt
└── README.md
```

---

## Mathematical background

This solver addresses problems of the form

$$
\max_{f}\; \int_a^b P(x, f, f')\,dx,
\quad f' \le 0,
$$

which arise in:

- **Monotone regression** — $P = -(f - g)^2$, find the best non-increasing fit.
- **Calculus of variations** — $P = L(x, f, f')$, a Lagrangian with a one-sided constraint on the derivative.
- **Optimal control** — $f$ as a state variable whose rate of change is bounded.
- **Shape-constrained estimation** — density estimation under monotonicity.

The inequality constraint $f' \le 0$ is a **unilateral** constraint in the calculus of variations; the Euler–Lagrange equation acquires a complementary slackness term (a non-negative Lagrange multiplier $\lambda(x)$ times the active constraint), yielding an **obstacle-type** problem. The piecewise-linear discretization reduces this to a finite-dimensional quadratic/constrained nonlinear program.

### What this solver does *not* handle (yet)

- General integral constraints $\int g(x, f, f')\,dx \le C$
- Two-sided bounds on $f'$
- Higher-order derivatives ($f''$)
- Partial differential equations
- Automatic differentiation of $P$ (finite differences are used internally by SLSQP for the Jacobian)

---

## License

MIT
