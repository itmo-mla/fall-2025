from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y.T


def polynomial_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> np.ndarray:
    return (gamma * (X @ Y.T) + coef0) ** degree


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    D2 = np.maximum(X2 + Y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * D2)


@dataclass
class SVMResult:
    alphas: np.ndarray
    b: float
    support_idx: np.ndarray
    w: np.ndarray | None  # only for linear kernel


def solve_svm_dual(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    kernel: str = "linear",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    tol_sv: float = 1e-6,
) -> SVMResult:
    """
    Soft-margin SVM via dual:

      maximize  sum_i a_i - 1/2 sum_{i,j} a_i a_j y_i y_j K(x_i, x_j)
      s.t.      0 <= a_i <= C
                sum_i a_i y_i = 0

    Solved by scipy.optimize.minimize on the negative objective.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]

    if kernel == "linear":
        K = linear_kernel(X, X)
    elif kernel == "poly":
        K = polynomial_kernel(X, X, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == "rbf":
        K = rbf_kernel(X, X, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel='{kernel}' (use 'linear', 'poly', 'rbf').")

    Q = (y[:, None] * y[None, :]) * K  # (n,n)

    def objective(a: np.ndarray) -> float:
        # negative of dual (minimize)
        return float(0.5 * a @ (Q @ a) - np.sum(a))

    def grad(a: np.ndarray) -> np.ndarray:
        return (Q @ a) - np.ones_like(a)

    # constraints: y^T a = 0
    def cons_fun(a: np.ndarray) -> float:
        return float(y @ a)

    def cons_jac(a: np.ndarray) -> np.ndarray:
        return y.copy()

    try:
        from scipy.optimize import minimize
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy is required for task-03 (scipy.optimize.minimize).") from e

    bounds = [(0.0, float(C)) for _ in range(n)]
    constraints = [{"type": "eq", "fun": cons_fun, "jac": cons_jac}]

    # initialization: feasible a=0
    a0 = np.zeros(n, dtype=float)

    res = minimize(
        objective,
        a0,
        jac=grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError(f"SVM dual optimization failed: {res.message}")

    a = np.asarray(res.x, dtype=float)
    support = np.where(a > tol_sv)[0]

    # compute bias using margin SVs (0 < a_i < C)
    margin_sv = support[(a[support] < (C - tol_sv))]
    if len(margin_sv) == 0:
        # fallback: use all support vectors
        margin_sv = support

    # b = y_i - sum_j a_j y_j K(x_j, x_i)
    decision_on_sv = (a * y) @ K[:, margin_sv]
    b_vals = y[margin_sv] - decision_on_sv
    b = float(np.mean(b_vals)) if len(b_vals) else 0.0

    w = None
    if kernel == "linear":
        # w = sum a_i y_i x_i
        w = (a * y) @ X
        w = np.asarray(w, dtype=float)

    return SVMResult(alphas=a, b=b, support_idx=support.astype(int), w=w)


def decision_function(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sv: SVMResult,
    X: np.ndarray,
    kernel: str = "linear",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X = np.asarray(X, dtype=float)

    if kernel == "linear" and sv.w is not None:
        return (X @ sv.w) + sv.b

    if kernel == "linear":
        K = linear_kernel(X_train, X)
    elif kernel == "poly":
        K = polynomial_kernel(X_train, X, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == "rbf":
        K = rbf_kernel(X_train, X, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel='{kernel}'.")

    return (sv.alphas * y_train) @ K + sv.b


def predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sv: SVMResult,
    X: np.ndarray,
    kernel: str = "linear",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    f = decision_function(X_train, y_train, sv, X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    return np.where(f >= 0.0, 1.0, -1.0)


