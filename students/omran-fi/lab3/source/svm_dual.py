from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


KernelType = Literal["linear", "rbf", "poly"]


@dataclass
class DualSVMConfig:
    kernel: KernelType = "linear"
    C: float = 1.0
    gamma: float = 1.0          # for rbf
    degree: int = 3             # for poly
    coef0: float = 1.0          # for poly
    eps_sv: float = 1e-7        # support vector threshold
    maxiter: int = 2000
    ftol: float = 1e-9


class DualSVM:
    """
    Soft-margin SVM solved in the dual form using constrained optimization.

    Dual problem:
        minimize   1/2 * (lambda*y)^T K (lambda*y) - sum(lambda_i)
        s.t.       sum(lambda_i * y_i) = 0
                  0 <= lambda_i <= C

    Decision function:
        f(x) = sum_i lambda_i y_i K(x, x_i) + b
        y_hat = sign(f(x))
    """

    def __init__(self, config: DualSVMConfig):
        self.cfg = config

        # learned parameters
        self.lmbda: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        self.w: Optional[np.ndarray] = None  # only for linear kernel

        # training cache
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    # ----------------- kernels -----------------
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return X1 @ X2.T

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # pairwise squared distances in a vectorized way
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dists = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
        return np.exp(-self.cfg.gamma * dists)

    def _poly_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return (X1 @ X2.T + self.cfg.coef0) ** self.cfg.degree

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.cfg.kernel == "linear":
            return self._linear_kernel(X1, X2)
        if self.cfg.kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        if self.cfg.kernel == "poly":
            return self._poly_kernel(X1, X2)
        raise ValueError(f"Unknown kernel: {self.cfg.kernel}")

    # ----------------- training -----------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DualSVM":
        """
        X: (n_samples, n_features)
        y: labels in {-1, +1}
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if not np.all(np.isin(y, [-1.0, 1.0])):
            raise ValueError("y must be in {-1, +1}")

        self.X_train = X
        self.y_train = y

        n = X.shape[0]
        K = self._kernel(X, X)

        # Q = (y y^T) ⊙ K
        Q = (y[:, None] * y[None, :]) * K

        def objective(lmbda: np.ndarray) -> float:
            # 1/2 lambda^T Q lambda - 1^T lambda
            return 0.5 * (lmbda @ (Q @ lmbda)) - np.sum(lmbda)

        def gradient(lmbda: np.ndarray) -> np.ndarray:
            return (Q @ lmbda) - np.ones(n)

        # Constraint: y^T lambda = 0
        lc = LinearConstraint(y.reshape(1, -1), 0.0, 0.0)
        bounds = Bounds(np.zeros(n), np.full(n, self.cfg.C))

        res = minimize(
            fun=objective,
            x0=np.zeros(n),
            jac=gradient,
            bounds=bounds,
            constraints=[lc],
            method="SLSQP",
            options={"maxiter": self.cfg.maxiter, "ftol": self.cfg.ftol, "disp": False},
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.lmbda = res.x

        # choose support vectors for b:
        eps = self.cfg.eps_sv
        margin_mask = (self.lmbda > eps) & (self.lmbda < self.cfg.C - eps)
        if np.any(margin_mask):
            idxs = np.where(margin_mask)[0]
        else:
            # fallback: any sv
            sv_mask = self.lmbda > eps
            idxs = np.where(sv_mask)[0]

        if idxs.size == 0:
            self.b = 0.0
        else:
            # b = mean_i [ y_i - sum_j lambda_j y_j K(x_j, x_i) ]
            # using precomputed K: sum_j (lambda_j y_j K_{j,i})
            lmbda_y = self.lmbda * y
            b_vals = y[idxs] - (K[:, idxs].T @ lmbda_y)
            self.b = float(np.mean(b_vals))

        # linear hyperplane parameters
        if self.cfg.kernel == "linear":
            self.w = np.sum((self.lmbda * y)[:, None] * X, axis=0)
        else:
            self.w = None

        return self

    # ----------------- inference -----------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.lmbda is None or self.b is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)

        if self.cfg.kernel == "linear" and self.w is not None:
            return X @ self.w + self.b

        K_test = self._kernel(X, self.X_train)   # (n_test, n_train)
        return K_test @ (self.lmbda * self.y_train) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1)

    def get_hyperplane(self) -> Tuple[np.ndarray, float]:
        """
        Return (w, b) only for linear kernel.
        """
        if self.cfg.kernel != "linear" or self.w is None or self.b is None:
            raise ValueError("Hyperplane parameters are available only for linear kernel.")
        return self.w, self.b
