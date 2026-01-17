from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .kernels import Kernel, LinearKernel

Array = np.ndarray


def _ensure_2d(X: Array) -> Array:
    X = np.asarray(X)
    if X.ndim == 1:
        return X[None, :]
    return X


def _ensure_pm_one(y: Array) -> Array:
    """Ensure labels are in {-1, 1}.

    Accepts {-1, 1} or {0, 1}. Raises ValueError otherwise.
    """
    y = np.asarray(y).reshape(-1)
    uniq = set(np.unique(y).tolist())
    if uniq.issubset({-1, 1}):
        return y.astype(int, copy=False)
    if uniq.issubset({0, 1}):
        return (y * 2 - 1).astype(int)
    raise ValueError(f"Expected binary labels in {{-1,1}} or {{0,1}}, got {sorted(uniq)}")


@dataclass(slots=True)
class SVM:
    """Soft-margin SVM via the dual problem (SLSQP).

    Solves:
        max_a  sum_i a_i - 1/2 sum_{ij} a_i a_j y_i y_j K(x_i, x_j)
        s.t.   0 <= a_i <= C,   sum_i a_i y_i = 0

    Notes:
      * Works for arbitrary kernels K.
      * For linear kernel additionally exposes `w_`.
      * Expects labels in {-1, 1} (or {0, 1} which will be converted).
    """

    C: float = 1.0
    kernel: Kernel = field(default_factory=LinearKernel)
    maxiter: int = 1000
    ftol: float = 1e-3
    support_threshold: float = 1e-6

    # learned params
    alphas_: Optional[Array] = field(init=False, default=None)
    support_mask_: Optional[Array] = field(init=False, default=None)
    support_indices_: Optional[Array] = field(init=False, default=None)
    support_vectors_: Optional[Array] = field(init=False, default=None)
    support_labels_: Optional[Array] = field(init=False, default=None)
    dual_coef_: Optional[Array] = field(init=False, default=None)  # alpha_i * y_i on SVs
    intercept_: float = field(init=False, default=0.0)
    w_: Optional[Array] = field(init=False, default=None)

    # training cache
    _X_train: Optional[Array] = field(init=False, default=None, repr=False)
    _y_train: Optional[Array] = field(init=False, default=None, repr=False)

    def fit(self, X: Array, y: Array) -> "SVM":
        X = _ensure_2d(np.asarray(X, dtype=float))
        y = _ensure_pm_one(y)

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)=}, {len(y)=}")

        n = len(X)
        K = np.asarray(self.kernel(X, X), dtype=float)
        y_col = y[:, None].astype(float)
        G = (y_col @ y_col.T) * K  # G_ij = y_i y_j K_ij

        def objective(alpha: Array) -> float:
            # minimize: 0.5 a^T G a - 1^T a
            return 0.5 * float(alpha @ (G @ alpha)) - float(alpha.sum())

        def gradient(alpha: Array) -> Array:
            # grad = G a - 1
            return (G @ alpha) - 1.0

        bounds = [(0.0, float(self.C))] * n
        constraints = ({"type": "eq", "fun": lambda a: float(a @ y), "jac": lambda a: y.astype(float)},)

        res = minimize(
            fun=objective,
            x0=np.zeros(n, dtype=float),
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": int(self.maxiter), "ftol": float(self.ftol)},
        )

        if not res.success:
            raise ArithmeticError(f"SVM optimization failed: {res.message}")

        alpha = np.asarray(res.x, dtype=float)

        support_mask = alpha > self.support_threshold
        support_idx = np.flatnonzero(support_mask)

        if support_idx.size == 0:
            raise ArithmeticError(
                "No support vectors found. Try increasing C or decreasing support_threshold."
            )

        X_sv = X[support_idx]
        y_sv = y[support_idx]
        coef_sv = alpha[support_idx] * y_sv

        # Prefer margin SVs (0 < alpha < C) for a stabler bias estimate.
        margin_mask = support_mask & (alpha < (self.C - self.support_threshold))
        bias_idx = np.flatnonzero(margin_mask)
        if bias_idx.size == 0:
            bias_idx = support_idx

        K_bias = np.asarray(self.kernel(X[bias_idx], X_sv), dtype=float)
        decision_no_bias = K_bias @ coef_sv
        intercept = float(np.mean(y[bias_idx] - decision_no_bias))

        # Store learned state
        self._X_train = X
        self._y_train = y
        self.alphas_ = alpha
        self.support_mask_ = support_mask
        self.support_indices_ = support_idx
        self.support_vectors_ = X_sv
        self.support_labels_ = y_sv
        self.dual_coef_ = coef_sv
        self.intercept_ = intercept

        # Only meaningful for linear kernel
        if isinstance(self.kernel, LinearKernel):
            self.w_ = (alpha * y).T @ X
        else:
            self.w_ = None

        return self

    def decision_function(self, X: Array) -> Array:
        self._check_is_fitted()
        X2 = _ensure_2d(np.asarray(X, dtype=float))
        K = np.asarray(self.kernel(X2, self.support_vectors_), dtype=float)
        scores = K @ self.dual_coef_ + self.intercept_
        return scores

    def predict(self, X: Array) -> Array:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1).astype(int)

    def _check_is_fitted(self) -> None:
        if self.support_vectors_ is None or self.dual_coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
