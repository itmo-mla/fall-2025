from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def add_bias_last(X: np.ndarray) -> np.ndarray:
    """
    Lecture style: F = [X, 1] (bias column appended as last column).
    """
    n = X.shape[0]
    return np.hstack([X, np.ones((n, 1), dtype=X.dtype)])


@dataclass
class FitResult:
    w: np.ndarray
    n_iter: int
    converged: bool
    history: dict[str, list[float]]  # Q, step_norm


class LogisticRegressionLecture:
    """
    Logistic regression exactly in lecture notation with y in {-1, +1}:

      Q(w) = sum_i log(1 + exp(- y_i * <w, x_i>))
      sigma_i = sigmoid( y_i * <w, x_i> )

    Newton:
      grad_j = - sum_i (1 - sigma_i) * y_i * f_j(x_i)
      Hess_jk = sum_i (1 - sigma_i) * sigma_i * f_j(x_i) * f_k(x_i)

    IRLS (as in slides):
      gamma_i = sqrt((1 - sigma_i) * sigma_i)
      F_tilde = diag(gamma) * F
      y_tilde_i = y_i * sqrt((1 - sigma_i) / sigma_i)
      w <- w + h_t * (F_tilde^T F_tilde)^(-1) * F_tilde^T y_tilde
    """

    def __init__(
        self,
        method: Literal["newton", "irls"] = "newton",
        max_iter: int = 100,
        tol: float = 1e-8,
        q_tol: float = 1e-10,   # NEW: tolerance on change of Q
        ridge: float = 1e-10,
        step_size: float = 1.0,
    ) -> None:
        self.method = method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.step_size = float(step_size)
        self.q_tol = float(q_tol)

        self.w: Optional[np.ndarray] = None  # includes bias as last element

    def _check_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1).astype(int)
        uniq = set(np.unique(y).tolist())
        if not uniq.issubset({-1, 1}):
            raise ValueError("y must be in {-1, +1} for lecture-style formulas.")
        return y

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        F = add_bias_last(np.asarray(X, dtype=np.float64))
        return F @ self.w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probability of class +1 as sigmoid(<w, x>).
        """
        s = self.decision_function(X)
        p_pos = sigmoid(s)
        return np.vstack([1.0 - p_pos, p_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.decision_function(X)
        return np.where(s >= 0, 1, -1).astype(int)

    def _Q(self, F: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        margins = y * (F @ w)
        # Q = sum log(1 + exp(-margin))
        # stable: logaddexp(0, -margin)
        return float(np.sum(np.logaddexp(0.0, -margins)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        X = np.asarray(X, dtype=np.float64)
        y = self._check_y(y)

        F = add_bias_last(X)  # (n, d+1)
        n_features = F.shape[1]

        # init w (zeros)
        w = np.zeros(n_features, dtype=np.float64)

        history = {"Q": [], "step_norm": []}
        converged = False
        prev_Q = None

        for it in range(1, self.max_iter + 1):
            w_old = w.copy()

            margins = y * (F @ w)
            sigma_i = sigmoid(margins)
            sigma_i = np.clip(sigma_i, 1e-12, 1.0 - 1e-12)

            if self.method == "newton":
                # grad = - F^T [ (1 - sigma) * y ]
                a = (1.0 - sigma_i) * y  # shape (n,)
                grad = -(F.T @ a)        # shape (d+1,)

                # Hess = F^T diag((1-sigma)*sigma) F
                wdiag = (1.0 - sigma_i) * sigma_i  # shape (n,)
                H = F.T @ (F * wdiag[:, None])
                H = H + self.ridge * np.eye(n_features)

                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.pinv(H) @ grad

                w = w - self.step_size * step

            elif self.method == "irls":
                # gamma_i = sqrt((1-sigma)*sigma)
                gamma = np.sqrt((1.0 - sigma_i) * sigma_i)

                # F_tilde = diag(gamma) F (avoid diag explicitly)
                F_tilde = F * gamma[:, None]

                # y_tilde_i = y_i * sqrt((1-sigma)/sigma)
                y_tilde = y * np.sqrt((1.0 - sigma_i) / sigma_i)

                A = F_tilde.T @ F_tilde + self.ridge * np.eye(n_features)
                b = F_tilde.T @ y_tilde

                try:
                    delta = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(A) @ b

                w = w + self.step_size * delta
            else:
                raise ValueError("method must be 'newton' or 'irls'")

            step_norm = float(np.linalg.norm(w - w_old))
            Qv = self._Q(F, y, w)

            history["Q"].append(Qv)
            history["step_norm"].append(step_norm)

            if prev_Q is not None and abs(prev_Q - Qv) < self.q_tol:
                converged = True
                break
            prev_Q = Qv

            if step_norm < self.tol:
                converged = True
                break

        self.w = w.copy()
        return FitResult(w=w, n_iter=len(history["Q"]), converged=converged, history=history)
