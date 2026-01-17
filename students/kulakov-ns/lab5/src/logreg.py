from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import *


@dataclass
class FitResult:
    w: np.ndarray
    n_iter: int
    history: pd.DataFrame


def newton_raphson_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    ridge: float = 1e-10,
    w0: np.ndarray | None = None,
) -> FitResult:
    """Logistic regression via Newton-Raphson on NLL."""
    n_features = X.shape[1]
    w = np.zeros(n_features) if w0 is None else w0.astype(float).copy()

    rows = []
    for it in range(1, max_iter + 1):
        nll = neg_log_likelihood(X, y, w)
        g = grad_nll(X, y, w)
        H = hess_nll(X, w) + ridge * np.eye(n_features)

        step = np.linalg.solve(H, g)
        w_new = w - step

        step_norm = float(np.linalg.norm(step))
        rows.append({"iter": it, "nll": nll, "step_norm": step_norm})

        w = w_new
        if step_norm < tol:
            break

    hist = pd.DataFrame(rows)
    return FitResult(w=w, n_iter=len(hist), history=hist)


def irls_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    ridge: float = 1e-10,
    w0: np.ndarray | None = None,
) -> FitResult:
    """Logistic regression via IRLS (iteratively reweighted least squares)."""
    n_features = X.shape[1]
    w = np.zeros(n_features) if w0 is None else w0.astype(float).copy()

    rows = []
    for it in range(1, max_iter + 1):
        eta = X @ w
        p = sigmoid(eta)
        W = np.clip(p * (1 - p), 1e-12, None)

        z = eta + (y - p) / W  # working response

        XTWX = X.T @ (X * W[:, None]) + ridge * np.eye(n_features)
        XTWz = X.T @ (W * z)
        w_new = np.linalg.solve(XTWX, XTWz)

        nll = neg_log_likelihood(X, y, w)
        step = w_new - w
        step_norm = float(np.linalg.norm(step))
        rows.append({"iter": it, "nll": nll, "step_norm": step_norm})

        w = w_new
        if step_norm < tol:
            break

    hist = pd.DataFrame(rows)
    return FitResult(w=w, n_iter=len(hist), history=hist)