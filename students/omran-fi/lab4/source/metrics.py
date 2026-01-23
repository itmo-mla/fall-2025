from __future__ import annotations

import numpy as np


def reconstruction_mse(X: np.ndarray, X_rec: np.ndarray) -> float:
    err = X - X_rec
    return float(np.mean(err * err))


def subspace_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Distance between two k-dim subspaces in R^d using projection matrices:
    ||P1 - P2||_F
    W: shape (k, d) with orthonormal rows.
    """
    # Convert to column-orthonormal basis: Q shape (d, k)
    Q1 = W1.T
    Q2 = W2.T
    P1 = Q1 @ Q1.T
    P2 = Q2 @ Q2.T
    return float(np.linalg.norm(P1 - P2, ord="fro"))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))
