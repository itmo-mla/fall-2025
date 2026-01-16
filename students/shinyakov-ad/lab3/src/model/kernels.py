from __future__ import annotations

import numpy as np


class LinearKernel:
    def transform(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return np.dot(X1, X2.T)


class PolynomialKernel:
    def __init__(self, degree: int = 3):
        self.degree = degree

    def transform(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return (np.dot(X1, X2.T) + 1.0) ** self.degree


class RBFKernel:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def transform(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dists)



