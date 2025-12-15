from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Array = np.ndarray


class Kernel(Protocol):
    """Kernel callable returning the Gram matrix K(X, Z)."""

    def __call__(self, X: Array, Z: Array) -> Array:  # pragma: no cover
        ...


def _as_2d(a: Array) -> Array:
    a = np.asarray(a)
    return a[None, :] if a.ndim == 1 else a


@dataclass(frozen=True, slots=True)
class LinearKernel:
    """K(x, z) = x^T z"""

    def __call__(self, X: Array, Z: Array) -> Array:
        X2 = _as_2d(X)
        Z2 = _as_2d(Z)
        return X2 @ Z2.T


@dataclass(frozen=True, slots=True)
class PolynomialKernel:
    """K(x, z) = (gamma * x^T z + coef0)^degree."""

    degree: int = 2
    gamma: float = 1.0
    coef0: float = 0.0

    def __call__(self, X: Array, Z: Array) -> Array:
        X2 = _as_2d(X)
        Z2 = _as_2d(Z)
        return (self.gamma * (X2 @ Z2.T) + self.coef0) ** self.degree


@dataclass(frozen=True, slots=True)
class RBFKernel:
    """K(x, z) = exp(-gamma * ||x - z||^2)."""

    gamma: float = 1.0

    def __call__(self, X: Array, Z: Array) -> Array:
        X2 = _as_2d(X).astype(float, copy=False)
        Z2 = _as_2d(Z).astype(float, copy=False)

        # ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x^T z
        x_norm = np.sum(X2 * X2, axis=1)[:, None]
        z_norm = np.sum(Z2 * Z2, axis=1)[None, :]
        sq_dists = x_norm + z_norm - 2.0 * (X2 @ Z2.T)
        return np.exp(-self.gamma * sq_dists)





# Backwards-compatible alias for the "SquaredKernel" from the original solution
@dataclass(frozen=True, slots=True)
class SquaredKernel(PolynomialKernel):
    degree: int = 2
    gamma: float = 1.0
    coef0: float = 0.0
