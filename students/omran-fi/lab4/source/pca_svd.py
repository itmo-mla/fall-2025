from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PCAResult:
    mean_: np.ndarray
    components_: np.ndarray          # shape: (n_components, n_features)
    singular_values_: np.ndarray     # shape: (min(n_samples, n_features),)
    explained_variance_: np.ndarray  # shape: (n_components_total,)
    explained_variance_ratio_: np.ndarray  # shape: (n_components_total,)

    def transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        Xc = X - self.mean_
        W = self.components_ if n_components is None else self.components_[:n_components]
        return Xc @ W.T

    def inverse_transform(self, Z: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        W = self.components_ if n_components is None else self.components_[:n_components]
        return Z @ W + self.mean_


def pca_via_svd(X: np.ndarray) -> PCAResult:
    """
    PCA by SVD:
    X_centered = U S V^T
    principal axes = V
    explained_variance = (S^2) / (n_samples - 1)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array.")

    n_samples, n_features = X.shape
    mean_ = X.mean(axis=0)
    Xc = X - mean_

    # full_matrices=False -> economical SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Vt shape: (k, n_features), rows are PCs
    components_ = Vt  # each row is a principal axis

    # explained variance of each PC
    denom = max(n_samples - 1, 1)
    explained_variance_ = (S ** 2) / denom
    total_var = explained_variance_.sum() if explained_variance_.sum() > 0 else 1.0
    explained_variance_ratio_ = explained_variance_ / total_var

    return PCAResult(
        mean_=mean_,
        components_=components_,
        singular_values_=S,
        explained_variance_=explained_variance_,
        explained_variance_ratio_=explained_variance_ratio_,
    )


def effective_dimensionality(
    explained_variance_ratio: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """
    Effective dimensionality as the minimal k such that cumulative explained variance >= threshold.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in (0, 1].")

    cumsum = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cumsum, threshold) + 1)
    return min(k, explained_variance_ratio.size)
