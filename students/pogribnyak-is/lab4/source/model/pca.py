from dataclasses import dataclass

import numpy as np
from typing import Optional

from model.component_selector import ComponentSelector, VarianceThresholdSelector
from model.decomposition import Decomposition, SVDDecomposition


@dataclass(frozen=True)
class LinearProjector:
    components: np.ndarray
    mean: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray: return (X - self.mean) @ self.components.T


class PCA:
    def __init__(
        self,
        decomposer: Optional[Decomposition] = None,
        selector: Optional[ComponentSelector] = None
    ):
        self.decomposer = decomposer or SVDDecomposition()
        self.selector = selector or VarianceThresholdSelector(0.95)
        self.components_: Optional[np.ndarray] = None
        self.projector_: Optional[LinearProjector] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    @staticmethod
    def _compute_explained_variance(S: np.ndarray, n_samples: int) -> np.ndarray:
        explained_variance = (S ** 2) / (n_samples - 1)
        return explained_variance / explained_variance.sum()

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = self.decomposer.compute(Xc)
        ratio = self._compute_explained_variance(S, X.shape[0])
        k = self.selector.select(ratio)
        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = ratio[:k]
        assert self.components_ is not None
        assert self.mean_ is not None
        self.projector_ = LinearProjector(self.components_, self.mean_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.projector_ is not None, "Вызовите fit() перед transform()!"
        return self.projector_.transform(X)
