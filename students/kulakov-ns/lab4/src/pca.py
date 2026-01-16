import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got shape={x.shape}")
    return x

def _cumulative_ratio(explained: np.ndarray) -> np.ndarray:
    total = float(np.sum(explained))
    if total <= 0:
        return np.zeros_like(explained, dtype=float)
    return np.cumsum(explained) / total



class CustomPCA:
    def __init__(self, n_components: int, eps: float = 1e-3, whiten: bool = False):
        self.n_components: int = n_components
        self.eps: float = eps
        self.whiten: bool = False
        self.eff_size: int = n_components

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "CustomPCA":
        X = _as_2d(X)
        n_samples, _ = X.shape
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to fit PCA")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        explained = (S**2) / (n_samples - 1)
        cum = _cumulative_ratio(explained)
        self.eff_size = np.argmax(1 - cum < self.eps)

        k = self.n_components

        self.components_ = Vt[:k, :]
        self.singular_values_ = S[:k]
        self.explained_variance_ = explained[:k]
        total = float(np.sum(explained))
        self.explained_variance_ratio_ = explained[:k] / total if total > 0 else np.zeros(k)

        return self


    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = _as_2d(X)
        Xc = X - self.mean_
        Z = Xc @ self.components_.T 

        if self.whiten:
            s = np.where(self.singular_values_ > 0, self.singular_values_, 1.0)
            Z = Z / s

        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        Z = _as_2d(Z)

        if self.whiten:
            s = np.where(self.singular_values_ > 0, self.singular_values_, 1.0)
            Z = Z * s

        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.components_ is None or self.singular_values_ is None:
            raise RuntimeError("CustomPCA is not fitted yet. Call fit() first.")

