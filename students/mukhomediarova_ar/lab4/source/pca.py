import numpy as np


class PCASVD:
    """
    PCA implementation based on singular value decomposition (SVD).

    Parameters
    ----------
    n_components : int | float | None
        - int  : number of principal components to keep;
        - float: fraction of explained variance in (0, 1]; the smallest
                 number of components with cumulative explained variance
                 >= n_components will be chosen;
        - None : keep all components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def fit(self, X: np.ndarray) -> "PCASVD":
        """
        Fit PCA model using SVD.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        """
        X = np.asarray(X, dtype=float)
        n_samples, _ = X.shape

        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Full-matrices=False gives compact SVD: shapes (n_samples, n_features, n_features)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.singular_values_ = S
        # Explained variance for each principal component
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_variance = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        # Determine number of components to keep
        if self.n_components is None:
            n_components = Vt.shape[0]
        elif isinstance(self.n_components, float):
            if not 0.0 < self.n_components <= 1.0:
                raise ValueError("If n_components is float, it must be in (0, 1].")
            cumulative = np.cumsum(self.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumulative, self.n_components) + 1)
        else:
            n_components = int(self.n_components)
            if n_components < 1 or n_components > Vt.shape[0]:
                raise ValueError("n_components must be in [1, n_features].")

        self.n_components_ = n_components
        self.components_ = Vt[:n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to the principal components space.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("The model must be fitted before calling transform.")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Project data back to the original feature space.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("The model must be fitted before calling inverse_transform.")

        X_transformed = np.asarray(X_transformed, dtype=float)
        return np.dot(X_transformed, self.components_) + self.mean_


class PCAEigen:
    """
    "Reference" PCA implementation via eigen-decomposition of the covariance matrix.

    This implementation is mathematically equivalent to SVD-based PCA and is used
    to demonstrate this equivalence numerically.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def fit(self, X: np.ndarray) -> "PCAEigen":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix of shape (n_features, n_features)
        cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # eigh is used for symmetric matrices, returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues/eigenvectors in descending order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        total_variance = eigvals.sum()
        self.explained_variance_ = eigvals
        self.explained_variance_ratio_ = eigvals / total_variance

        if self.n_components is None:
            n_components = n_features
        elif isinstance(self.n_components, float):
            if not 0.0 < self.n_components <= 1.0:
                raise ValueError("If n_components is float, it must be in (0, 1].")
            cumulative = np.cumsum(self.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumulative, self.n_components) + 1)
        else:
            n_components = int(self.n_components)
            if n_components < 1 or n_components > n_features:
                raise ValueError("n_components must be in [1, n_features].")

        self.n_components_ = n_components
        # Components are eigenvectors (each column is eigenvector)
        self.components_ = eigvecs[:, :n_components].T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("The model must be fitted before calling transform.")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("The model must be fitted before calling inverse_transform.")

        X_transformed = np.asarray(X_transformed, dtype=float)
        return np.dot(X_transformed, self.components_) + self.mean_


def choose_effective_dim(explained_variance_ratio: np.ndarray, threshold: float = 0.95) -> int:
    """
    Choose the minimal number of components such that cumulative explained variance
    exceeds the given threshold.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")

    cumulative = np.cumsum(explained_variance_ratio)
    return int(np.searchsorted(cumulative, threshold) + 1)

