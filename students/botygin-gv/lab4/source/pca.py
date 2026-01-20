import numpy as np
from scipy.linalg import svd


class CustomPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD
        U, s, Vt = svd(X_centered, full_matrices=False)
        self.singular_values_ = s

        # Compute explained variance ratio
        total_var = np.sum(s ** 2)
        explained_var = (s ** 2) / total_var
        self.explained_variance_ratio_ = explained_var

        if self.n_components is None:
            self.n_components = min(X.shape)
        else:
            self.n_components = min(self.n_components, min(X.shape))

        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def effective_dimension(self, threshold=0.95):
        cumsum = np.cumsum(self.explained_variance_ratio_)
        return int(np.argmax(cumsum >= threshold)) + 1