import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        centered_X = X - self.mean_

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        U, S, Vt = np.linalg.svd(centered_X, full_matrices=False)

        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (n_samples - 1)
        total_variance = np.sum(S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X: np.ndarray):
        centered_X = X - self.mean_
        return centered_X @ self.components_.T

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
