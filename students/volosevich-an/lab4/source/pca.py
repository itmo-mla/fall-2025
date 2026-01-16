import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.components_ = None
        self.n_components = n_components

    def svd(self, X):
        return np.linalg.svd(X - np.mean(X, axis=0), full_matrices=False)

    def fit(self, X):
        U, S, VT = self.svd(X)

        if self.n_components is None:
            self.n_components = self.effective_dimension(X)
            print(f"Effective dimension determined: {self.n_components}")
    
        self.components_ = VT[:self.n_components]
    
    def transform(self, X):
        return np.dot(X - np.mean(X, axis=0), self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def explained_variance(self, X):
        U, S, VT = self.svd(X)
        return S**2 / (X.shape[0] - 1)
    
    def effective_dimension(self, X, threshold=0.95):
        explained_variance = self.explained_variance(X)
        total_variance = np.sum(explained_variance)
        cumulative_variance = np.cumsum(explained_variance) / total_variance
        effective_dim = np.searchsorted(cumulative_variance, threshold) + 1
        return effective_dim
    