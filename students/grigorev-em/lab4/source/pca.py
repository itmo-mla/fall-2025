import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.components_ = None
        self.lambdas = None
        self.n_components = n_components
        self.mean = None

    def fit(self, X):
        self.mean = X.mean(axis=0)

        U, s, Vh = np.linalg.svd((X - self.mean), full_matrices=False)
        self.components_ = Vh
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components_[:self.n_components].T

    def fit_transform(self, X):
        return self.fit(X).transform(X)