import numpy as np
import matplotlib.pyplot as plt


class CustomPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):

        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])

        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        self.components_ = Vt
        self.singular_values_ = np.sort(s)[::-1]

        return self

    def transform(self, X):
        return X @ self.components_[:self.n_components].T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def reconstruct(self, X_pca):
        return X_pca @ self.components_