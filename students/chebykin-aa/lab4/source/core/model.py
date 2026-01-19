import numpy as np

class OwnPCA:
    def __init__(
            self,
            variance_threshold=0.95
    ):
        self.variance_threshold = variance_threshold
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_eff_ = None

    def fit(self, X):
        # Центрирование данных
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD
        U, D, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt
        self.singular_values_ = D

        # Объяснённая дисперсия
        self.explained_variance_ = (D ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        # Определение эффективной размерности
        cumulative = np.cumsum(self.explained_variance_ratio_)
        self.n_components_eff_ = np.searchsorted(cumulative, self.variance_threshold) + 1

        return self

    def transform(self, X, n_components=None):
        X_centered = X - self.mean_
        if n_components is None:
            n_components = self.n_components_eff_
        return X_centered @ self.components_[:n_components].T