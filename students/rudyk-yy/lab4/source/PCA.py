import numpy as np

class PCA:


    def __init__(self, n_components=None, eps=None):
        self.n_components = n_components
        self.eps = eps

    def fit(self, X):
        # центрирование
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # SVD
        V, S, U_T = np.linalg.svd(Xc, full_matrices=False)

        # собственные значения
        self.singular_values_ = S
        self.explained_variance_ = S**2
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        # выбор числа компонент
        if self.n_components is None:
            if self.eps is None:
                self.n_components_ = X.shape[1]
            else:
                total = self.explained_variance_.sum()
                residual = np.array([
                    self.explained_variance_[m+1:].sum() / total
                    for m in range(len(self.explained_variance_))
                ])

                self.n_components_ = np.argmax(residual <= self.eps)
                if self.n_components_ == 0:
                    self.n_components_ = 1
        else:
            self.n_components_ = self.n_components

        # главные направления
        self.components_ = U_T[:self.n_components_]

        return self

    def transform(self, X):
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)