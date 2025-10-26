"""
PCA через SVD — собственная реализация.
Не использует sklearn для PCA (только numpy.linalg.svd).
"""
import numpy as np

class PCA_SVD:
    """Простая реализация PCA через SVD.

    Методы:
      - fit(X): подгоняет модель (центрирует X, считает SVD)
      - transform(X, n_components): проецирует X на m главных компонент
      - inverse_transform(Z): восстанавливает X из представления Z
      - explained_variance_: собственные значения (λ_j)
      - explained_variance_ratio_: доли объяснённой дисперсии
      - effective_dim(eps): минимальное m такое, что остаточная доля ≤ eps
    """

    def __init__(self, center=True):
        self.center = center
        self.mean_ = None
        self.components_ = None  # U (n_features x n_components) — собственные векторы признаков
        self.singular_values_ = None  # sqrt(λ_j)
        self.explained_variance_ = None  # λ_j
        self.explained_variance_ratio_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        if self.center:
            self.mean_ = X.mean(axis=0)
            Xc = X - self.mean_
        else:
            self.mean_ = np.zeros(n_features)
            Xc = X.copy()

        # SVD: Xc = V D U^T, but numpy returns U, s, Vt for Xc
        # We'll compute economy SVD
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        # s: singular values (length = min(n_samples, n_features))
        # eigenvalues of X^T X are λ_j = s_j^2
        self.singular_values_ = s
        self.components_ = Vt.T  # columns are principal directions in feature space
        self.explained_variance_ = (s ** 2) / (n_samples - 1)
        total_var = self.explained_variance_.sum()
        if total_var <= 0:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X, n_components=None):
        X = np.asarray(X, dtype=float)
        if self.mean_ is None:
            raise RuntimeError("PCA not fitted yet. Call fit(X) first.")
        Xc = X - self.mean_
        if n_components is None:
            W = self.components_
        else:
            W = self.components_[:, :n_components]
        # project: Xc dot W
        return Xc.dot(W)

    def inverse_transform(self, Z):
        # Z shape: (n_samples, m); components_[:, :m] shape: (n_features, m)
        m = Z.shape[1]
        W = self.components_[:, :m]
        Xc_hat = Z.dot(W.T)
        return Xc_hat + self.mean_

    def effective_dim(self, eps=0.01):
        """Найти минимальное m такое, что сумма λ_{m+1..n}/sum(λ) <= eps.
        Соответствует определению в методичке: Em <= eps.
        """
        if self.explained_variance_ is None:
            raise RuntimeError("PCA not fitted yet. Call fit(X) first.")
        cum = np.cumsum(self.explained_variance_)
        total = cum[-1]
        # хотим минимальный m: (total - cum[m-1]) / total <= eps
        # т.е. cum[m-1] / total >= 1 - eps
        target = 1.0 - eps
        m = np.searchsorted(cum / total, target) + 1  # +1 потому что searchsorted даёт индекс
        m = int(min(max(1, m), len(self.explained_variance_)))
        return m

    def reconstruction_error(self, X, m=None):
        """Сумма квадратов ошибок восстановления для m компонент (средняя по сэмплам).
        Если m is None, используем все компоненты (ошибка ~ 0 up to numerical).
        """
        X = np.asarray(X, dtype=float)
        if m is None:
            m = self.n_features_
        Z = self.transform(X, n_components=m)
        X_hat = self.inverse_transform(Z)
        err = ((X - X_hat) ** 2).sum(axis=1).mean()
        return err