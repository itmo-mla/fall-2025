from typing import Union, Optional, Any, Literal, Type

import numpy as np
from scipy.linalg import svd, eigh

from sklearn.base import BaseEstimator, TransformerMixin

#==================================================================

__all__ = [
    "MyPCA"
]

#==================================================================

#========== PCA ==========

class MyPCA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int | float | Literal['profile_likelihood'] | None = None,
        svd_solver: Literal["full", "covariance_eigh"] = "full",
    ):
        self.n_components = n_components
        self.svd_solver = svd_solver

    def fit(self, X, y=None):
        X = np.asarray(X)
        n, d = X.shape

        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_

        if self.svd_solver == "full":
            _, s, Vh = svd(Xc, full_matrices=False)
            # determine number of components
            k = self._determine_n_components(s)
            
            self.components_ = Vh.T[:, :k]
            s_sq = (s * s) / (n - 1)
            self.explained_variance_ = s_sq[:k]
            self.explained_variance_ratio_ = s_sq[:k] / s_sq.sum() 
            self.singular_values_ = s[:k]

        elif self.svd_solver == "covariance_eigh":
            C = (Xc.T @ Xc) / (n - 1)
            w, V = eigh(C)
            idx = np.argsort(w)[::-1]
            k = self._determine_n_components(w[idx], n=n)
            idx = idx[:k]

            self.components_ = V[:, idx]
            self.explained_variance_ = w[idx]
            self.explained_variance_ratio_ = w[idx] / w.sum()
            self.singular_values_ = np.sqrt(w[idx] * (n - 1))

        else:
            raise ValueError("Unknown svd_solver")

        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z)
        return (Z @ self.components_.T + self.mean_).squeeze()
    
    def _determine_n_components(self, s: np.ndarray, n=None, min_seg=1, eps=1e-12):
        if isinstance(self.n_components, int):
            self.n_components_ = self.n_components
            return self.n_components_
        
        elif isinstance(self.n_components, float):
            if self.svd_solver == "full":
                s_sq = s * s
                explained_var = np.cumsum(s_sq) / s_sq.sum()
                 
            elif self.svd_solver == "covariance_eigh":
                explained_var = np.cumsum(s) / s.sum()

            self.n_components_ = np.flatnonzero(explained_var >= self.n_components)[0] + 1
            return self.n_components_
        
        elif self.n_components == 'profile_likelihood':            
            n = s.size
            if n < 2 * min_seg:
                raise ValueError("Insufficient amount of values with given min_seg.")

            if self.svd_solver == "covariance_eigh":
                # привожу к сингулярным числам, потому что у них 
                # масштаб меньше собственных чисел, и для них оценка получается ровнее
                # s = np.sqrt(s * (n - 1))
                s = np.log(s) # эмпирически получается, что с логарифмом оценка точнее
                # можно в целом как угодно попробовать задать масштаб шкалы, главное, чтобы 
                # не было слишком большого перепада между значениями
                # заданной меры ошибки, индуцированной моделью (в нашем случае ошибки = сингулярные числа
                # хотя в оригинале используется reconstruction loss, который в случае PCA равен сингулярным числам).
                
            S = np.cumsum(s)
            Q = np.cumsum(s * s)
            S_total = S[-1]
            Q_total = Q[-1]

            Ls = np.arange(min_seg, n - min_seg + 1)  # L in [min_seg .. n-min_seg]
            S1 = S[Ls - 1]
            Q1 = Q[Ls - 1]
            n1 = Ls

            S2 = S_total - S1
            Q2 = Q_total - Q1
            n2 = n - n1

            # mu1 = S1 / n1
            # mu2 = S2 / n2

            sse1 = Q1 - (S1 * S1) / n1
            sse2 = Q2 - (S2 * S2) / n2
            sse = sse1 + sse2

            sigma2 = sse / n
            sigma2 = np.maximum(sigma2, eps)  # защита от log(0)

            ll = -(n / 2.0) * (np.log(2 * np.pi) + np.log(sigma2) + 1.0)

            idx = np.argmax(ll)
            L_star = int(Ls[idx])

            if L_star == n:
                raise ValueError("profile likelihood method couldn't find an optimal intrinsic dimensionality, ",
                                 "or initial dataset's dimensionality is already sufficient enough.")
            self.n_components_ = L_star
            self.likelihood_profile_ = ll

            return L_star
        
#==================================================================
