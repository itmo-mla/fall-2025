import os
from pathlib import Path
from typing import Literal, Self


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

import cvxpy as cp

#==============================================================================#

__all__ = [
    "Kernel",
    "MySVC"
]

RANDOM_SEED = 18012026

#==============================================================================#

#========== Kernels ==========#

class Kernel:

    def __init__(
        self, 
        kernel: Literal['linear', 'poly', 'rbf', 'sigmoid'],
        gamma: float,
        *,
        degree: int = 3,
        coef0: float = 0.0,
        p: int = 2

    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.p = p

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A)
        B = np.asarray(B)

        if self.kernel == 'linear':
            return self.linear(A, B)
        elif self.kernel == 'poly':
            return self.poly(A, B)
        elif self.kernel == 'rbf':
            return self.rbf(A, B)
        # elif self.kernel == 'sigmoid':
            # return self.sigmoid(A, B)
        else:
            raise ValueError(f"Uknown value for kernel: {self.kernel}. ",
                             "Available kernels are ['linear', 'poly', 'rbf', 'sigmoid']")

    def linear(self, A: np.ndarray, B: np.ndarray):
        return A @ B.T # equivalent to np.dot, but should be a bit faster (docs recomendation)
    
    def poly(self, A: np.ndarray, B: np.ndarray):
        return np.power(self.gamma * (A @ B.T) + self.coef0, self.degree)
    
    def rbf(self, A: np.ndarray, B: np.ndarray):
        cdists = self._cdist(A, B)
        return np.exp(-self.gamma * cdists)

    # def sigmoid(self, A: np.ndarray, B: np.ndarray):
    #     return np.tanh(self.gamma * (A @ B.T) + self.coef0)
    
    def _cdist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Pairwise distances between rows of A and rows of B under Lp norm."""
        if self.p == 2:
            return np.maximum( # no sqrt for small rbf optimization
                    0.0, 
                    (A * A).sum(axis=1, keepdims=True) +   # (a_h, 1)   (A * A) is slightly faster than np.power(A, 2)
                    (B * B).sum(axis=1, keepdims=True).T + # (1, b_w)
                    -2 * (A @ B.T)
                )
        else:
            return np.power(cdist(A, B, metric='minkowski', p=self.p), 2)

#==============================================================================#

#========== SVC ==========#

class MySVC(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'poly', 'rbf'] = 'linear',
        degree: int = 3,
        gamma: Literal['scale', 'auto'] | float = 'scale',
        coef0: float = 0.0,
        # probability: bool = False,
        # class_weight: dict | Literal['balanced'] = None,
        tol: float = 1e-6,
        max_iter: int = 30000,
        random_state: int | None = None,
        eps: float = 1e-8
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        # self.probability = probability
        # self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.eps = eps

    def fit(self, X, y) -> Self:
        X = np.asarray(X)
        y = np.asarray(y)

        self._init_kernel(X)

        # Transform y into appropriate form for optimization task
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"target contains more than 2 classes. Currently SVC supports only binary tasks.")
        
        self.classes_ = classes
        y_pm1 = np.where(y == classes[0], -1.0, 1.0)

        # Calculate Gram matrix
        gram = self.kernel_fn(X, X)

        G = np.outer(y_pm1, y_pm1) * gram
        # G = np.diag(y_pm1) @ gram @ np.diag(y_pm1)
        G = 0.5 * (G + G.T) # small symmetry normalization trick

        lambda_ = self._solve_dual(G, y_pm1)
        
        sv_tol = max(self.eps, 10*self.tol)

        sv_idx = np.where(lambda_ > sv_tol)[0] # indices of support vectors (all support vectors)
        sv_edge_idx = np.where((sv_tol < lambda_) & (lambda_ < self.C - sv_tol))[0] # free support vectors
        if sv_edge_idx.size == 0:
            print("!!! No free support vectors (suspicious) !!!")
            sv_edge_idx = sv_idx

        sv_coef = (lambda_[sv_idx] * y_pm1[sv_idx]) # (n_sv, )

        decision_on_edge = sv_coef @ gram[np.ix_(sv_idx, sv_edge_idx)] # (n_sv, ) @ (n_sv, n_edge) -> (n_edge,)

        if self.kernel == 'linear':
            self.coef_ = sv_coef @ X[sv_idx]
        self.intercept_ = np.mean(y_pm1[sv_edge_idx] - decision_on_edge) 

        self.support_ = sv_idx # indices of support vectors
        self.support_vectors_ = X[sv_idx]
        self.dual_coef_ = sv_coef

        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X[None, :]
        preds_pm1 = np.sign(self.decision_function(X)).astype(np.int64)
        preds = np.where(preds_pm1 == -1, self.classes_[0], self.classes_[1])
        return preds
        
    
    def decision_function(self, X):
        if X.ndim == 1:
            X = X[None, :]

        if self.kernel == 'linear':
            return X @ self.coef_ + self.intercept_
        
        gram = self.kernel_fn(self.support_vectors_, X) # (n_support, n_features), (n_test, n_features) -> (n_support, n_test)
        lambda_y_gram = self.dual_coef_ @ gram           # (n_support,) @ (n_support, n_test) -> (n_test, )
        
        decision_values = lambda_y_gram + self.intercept_ # -> (n_test,)
        return decision_values

    ### ========= Helpers ========== ###

    def _compute_gamma(self, X: np.ndarray):
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        
        n_features = X.shape[1]
        if self.gamma == 'auto':
            return 1. / n_features
        if self.gamma == 'scale':
            X_var = X.var()
            return 1. / (n_features * X_var) if X_var != 0 else 1.0
        return 0.1
    

    def _init_kernel(self, X: np.ndarray):
        gamma = self._compute_gamma(X)
        self.kernel_fn = Kernel(
            self.kernel,
            gamma,
            degree=self.degree,
            coef0=self.coef0,
            p=2                 # always Euclidean distance by default
        )


    def _solve_dual(self, G: np.ndarray, y: np.ndarray):
        lambda_ = cp.Variable(G.shape[0])
        constraints = [
            lambda_ >= 0,
            lambda_ <= self.C,
            lambda_ @ y == 0
        ]

        obj = cp.Minimize((1/2)*cp.quad_form(lambda_, cp.psd_wrap(G)) - cp.sum(lambda_))

        problem = cp.Problem(obj, constraints)
        problem.solve(cp.OSQP, eps_abs=self.tol, eps_rel=self.tol, max_iter=self.max_iter) # cp.CVXOPT
        
        return np.clip(lambda_.value, 0.0, self.C)

#==============================================================================#