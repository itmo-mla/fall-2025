from typing import Literal, Self

import numpy as np
from scipy.special import expit

import statsmodels.api as sm

from sklearn.base import BaseEstimator, ClassifierMixin


#==============================================================================#

__all__ = [
    "MyLogisticRegression",
    "SMGLMLogitClassifier"
]

RANDOM_SEED = 18012026

#==============================================================================#

#========== Logistic Regression ==========#

class MyLogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        C: float = 1.0,
        solver: Literal['newton', 'irls'] = 'newton',
        tol: float = 1e-5,
        max_iter: int = 10000,
        random_state: int | None = None,
        eps: float = 1e-8
    ):
        self.C = C
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.eps = eps

        self.penalize_intercept_ = False

    def fit(self, X, y) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes = np.sort(np.unique(y))
        if classes.size != 2:
            raise ValueError("target contains more than 2 classes. Currently Logistic Regression supports only binary tasks.")
        self.classes_ = classes

        # map y to {0,1} as Murphy assumes
        y01 = (y == classes[1]).astype(float)

        Xb = self._ensure_intercept_column(X)
        n, d = Xb.shape
        self.n_features_in_ = X.shape[1]

        if self.C <= 0:
            raise ValueError("C must be > 0.")
        # lam = 1.0 / self.C  # Murphy: lambda = 1/C
        lam = 1.0 / (2.0 * n * self.C) # sklearn-like

        w = np.zeros(d, dtype=float)

        self.loss_history_ = []
        self.n_iter_ = 0

        if self.solver == "newton":
            w = self._fit_newton(Xb, y01, w, lam)
        elif self.solver == "irls":
            w = self._fit_irls(Xb, y01, w, lam)
        else:
            raise ValueError("solver must be one of {'newton','irls'}")

        self.intercept_ = float(w[0])
        self.coef_ = w[1:].reshape(1, -1)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._ensure_intercept_column(X)
        w = self._packed_coef()
        # return a = Xw (logits)
        return Xb @ w

    def predict_proba(self, X):
        a = self.decision_function(X)
        mu = expit(a)
        return np.vstack([1.0 - mu, mu]).T

    def predict(self, X):
        mu = self.predict_proba(X)[:, 1]
        y01_pred = (mu >= 0.5).astype(int)
        return np.where(y01_pred == 1, self.classes_[1], self.classes_[0])

    # ----------------- Newton solver -----------------

    def _fit_newton(self, X, y, w, lam):
        n = X.shape[0]
        reg_mask = np.ones_like(w)
        if not self.penalize_intercept_:
            reg_mask[0] = 0.0

        prev_loss = self._pnll_mean(X, y, w, lam, reg_mask)

        for it in range(1, self.max_iter + 1):
            a = X @ w
            mu = expit(a)

            s = np.clip(mu * (1.0 - mu), self.eps, None)

            g_data = (X.T @ (mu - y)) / n

            g = g_data + 2.0 * lam * (reg_mask * w)

            H_data = ((X.T * s) @ X) / n

            H = H_data + 2.0 * lam * np.diag(reg_mask)

            delta = self._solve(H, g)

            step = 1.0
            while True:
                w_new = w - step * delta
                new_loss = self._pnll_mean(X, y, w_new, lam, reg_mask)
                if new_loss <= prev_loss or step < 1e-8:
                    break
                step *= 0.5

            update = w_new - w
            w = w_new
            self.loss_history_.append(new_loss)
            self.n_iter_ = it

            if np.linalg.norm(update) <= self.tol * (1.0 + np.linalg.norm(w)):
                break
            if abs(prev_loss - new_loss) <= self.tol * (1.0 + abs(new_loss)):
                break
            prev_loss = new_loss

        return w

    # ----------------- IRLS solver -----------------

    def _fit_irls(self, X, y, w, lam):
        n = X.shape[0]
        reg_mask = np.ones_like(w)
        if not self.penalize_intercept_:
            reg_mask[0] = 0.0

        prev_loss = self._pnll_mean(X, y, w, lam, reg_mask)

        for it in range(1, self.max_iter + 1):
            w_old = w.copy()

            # Murphy Alg 2: a, mu, s, z
            a = X @ w
            mu = expit(a)
            s = np.clip(mu * (1.0 - mu), self.eps, None)
            z = a + (y - mu) / s

            XTSX = (X.T * s) @ X
            XTSz = X.T @ (s * z)

            A = XTSX + (2.0 * n * lam) * np.diag(reg_mask)
            b = XTSz
            w = self._solve(A, b)

            new_loss = self._pnll_mean(X, y, w, lam, reg_mask)

            # optional damping if PNLL increases
            if new_loss > prev_loss:
                step = 1.0
                w_try = w.copy()
                while new_loss > prev_loss and step > 1e-8:
                    step *= 0.5
                    w_try = (1.0 - step) * w_old + step * w
                    new_loss = self._pnll_mean(X, y, w_try, lam, reg_mask)
                w = w_try

            update = w - w_old
            self.loss_history_.append(new_loss)
            self.n_iter_ = it

            if np.linalg.norm(update) <= self.tol * (1.0 + np.linalg.norm(w)):
                break
            if abs(prev_loss - new_loss) <= self.tol * (1.0 + abs(new_loss)):
                break
            prev_loss = new_loss

        return w

    # ----------------- Helpers -----------------

    def _ensure_intercept_column(self, X: np.ndarray) -> np.ndarray:
        # If first col is ones, assume bias already included
        if X.shape[1] >= 1 and np.allclose(X[:, 0], 1.0, atol=1e-12, rtol=0.0):
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def _packed_coef(self) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise AttributeError("Model is not fitted yet.")
        return np.concatenate([[self.intercept_], self.coef_.ravel()])

    def _pnll_mean(self, X, y, w, lam, reg_mask) -> float:
        a = X @ w
        mu = np.clip(expit(a), self.eps, 1.0 - self.eps)
        nll_mean = -np.mean(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu))
        reg = lam * float((reg_mask * w) @ (reg_mask * w))
        return nll_mean + reg

    def _solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x
        

#===== Statsmodels Reference Model Wrapper =====#

class SMGLMLogitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        solver: Literal["irls", "newton"] = "irls",
        tol: float = 1e-8,
        max_iter: int = 100
    ):
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes = np.sort(np.unique(y))
        if classes.size != 2:
            raise ValueError("Only binary classification is supported.")
        self.classes_ = classes

        y01 = (y == classes[1]).astype(float)

        # add intercept column (like your implementation)
        Xb = self._ensure_intercept_column(X)
        self.n_features_in_ = X.shape[1]

        model = sm.GLM(y01, Xb, family=sm.families.Binomial())

        method = "IRLS" if self.solver == "irls" else "newton"
        res = model.fit(method=method, maxiter=self.max_iter, tol=self.tol)

        self.result_ = res
        params = np.asarray(res.params, dtype=float)

        self.intercept_ = float(params[0])
        self.coef_ = params[1:].reshape(1, -1)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._ensure_intercept_column(X)
        w = np.concatenate([[self.intercept_], self.coef_.ravel()])
        return Xb @ w

    def predict_proba(self, X):
        logits = self.decision_function(X)
        p1 = expit(logits)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        y01_pred = (p1 >= 0.5).astype(int)
        return np.where(y01_pred == 1, self.classes_[1], self.classes_[0])

    @staticmethod
    def _ensure_intercept_column(X: np.ndarray) -> np.ndarray:
        if X.shape[1] >= 1 and np.allclose(X[:, 0], 1.0, atol=1e-12, rtol=0.0):
            return X
        return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])

#==============================================================================#