import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

class DualSVM:
    def __init__(self, C=1.0, kernel="linear", gamma=1.0, tol=1e-6,
                 solver="SLSQP", standardize=True, maxiter=2000, ftol=1e-9):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.tol = float(tol)
        self.solver = solver
        self.standardize = bool(standardize)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)

        self.a_ = None         
        self.b_ = None         
        self.w_ = None         
        self.sv_idx_ = None    
        self.X_ = None         
        self.y_ = None         
        self.opt_result_ = None

        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y):
        X = self._as_2d_float(X)
        y = self._as_1d_float(y)

        y = self._ensure_pm_one(y)

        if self.standardize:
            Xs = self._fit_standardize(X)
        else:
            Xs = X.copy()

        n = Xs.shape[0]

        K = self._kernel_matrix(Xs, Xs)
        Q = (y[:, None] * y[None, :]) * K

        def obj(a):
            return 0.5 * a @ (Q @ a) - np.sum(a)

        def grad(a):
            return (Q @ a) - np.ones_like(a)

        lin_con = LinearConstraint(y.reshape(1, -1), lb=[0.0], ub=[0.0])

        bounds = Bounds(np.zeros(n), np.ones(n) * self.C)

        a0 = np.zeros(n)

        res = minimize(
            obj, a0, jac=grad,
            constraints=[lin_con],
            bounds=bounds,
            method=self.solver,
            options={"maxiter": self.maxiter, "ftol": self.ftol}
        )

        self.opt_result_ = res
        a = res.x

        sv = np.where(a > self.tol)[0]
        free_sv = np.where((a > self.tol) & (a < self.C - self.tol))[0]

        def f_train_at_i(i):
            return np.sum(a * y * K[:, i])

        if free_sv.size > 0:
            bs = [y[i] - f_train_at_i(i) for i in free_sv]
            b = float(np.mean(bs))
        elif sv.size > 0:
            bs = [y[i] - f_train_at_i(i) for i in sv]
            b = float(np.mean(bs))
        else:
            b = 0.0

        self.a_ = a
        self.b_ = b
        self.sv_idx_ = sv
        self.X_ = Xs
        self.y_ = y

        if self.kernel == "linear":
            self.w_ = (a * y) @ Xs
        else:
            self.w_ = None

        return self

    def decision_function(self, X):
        self._check_is_fitted()
        X = self._as_2d_float(X)

        if self.standardize:
            Xs = self._transform_standardize(X)
        else:
            Xs = X

        sv = self.sv_idx_
        if sv.size == 0:
            return np.zeros(Xs.shape[0]) + self.b_

        a_sv = self.a_[sv]
        y_sv = self.y_[sv]
        X_sv = self.X_[sv]

        if self.kernel == "linear" and self.w_ is not None:
            return Xs @ self.w_ + self.b_

        K = self._kernel_matrix(X_sv, Xs)
        scores = (a_sv * y_sv) @ K + self.b_
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def _kernel_matrix(self, X, Z):
        if self.kernel == "linear":
            return X @ Z.T
        elif self.kernel == "rbf":
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Z_norm = np.sum(Z**2, axis=1).reshape(1, -1)
            sq_dists = X_norm + Z_norm - 2 * (X @ Z.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    @staticmethod
    def _ensure_pm_one(y):
        uniq = np.unique(y)
        if set(uniq.tolist()) == {0.0, 1.0}:
            return np.where(y > 0, 1.0, -1.0)
        if set(uniq.tolist()) == {-1.0, 1.0}:
            return y
        raise ValueError("y must be in {-1,+1} or {0,1}")

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _as_1d_float(y):
        y = np.asarray(y, dtype=float).reshape(-1)
        return y

    def _fit_standardize(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (X - self.mean_) / self.scale_

    def _transform_standardize(self, X):
        return (X - self.mean_) / self.scale_

    def _check_is_fitted(self):
        if self.a_ is None or self.X_ is None or self.y_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
