import numpy as np
from scipy.optimize import minimize


class SVMDual:
    def __init__(self, C=1.0, kernel='linear', gamma=None, degree=3, coef0=0, tol=0.001, verbose=False):
        self.C = C                 # параметр регуляризации
        self.kernel = kernel       # тип ядра
        self.gamma = gamma         # параметр для RBF ядра
        self.degree = degree       # степень полинома
        self.coef0 = coef0         # смещение 
        self.tol = tol             # порог для определения опорных векторов  
        self.verbose = verbose     # флаг вывода отладочной информации

        self.alpha = None
        self.b = None
        self.w = None
        self.X = None
        self.y = None
        self.support_idx = None

    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + self.coef0) ** self.degree
        else:
            raise ValueError("Unknown kernel")

    def _gram_matrix(self, X, y):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = y[i] * y[j] * self._kernel(X[i], X[j])
        return K

    def _objective(self, alpha, K):
        return 0.5 * alpha @ K @ alpha - np.sum(alpha)

    def _grad(self, alpha, K):
        return K @ alpha - np.ones_like(alpha)
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.X = X
        self.y = y
        n, d = X.shape

        if self.kernel == 'rbf' and (self.gamma is None):
            self.gamma = 1.0 / (d * np.var(X))

        K = self._gram_matrix(X, y) + 1e-12 * np.eye(n)

        alpha0 = np.zeros(n)
        bounds = [(0, self.C)] * n
        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}

        result = minimize(
            fun=lambda a: self._objective(a, K),
            x0=alpha0,
            jac=lambda a: self._grad(a, K),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 2000}
        )

        if self.verbose:
            print(result.message)

        self.alpha = result.x
        self.support_idx = np.where(self.alpha > self.tol)[0]

        if self.kernel == 'linear':
            self.w = np.sum((self.alpha * y)[:, None] * X, axis=0)

        sv_mask = (self.alpha > self.tol) & (self.alpha < self.C - self.tol)
        idxs = np.where(sv_mask)[0] if np.any(sv_mask) else self.support_idx

        bs = []
        for i in idxs:
            if self.kernel == 'linear':
                val = np.dot(self.w, X[i])
            else:
                # RBF или Poly
                val = np.sum(self.alpha * y * np.array([self._kernel(X[j], X[i]) for j in range(n)]))
            bs.append(y[i] - val)

        self.b = np.mean(bs)

    def decision(self, X_new):
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new = X_new[None, :]

        if self.kernel == 'linear':
            return X_new @ self.w + self.b

        out = np.zeros(len(X_new))
        for i, x in enumerate(X_new):
            s = np.sum(self.alpha * self.y * np.array([self._kernel(self.X[j], x) for j in range(len(self.X))]))
            out[i] = s + self.b
        return out

    def predict(self, X_new):
        return np.sign(self.decision(X_new))
