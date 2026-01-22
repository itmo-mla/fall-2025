import numpy as np

class LogisticRegressionIRLS():

    def __init__(self, h_t=1e-3, iter=1000, tol=1e-6):
        self.h_t = h_t
        self.iter = iter
        self.tol = tol
        self.w = None

    def fit(self, X, y):
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]

        for t in range(self.iter):
            sigma = self.__sigmoid__(y * (X @ self.w))
            sigma = np.clip(sigma, self.tol, 1 - self.tol)
            gamma = np.sqrt(sigma * (1 - sigma))
            F_tilda = gamma[:, None] * X
            y_tilda = (y - sigma) / gamma
            delta_w = np.linalg.lstsq(F_tilda, y_tilda, rcond=None)[0]

            if np.linalg.norm(delta_w) < self.tol:
                break

            self.w += self.h_t * delta_w

    def predict(self, X):
        return (self.__sigmoid__(X @ self.w) >= 0.5).astype(int)
    
    def __sigmoid__(self, X):
        return 1 / (1 + np.exp(-X))

class NewtonRaphsonLogisticRegression:
    def __init__(self, h_t=1e-3, iter=100, tol=1e-6):
        self.h_t = h_t
        self.iter = iter
        self.tol = tol
        self.w = None

    def __sigmoid__(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])

        for _ in range(self.iter):
            mu = self.__sigmoid__(X @ self.w)

            grad = X.T @ (y - mu)
            W = mu * (1 - mu)
            H = X.T @ (W[:, None] * X)

            delta = np.linalg.solve(H, grad)

            if np.linalg.norm(delta) < self.tol:
                break

            self.w += self.h_t * delta

    def predict(self, X):
        return (self.__sigmoid__(X @ self.w) >= 0.5).astype(int)
