import numpy as np


class LogisticRegression:

    def __init__(self, method="newton", max_iter=20, tol=1e-6, lern_rate=0.1):
        if method not in ("newton", "irls"):
            raise ValueError("method must be 'newton' or 'irls'")
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.lern_rate = lern_rate
        self.w = None

    @staticmethod
    def _sigmoid(z):

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
      
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

        for _ in range(self.max_iter):
            w_old = self.w.copy()


            margins = y * (X @ self.w)
            sigma = self._sigmoid(margins)
            sigma = np.clip(sigma, 1e-8, 1 - 1e-8)  
            if self.method == "newton":
                try:
                    self._newton_step(X, y, sigma)
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered in Newton step; skipping update.")   
            else:
                try:
                    self._irls_step(X, y, sigma)
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered in IRLS step; skipping update.")

            if np.linalg.norm(self.w - w_old) < self.tol:
                print("Update less then tol, stops training")
                break

        return self

    def _newton_step(self, X, y, sigma):
        
        grad = -X.T @ (y/sigma)
        d = np.diag(sigma * (1 - sigma))
        hessian = X.T @ d @ X

        hessian += 1e-8 * np.eye(hessian.shape[0])

        delta = np.linalg.inv(hessian) @ grad
        self.w -=  self.lern_rate * delta


    def _irls_step(self, X, y, sigma):
        gamma = np.sqrt(sigma * (1 - sigma))

        X_mod = np.diag(gamma) @ X
        y_mod = y * (gamma / sigma)

        self.w += self.lern_rate * (np.linalg.inv(X_mod.T @ X_mod) @ X_mod.T @ y_mod)


    def predict(self, X):
        return np.where((X @ self.w) >= 0, 1, -1)
