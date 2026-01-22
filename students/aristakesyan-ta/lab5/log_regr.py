import numpy as np


class LogisticRegressionNewtonRaphson:
    def __init__(self, max_iter=100, tol=1e-6, C=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.beta = None
        self.history = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        n, p = X.shape
        self.beta = np.zeros(p)
        
        # L2 regularization strength
        # Sklearn minimizes: C * Loss + 0.5 * ||w||^2
        # Equivalent to minimizing: Loss + (0.5/C) * ||w||^2
        # So lambda = 1/C
        lam = 1.0 / self.C
        
        for iteration in range(self.max_iter):
            eta = X @ self.beta
            mu = self.sigmoid(eta)
            
            w = np.maximum(mu * (1 - mu), 1e-10)
            W = np.diag(w)
            
            # Gradient of Loss: X^T (y - mu) (for maximization)
            # Gradient of Regularizer: -lam * beta
            # Total Gradient: X^T (y - mu) - lam * beta
            
            reg_vec = lam * self.beta
            reg_vec[0] = 0 # Do not regularize intercept
            
            gradient = X.T @ (y - mu) - reg_vec
            
            # Hessian of Loss: - X^T W X
            # Hessian of Regularizer: - lam * I
            # Total Hessian: - (X^T W X + lam * I)
            # Update: beta = beta - H^-1 * grad
            # beta = beta + (X^T W X + lam * I)^-1 * (X^T(y-mu) - lam*beta)
            
            hessian = X.T @ W @ X
            hessian_reg = hessian + lam * np.eye(p)
            hessian_reg[0, 0] -= lam # Do not regularize intercept
            
            try:
                delta = np.linalg.solve(hessian_reg, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hessian_reg, gradient, rcond=None)[0]
            
            self.beta += delta
            
            diff = np.linalg.norm(delta)
            self.history.append(diff)
            
            if diff < self.tol:
                break
                
        return self
    
    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return self.sigmoid(X @ self.beta)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionIRLS:
    def __init__(self, max_iter=100, tol=1e-6, C=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.beta = None
        self.history = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        n, p = X.shape
        self.beta = np.zeros(p)
        lam = 1.0 / self.C
        
        for iteration in range(self.max_iter):
            eta = X @ self.beta
            mu = self.sigmoid(eta)
            
            w = np.maximum(mu * (1 - mu), 1e-10)
            W = np.diag(w)
            
            # IRLS with regularization is equivalent to Newton-Raphson
            # We use the same update rule for stability
            
            reg_vec = lam * self.beta
            reg_vec[0] = 0
            
            gradient = X.T @ (y - mu) - reg_vec
            
            hessian = X.T @ W @ X
            hessian_reg = hessian + lam * np.eye(p)
            hessian_reg[0, 0] -= lam
            
            try:
                delta = np.linalg.solve(hessian_reg, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hessian_reg, gradient, rcond=None)[0]
            
            self.beta += delta
            
            diff = np.linalg.norm(delta)
            self.history.append(diff)
            
            if diff < self.tol:
                break
                
        return self
    
    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return self.sigmoid(X @ self.beta)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)