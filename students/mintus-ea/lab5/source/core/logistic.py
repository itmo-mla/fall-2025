import numpy as np

class LogisticRegression:
    def __init__(self, method='newton', max_iter=100, tol=1e-4, lambda_reg=0.0):
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.w = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        X = self._add_intercept(X)
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        
        for i in range(self.max_iter):
            z = np.dot(X, self.w)
            p = self._sigmoid(z)
            
            # Gradient: X^T * (p - y) + lambda * w
            grad = np.dot(X.T, (p - y)) 
            grad[1:] += self.lambda_reg * self.w[1:]
            grad /= n_samples
            
            # Hessian: X^T * W * X + lambda * I
            W_diag = p * (1 - p)
            W_diag = np.maximum(W_diag, 1e-5)
            
            H = np.dot(X.T * W_diag, X)
            
            # Add regularization
            reg_matrix = np.eye(n_features) * self.lambda_reg
            reg_matrix[0, 0] = 0
            H += reg_matrix
            
            H /= n_samples
            
            # Update: w_new = w_old - H^-1 * grad
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
                
            self.w -= delta
            
            # Log Loss
            p_safe = np.clip(p, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))
            loss += (self.lambda_reg / (2 * n_samples)) * np.sum(self.w[1:]**2)
            self.loss_history.append(loss)
            
            if np.linalg.norm(delta) < self.tol:
                break
                
        return self

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self._sigmoid(np.dot(X, self.w))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
