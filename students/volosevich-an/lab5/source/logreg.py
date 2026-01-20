import numpy as np


class LogisticRegression:
    def __init__(self, max_iter=1000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit_newton_raphson(self, X, y):
        n_samples, n_features = X.shape
        
        X_design = np.c_[np.ones(n_samples), X]
        theta = np.zeros(n_features + 1)
        
        for i in range(self.max_iter):
            linear_pred = np.dot(X_design, theta)
            p = self._sigmoid(linear_pred)
            
            # Градиент
            gradient = np.dot(X_design.T, (p - y))

            # p*(1-p)
            W = p * (1 - p)
            #  X^T * W * X
            hessian = X_design.T @ (W[:, None] * X_design)
            
            # Регуляризация 
            hessian += 1e-8 * np.eye(hessian.shape[0])
            
            # H * Δθ = -∇J
            delta_theta = np.linalg.solve(hessian, -gradient)
            theta += delta_theta

            # Досрочно завершаем алгоритм при достижении сходимости
            if np.linalg.norm(delta_theta) < self.tol:
                print(f"Convergence achieved at iter {i}")
                break
        
        self.bias = theta[0]
        self.weights = theta[1:]

    def fit_irls(self, X, y):
        n_samples, n_features = X.shape
        X_design = np.c_[np.ones(n_samples), X]
        theta = np.zeros(n_features + 1)
        
        for i in range(self.max_iter):
            eta = np.dot(X_design, theta) 
            p = self._sigmoid(eta)
            
            # Приближение отклика
            z = eta + (y - p) / (p * (1 - p) + 1e-8)
            W = p * (1 - p)
            
            # Считаем взвешенную линейную регрессию
            W_sqrt = np.sqrt(W)
            X_weighted = W_sqrt[:, None] * X_design
            z_weighted = W_sqrt * z
            
            # (X^T W X)θ = X^T W z
            theta_new = np.linalg.lstsq(X_weighted, z_weighted, rcond=None)[0]
            
            # Досрочно завершаем алгоритм при достижении сходимости
            if np.linalg.norm(theta_new - theta) < self.tol:
                print(f"Convergence achieved at iter {i}")
                break
            theta = theta_new
        
        self.bias = theta[0]
        self.weights = theta[1:]

    # Возвращаем вероятности
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    # Предскзание вида 0 / 1
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    