import numpy as np
from scipy.optimize import minimize
from .kernels import linear_kernel

class SVM:
    def __init__(self, kernel='linear', C=1.0, **kernel_params):
        self.kernel_name = kernel
        self.C = C
        self.kernel_params = kernel_params
        self.kernel_func = self._get_kernel(kernel)
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.X_train = None

    def _get_kernel(self, name):
        if name == 'linear':
            return linear_kernel
        elif callable(name):
            return name
        from .kernels import rbf_kernel, polynomial_kernel
        if name == 'rbf':
            return lambda x, y: rbf_kernel(x, y, **self.kernel_params)
        elif name == 'poly':
            return lambda x, y: polynomial_kernel(x, y, **self.kernel_params)
        return linear_kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X_train = X
        # Compute Gram matrix
        K = self.kernel_func(X, X)

        P = np.outer(y, y) * K
        
        def objective(alpha):
            return 0.5 * np.sum(np.outer(alpha, alpha) * P) - np.sum(alpha)
        
        def objective_grad(alpha):
            return np.dot(P, alpha) - np.ones(n_samples)

        # Optimization constraints
        # 1. sum(alpha_i * y_i) = 0
        constraints = ({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y})
        
        # 2. 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Initial guess
        alpha0 = np.zeros(n_samples)
        
        # Solve
        # Using trust-constr or SLSQP for constrained optimization
        res = minimize(lambda a: 0.5 * np.dot(a, np.dot(P, a)) - np.sum(a), 
                       alpha0, 
                       method='SLSQP', 
                       bounds=bounds, 
                       constraints=constraints,
                       jac=objective_grad)
        
        self.alpha = res.x
        
        # Support vectors have non-zero lagrange multipliers
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]
        
        # Calculate intercept b
        # b = mean(y_k - sum(alpha_i * y_i * K(x_i, x_k))) for support vectors
        if len(self.alpha) > 0:
            # Recalculate K for SVs
            K_sv = self.kernel_func(self.support_vectors, self.support_vectors)
            
            # Prediction part without bias for SVs: sum(alpha_i * y_i * K(x_i, x_k))
            decision_values = np.dot(self.alpha * self.support_vector_labels, K_sv) # (N_sv,)
            
            self.b = np.mean(self.support_vector_labels - decision_values)
        else:
            self.b = 0
            
        # For linear kernel, we can compute w directly
        if self.kernel_name == 'linear':
            self.w = np.dot(self.alpha * self.support_vector_labels, self.support_vectors)
        
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X)) + self.b
            
        K_pred = self.kernel_func(X, self.support_vectors)
        return np.dot(K_pred, self.alpha * self.support_vector_labels) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
