import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist

class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma=0.1, degree=3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.lambdas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0
        
    def _kernel_function(self, X, Y=None):
        if Y is None:
            Y = X
            
        match self.kernel: 
            case 'linear':
                return X @ Y.T
            case 'rbf':
                pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
                return np.exp(-self.gamma * pairwise_sq_dists)
            case _:
                raise ValueError(f"Неизвестное ядро: {self.kernel}")
    
    def _objective_function(self, lambdas, K, y):
        term1 = -np.sum(lambdas)
        
        term2 = 0.5 * (lambdas * y) @ K @ (lambdas * y)
        
        return term1 + term2
    
    def _objective_gradient(self, lambdas, K, y):
        return -1 + y * (K @ (lambdas * y))
    
    def fit(self, X, y):
        n_samples, _ = X.shape
        
        K = self._kernel_function(X)
        
        initial_lambdas = np.zeros(n_samples)
        
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        constraints = LinearConstraint(y.reshape(1, -1), lb=0, ub=0)
        
        result = minimize(
            fun=self._objective_function,
            x0=initial_lambdas,
            args=(K, y),
            method='SLSQP',
            jac=self._objective_gradient,
            bounds=bounds,
            constraints=[constraints],
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        self.lambdas = result.x
        
        support_vector_indices = np.where(self.lambdas > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_lambdas = self.lambdas[support_vector_indices]
        
        margin_indices = np.where((self.lambdas > 1e-5) & (self.lambdas < self.C - 1e-5))[0]
        
        if len(margin_indices) > 0:
            b_values = []
            for idx in margin_indices:
                x_k = X[idx].reshape(1, -1)
                K_ik = self._kernel_function(self.support_vectors, x_k).flatten()
                
                f_xk = np.sum(self.support_lambdas * self.support_vector_labels * K_ik)
                b_k = f_xk - y[idx]
                b_values.append(b_k)
            
            self.b = np.mean(b_values)
        else:
            b_values = []
            for i in range(len(self.support_vectors)):
                x_k = self.support_vectors[i].reshape(1, -1)
                K_ik = self._kernel_function(self.support_vectors, x_k).flatten()
                
                f_xk = np.sum(self.support_lambdas * self.support_vector_labels * K_ik)
                b_k = f_xk - self.support_vector_labels[i]
                b_values.append(b_k)
            
            self.b = np.mean(b_values)

        if self.kernel == 'linear':
            self.coef_ = (self.support_lambdas * self.support_vector_labels) @ self.support_vectors
            self.coef_ = self.coef_.reshape(1, -1)
        else:
            self.coef_ = None
    
    def predict(self, X):
        K = self._kernel_function(X, self.support_vectors)
        
        decision_values = np.sum(
            (self.support_lambdas * self.support_vector_labels) * K, 
            axis=1
        ) - self.b
        
        return np.sign(decision_values).astype(int)
    
    def decision_function(self, X):
        K = self._kernel_function(X, self.support_vectors)
        return np.sum(
            (self.support_lambdas * self.support_vector_labels) * K, 
            axis=1
        ) - self.b