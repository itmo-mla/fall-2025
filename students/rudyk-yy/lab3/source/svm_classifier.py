from scipy.optimize import minimize
import numpy as np

class SvmClassifier:
    def __init__(self, X,y, kernel='linear', C=1.0,d =3):
        self.X = X
        self.d=d
        self.y = y
        self.kernel = kernel
        self.C = C
    def _objective_function(self, alpha, kernel_matrix):
        alpha = np.array(alpha)
        y = np.array(self.y)
        kernel_matrix = np.array(kernel_matrix)
        bigsum = np.dot(alpha * y, np.dot(kernel_matrix, alpha * y))
        return -np.sum(alpha) + 0.5 * bigsum
    def _objective_gradient(self, alpha, kernel_matrix):
        alpha = np.array(alpha)
        y = np.array(self.y)
        kernel_matrix = np.array(kernel_matrix)
        grad = -np.ones_like(alpha) + np.dot(kernel_matrix, alpha * y) * y
        return grad
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return sum(a*b for a, b in zip(x1, x2))
        elif self.kernel == 'constant':
            return 1
        elif self.kernel == 'polynomial':
            return (sum(a*b for a, b in zip(x1, x2)) + 1) ** self.d
        elif self.kernel == 'rbf':
            diff = np.array(x1) - np.array(x2)
            return np.exp(-np.dot(diff, diff))
        elif self.kernel == 'custom':
            a = sum(a*b for a, b in zip(x1, x2))
            return a**3 + 2*a**2 + 3*a + 1
    
    def _compute_kernel_matrix(self):
        n_samples = len(self.X)
        kernel_matrix = [[0]*n_samples for _ in range(n_samples)]
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i][j] = self._kernel_function(self.X[i], self.X[j])
                
        return kernel_matrix
    def fit(self):
        n_samples = len(self.X)
        initial_alpha = [0.0] * n_samples
        bounds = [(0, self.C) for _ in range(n_samples)]
        kernel_matrix = self._compute_kernel_matrix()
        constraints = {'type': 'eq', 'fun': lambda alpha: sum(alpha[i] * self.y[i] for i in range(n_samples))}
        result = minimize(fun=self._objective_function,
                          x0=initial_alpha,
                          args=(kernel_matrix,),
                          bounds=bounds,
                          constraints=constraints,
                            method='SLSQP',
                            jac=self._objective_gradient

                        )
        self.alpha = result.x
        support_indices = [i for i in range(n_samples) if self.alpha[i] > 1e-5]
        if not support_indices:
            self.b = 0
        else:
            b_sum = 0
            for i in support_indices:
                decision = sum(self.alpha[j] * self.y[j] * self._kernel_function(self.X[j], self.X[i]) for j in range(n_samples))
                b_sum += self.y[i] - decision
            self.b = b_sum / len(support_indices)
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            decision_value = 0
            for i in range(len(self.X)):
                k = self._kernel_function(self.X[i], x)
                decision_value += self.alpha[i] * self.y[i] * k 
            decision_value += self.b
            predictions.append(1 if decision_value >= 0 else -1)
        return predictions

        
