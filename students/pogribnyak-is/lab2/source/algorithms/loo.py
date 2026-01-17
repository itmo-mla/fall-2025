import numpy as np
from typing import List, Tuple
from model.knn import KNN


class LeaveOneOut:
    def __init__(self, k_range: List[int]):
        self.k_range = k_range
        self.errors: List[float] = []
        self.best_k: int = 1
        self.best_error: float = float('inf')
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, List[float]]:
        n_samples = len(X)
        errors_by_k = []
        
        iterator = self.k_range
        
        for k in iterator:
            errors = []
            for i in range(n_samples):
                X_train_loo = np.delete(X, i, axis=0)
                y_train_loo = np.delete(y, i)
                
                X_test_loo = X[i:i+1]
                y_test_loo = y[i]
                
                knn = KNN(k=k)
                knn.fit(X_train_loo, y_train_loo)
                prediction = knn.predict(X_test_loo)[0]
                
                error = 1 if prediction != y_test_loo else 0
                errors.append(error)

            empirical_risk = np.mean(errors)
            errors_by_k.append(empirical_risk)
        
        best_idx = np.argmin(errors_by_k)
        self.best_k = self.k_range[best_idx]
        self.best_error = errors_by_k[best_idx]
        self.errors = errors_by_k
        
        return self.best_k, self.best_error, errors_by_k
    
    def get_best_k(self) -> int:
        if self.best_k is None: raise ValueError("LOO еще не выполнен. Вызовите evaluate() сначала.")
        return self.best_k

