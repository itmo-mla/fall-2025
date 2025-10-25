import numpy as np

class HingeLoss():
    def __call__(
        self, 
        margins: np.ndarray
    )-> np.ndarray:
        """Метод, вычисляющий функцию потерь"""
        return np.sum(np.maximum(0, 1 - margins)) / len(margins)
        
    def get_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    )-> np.ndarray:
        """Метод, вычисляющий градиент по w"""
        return -(X[mask].T @ y[mask]) / len(y)

    def get_optimal_h(
        self,
        X: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
    )-> float:  
        """Метод, вычисляющий градиент по h"""
        margin = 1 - y * np.dot(X, w)
        s = np.dot(X, X)
        if margin > 0 and s > 0:
            return margin / s
        else:
            return 0.0