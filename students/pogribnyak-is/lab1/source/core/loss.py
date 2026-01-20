import numpy as np


class LossFunction:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: raise NotImplementedError

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float): raise NotImplementedError

    def gradient_single(self, x: np.ndarray, y: float, w: np.ndarray, b: float): raise NotImplementedError


class HingeLoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        margins = 1 - y_true * y_pred
        return np.mean(np.maximum(0.0, margins))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
        y_pred = X @ w + b
        mask = (1 - y * y_pred) > 0  # активные объекты (нарушают margin)

        if not np.any(mask):
            return np.zeros_like(w), 0.0

        X_active = X[mask]
        y_active = y[mask]

        grad_w = -np.mean(y_active[:, None] * X_active, axis=0)
        grad_b = -np.mean(y_active)

        return grad_w, grad_b

    def gradient_single(self, x: np.ndarray, y: float, w: np.ndarray, b: float):
        margin = y * (np.dot(w, x) + b)
        if margin >= 1:
            return np.zeros_like(w), 0.0
        return -y * x, -y



