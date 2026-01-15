import numpy as np
from typing import Optional
from regularization import L2Regularizer


class RecurrentLogisticLoss:
    def __init__(self, smoothing: float = 1e-3, regularizer: Optional[L2Regularizer] = None):
        self.smoothing = float(smoothing)
        self.regularizer = regularizer
        self.history = []

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        z = y_true * y_pred
        return float(np.log1p(np.exp(-z)).mean())

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        z = y_true * y_pred
        grad_scores = (-y_true) / (1.0 + np.exp(z))
        return grad_scores.reshape(-1, 1) / n

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        batch_loss = self(y_true, y_pred)

        current_loss = batch_loss
        if self.history:
            weight = max(min(self.smoothing * y_true.shape[0], 1.0), self.smoothing)
            current_loss = weight * batch_loss + (1.0 - weight) * self.history[-1]

        if self.regularizer is not None:
            current_loss += self.regularizer(weights)

        self.history.append(current_loss)
        return current_loss
