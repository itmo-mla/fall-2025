import numpy as np
from typing import Optional

class L2Regularizer:
    def __init__(self, reg_strength: float = 1e-3):
        self.reg_strength = float(reg_strength)

    def __call__(self, w: np.ndarray) -> float:
        return self.reg_strength * np.sum(w ** 2)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 2.0 * self.reg_strength * w

class RecurrentLogisticLoss:
    def __init__(self, smoothing: float = 1e-3, regularizer: Optional[L2Regularizer] = None):
        self.smoothing = float(smoothing)
        self.regularizer = regularizer
        self.history = []          # batch-level smoothed loss history
        self.epoch_train_loss = [] # epoch-level train loss
        self.val_history = []      # epoch-level val loss

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true, y_pred are 1d arrays
        z = y_true * y_pred
        # stable logistic: log(1 + exp(-z))
        return float(np.log1p(np.exp(-z)).mean())

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dL/df = -y * sigmoid(-y*f); return per-sample gradients averaged over batch
        n = y_true.shape[0]
        z = y_true * y_pred
        denom = 1.0 + np.exp(z)  # 1 + exp(y * f)
        # sigmoid(-y*f) = 1/(1+exp(y*f)) = 1/denom
        grad_scores = (-y_true) / denom  # shape (n,)
        grad_scores = grad_scores.reshape(-1, 1) / n
        return grad_scores  # shape (n,1)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        batch_loss = self(y_true, y_pred)
        if self.history:
            batch_weight = max(min(self.smoothing * y_true.shape[0], 1.0), self.smoothing)
            current_loss = batch_weight * batch_loss + (1.0 - batch_weight) * self.history[-1]
        else:
            current_loss = batch_loss

        if self.regularizer is not None and weights is not None:
            current_loss += self.regularizer(weights)

        self.history.append(current_loss)
        return current_loss
