from typing import Optional
import numpy as np

from .regularization import Regularization


class BinaryLoss:
    def __init__(self, loss_lambda: float = 1e-3, regularization: Optional[Regularization] = None):
        self.loss_lambda = loss_lambda
        self.reg = regularization

        self.history = []
        self.val_history = []

    def calc(self, y_true: np.ndarray, y_pred: np.ndarray):
        return (y_pred * y_true < 0).mean()

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.expand_dims(np.sign(y_pred) * (y_pred * y_true < 0), axis=-1) / y_true.shape[0]

    def get_loss(self, y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None):
        if self.history:
            batch_coef = max(min(self.loss_lambda * y_true.shape[0], 1.0), self.loss_lambda)
            new_loss = batch_coef * self.calc(y_true, y_pred) + (1 - batch_coef) * self.history[-1]
        else:
            new_loss = self.calc(y_true, y_pred)

        if self.reg is not None:
            if weights is None:
                raise ValueError("Provide model weights for regularization")
            new_loss += self.reg(weights)

        self.history.append(new_loss)
        return new_loss

    def val_loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        loss = self.calc(y_true, y_pred)
        self.val_history.append(loss)
        return loss

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self.calc(y_true, y_pred)
