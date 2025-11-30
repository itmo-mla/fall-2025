from abc import ABC, abstractmethod
import numpy as np
from .regularization import REGULARIZATIONS
from .margin import BaseMargin

class BaseLoss(ABC):
    NAME = None

    def __init__(self, regularization: str = None, reg_coef = 0.01):
        self.regularization = REGULARIZATIONS[regularization](reg_coef) if regularization is not None else None
        self.history = []

    def calculate_loss(self, predictions, true_label, weights):
        loss = self.__calculate__(predictions, true_label)

        if (self.regularization is not None):
            loss += self.regularization(weights)
        
        self.history.append(loss)

        return loss
    
    def calculate_derivative(self, predictions, true_labels, weights, X_train_thing):
        base_grad = self.__derivative__(predictions, true_labels, weights, X_train_thing)
        if self.regularization is not None:
            base_grad += self.regularization.derivative(weights)
        return base_grad

        
    @abstractmethod
    def __calculate__(self, predictions, true_labels):
        raise NotImplementedError("Loss calculation is not implemented")
    
    @abstractmethod
    def __derivative__(self, predictions, true_labels, weights, X):
        raise NotImplementedError("Loss calculation is not implemented")
    
    def __call__(self, prediction, true_label, weights):
        return self.calculate_loss(prediction, true_label, weights)

class MSE(BaseLoss):
    NAME = "mse"

    def __init__(self, regularization=None, reg_coef=0.01, margin: BaseMargin=None):
        super().__init__(regularization=regularization, reg_coef=reg_coef)
        self.margin = margin

    def __calculate__(self, predictions, true_labels):
        loss = predictions - true_labels
        return np.mean(loss ** 2)

    def __derivative__(self, predictions, true_labels, weights, X):
        return 2 * (predictions - true_labels) / len(predictions)

class HingeLoss(BaseLoss):
    NAME = "hinge"

    def __init__(self, regularization=None, reg_coef=0.01, margin: BaseMargin=None):
        super().__init__(regularization=regularization, reg_coef=reg_coef)
        self.margin = margin

    def __calculate__(self, predictions, true_labels):
        loss = np.maximum(0, 1 - self.margin.calculate(predictions, true_labels))
        return np.mean(loss)

    def __derivative__(self, predictions, true_labels, weights, X):
        margin = self.margin.calculate(predictions, true_labels)
        mask = (margin < 1).astype(float)
        return -np.mean((true_labels * mask)[:, np.newaxis] * X, axis=0)
    
LOSSES = {
    v.NAME: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, BaseLoss) and v is not BaseLoss
}