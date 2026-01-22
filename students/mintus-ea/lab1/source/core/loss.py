import numpy as np

class Loss:
    def __init__(self):
        pass
    
    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def gradient(self, y_true, y_pred, X):
        raise NotImplementedError

class QuadraticLoss(Loss):
    def loss(self, y_true, y_pred):
        # y_true in {-1, 1}
        return (y_true - y_pred) ** 2

    def gradient(self, y_true, y_pred, X):
        # dL/dw = 2 * (y_pred - y_true) * X
        # Returns gradient for a single sample or batch
        if len(X.shape) == 1:
            return 2 * (y_pred - y_true) * X
        else:
            return 2 * (y_pred - y_true)[:, np.newaxis] * X
