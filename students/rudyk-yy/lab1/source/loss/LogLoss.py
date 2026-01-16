from loss.Loss import Loss
import numpy as np

class LogLoss(Loss):
    def loss(self, X):
        return np.log2(1 + np.exp(-X))

    def derivative(self, X):
        return -1 / (np.log(2) * (1 + np.exp(X)))