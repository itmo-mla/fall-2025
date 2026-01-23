import numpy as np


class Loss:
    def loss(self):
        raise NotImplementedError("Please Implement this method")

    def derivative(self):
        raise NotImplementedError("Please Implement this method")


class LogLoss(Loss):
    def loss(self, x):
        return np.log(1 + np.exp(-x))

    def derivative(self, x):
        return -1 / (1 + np.exp(x))
