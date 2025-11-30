from optimizer.optimizer import Optimizer
import numpy as np


class Nesterov(Optimizer):
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.v = None

    def update(self, weights, gradient, learning_rate):
        if self.v is None:
            self.v = np.zeros_like(weights)
        self.v = self.gamma * self.v + (1 - self.gamma) * gradient(weights - self.gamma * self.v)
        weights -= learning_rate * self.v
        return weights