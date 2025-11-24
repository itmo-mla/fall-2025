from optimizer.optimizer import Optimizer
import numpy as np
class Momentum():
    def __init__(self, gamma=0.5):
        self.gamma = gamma
        self.v = None

    def update(self, weights, grad, learning_rate):
        # grad — это уже np.ndarray
        if self.v is None:
            self.v = np.zeros_like(weights)
        self.v = self.gamma * self.v + (1 - self.gamma) * grad(weights)
        return weights - learning_rate * self.v