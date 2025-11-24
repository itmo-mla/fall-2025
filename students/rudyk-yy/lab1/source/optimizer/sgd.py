from optimizer.optimizer import Optimizer
import numpy as np

class SgdOptimizer(Optimizer):
    def __init__(self, gamma=0.9):
        self.gamma = gamma

    def update(self, weights, gradient, learning_rate):
        weights -= learning_rate * gradient(weights)
        return weights