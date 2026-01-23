import numpy as np


class Optimizer:
    def step(self, weights: np.ndarray, grad: np.ndarray):
        raise NotImplementedError("Please Implement this method")


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, weights, grad_fn):
        return weights - self.lr * grad_fn(weights)
