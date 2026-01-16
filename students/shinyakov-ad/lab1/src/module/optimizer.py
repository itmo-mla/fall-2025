from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    NAME = None

    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def step(self, weights: np.array, gradient: np.array):
        raise NotImplementedError("Optimizer step function is not implemented")

    def __call__(self, weights, gradient):
        return self.step(weights, gradient)

class SGDOptimizer(BaseOptimizer):
    NAME = "sgd"

    def __init__(self, learning_rate = 0.001, gamma = 0.9):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.velocity = 0

    def step(self, weights, grad_func):
        self.velocity = (1 - self.gamma) * grad_func(weights) + self.gamma * self.velocity
        return weights - self.learning_rate * self.velocity

class NAGOptimizer(BaseOptimizer):
    NAME = "nag"

    def __init__(self, learning_rate=0.001, gamma=0.9):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.velocity = 0

    def step(self, weights, grad_func):
        lookahead_weights = weights - self.learning_rate * self.gamma * self.velocity
        grad = grad_func(lookahead_weights)
        self.velocity = self.gamma * self.velocity + (1 - self.gamma) * grad
        return weights - self.learning_rate * self.velocity

OPTIMIZERS = {
    v.NAME: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, BaseOptimizer) and v is not BaseOptimizer
}