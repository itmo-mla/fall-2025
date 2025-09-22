import numpy as np

class Optimizer():
  def __init__(self):
    pass

  def step(weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Please Implement this method")


class SGDOptimizer(Optimizer):
  def __init__(self, learning_rate=0.01):
    self.lr = learning_rate

  def step(self, weights: np.ndarray, grad_fn: np.ndarray) -> np.ndarray:
    return weights - self.lr * grad_fn(weights)


class MomentumOptimizer(Optimizer):
  """ Метод моментов (Поляк) """
  def __init__(self, learning_rate=0.01, momentum=0.5):
    self.lr = learning_rate
    self.momentum = momentum

  def step(self, weights: np.ndarray, grad_fn) -> np.ndarray:
    return weights - self.lr * ((self.momentum * weights) + (1 - self.momentum) * grad_fn(weights))


class NAGOptimizer(Optimizer):
  """ Nesterov Accelerated Gradient """
  def __init__(self, learning_rate=0.01, momentum=0.5):
    self.lr = learning_rate
    self.momentum = momentum

  def step(self, weights: np.ndarray, grad_fn) -> np.ndarray:
    return weights - self.lr * ((self.momentum * weights) + (1 - self.momentum) * grad_fn(weights * self.momentum * self.lr))

