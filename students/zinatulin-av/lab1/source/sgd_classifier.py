import numpy as np
from typing import Callable
from tqdm import tqdm
import random
from functools import cmp_to_key
from source.optimizers import SGDOptimizer
from source.losses import LogLoss

class SimpleSGDClassifier:
  def __init__(self, optimizer = SGDOptimizer(), loss = LogLoss(), penalty='l2', alpha=0.01, max_iter=100, forget_rate=0.01, ordering='random', weight_init='random'):
    """
      penalty: None | 'l2'
      ordering: 'random' | 'large_margin_first'
      weight_init: 'random' | 'corr' | 'multistart'
    """
    self.optimizer = optimizer
    self.penalty = penalty
    self.alpha = alpha
    self.max_iter = max_iter
    self.loss = loss
    self.forget_rate = forget_rate
    self.weight_init = weight_init
    self.ordering = ordering
    self.w = np.array([])
    self.w0 = 0
    self.Q = 0

  def fit(self, X, y, loss_callback):
    match self.weight_init:
      case 'random':
        self.w = np.random.normal(0, 2/X.shape[1], X.shape[1])
      case 'corr':
        self.w = (X.T @ y) / np.sum(X**2, axis=0)
      case 'multistart':
        models = [SimpleSGDClassifier(max_iter=1) for _ in range(10)]
        best_params = models[0].w
        best_Q = float('inf')
        for model in models:
          model.fit(X, y, None)
          if model.Q < best_Q:
            best_params = model.w
            best_Q = model.Q
        self.w = best_params
      case _:
        raise NotImplementedError('Incorrect ordering param')
    self.w0 = 0
    _xy = list(zip(X, y))

    random.shuffle(_xy)
    self.Q = 0
    for x, y in _xy:
      self.Q += self.loss.loss(x @ self.w + self.w0) / len(_xy)

    for epoch in tqdm(range(self.max_iter)):

      match self.ordering:
        case 'random':
          random.shuffle(_xy)
        case 'large_margin_first':
          def compare(a, b):
            return (b[0] @ self.w + self.w0) * y - (a[0] @ self.w + self.w0) * y
          _xy = sorted(_xy, key=cmp_to_key(compare))
        case _:
          raise NotImplementedError('Incorrect ordering param')

      for x, y in _xy:

        match self.penalty:
          case 'l2':
            self.w = self.optimizer.step(self.w * (1 - self.optimizer.lr * self.alpha), lambda w: self.loss.derivative((x @ w + self.w0) * y) * x * y + self.alpha * w)
          case None:
            self.w = self.optimizer.step(self.w, lambda w: self.loss.derivative((x @ w + self.w0) * y) * x * y)
          case _:
            raise NotImplementedError('Incorrect penalty param')

        self.w0 = self.optimizer.step(self.w0, lambda w0: self.loss.derivative((x @ self.w + w0) * y) * y)
        self.Q = self.forget_rate * self.loss.loss((x @ self.w + self.w0) * y) + (1 - self.forget_rate) * self.Q;

        if loss_callback:
          loss_callback(self.Q)

  def predict(self, X):
    def sign(x):
      return x / np.abs(x)
    return sign(X @ self.w + self.w0)
