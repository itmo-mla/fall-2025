import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod


class OptimizerContext:
    def __init__(
            self,
            X_batch: Optional[np.ndarray] = None,
            y_batch: Optional[np.ndarray] = None,
            loss_fn: Optional[Callable] = None,
            reg_fn: Optional[Callable] = None,
            bias: Optional[np.ndarray] = None,
    ):
        self.X_batch = X_batch
        self.y_batch = y_batch
        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.bias = bias


class Optimizer(ABC):
    @abstractmethod
    def update(self, name: str, param: np.ndarray, grad: np.ndarray,
               ctx: Optional[OptimizerContext] = None) -> np.ndarray:
        pass


class MomentumOptimizer(Optimizer):
    def __init__(self, lr: float = 1e-4, momentum: float = 0.5):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, name: str, param: np.ndarray, grad: np.ndarray,
               ctx: Optional[OptimizerContext] = None) -> np.ndarray:
        v = self.velocity.get(name, np.zeros_like(param))
        v = self.momentum * v + self.lr * grad
        self.velocity[name] = v
        return param - v


class FastOptimizer(Optimizer):
    def __init__(self, lr: float = 1e-4, n_search_steps: int = 10):
        self.lr = lr
        self.n_search_steps = n_search_steps

    def update(self, name: str, param: np.ndarray, grad: np.ndarray,
               ctx: Optional[OptimizerContext] = None) -> np.ndarray:
        if name != "weights" or ctx is None:
            return param - self.lr * grad

        if ctx.X_batch is None or ctx.y_batch is None or ctx.loss_fn is None:
            return param - self.lr * grad

        best_loss = float('inf')
        best_param = param.copy()

        lrs = self.lr * np.logspace(-2, 2, self.n_search_steps)
        for lr_try in lrs:
            trial = param - lr_try * grad
            trial_scores = (ctx.X_batch @ trial + ctx.bias).ravel()
            trial_loss = ctx.loss_fn(ctx.y_batch, trial_scores)
            if ctx.reg_fn is not None:
                trial_loss += ctx.reg_fn(trial)

            if trial_loss < best_loss:
                best_loss = trial_loss
                best_param = trial.copy()

        return best_param
