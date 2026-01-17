from typing import Optional, Protocol

import numpy as np

from core.loss import LossFunction
from core.regularizer import Regularizer


class OptimizerContext(Protocol):
    X_batch: Optional[np.ndarray]
    y_batch: Optional[np.ndarray]
    loss_fn: Optional[LossFunction]
    regularizer: Optional[Regularizer]
    current_loss: Optional[float]


class Optimizer:
    def update(self, w, b, grad_w, grad_b, ctx: Optional[OptimizerContext] = None): raise NotImplementedError

    def reset(self): raise NotImplementedError


class SGDWithMomentum(Optimizer):
    def __init__(self, lr: float = 1e-4, momentum: float = 0.2):
        self.lr = lr
        self.momentum = momentum
        self.v_w = None
        self.v_b = None

    def update(self, w, b, grad_w, grad_b, ctx: Optional[OptimizerContext] = None):
        if self.v_w is None:
            self.v_w = np.zeros_like(w)
            self.v_b = 0.0
        self.v_w = self.momentum * self.v_w - self.lr * grad_w
        self.v_b = self.momentum * self.v_b - self.lr * grad_b
        return w + self.v_w, b + self.v_b

    def reset(self):
        self.v_w = None
        self.v_b = None


class SteepestDescent(Optimizer):
    def __init__(self, alpha_init=1.0, beta=0.5, c=1e-4, max_iters=20):
        self.alpha_init = alpha_init
        self.beta = beta
        self.c = c
        self.max_iters = max_iters

    def update(self, w, b, grad_w, grad_b, ctx: Optional[OptimizerContext] = None):
        if ctx is None or ctx.loss_fn is None: return w - self.alpha_init * grad_w, b - self.alpha_init * grad_b

        d_w = -grad_w
        d_b = -grad_b
        grad_norm_sq = np.dot(grad_w, grad_w) + grad_b ** 2
        alpha = self.alpha_init

        for _ in range(self.max_iters):
            w_new = w + alpha * d_w
            b_new = b + alpha * d_b
            X = ctx.X_batch
            if X.ndim == 1: X = X.reshape(1, -1)
            loss_new = ctx.loss_fn.compute(ctx.y_batch, X @ w_new + b_new)
            if ctx.regularizer: loss_new += ctx.regularizer.penalty(w_new)
            if loss_new <= ctx.current_loss - self.c * alpha * grad_norm_sq: return w_new, b_new
            alpha *= self.beta
        return w + alpha * d_w, b + alpha * d_b

    def reset(self): pass
