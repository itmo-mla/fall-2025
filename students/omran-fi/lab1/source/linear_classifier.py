from __future__ import annotations

import math
import numpy as np
from source.config import FitConfig


def margins_all(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector margins: y*(Xw). Returns shape (n,)."""
    return (X @ w).reshape(-1) * y.reshape(-1)


def quadratic_margin_loss(w: np.ndarray, x: np.ndarray, y: float) -> float:
    """
    L = (1 - m)^2,  m = y*(w^T x)
    w: (d,1), x: (d,1)
    """
    m = float(y * (w.T @ x)[0, 0])
    return float((1.0 - m) ** 2)


def quadratic_margin_grad(w: np.ndarray, x: np.ndarray, y: float) -> np.ndarray:
    """
    ∇w L = -2(1-m) y x
    returns (d,1)
    """
    m = float(y * (w.T @ x)[0, 0])
    return -2.0 * (1.0 - m) * float(y) * x


class LinearClassifier:
    def __init__(self, n_features: int):
        self.n_features = int(n_features)
        self.w: np.ndarray | None = None  # (d,1)
        self.v: np.ndarray | None = None  # (d,1)

        self.Q: float | None = None

        self.loss_history: list[float] = []
        self.Q_history: list[float] = []

    def init_random(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self.n_features)
        self.w = rng.uniform(-scale, scale, size=(self.n_features, 1)).astype(float)
        self.v = np.zeros_like(self.w)

    def init_with(self, w: np.ndarray) -> None:
        w = np.asarray(w, dtype=float).reshape(-1, 1)
        if w.shape[0] != self.n_features:
            raise ValueError("Wrong w shape for init.")
        self.w = w.copy()
        self.v = np.zeros_like(self.w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.w
        y_pred = np.sign(scores)
        y_pred[y_pred == 0] = 1
        return y_pred

    def _pick_index(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, cfg: FitConfig, step: int) -> int:
        """
        Stable margin sampling:
        - warmup: uniform sampling first margin_warmup steps
        - softmax on -|margin|/T  (small |margin| => higher prob)
        - mix with uniform: p = (1-mix)*p_soft + mix*uniform
        """
        n = X.shape[0]
        if not cfg.use_margin_sampling:
            return int(rng.integers(0, n))

        if getattr(cfg, "margin_warmup", 0) > 0 and step < int(cfg.margin_warmup):
            return int(rng.integers(0, n))

        m = margins_all(self.w, X, y)  # (n,)
        T = max(float(getattr(cfg, "margin_temperature", 1.0)), 1e-9)

        scores = -np.abs(m) / T
        scores = scores - float(np.max(scores))  # stability
        p = np.exp(scores)
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= 0:
            return int(rng.integers(0, n))
        p = p / s

        mix = float(np.clip(getattr(cfg, "margin_mix", 0.0), 0.0, 1.0))
        if mix > 0.0:
            p = (1.0 - mix) * p + mix * (np.ones_like(p) / len(p))
            p = p / float(np.sum(p))

        return int(rng.choice(np.arange(n), p=p))

    def fit(self, X: np.ndarray, y: np.ndarray, cfg: FitConfig) -> None:
        if self.w is None:
            self.init_random(cfg.seed)

        rng = np.random.default_rng(cfg.seed)

        # init Q on small subset
        if self.Q is None:
            idxs = rng.choice(np.arange(X.shape[0]), size=min(30, X.shape[0]), replace=False)
            losses = [
                quadratic_margin_loss(self.w, X[i].reshape(-1, 1), float(y[i, 0]))
                for i in idxs
            ]
            self.Q = float(np.mean(losses))

        # ensure v exists
        if self.v is None:
            self.v = np.zeros_like(self.w)

        for step in range(cfg.n_iter):
            i = self._pick_index(X, y, rng, cfg, step)
            x = X[i].reshape(-1, 1)
            yi = float(y[i, 0])

            loss = quadratic_margin_loss(self.w, x, yi)

            # -------- learning rate (russian fastest) --------
            lr = float(cfg.lr)
            if cfg.use_fastest_step:
                # lr = 1 / (||x||^2 + eps [+ l2])
                x2 = float((x.T @ x)[0, 0])
                eps = float(getattr(cfg, "fastest_eps", 1e-12))
                denom = x2 + eps

                if bool(getattr(cfg, "fastest_l2_in_denom", True)):
                    denom += float(cfg.l2)

                lr = 1.0 / denom

           
            g = quadratic_margin_grad(self.w, x, yi)

            # -------- update --------
            if cfg.use_momentum:
                if cfg.nesterov:
                    # v = gamma*v + (1-gamma)*grad(w - lr*gamma*v)
                    w_look = self.w - lr * float(cfg.gamma) * self.v
                    g_look = quadratic_margin_grad(w_look, x, yi)

                    self.v = float(cfg.gamma) * self.v + (1.0 - float(cfg.gamma)) * g_look
                    self.w = self.w * (1.0 - lr * float(cfg.l2)) - lr * self.v
                else:
                    # classical momentum (also russian-style scaling)
                    self.v = float(cfg.gamma) * self.v + (1.0 - float(cfg.gamma)) * g
                    self.w = self.w * (1.0 - lr * float(cfg.l2)) - lr * self.v
            else:
                # plain SGD with weight decay
                self.w = self.w * (1.0 - lr * float(cfg.l2)) - lr * g

            # recursive quality functional Q
            self.Q = float(cfg.lambda_q) * float(loss) + (1.0 - float(cfg.lambda_q)) * float(self.Q)

            self.loss_history.append(float(loss))
            self.Q_history.append(float(self.Q))
