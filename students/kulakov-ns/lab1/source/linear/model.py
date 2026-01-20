from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .init import init_correlation, init_multistart, init_random


@dataclass
class NAGOptimizer:
    weights: np.ndarray
    learning_rate: float
    l2_reg_param: float
    gamma: float

    def __post_init__(self) -> None:
        self.v = np.zeros_like(self.weights)

    def step(self, grad: np.ndarray) -> None:
        self.v = self.gamma * self.v + (1.0 - self.gamma) * grad
        self.weights -= self.learning_rate * grad


InitMethod = Literal["random", "corr", "multistart"]
BatchGeneration = Literal["random", "margin"]
PredictMode = Literal["class", "probability"]


class LinearClassifier:
    def __init__(
        self,
        learning_rate: float = 0.1,
        l2_reg_param: float = 1.0,
        Q_param: float = 0.001,
        gamma: float = 0.9,
        *,
        random_state: int | None = None,
    ):
        self.learning_rate = float(learning_rate)
        self.l2_reg_param = float(l2_reg_param)
        self.Q_param = float(Q_param)
        self.gamma = float(gamma)

        self.weights: np.ndarray | None = None
        self.Q: float | None = None

        self._rng = np.random.default_rng(random_state)

    # initialization
    def _init_weights(
        self,
        n_features: int,
        weights_init_method: InitMethod,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        n_attempts: int | None = None,
    ) -> None:
        if weights_init_method == "random":
            self.weights = init_random(n_features, self._rng)
            return

        if weights_init_method == "corr":
            if X is None or y is None:
                raise ValueError("X and y are required for 'corr' initialization")
            self.weights = init_correlation(X, y)
            return

        if weights_init_method == "multistart":
            if X is None or y is None:
                raise ValueError("X and y are required for 'multistart' initialization")
            if not n_attempts:
                raise ValueError("n_attempts must be provided for 'multistart' initialization")

            def score_fn(w: np.ndarray) -> float:
                return float(np.mean([self._loss_by_sample(w, xi, int(yi)) for xi, yi in zip(X, y)]))

            self.weights = init_multistart(n_features, attempts=int(n_attempts), rng=self._rng, score_fn=score_fn)
            return

        raise ValueError(f"Unknown weights_init_method: {weights_init_method!r}")

    def _init_Q(self, X: np.ndarray, y: np.ndarray, n_subsamples: int = 100) -> None:
        n = len(X)
        k = min(int(n_subsamples), n)
        idx = self._rng.choice(n, size=k, replace=False)
        self.Q = self._calculate_accurate_Q(X[idx], y[idx])

    def _margin(self, w: np.ndarray, x: np.ndarray, y: int) -> float:
        return float(y * (w @ x))

    def _get_margin(self, x: np.ndarray, y: int) -> float:
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._margin(self.weights, np.asarray(x, dtype=float), int(np.asarray(y).item()))

    def _get_batched_margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        return (self.weights @ X_arr.T) * y_arr

    def _get_loss_by_sample(self, x: np.ndarray, y: int) -> float:
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._loss_by_sample(self.weights, np.asarray(x, dtype=float), int(np.asarray(y).item()))

    def _get_grad_by_sample(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._grad_by_sample(self.weights, np.asarray(x, dtype=float), int(np.asarray(y).item()))

    def _loss_by_sample(self, w: np.ndarray, x: np.ndarray, y: int) -> float:
        # (1 - margin)^2 + lambda * ||w||^2
        m = self._margin(w, x, y)
        return float((1.0 - m) ** 2 + self.l2_reg_param * np.sum(w * w))

    def _grad_by_sample(self, w: np.ndarray, x: np.ndarray, y: int) -> np.ndarray:
        # (margin - 1) * y * x + lambda * w
        m = self._margin(w, x, y)
        return (m - 1.0) * y * x + self.l2_reg_param * w

    def _get_nag_grad_by_sample(self, optimizer: NAGOptimizer, x: np.ndarray, y: int) -> np.ndarray:
        assert self.weights is not None
        self.weights -= optimizer.learning_rate * optimizer.gamma * optimizer.v
        g = self._grad_by_sample(self.weights, x, y)
        self.weights += optimizer.learning_rate * optimizer.gamma * optimizer.v
        return g

    # Q handling
    def _calculate_accurate_Q(self, X: np.ndarray, y: np.ndarray) -> float:
        assert self.weights is not None
        return float(np.mean([self._loss_by_sample(self.weights, xi, int(yi)) for xi, yi in zip(X, y)]))

    def _get_current_Q(self) -> float:
        if self.Q is None:
            raise RuntimeError("Q is not initialized. Call fit() first.")
        return float(self.Q)

    def _update_Q(self, loss_value: float) -> None:
        if self.Q is None:
            self.Q = float(loss_value)
            return
        self.Q = self.Q_param * float(loss_value) + (1.0 - self.Q_param) * float(self.Q)

    # batching helpers
    def _get_random_idx(self, max_idx: int, n: int) -> np.ndarray:
        if n < max_idx:
            return self._rng.choice(max_idx, size=n, replace=False)
        return np.arange(max_idx)

    def _get_current_batch_indexes_by_margin(self, X: np.ndarray, y: np.ndarray, n_samples_per_iter: int) -> np.ndarray:
        n = len(y)
        if n <= n_samples_per_iter:
            return np.arange(n)

        assert self.weights is not None
        margins = np.abs(y * (X @ self.weights))

        m_min = float(margins.min())
        m_max = float(margins.max())
        if m_max - m_min < 1e-12:
            # all margins equal -> uniform sampling
            return self._rng.choice(n, size=n_samples_per_iter, replace=False)

        scaled = (margins - m_min) / (m_max - m_min)
        negative = 1.0 - scaled
        probs = negative / float(negative.sum())
        return self._rng.choice(n, size=n_samples_per_iter, replace=False, p=probs)

    def _epoch_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int, batch_generation: BatchGeneration) -> list[np.ndarray]:
        remaining = np.arange(len(X))
        batches: list[np.ndarray] = []

        while remaining.size:
            if batch_generation == "random":
                local = self._get_random_idx(remaining.size, batch_size)
            elif batch_generation == "margin":
                local = self._get_current_batch_indexes_by_margin(X[remaining], y[remaining], batch_size)
            else:
                raise ValueError(f"Unknown batch_generation: {batch_generation!r}")

            batches.append(remaining[local])
            remaining = np.delete(remaining, local)

        return batches

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iters: int = 10,
        batch_size: int = 1000,
        stop_threshold: float = 0.1,
        weights_init_method: InitMethod = "random",
        batch_generation: BatchGeneration = "random",
        n_attempts: int | None = None,
    ) -> list[float]:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=int).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X_train and y_train must have the same length")

        self._init_weights(X.shape[1], weights_init_method, X, y, n_attempts)
        assert self.weights is not None
        self._init_Q(X, y)

        optimizer = NAGOptimizer(self.weights, self.learning_rate, self.l2_reg_param, self.gamma)

        q_history: list[float] = []

        eps = 1e-12
        for _epoch in range(int(n_iters)):
            current_weights = float(np.sum(self.weights * self.weights))
            current_Q = self._get_current_Q()

            for batch_idx in self._epoch_batches(X, y, int(batch_size), batch_generation):
                for i in batch_idx:
                    xi = X[i]
                    yi = int(y[i])
                    loss_value = self._loss_by_sample(self.weights, xi, yi)
                    grad = self._get_nag_grad_by_sample(optimizer, xi, yi)
                    self._update_Q(loss_value)
                    optimizer.step(grad)

            new_weights = float(np.sum(self.weights * self.weights))
            if abs(new_weights - current_weights) / max(new_weights, eps) < float(stop_threshold):
                break

            new_Q = self._get_current_Q()
            if abs(new_Q - current_Q) / max(new_Q, eps) < float(stop_threshold) or new_Q > current_Q:
                break

            q_history.append(new_Q)

        return q_history

    def run_multistart(
        self,
        n_runs: int,
        X: np.ndarray,
        y: np.ndarray,
        n_iters: int,
        batch_size: int,
        stop_threshold: float,
        weights_init_method: InitMethod,
    ) -> None:
        best_Q: float | None = None
        best_w: np.ndarray | None = None

        for _ in range(int(n_runs) + 1):
            self.fit(X, y, n_iters, batch_size, stop_threshold, weights_init_method)
            current_Q = self._get_current_Q()
            if best_Q is None or current_Q < best_Q:
                best_Q = current_Q
                best_w = np.copy(self.weights)

        if best_w is not None:
            self.weights = best_w

    def predict(self, X: np.ndarray, mode: PredictMode) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=float)
        raw = self.weights @ X_arr.T

        if mode == "class":
            return np.sign(raw)
        if mode == "probability":
            return raw
        raise ValueError("mode must be 'class' or 'probability'")
