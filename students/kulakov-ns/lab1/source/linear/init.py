from __future__ import annotations

import numpy as np


def init_random(n_features: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=n_features)


def init_correlation(X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = X.T @ y
    denom = (X * X).sum(axis=0) + eps
    return num / denom


def init_multistart(
    n_features: int,
    score_fn,
    attempts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if attempts <= 0:
        raise ValueError("attempts must be positive")

    best_w: np.ndarray | None = None
    best_score: float | None = None

    for _ in range(attempts):
        w = init_random(n_features, rng)
        s = float(score_fn(w))
        if best_score is None or s < best_score:
            best_score = s
            best_w = w

    assert best_w is not None
    return best_w
