from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StandardizationParams:
    mean_: np.ndarray
    std_: np.ndarray


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit(X: np.ndarray, eps: float = 1e-12) -> StandardizationParams:
    mean_ = X.mean(axis=0)
    std_ = X.std(axis=0, ddof=0)
    std_ = np.where(std_ < eps, 1.0, std_)
    return StandardizationParams(mean_=mean_, std_=std_)


def standardize_transform(X: np.ndarray, params: StandardizationParams) -> np.ndarray:
    return (X - params.mean_) / params.std_


def center_fit(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def center_transform(X: np.ndarray, mean_: np.ndarray) -> np.ndarray:
    return X - mean_
