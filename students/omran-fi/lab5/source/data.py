from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  # in {-1, +1}
    y_test: np.ndarray   # in {-1, +1}
    feature_names: list[str]
    target_names: list[str]


def _fit_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def _apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _to_pm_one(y01: np.ndarray) -> np.ndarray:
    # 0 -> -1, 1 -> +1
    y01 = y01.astype(int).reshape(-1)
    return np.where(y01 == 1, 1, -1).astype(int)


def load_dataset(
    name: str = "breast_cancer",
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Datasets are downloaded/loaded at runtime via sklearn.datasets.
    Returns y in {-1, +1} to match lecture formulas.
    """
    name = name.strip().lower()

    if name == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float64)
        y01 = ds.target.astype(int)
        feature_names = list(ds.feature_names)
        target_names = list(ds.target_names)  # ['malignant','benign'] but labels are 0/1

    elif name == "iris_binary":
        ds = load_iris()
        X_all = ds.data.astype(np.float64)
        y_all = ds.target.astype(int)
        mask = y_all != 2  # keep classes 0 and 1 only
        X = X_all[mask]
        y01 = y_all[mask]
        feature_names = list(ds.feature_names)
        target_names = [ds.target_names[0], ds.target_names[1]]

    else:
        raise ValueError("Unknown dataset. Use 'breast_cancer' or 'iris_binary'.")

    y = _to_pm_one(y01)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Standardize using TRAIN stats only (avoid leakage)
    mean, std = _fit_standardize(X_train)
    X_train = _apply_standardize(X_train, mean, std)
    X_test = _apply_standardize(X_test, mean, std)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        feature_names=feature_names,
        target_names=target_names,
    )
