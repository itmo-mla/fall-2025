from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    # sklearn разрешено использовать только для подгрузки датасета
    from sklearn.datasets import make_classification
except Exception:  # pragma: no cover - sklearn может быть не установлен
    make_classification = None  # type: ignore


RNG_SEED = 42


def _generate_synthetic_dataset(
    n_samples: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Запасной вариант: простой двумерный датасет для бинарной классификации."""
    rng = np.random.default_rng(RNG_SEED)

    mean_neg = np.array([-1.5, -1.0])
    mean_pos = np.array([1.5, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    n_per_class = n_samples // 2
    x_neg = rng.multivariate_normal(mean_neg, cov, size=n_per_class)
    x_pos = rng.multivariate_normal(mean_pos, cov, size=n_per_class)

    x = np.vstack([x_neg, x_pos])
    y = np.concatenate(
        [np.full(n_per_class, -1.0, dtype=float), np.full(n_per_class, 1.0, dtype=float)]
    )
    return x, y


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Загрузка/генерация двумерного датасета для бинарной классификации.

    Приоритеты:
    1. make_classification из sklearn.datasets (2 информативных признака);
    2. простой синтетический двумерный датасет.
    """
    if make_classification is not None:
        print("Генерирую двумерный датасет с помощью sklearn.datasets.make_classification")
        x, y = make_classification(
            n_samples=400,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=1.8,
            flip_y=0.03,
            random_state=RNG_SEED,
        )
        # Переводим метки из {0, 1} в {-1, 1}
        y = np.where(y == 1, 1.0, -1.0).astype(float)
        return x.astype(float), y

    print("sklearn недоступен, использую синтетический двумерный датасет.")
    return _generate_synthetic_dataset()


def train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = RNG_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Разбиение на train / val / test."""
    assert x.shape[0] == y.shape[0]

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


def standardize_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Масштабирование признаков по train: вычитание среднего и деление на std."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0

    x_train_scaled = (x_train - mean) / std
    x_val_scaled = (x_val - mean) / std
    x_test_scaled = (x_test - mean) / std

    return x_train_scaled, x_val_scaled, x_test_scaled

