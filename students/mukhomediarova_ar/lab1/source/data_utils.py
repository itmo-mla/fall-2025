import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    # Датасет для лабораторной работы
    from sklearn.datasets import load_breast_cancer
except Exception:  # sklearn может быть не установлен
    load_breast_cancer = None  # type: ignore


RNG_SEED = 42


def _generate_synthetic_dataset(
    n_per_class: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация простого двумерного датасета для бинарной классификации.

    Класс -1 и класс +1 выбираются из двух двумерных нормальных распределений.
    """
    rng = np.random.default_rng(RNG_SEED)

    mean_neg = np.array([-1.0, -1.0])
    mean_pos = np.array([1.5, 1.5])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    x_neg = rng.multivariate_normal(mean_neg, cov, size=n_per_class)
    x_pos = rng.multivariate_normal(mean_pos, cov, size=n_per_class)

    x = np.vstack([x_neg, x_pos])
    y = np.concatenate([-np.ones(n_per_class), np.ones(n_per_class)])

    return x, y


def _load_csv_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загрузка датасета из CSV.

    - Целевой столбец: 'target', если он есть, иначе последний столбец.
    - Признаки: все остальные столбцы (числовые, переводим в float).
    - Метки приводятся к {-1, 1}.
    """
    df = pd.read_csv(csv_path)
    if "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]

    y_raw = df[target_col]
    x_df = df.drop(columns=[target_col])

    # Используем только числовые признаки
    x_df = x_df.select_dtypes(include=["number"]).astype(float)
    if x_df.shape[1] == 0:
        raise ValueError("В датасете нет числовых признаков для обучения.")

    x = x_df.to_numpy()

    unique_vals = sorted(pd.unique(y_raw))
    if len(unique_vals) != 2:
        raise ValueError(
            f"Ожидается бинарная классификация, найдено {len(unique_vals)} различных значений целевой переменной."
        )

    mapping = {unique_vals[0]: -1.0, unique_vals[1]: 1.0}
    y = y_raw.map(mapping).to_numpy(dtype=float)

    return x, y


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет для бинарной классификации.

    Приоритеты:
    1. `sklearn.datasets.load_breast_cancer` (основной вариант для ЛР);
    2. файл `data.csv` рядом с `main.py` (если вы хотите использовать свой датасет);
    3. синтетический двумерный датасет.
    """
    # 1. Основной вариант — Breast Cancer из sklearn
    if load_breast_cancer is not None:
        print("Загружаю датасет Breast Cancer из sklearn.datasets")
        ds = load_breast_cancer()
        x = ds["data"].astype(float)
        y_raw = ds["target"].astype(float)
        # Переводим метки {0, 1} -> {-1, 1}
        y = 2.0 * y_raw - 1.0
        return x, y

    # 2. Пытаемся использовать локальный CSV
    here = Path(__file__).resolve().parent
    csv_path = here / "data.csv"
    if csv_path.exists():
        print(f"sklearn не найден, загружаю датасет из CSV: {csv_path}")
        return _load_csv_dataset(csv_path)

    # 3. Fallback — синтетический датасет
    print("sklearn не найден и файл data.csv не существует, генерирую синтетический датасет.")
    return _generate_synthetic_dataset()


def train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = RNG_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разбиение на train / val / test.
    """
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
    """
    Масштабирование признаков: вычитаем среднее и делим на стандартное отклонение по train.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0

    x_train_scaled = (x_train - mean) / std
    x_val_scaled = (x_val - mean) / std
    x_test_scaled = (x_test - mean) / std

    return x_train_scaled, x_val_scaled, x_test_scaled

