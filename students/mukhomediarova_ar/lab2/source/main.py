from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    # sklearn разрешено использовать только для подгрузки датасета
    from sklearn.datasets import load_breast_cancer
except Exception:  # pragma: no cover - sklearn может быть не установлен
    load_breast_cancer = None  # type: ignore

try:
    # Запуск как пакет: python -m students.mukhomediarova_ar.lab2.source.main
    from .knn import ParzenWindowKNN, gaussian_kernel, weighted_majority_vote
    from .metrics import Array, EuclideanDistance, accuracy_score
    from .selection import CondensedPrototypeSelector
except ImportError:  # pragma: no cover - запуск напрямую как скрипта
    from knn import ParzenWindowKNN, gaussian_kernel, weighted_majority_vote  # type: ignore
    from metrics import Array, EuclideanDistance, accuracy_score  # type: ignore
    from selection import CondensedPrototypeSelector  # type: ignore


RNG_SEED = 42


def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _generate_synthetic_dataset(
    n_per_class: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Запасной вариант: простой двумерный датасет для бинарной классификации."""
    rng = np.random.default_rng(RNG_SEED)

    mean_neg = np.array([-1.0, -1.0])
    mean_pos = np.array([1.5, 1.5])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    x_neg = rng.multivariate_normal(mean_neg, cov, size=n_per_class)
    x_pos = rng.multivariate_normal(mean_pos, cov, size=n_per_class)

    x = np.vstack([x_neg, x_pos])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return x, y


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Загрузка датасета для классификации.

    Приоритеты:
    1. Breast Cancer из sklearn.datasets;
    2. простой синтетический двумерный датасет.
    """
    if load_breast_cancer is not None:
        print("Загружаю датасет Breast Cancer из sklearn.datasets")
        ds = load_breast_cancer()
        x = ds["data"].astype(float)
        y = ds["target"].astype(int)
        return x, y

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
    std[std == 0] = 1.0

    x_train_scaled = (x_train - mean) / std
    x_val_scaled = (x_val - mean) / std
    x_test_scaled = (x_test - mean) / std

    return x_train_scaled, x_val_scaled, x_test_scaled


def _pairwise_euclidean_distances(x: Array) -> Array:
    """Матрица попарных евклидовых расстояний между объектами."""
    x = np.asarray(x, dtype=float)
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    sq_norms = np.sum(x * x, axis=1, keepdims=True)
    dist_sq = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def loo_empirical_risk_parzen(
    k_values: Iterable[int],
    x: Array,
    y: Array,
) -> Tuple[np.ndarray, np.ndarray]:
    """Эмпирический риск (LOO) для KNN с окном Парзена для набора k."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    n = x.shape[0]
    if n <= 1:
        raise ValueError("Для LOO требуется как минимум два объекта")

    k_values_arr = np.array(list(k_values), dtype=int)
    if np.any(k_values_arr <= 0):
        raise ValueError("Все значения k должны быть положительными целыми числами")

    print("Вычисляю попарные расстояния для LOO...")
    dist = _pairwise_euclidean_distances(x)
    # Исключаем сам объект из соседей
    np.fill_diagonal(dist, np.inf)

    order = np.argsort(dist, axis=1)
    dist_sorted = np.take_along_axis(dist, order, axis=1)
    y_sorted = y[order]

    risks = np.empty_like(k_values_arr, dtype=float)

    for idx, k in enumerate(k_values_arr):
        k_eff = min(int(k), n - 1)
        neigh_dist = dist_sorted[:, :k_eff]  # (n_objects, k_eff)
        neigh_labels = y_sorted[:, :k_eff]

        # Ширина окна — расстояние до k‑го соседа
        h = neigh_dist[:, k_eff - 1]  # shape (n_objects,)
        h_safe = h.copy()
        h_safe[h_safe == 0.0] = 1.0
        args = neigh_dist / h_safe[:, None]

        weights = gaussian_kernel(args)

        errors = 0
        for i in range(n):
            pred = weighted_majority_vote(neigh_labels[i], weights[i])
            if pred != y[i]:
                errors += 1

        risk = errors / n
        risks[idx] = risk
        print(f"k={k_eff}: LOO‑риск={risk:.4f}")

    return k_values_arr, risks


def plot_risk_curve(k_values: np.ndarray, risks: np.ndarray, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, risks, marker="o")
    plt.xlabel("k")
    plt.ylabel("Эмпирический риск (LOO)")
    plt.title("Зависимость эмпирического риска от k (Parzen‑KNN)")
    plt.grid(alpha=0.3)
    out_path = _results_dir() / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"График эмпирического риска сохранён в {out_path}")


def plot_prototype_selection_2d(
    x_train: Array,
    y_train: Array,
    x_proto: Array,
    y_proto: Array,
    filename: str,
) -> None:
    """Простая 2D‑визуализация отбора эталонов по первым двум признакам."""
    x_train = np.asarray(x_train)
    x_proto = np.asarray(x_proto)

    if x_train.shape[1] < 2:
        print("Меньше двух признаков — пропускаю 2D‑визуализацию отбора эталонов.")
        return

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        alpha=0.35,
        label="Все объекты train",
    )
    plt.scatter(
        x_proto[:, 0],
        x_proto[:, 1],
        c=y_proto,
        cmap="coolwarm",
        edgecolor="black",
        marker="o",
        s=60,
        label="Эталоны",
    )
    plt.xlabel("Признак 1 (стандартизованный)")
    plt.ylabel("Признак 2 (стандартизованный)")
    plt.legend()
    plt.colorbar(scatter, label="Класс")
    plt.title("Отбор эталонов (Condensed NN, проекция на первые два признака)")
    out_path = _results_dir() / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Визуализация отбора эталонов сохранена в {out_path}")


def run_experiments() -> None:
    # 1. Датасет
    x, y = load_dataset()
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = train_val_test_split(x, y)

    x_train, x_val, x_test = standardize_features(x_train, x_val, x_test)

    print(
        f"Размерности выборок: train={x_train.shape[0]}, "
        f"val={x_val.shape[0]}, test={x_test.shape[0]}"
    )

    # 2. Подбор k методом LOO по обучающей выборке
    k_candidates = range(1, 31, 2)  # нечётные k от 1 до 29
    k_values, risks = loo_empirical_risk_parzen(k_candidates, x_train, y_train)
    best_k = int(k_values[np.argmin(risks)])
    print(f"Лучшее k по LOO: k={best_k}")
    plot_risk_curve(k_values, risks, "loo_risk_parzen_knn.png")

    # 3. Обучение Parzen‑KNN с найденным k и оценка качества
    metric = EuclideanDistance()
    parzen_knn = ParzenWindowKNN(k=best_k, metric=metric, kernel=gaussian_kernel)
    parzen_knn.fit(x_train, y_train)

    y_val_pred = parzen_knn.predict(x_val)
    y_test_pred = parzen_knn.predict(x_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Точность Parzen‑KNN: val_accuracy={val_acc:.4f}, test_accuracy={test_acc:.4f}")

    # 4. Отбор эталонов и сравнение качества
    selector = CondensedPrototypeSelector(k=1, max_iter=10, random_state=RNG_SEED)
    x_proto, y_proto, proto_mask = selector.select(x_train, y_train)
    print(
        f"Отбор эталонов: было объектов в train={x_train.shape[0]}, "
        f"стало эталонов={x_proto.shape[0]}"
    )

    parzen_knn_protos = ParzenWindowKNN(k=best_k, metric=metric, kernel=gaussian_kernel)
    parzen_knn_protos.fit(x_proto, y_proto)

    y_val_pred_p = parzen_knn_protos.predict(x_val)
    y_test_pred_p = parzen_knn_protos.predict(x_test)

    val_acc_p = accuracy_score(y_val, y_val_pred_p)
    test_acc_p = accuracy_score(y_test, y_test_pred_p)
    print(
        "Качество с отбором эталонов:\n"
        f"  val_accuracy={val_acc_p:.4f} (до: {val_acc:.4f})\n"
        f"  test_accuracy={test_acc_p:.4f} (до: {test_acc:.4f})"
    )

    # 5. Визуализация результата отбора эталонов
    plot_prototype_selection_2d(
        x_train,
        y_train,
        x_proto,
        y_proto,
        "prototype_selection_2d.png",
    )


def main() -> None:
    np.random.seed(RNG_SEED)
    run_experiments()


if __name__ == "__main__":
    main()

