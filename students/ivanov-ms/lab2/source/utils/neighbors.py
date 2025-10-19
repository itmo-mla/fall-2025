import numpy as np


def compute_distance_matrix(X):
    """
    Вычисление матрицы попарных расстояний между всеми объектами
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def find_k_neighbors(distances, k, exclude_self=True):
    """
    Нахождение k ближайших соседей для каждого объекта
    """
    n_samples = distances.shape[0]
    indices = np.zeros((n_samples, k), dtype=int)

    for i in range(n_samples):
        # Получаем расстояния от i-го объекта до всех остальных
        row_distances = distances[i].copy()

        if exclude_self:
            # Исключаем сам объект (расстояние до себя = 0)
            row_distances[i] = np.inf

        # Находим индексы k ближайших соседей
        neighbor_indices = np.argpartition(row_distances, k)[:k]
        # Сортируем по расстоянию
        sorted_neighbors = neighbor_indices[np.argsort(row_distances[neighbor_indices])]
        indices[i] = sorted_neighbors

    return indices


def find_neighbors_in_subset(distances, subset_indices, k, exclude_self=True):
    """
    Нахождение k ближайших соседей в подмножестве объектов
    """
    n_samples = distances.shape[0]
    n_subset = len(subset_indices)

    if n_subset <= k:  # Can't find k neighbors in subset with k elements or less
        return np.array([]), np.array([])

    indices = np.zeros((n_samples, k), dtype=int)
    neighbor_distances = np.zeros((n_samples, min(k, n_subset)))

    for i in range(n_samples):
        # Расстояния только до объектов в подмножестве
        subset_dists = distances[i, subset_indices]

        if exclude_self and i in subset_indices:
            subset_dists[subset_indices.index(i)] = np.inf

        # Находим k ближайших соседей в подмножестве
        neighbor_indices = np.argpartition(subset_dists, k)[:k]
        sorted_idx = neighbor_indices[np.argsort(subset_dists[neighbor_indices])]
        # Сохраняем индексы (в глобальной нумерации) и расстояния
        indices[i] = [subset_indices[idx] for idx in sorted_idx]
        neighbor_distances[i] = subset_dists[sorted_idx]

    return indices, neighbor_distances
