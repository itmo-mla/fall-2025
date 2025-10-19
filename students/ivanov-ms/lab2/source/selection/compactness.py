import math
import numpy as np

from utils.neighbors import compute_distance_matrix, find_k_neighbors, find_neighbors_in_subset


class CompactnessProfile:
    def __init__(self):
        self.X = None
        self.y = None
        self.n_samples = None
        self.distances = None

    def fit(self, X, y):
        """Вычисление матрицы расстояний для профиля компактности"""
        self.X = X
        self.y = y
        self.n_samples = len(y)
        self.distances = compute_distance_matrix(X)
        return self

    def compute_profile(self, max_m=None):
        """
        Вычисление профиля компактности Π(m)
        Π(m) = (1/L) * Σ_{i=1}^L [y_i ≠ y_i^{(m)}]
        где y_i^{(m)} - класс m-го соседа объекта x_i
        """
        if max_m is None:
            max_m = self.n_samples - 1

        # Находим всех соседей
        neighbor_indices = find_k_neighbors(self.distances, max_m, exclude_self=True)

        profile = np.zeros(max_m)

        for m in range(max_m):
            error_count = 0

            for i in range(self.n_samples):
                if self.y[i] != self.y[neighbor_indices[i, m]]:
                    error_count += 1

            profile[m] = error_count / self.n_samples

        return profile

    def compute_profile_for_subset(self, subset_indices, max_m=None):
        """
        Вычисление профиля компактности для подмножества объектов
        """
        if max_m is None or max_m > len(subset_indices) - 1:
            valid_m = len(subset_indices) - 1
        else:
            valid_m = max_m

        if len(subset_indices) == 0:
            return np.array([1.0] * max_m)  # Максимальная ошибка при пустом множестве

        # Находим соседей в подмножестве
        neighbor_indices, _ = find_neighbors_in_subset(self.distances, subset_indices, valid_m, exclude_self=True)

        profile = np.ones(max_m)
        for m in range(valid_m):
            error_count = 0

            for i in range(self.n_samples):
                if self.y[i] != self.y[neighbor_indices[i, m]]:
                    error_count += 1

            profile[m] = error_count / self.n_samples

        return profile

    def compute_ccv(self, subset_indices, k: int):
        """
        Вычисление CCV через профиль компактности
        CCV(X^L) = Σ_{m=1}^k Π(m) * (C_{L-1}^{l-1-m} / C_L^{l})

        Для случая l = L (все объекты как эталоны):
        C_L^{l} = C_L^L = 1
        """
        L, l = self.n_samples, max(len(subset_indices), k)

        profile = self.compute_profile_for_subset(subset_indices, max_m=k)

        ccv = 0.0
        for m in range(k):
            # Вычисляем комбинаторные коэффициенты
            weight = math.comb(L - 1, l - m - 1) / math.comb(L, l)
            ccv += profile[m] * weight

        return ccv
