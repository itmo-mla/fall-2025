import numpy as np

from .compactness import CompactnessProfile
from utils import find_neighbors_in_subset


class PrototypeSelector:
    def __init__(self, k=3):
        self.k = k
        self.profile_calc = None
        self.X = None
        self.y = None
        self.L = None
        self.distances = None

        self.omega_indices = None
        self.history = {}

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.L = len(y)
        self.profile_calc = CompactnessProfile()
        self.profile_calc.fit(X, y)
        self.distances = self.profile_calc.distances

        # Инициализация: по одному объекту от каждого класса
        self._initialize_prototypes()

        # Вычисляем начальный CCV
        current_ccv = self.profile_calc.compute_ccv(self.omega_indices, self.k)

        print("\nНачало отбора эталонов:")
        print(f"Инициализация: |Ω| = {len(self.omega_indices)}, CCV = {current_ccv:.4f}")

        iteration = 0
        ccv_history = [current_ccv]
        omega_sizes = [len(self.omega_indices)]

        # Жадное добавление с использованием профиля компактности
        while len(self.omega_indices) < self.L:
            iteration += 1
            best_candidate = None
            best_ccv = current_ccv

            # Находим кандидатов для добавления
            candidates = self._find_promising_candidates()

            for candidate in candidates:
                new_ccv = self.profile_calc.compute_ccv(self.omega_indices + [candidate], self.k)

                if new_ccv < best_ccv:
                    best_ccv = new_ccv
                    best_candidate = candidate

            if best_candidate is not None and best_ccv < current_ccv:
                self.omega_indices.append(best_candidate)
                current_ccv = best_ccv
                ccv_history.append(best_ccv)
                omega_sizes.append(len(self.omega_indices))

                print(
                    f"Итерация {iteration}: добавлен объект {best_candidate}, "
                    f"|Ω| = {len(self.omega_indices)}, CCV = {current_ccv:.4f}"
                )
            else:
                print(f"Остановка на итерации {iteration}: улучшений не найдено")
                print(f"Итоговый |Ω| = {len(self.omega_indices)}, CCV = {current_ccv:.4f}")
                break

        self.history = {
            "ccv": ccv_history,
            "omega_sizes": omega_sizes
        }

        return self

    def _initialize_prototypes(self):
        """Инициализация: по одному объекту от каждого класса"""
        unique_classes = np.unique(self.y)
        self.omega_indices = []

        for cls in unique_classes:
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) > 0:
                # Выбираем объект, наиболее близкий к центроиду класса
                centroid = np.mean(self.X[class_indices], axis=0)
                distances_to_centroid = [np.linalg.norm(self.X[i] - centroid) for i in class_indices]
                best_idx = class_indices[np.argmin(distances_to_centroid)]
                self.omega_indices.append(best_idx)

        return self.omega_indices

    def _find_promising_candidates(self, n_candidates=20):
        """Нахождение перспективных кандидатов для добавления"""
        candidates = np.array([i for i in range(self.L) if i not in self.omega_indices])

        if len(candidates) <= n_candidates:
            return candidates

        # Оцениваем "полезность" каждого кандидата
        candidate_scores = np.zeros_like(candidates)

        for i, candidate in enumerate(candidates):
            # Быстрая оценка: смотрим, сколько объектов будут лучше классифицированы
            improvement = self._estimate_improvement(candidate)
            candidate_scores[i] = -improvement

        # Лучшие по полезности
        candidates_order_idx = np.argpartition(candidate_scores, n_candidates)[:n_candidates]
        sorted_candidates_idx = candidates_order_idx[np.argsort(candidate_scores[candidates_order_idx])]
        return candidates[sorted_candidates_idx]

    def _estimate_improvement(self, candidate):
        """Быстрая оценка улучшения от добавления кандидата"""
        improvement = 0

        # Находим соседей в текущем множестве
        current_neighbors, _ = find_neighbors_in_subset(self.distances, self.omega_indices, 1, exclude_self=True)

        # Находим, для каких объектов кандидат станет ближайшим соседом
        for i in range(self.L):
            if i == candidate:
                continue

            dist_to_candidate = self.distances[i, candidate]
            candidate_class = self.y[candidate]

            # Проверяем, станет ли кандидат ближе, чем текущие соседи
            if current_neighbors.shape[1] > 0:
                current_neighbor_idx = current_neighbors[i, 0]
                current_neighbor_dist = self.distances[i, current_neighbor_idx]
                current_neighbor_class = self.y[current_neighbor_idx]
                if dist_to_candidate < current_neighbor_dist and candidate_class != current_neighbor_class:
                    # Если кандидат правильного класса, это улучшение
                    if candidate_class == self.y[i]:
                        improvement += 1
                    elif current_neighbor_class == self.y[i]:
                        improvement -= 1

        return improvement

    def get_prototypes(self):
        return self.X[self.omega_indices], self.y[self.omega_indices]
