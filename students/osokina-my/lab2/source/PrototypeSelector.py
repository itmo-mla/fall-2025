import math
import numpy as np


MIN_LEFT_SAMPLES = 20


class PrototypeSelector:
    def __init__(self, X, y):
        self.prototypes = None
        self.prototype_labels = None
        # Изначально все индексы прототипы
        self.prototypes_indices = list(range(len(X)))
        self.prototypes_indices_history = []
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        # Количество объектов в каждом классе
        self.class_counts = {}
        unique_classes = np.unique(y)
        for cls in unique_classes:
            self.class_counts[cls] = np.sum(y == cls)

        # Вычисляем расстояния при инициализации
        self._compute_distance()

    def _compute_distance(self):
        """Вычисление матрицы расстояний между всеми объектами"""
        n_samples = len(self.X_train)
        self.distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Евклидово расстояние между точкой i и всеми остальными
            self.distance_matrix[i] = np.sqrt(np.sum((self.X_train[i] - self.X_train) ** 2, axis=1))
        
        # Сортировка индексов по расстоянию
        self._sorted_indices = np.argsort(self.distance_matrix, axis=1)
        self._sorted_y = self.y_train[self._sorted_indices]

    def _comb(self, n, k):
        """Вычисление биномиального коэффициента C(n, k)"""
        if k > n or k < 0:
            return 0
        return math.comb(n, k)

    def _calculate_sample_contribution(self, i, prototypes_mask, k_neighbors, L):
        """Вычисление вклада отдельного объекта в CCV"""
        # Получаем отсортированные индексы для объекта i
        sorted_idx = self._sorted_indices[i]
        
        # Находим позиции прототипов среди отсортированных соседей
        ref_positions = np.where(prototypes_mask[sorted_idx])[0]
        
        y_i = self.y_train[i]
        l = self.class_counts[y_i]
        
        T_i = 0.0
        valid_neighbors = 0
        
        # Перебираем прототипы по порядку расстояния
        for pos in ref_positions:
            idx = sorted_idx[pos]

            # Пропускаем сам объект, если он является прототипом (Leave-One-Out)
            if idx == i:
                continue

            valid_neighbors += 1
            m = valid_neighbors
            
            # Если мы уже нашли нужное количество соседей, выходим
            if m > k_neighbors:
                break

            # Проверяем, совпадает ли метка m-го прототипа с меткой x_i
            if self.y_train[idx] != y_i:
                # Вычисляем комбинаторный коэффициент
                # C(L-1-m, l-1) / C(L-1, l-1)
                numerator = self._comb(L - 1 - m, l - 1)
                denominator = self._comb(L - 1, l - 1)

                if denominator > 0:
                    weight = numerator / denominator
                    T_i += weight
                    
        return T_i

    def _ccv_on_prototypes(self, _prototypes_indices):
        """
        Полная кросс-валидация на эталонных элементах
        """

        L = len(self.X_train)  # Общее количество объектов
        k_neighbors = 1  # Число соседей для проверки
        total_ccv = 0.0

        # Создаем маску прототипов
        prototypes_mask = np.zeros(L, dtype=bool)
        prototypes_mask[_prototypes_indices] = True

        # Проходим по всем объектам выборки для вычисления общего CCV
        for i in range(L):
            total_ccv += self._calculate_sample_contribution(i, prototypes_mask, k_neighbors, L)

        return total_ccv / L

    def _find_best_prototype_to_remove(self, current_prototypes):
        """Поиск лучшего прототипа для удаления на текущем шаге"""
        best_ccv_score = float('inf')
        best_prototypes_indices = current_prototypes
        best_removed_idx = -1

        # Пробуем удалить каждый прототип по очереди
        for j in range(len(current_prototypes)):
            new_prototypes_indices = current_prototypes[:j] + current_prototypes[j + 1:]
            ccv_score = self._ccv_on_prototypes(new_prototypes_indices)

            if ccv_score < best_ccv_score:
                best_ccv_score = ccv_score
                best_prototypes_indices = new_prototypes_indices
                best_removed_idx = current_prototypes[j]
        
        return best_ccv_score, best_prototypes_indices, best_removed_idx

    def select_prototypes(self):
        """
        Итеративно удаляет элементы из списка эталонов, минимизируя значение CCV
        """
        self.prototypes_indices_history = []
        self.ccv_scores = []

        # Вычисляем начальный CCV
        current_ccv = self._ccv_on_prototypes(self.prototypes_indices)
        self.ccv_scores.append(current_ccv)
        print(f"Начальный CCV: {current_ccv:.6f}")

        MAX_ITERATIONS = len(self.X_train) - MIN_LEFT_SAMPLES
        iteration = 0

        while len(self.prototypes_indices) > MIN_LEFT_SAMPLES and iteration < MAX_ITERATIONS:
            current_prototypes = self.prototypes_indices.copy()
            
            best_ccv_score, best_prototypes_indices, best_removed_idx = self._find_best_prototype_to_remove(current_prototypes)

            # Удаляем жадно, по min score
            self.prototypes_indices = best_prototypes_indices
            current_ccv = best_ccv_score
            self.prototypes_indices_history.append(best_prototypes_indices.copy())
            self.ccv_scores.append(best_ccv_score)

            print(
                f"Итерация {iteration}: удален элемент {best_removed_idx}, CCV={best_ccv_score:.6f}, прототипов: {len(best_prototypes_indices)}")
            iteration += 1

        # После выбора сохраняем прототипы и их метки
        self.prototypes = self.X_train[self.prototypes_indices]
        self.prototype_labels = self.y_train[self.prototypes_indices]

        return self.prototypes, self.prototype_labels, self.prototypes_indices
