import numpy as np
from models import ParzenWindowKNN


class PrototypeSelection:
    """
    Алгоритм отбора эталонов (Hart's algorithm).
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self._reduce_dataset()

    def _reduce_dataset(self):
        X_reduced, y_reduced = self._initialize_boundary_set()
        if len(X_reduced) == 0:
            return self.X_train, self.y_train

        while True:
            changed = False
            for i in range(len(X_reduced) - 1, -1, -1):
                X_temp = np.delete(X_reduced, i, axis=0)
                y_temp = np.delete(y_reduced, i)
                if len(X_temp) == 0:
                    break

                knn_temp = ParzenWindowKNN(k=1)
                knn_temp.fit(X_temp, y_temp)
                pred = knn_temp.predict([X_reduced[i]])
                if pred[0] != y_reduced[i]:
                    X_reduced = np.delete(X_reduced, i, axis=0)
                    y_reduced = np.delete(y_reduced, i)
                    changed = True
            if not changed:
                break
        return X_reduced, y_reduced

    def _initialize_boundary_set(self):
        """
        Инициализация набора граничных точек.
        Для каждой точки из класса A находим ближайшую точку из другого класса B.
        Добавляем обе точки (A и B).
        """
        X_boundary = []
        y_boundary = []
        n_samples = self.X_train.shape[0]

        for i in range(n_samples):
            x_i = self.X_train[i]
            y_i = self.y_train[i]

            # Находим ближайшую точку из другого класса
            min_dist = np.inf
            nearest_other_idx = -1

            for j in range(n_samples):
                if self.y_train[j] != y_i:
                    dist = np.linalg.norm(x_i - self.X_train[j])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_other_idx = j

            if nearest_other_idx != -1:
                X_boundary.append(x_i)
                y_boundary.append(y_i)
                X_boundary.append(self.X_train[nearest_other_idx])
                y_boundary.append(self.y_train[nearest_other_idx])

        if len(X_boundary) > 0:
            X_boundary = np.array(X_boundary)
            y_boundary = np.array(y_boundary)
            # Используем уникальные строки
            _, unique_indices = np.unique(X_boundary, axis=0, return_index=True)
            X_boundary = X_boundary[unique_indices]
            y_boundary = y_boundary[unique_indices]
        else:
            X_boundary = np.array(X_boundary)
            y_boundary = np.array(y_boundary)

        return X_boundary, y_boundary
