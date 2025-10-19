import numpy as np


def _gaussian_kernel(r):
    return np.exp(-2 * r ** 2)


class ParzenWindowKNN:
    def __init__(self, k=5, kernel=_gaussian_kernel):
        self.k = k
        self.kernel = kernel
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        if X.shape[0] == 0:
            raise ValueError("Train doesn't contain any samples!")
        if X.shape[0] == 1:
            raise ValueError("Train contain only one sample!")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y contains different amount of samples!")

        self.X_train = X
        self.y_train = y
        if self.k >= self.X_train.shape[0] - 1:
            self.k = self.X_train.shape[0] - 1

        return self

    def predict(self, X):
        predictions = []
        for x in X:
            # Вычисляем расстояния до всех точек обучающей выборки
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Находим k ближайших соседей
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_distances = distances[k_indices]
            k_labels = self.y_train[k_indices]

            # Используем расстояние до k-го соседа как ширину окна
            sigma = k_distances[-1] if len(k_distances) > 0 else 1.0
            if sigma == 0:  # Избегаем деления на ноль
                sigma = 1e-8

            # Вычисляем веса с помощью ядра
            weights = self.kernel(k_distances / sigma)

            # Взвешенное голосование
            class_weights = {}
            for label, weight in zip(k_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight

            # Выбираем класс с максимальным весом
            predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)

        return np.array(predictions)
