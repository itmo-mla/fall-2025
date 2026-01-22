import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from source.utils import euclidean_distance, gaussian_kernel
from source.selector import greedy_prototype_selection


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Сохраняем тренировочные данные.
        X: numpy array формы (n_samples, n_features)
        y: numpy array формы (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def ccv_fit(self, X: np.ndarray, y: np.ndarray, max_prototypes: int = 50, e: float = 0.1):
        """
        Жадная стратегия добавления эталонов.
        """
        best_inds = greedy_prototype_selection(self, X, y, max_prototypes, e)
        self.X_train, self.y_train = X[best_inds], y[best_inds]

    def predict(self, X):
        """
        Предсказываем классы для каждого примера в X
        """
        predictions = []
        for x in X:
            # Вычисляем расстояния до всех точек в тренировочном наборе
            distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])
            
            # Находим индексы k ближайших соседей
            k_indices = distances.argsort()[:self.k]

            # Получаем метки и расстояния ближайших соседей
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            k_nearest_distances = distances[k_indices]

            # Используем переменное окно: h = расстояние до k-го соседа
            h = k_nearest_distances[-1] + 1e-8  # небольшая поправка, чтобы избежать деления на ноль

            # Вычисляем веса через гауссово ядро
            weights = gaussian_kernel(k_nearest_distances / h)
            
            # Суммируем веса для каждого класса
            class_weights = {}
            for label, weight in zip(k_nearest_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            
            # Выбираем класс с максимальным весом
            predicted_class = max(class_weights, key=class_weights.get)
            predictions.append(predicted_class)

        return np.array(predictions)
