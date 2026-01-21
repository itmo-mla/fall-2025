import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from source.utils import euclidean_distance, gaussian_kernel


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

    def ccv_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Итеративное удаление объектов из тренировочной выборки
        пока точность на валидации не падает, объекты удаляются.
        """
        # Приведение признаков
        X_curr = X.copy()
        y_curr = y.copy()

        # Начальная точность
        self.fit(X_curr, y_curr)

        y_pred = self.predict(X_curr)
        best_acc = np.mean(y_pred == y_curr)

        changed = True
        while changed:
            changed = False
            # Проходим по объектам по одному
            for i in range(len(X_curr)):
                X_try = np.delete(X_curr, i, axis=0)
                y_try = np.delete(y_curr, i, axis=0)

                self.fit(X_try, y_try)

                y_pred = self.predict(X_curr)
                acc = np.mean(y_pred == y_curr)

                # Если точность не уменьшилась — удаляем объект
                if acc >= best_acc:
                    X_curr = X_try
                    y_curr = y_try
                    best_acc = acc
                    changed = True
                    break  # начинаем цикл заново

        self.X_train, self.y_train = X_curr, y_curr

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
