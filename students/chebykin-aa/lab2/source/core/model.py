from typing import Any

import numpy as np

from .utils import minkowski_dist

class OwnKNeighborsClassifier():
    def __init__(
        self,
        weights: Any,
        metric: str = "minkowski",
        k_neighbors: int = 2,
        p: float = 1
    ):
        # Инициализируем параметры
        if metric == "minkowski":
            self.metric = minkowski_dist
        else:
            raise ValueError(
                f"Параметр 'metric' должен принимать значение из списка: "
                f"['minkowski']"
            )   

        if k_neighbors <= 0:
            raise ValueError(
                "Параметр 'k_neighbors' должен быть больше 0!"
            )
        if p <= 0:
            raise ValueError(
                "Параметр 'p' должен быть больше 0!"
            )
        
        self.weights = weights
        self.k = k_neighbors
        self.p = p

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """Метод обучения модели"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(
        self,
        X: np.ndarray
    )-> np.ndarray:
        """Метод предсказания модели"""
        preds = [0]*len(X)
        for obj_idx, obj in enumerate(np.array(X)):
            # Посчитаем расстояния
            dists = self.metric(self.X_train, obj, p = self.p)

            # Получим расстояние до k+1 ближайших соседей
            nn_ixds = np.argsort(dists)[:self.k+1]
            # Получим расстояние до k+1 соседа
            h = dists[nn_ixds[-1]]
            # Убираем k+1 соседа
            nn_ixds = nn_ixds[:-1]

            # Получаем веса соседей
            weights = self.weights(dists[nn_ixds] / h)
            # Проводим голосование
            labels = self.y_train[nn_ixds]
            class_votes = {label: np.sum(weights[labels == label]) for label in np.unique(labels)}
            preds[obj_idx] = max(class_votes, key=class_votes.get)

        return np.array(preds)