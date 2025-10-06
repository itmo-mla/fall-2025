import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from collections import defaultdict
from tqdm import tqdm


class Metric(ABC):
    @abstractmethod
    def _get_distance(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        pass


    def get_distance(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        return self._get_distance(obj1, obj2)


    def get_distances(self, obj1: np.ndarray, objects: list[np.ndarray]) -> np.ndarray:
        return np.array([self.get_distance(obj1, obj) for obj in objects])
    

class MinkowskyMetric(Metric):
    def __init__(self, p: int):
        self.p = p


    def _get_distance(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        return np.sum(np.abs(obj1 - obj2)**self.p) ** (1/self.p)
    

class EuclideanMetric(MinkowskyMetric):
    def __init__(self):
        self.p = 2


class ParzenWindow:
    def __init__(self, k: int, metric_estimator: Metric, kernel_function: Callable):
        self.k = k
        self.metric_estimator = metric_estimator
        self.kernel_function = np.vectorize(kernel_function)


    def _get_arguments(self, obj: np.ndarray, objects: list[np.ndarray]) -> np.ndarray:
        distances = self.metric_estimator.get_distances(obj, objects)
        closest_objects = np.argsort(distances)
        max_distance = distances[closest_objects[self.k]]

        return distances / max_distance


    def get_weights(self, obj: np.ndarray, objects: list[np.ndarray]) -> np.ndarray:
        kernel_args = self._get_arguments(obj, objects)

        return self.kernel_function(kernel_args)


class ParzenKNN:
    def __init__(self, k: int, metric_estimator: Metric, kernel_function: Callable):
        self.window_function = ParzenWindow(k, metric_estimator, kernel_function)


    def predict(self, obj1: np.ndarray, objects: np.ndarray, classes: np.ndarray, n_anchor_objects: int | None = None) -> int:
        if len(objects) != len(classes):
            raise ValueError("len(objects) != len(classes)")
        
        if n_anchor_objects:
            random_samples_idx = np.random.choice(len(objects), n_anchor_objects, replace=False)
            objects = objects[random_samples_idx]
            classes = classes[random_samples_idx]
        
        weights = self.window_function.get_weights(obj1, objects)
        weighted_sum = defaultdict(int)
        for class_name in set(classes):
            weighted_sum[str(class_name)] = np.sum(weights[classes == class_name])

        return list(weighted_sum.keys())[np.argmax(list(weighted_sum.values()))]


    def predict_bathced(self, test_objects: np.ndarray, anchor_objects: np.ndarray, classes: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([self.predict(obj, anchor_objects, classes, **kwargs) for obj in tqdm(test_objects, total=len(test_objects))])

