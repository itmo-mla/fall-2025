import numpy as np


class KNN():
    def __init__(self, k=5):
        self.X = None
        self.Y = None
        self.k = k

    def kernel(self, x):
        return 1 / (2*np.pi) * np.exp(- 2 * x ** 2)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def get_distances(self, X, xi):
        return np.linalg.norm(X - np.array(xi), axis=1)


    def _single_predict(self, x):
        X_dist = self.get_distances(self.X, x)
        sorted_distances = np.argsort(X_dist)
        h = X_dist[sorted_distances[self.k]]
        weights = self.kernel(X_dist / h)
        ans = None
        max_sum = 0
        class_weights = {}
        for idx, weight in enumerate(weights):
            label = self.Y[idx]
            class_weights[label] = class_weights.get(label, 0) + weight

        return max(class_weights, key=class_weights.get)
    
    def predict(self, X) -> np.ndarray:
        return np.array([self._single_predict(x) for x in X])