import numpy as np
import math


class KNN():
    def __init__(self, k=5):
        self.X = None
        self.Y = None
        self.k = k
        self.distances = None

    def kernel(self, x):
        return 1 / (2*np.pi) * np.exp(- 2 * x ** 2)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_samples = len(Y)
        self.distances = self._compute_distance_matrix()

    def get_distances(self, X, xi):
        return np.linalg.norm(X - np.array(xi), axis=1)
    
    def _compute_distance_matrix(self):
        n = self.X.shape[0]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.X[i] - self.X[j])
                distances[i, j] = distances[j, i] = dist
        np.fill_diagonal(distances, np.inf)
        return distances

    def _single_predict(self, x):
        X_dist = self.get_distances(self.X, x)
        sorted_distances = np.argsort(X_dist)
        h = X_dist[sorted_distances[self.k]]
        weights = self.kernel(X_dist / h)
        class_weights = {}
        for idx, weight in enumerate(weights):
            label = self.Y[idx]
            class_weights[label] = class_weights.get(label, 0) + weight

        return max(class_weights, key=class_weights.get)
    
    def predict(self, X) -> np.ndarray:
        return np.array([self._single_predict(x) for x in X])
    
    def get_neighbours(self, i, proto_idx=None):

        neighbours = np.argsort(self.distances[i])[:-1]
        if proto_idx:
            neighbours = [idx for idx in neighbours if idx in proto_idx]
        return neighbours
    
    def _compute_profile(self, m, proto_idx=None):
        if m > self.n_samples - 1 or (proto_idx is not None and m > len(proto_idx)):
            raise ValueError('m больше чем размер X_train')
        
        summ = 0

        for i in range(self.n_samples):
            y_m = self.Y[self.get_neighbours(i, proto_idx)[m-1]]
            if y_m != self.Y[i]:
                summ += 1
        
        return summ / self.n_samples
    
    def compute_ccv(self, proto_idx=None):
        L = self.n_samples
        if proto_idx is None:
            proto_idx = list(range(L))
        else:
            proto_idx = list(proto_idx)

        ell = len(proto_idx)
        if ell == 0 or ell >= L:
            raise ValueError("Размер ell должен быть от 1 до L-1")

        k = L - ell

        total = 0.0
        for i in range(L):
            neighbours = self.get_neighbours(i, proto_idx)
            for m in range(1, min(k, len(neighbours)) + 1):
                y_m = self.Y[neighbours[m - 1]]
                if y_m != self.Y[i]:
                    num = math.comb(L - 1 - m, ell - 1) if (L - 1 - m >= ell - 1 and ell - 1 >= 0) else 0
                    den = math.comb(L - 1, ell) if (L - 1 >= ell) else 0
                    weight = num / den if den > 0 else 0
                    total += weight

        return total / L
