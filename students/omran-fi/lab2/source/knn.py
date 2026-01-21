import numpy as np
from scipy.spatial.distance import cdist


class my_KNN:
    """
    KNN classifier with:
      - simple voting
      - Parzen window with fixed bandwidth
      - Parzen window with variable bandwidth (h = distance to k-th neighbor)
    """

    def __init__(self, neighbours=1, mode="simple", h=2.0, eps=1e-12):
        self.neighbours = int(neighbours)
        self.h = float(h)
        self.mode = mode
        self.eps = float(eps)

    def gaussian_kernel(self, d, h):
        return np.exp(-0.5 * (d / (h + self.eps)) ** 2)

    def parzen_window(self, all_dist, k_distances):
        """
        all_dist: distances from all train points to x, shape (n_train, 1)
        k_distances: distances of k nearest neighbors to x, shape (k,)
        """
        if self.mode == "simple":
            return np.ones_like(k_distances)

        if self.mode == "parzen_variable":
            sorted_d = np.sort(all_dist, axis=0)  # (n,1)

            idx = min(max(self.neighbours - 1, 0), sorted_d.shape[0] - 1)
            h = float(sorted_d[idx, 0])
            h = max(h, self.eps)
            return (1 / np.sqrt(2 * np.pi)) * self.gaussian_kernel(k_distances, h)

        if self.mode == "parzen_fixed":
            return (1 / np.sqrt(2 * np.pi)) * self.gaussian_kernel(k_distances, self.h)

        raise ValueError(f"Unknown mode: {self.mode}")

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).astype(int)
        self.classes_ = np.unique(self.y_train)
        self.n_classes_ = len(self.classes_)

    def predict_single(self, x):
        x = np.asarray(x).reshape(1, -1)
        dist = cdist(self.X_train, x)  

        k = self.neighbours
        nn_ids = np.argsort(dist, axis=0)[:k].flatten()
        nn_distances = dist[nn_ids, 0]
        nn_labels = self.y_train[nn_ids]

        weights = self.parzen_window(dist, nn_distances)

        class_sum = np.zeros(self.n_classes_, dtype=float)
        for w, lbl in zip(weights, nn_labels):
            class_sum[lbl] += w

        return int(np.argmax(class_sum))

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        pred = np.empty(X_test.shape[0], dtype=int)
        for i in range(X_test.shape[0]):
            pred[i] = self.predict_single(X_test[i])
        return pred
