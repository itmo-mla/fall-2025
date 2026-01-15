import numpy as np

class KNNParzen:
    def __init__(self, k=5, eps=1e-8):
        self.k = int(k)
        self.eps = eps

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def _gaussian_kernel(self, distances, h):
        h = max(h, self.eps)
        q = (distances / h) ** 2
        return np.exp(-2 * q)

    def _predict_point(self, x):
        dists = np.linalg.norm(self.X - x, axis=1)
        if self.k <= 0:
            h = np.max(dists) + self.eps
        else:
            sorted_idxs = np.argsort(dists)
            kth_idx = sorted_idxs[min(self.k - 1, len(dists) - 1)]
            h = dists[kth_idx]
            if h < self.eps:
                zeros = np.where(dists <= self.eps)[0]
                if zeros.size > 0:
                    vals, counts = np.unique(self.y[zeros], return_counts=True)
                    return vals[np.argmax(counts)]
                h = self.eps
        weights = self._gaussian_kernel(dists, h)
        class_weights = {c: 0.0 for c in self.classes_}
        for w, label in zip(weights, self.y):
            class_weights[label] += w
        return max(class_weights.items(), key=lambda t: t[1])[0]


    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_point(x) for x in X]
        return np.array(preds)