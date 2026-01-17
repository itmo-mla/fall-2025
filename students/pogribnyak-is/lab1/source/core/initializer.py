import numpy as np


class WeightInitializer:
    @staticmethod
    def default_bias(): return 0.0

    def initialize(self, **kwargs): raise NotImplementedError


class RandomInitializer(WeightInitializer):
    def __init__(self, scale: float = 0.01):
        self.scale = scale

    def initialize(self, X=None, y=None, n_features=None, seed=None, **_):
        if seed is not None: np.random.seed(seed)
        if n_features is None and X is not None: n_features = X.shape[1]
        w = np.random.normal(0, self.scale, size=n_features).astype(np.float32)
        return w, self.default_bias()


class CorrelationInitializer(WeightInitializer):
    def initialize(self, X=None, y=None, **_):
        correlations = np.array([
            np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0
            for i in range(X.shape[1])
        ])
        norm = np.linalg.norm(correlations)
        w = (correlations / norm * 0.1) if norm > 0 else np.zeros(X.shape[1])
        return w.astype(np.float32), self.default_bias()

