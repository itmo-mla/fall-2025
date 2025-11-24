import numpy as np

class LinearClassifier:
    def __init__(
        self, 
        learning_rate=0.001, 
        lm=0.001, 
        gamma=0.9,
        l2_param=0.00001, 
        epochs=10000,
        sampling_strategy="random", 
        random_state=None,
        init_method="random"
    ):
        self.h = learning_rate
        self.lm = lm
        self.gamma = gamma
        self.l2_param = l2_param
        self.epochs = epochs
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.init_method = init_method
        self._rng = np.random.default_rng(random_state)
        self.Q_plot = []

    def get_margin(self, w, xi, yi):
        return np.dot(w, xi) * yi

    def _get_loss(self, w, xi, yi):
        return (1. - self.get_margin(w, xi, yi))**2 + self.l2_param * np.sum(w ** 2)

    def _get_df(self, w, xi, yi):
        return -2. * (1. - self.get_margin(w, xi, yi)) * xi * yi + self.l2_param * w

    def _init_weights(self, X, y):
        match self.init_method:
            case "correlation":
                corrs = np.array([np.corrcoef(X[:, j], y)[0, 1] if np.std(X[:, j]) > 0 else 0 for j in range(X.shape[1])])
                corrs = np.nan_to_num(corrs)
                if np.sum(np.abs(corrs)) > 0:
                    self.w = corrs / np.linalg.norm(corrs)
                else:
                    self.w = self._rng.standard_normal(X.shape[1])
            case "random":
                self.w = self._rng.standard_normal(X.shape[1])
            case _:
                raise ValueError(f"Unknown weights init method: {self.init_method}")
        self.v = np.zeros(X.shape[1])

    def _choose_index(self, X, y):
        match self.sampling_strategy:
            case "random":
                return self._rng.integers(len(X))
            case "margin":
                margins = np.abs([self.get_margin(self.w, xi, yi) for xi, yi in zip(X, y)])
                eps = 1e-8
                probs = 1 / (margins + eps)
                probs /= probs.sum()
                return self._rng.choice(len(X), p=probs)
            case _:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _sgd_step(self, X, y):
        k = self._choose_index(X, y)
        future_w = self.w - self.h * self.gamma * self.v
        err = self._get_loss(future_w, X[k], y[k])
        self.v = self.gamma * self.v + (1 - self.gamma) * self._get_df(future_w, X[k], y[k])
        self.w = self.w * (1. - self.h * self.l2_param) - self.h * self.v
        self.Q = self.lm * err + (1. - self.lm) * self.Q
        self.Q_plot.append(self.Q)

    def fit(self, X, y):
        self._init_weights(X, y)
        self.Q = np.mean([self._get_loss(self.w, xi, yi) for xi, yi in zip(X, y)])
        self.Q_plot = [self.Q]

        for _ in range(self.epochs):
            self._sgd_step(X, y)

    def multistart_fit(self, X, y, n_restarts=5):
        best_Q = np.inf
        best_w = None
        best_Q_plot = None

        for _ in range(n_restarts):
            self.fit(X, y)
            if self.Q < best_Q:
                best_Q = self.Q
                best_w = self.w.copy()
                best_Q_plot = self.Q_plot.copy()

        self.w = best_w
        self.Q = best_Q
        self.Q_plot = best_Q_plot

    def predict(self, X):
        return [np.sign(np.dot(self.w, xi)) for xi in X]
    