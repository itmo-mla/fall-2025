import numpy as np
from tqdm import tqdm


class MetricsEstimator:
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.accuracy = self.get_accuracy(y_true, y_pred)
        self.precision = self.get_precision(y_true, y_pred)
        self.recall = self.get_recall(y_true, y_pred)
        self.f1_score = self.get_f1_score(y_true, y_pred)

    def get_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum(y_true == y_pred) / len(y_true)

    def get_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) * (y_pred == 1))
        fp = np.sum((y_true == -1) * (y_pred == 1))
        return tp / (tp + fp)

    def get_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) * (y_pred == 1))
        fn = np.sum((y_true == 1) * (y_pred == -1))
        return tp / (tp + fn)

    def get_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = self.get_precision(y_true, y_pred)
        recall = self.get_recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall)

    def __str__(self):
        return f"accuracy = {self.accuracy}\nprecision = {self.precision}\nrecall = {self.recall}\nf1_score = {self.f1_score}"


class NAGOptimizer:
    def __init__(self, weights: np.ndarray, learning_rate: float, l2_reg_param: float, gamma: float):
        self.weights = weights
        self.v = np.zeros_like(self.weights)
        self.learning_rate = learning_rate
        self.l2_reg_param = l2_reg_param
        self.gamma = gamma

    def step(self, current_grad: np.ndarray) -> None:
        self.v = self.gamma * self.v + (1 - self.gamma) * current_grad
        np.subtract(self.weights, self.learning_rate * current_grad, out=self.weights)
        np.subtract(self.weights, self.learning_rate * self.v, out=self.weights)


class LinearClassifier:
    def __init__(
        self, learning_rate: float = 0.1, l2_reg_param: float = 1.0, Q_param: float = 0.001, gamma: float = 0.9
    ):
        self.learning_rate = learning_rate
        self.l2_reg_param = l2_reg_param
        self.Q_param = Q_param
        self.gamma = gamma
        self.weights = None
        self.Q = None

    def _init_weights(self, n_features: int, weights_init_method: str, X: np.ndarray = None, y: np.ndarray = None, n_attempts = None):
        if weights_init_method == "random":
            self.weights = np.random.randn(n_features)
        elif weights_init_method == "corr":
            self.weights = np.dot(X.T, y) / np.sum(X * X, axis=0)
        elif weights_init_method == "multistart":
            best_Q = None
            best_weights = None
            for _ in range(n_attempts):
                self.weights = np.random.randn(n_features)
                current_Q = self._calculate_accurate_Q(X, y)
                if best_Q is None or current_Q < best_Q:
                    best_Q = current_Q
                    best_weights = self.weights
            self.weights = best_weights

    def _calculate_accurate_Q(self, X: np.ndarray, y: np.ndarray) -> None:
        return np.array([self._get_loss_by_sample(pair[0], pair[1]) for pair in zip(X, y)]).mean()

    def _init_Q(self, X: np.ndarray, y: np.ndarray, n_subsamples: int = 100) -> None:
        random_samples_idx = np.random.choice(len(X), n_subsamples, replace=False)
        self.Q = self._calculate_accurate_Q(X[random_samples_idx], y[random_samples_idx])

    def _get_margin(self, x: np.ndarray, y: int) -> float:
        return np.dot(self.weights, x) * y

    def _get_batched_margin(self, X: np.ndarray[tuple[int, int]], y: np.ndarray) -> np.ndarray[float]:
        return np.dot(self.weights, X.T) * y

    def _get_loss_by_sample(self, x: np.ndarray, y: int) -> float:
        return (1 - self._get_margin(x, y)) ** 2 + self.l2_reg_param * np.pow(self.weights, 2).sum()

    def _get_grad_by_sample(self, x: np.ndarray, y: int) -> np.ndarray[float]:
        return (self._get_margin(x, y) - 1) * y * x + self.l2_reg_param * self.weights

    def _get_nag_grad_by_sample(self, nag_optimizer: NAGOptimizer, x: np.ndarray, y: int) -> np.ndarray[float]:
        np.subtract(self.weights, nag_optimizer.learning_rate * nag_optimizer.gamma * nag_optimizer.v, out=self.weights)
        nag_grad = self._get_grad_by_sample(x, y)
        np.add(self.weights, nag_optimizer.learning_rate * nag_optimizer.gamma * nag_optimizer.v, out=self.weights)

        return nag_grad

    def _get_current_Q(self) -> float | None:
        return self.Q

    def _update_Q(self, current_loss: float) -> None:
        self.Q = self.Q_param * current_loss + (1 - self.Q_param) * self.Q

    @staticmethod
    def _shuffle_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        new_indexes = np.random.permutation(len(y))
        return X[new_indexes], y[new_indexes]

    def _get_current_batch_indexes_by_margin(self, X: np.ndarray, y: np.ndarray, n_samples_per_iter: int) -> np.ndarray:
        if len(y) <= n_samples_per_iter:
            return np.arange(len(y))

        margins = np.abs(self._get_batched_margin(X, y))
        scaled_margins = (margins - margins.min()) / (margins.max() - margins.min())
        negative_margins = 1 - scaled_margins
        probs = negative_margins / negative_margins.sum()
        samples_idx = np.random.choice(np.arange(len(y)), size=n_samples_per_iter, replace=False, p=probs)
        return samples_idx

    @staticmethod
    def _get_random_idx(max_idx: int, n: int) -> np.ndarray:
        if n < max_idx:
            return np.random.choice(max_idx, n, replace=False)
        return np.arange(max_idx)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iters=10,
        batch_size=1000,
        stop_threshold: float = 0.1,
        weights_init_method: str = "random",
        batch_generation: str = "random",
        n_attempts: int = None
    ) -> list[float]:
        self._init_weights(X_train.shape[-1], weights_init_method, X_train, y_train, n_attempts)
        self._init_Q(X_train, y_train)

        optimizer = NAGOptimizer(self.weights, self.learning_rate, self.l2_reg_param, self.gamma)

        print(f"Initial weights: {self.weights}")
        print(f"Initial Q: {self.Q}")

        Q = []

        for iter in range(n_iters):
            print(f"Iteration {iter + 1}")
            current_weights = np.sum(self.weights**2)
            current_Q = self._get_current_Q()

            batch_idx = self._shuffle_samples(X_train, y_train)
            untrained_X, untrained_y = X_train, y_train

            while len(untrained_X):
                if batch_generation == "random":
                    batch_idx = self._get_random_idx(len(untrained_X), batch_size)
                elif batch_generation == "margin":
                    batch_idx = self._get_current_batch_indexes_by_margin(untrained_X, untrained_y, batch_size)

                batch_X, batch_y = untrained_X[batch_idx], untrained_y[batch_idx]
                untrained_X = np.delete(untrained_X, batch_idx, axis=0)
                untrained_y = np.delete(untrained_y, batch_idx)

                for res in tqdm(zip(batch_X, batch_y), total=len(batch_y)):
                    sample = res[0]
                    target = res[1]
                    current_loss = self._get_loss_by_sample(sample, target)
                    current_grad = self._get_nag_grad_by_sample(optimizer, sample, target)

                    self._update_Q(current_loss)
                    optimizer.step(current_grad)

            new_weights = np.sum(self.weights**2)
            if np.abs(new_weights - current_weights) / new_weights < stop_threshold:
                print("Early stopping by weights")
                break

            new_Q = self._get_current_Q()
            if np.abs(new_Q - current_Q) / new_Q < stop_threshold or new_Q > current_Q:
                print("Early stopping by Q")
                break
            Q.append(new_Q)

        print(f"Final weights: {self.weights=}")
        print(f"Final Q: {self._get_current_Q()=}")
        return Q


    def run_multistart(
        self,
        n_runs: int,
        X: np.ndarray,
        y: np.ndarray,
        n_iters: int,
        batch_size: int,
        stop_threshold: float,
        weights_init_method: str,
    ) -> None:
        best_Q = None
        best_weights = None
        self.fit(X, y, n_iters, batch_size, stop_threshold, weights_init_method)
        for _ in range(n_runs):
            self.fit(X, y, n_iters, batch_size, stop_threshold, weights_init_method)
            current_Q = self._get_current_Q()

            if best_Q is None or best_Q > current_Q:
                best_Q = current_Q
                best_weights = self.weights

        self.weights = best_weights
        print(f"Best Q: {best_Q}")

    def predict(self, X: np.ndarray, mode: str) -> int | list[int]:
        raw_product = np.dot(self.weights, X.T)

        if mode == "class":
            return np.sign(raw_product)
        elif mode == "probability":
            return raw_product
