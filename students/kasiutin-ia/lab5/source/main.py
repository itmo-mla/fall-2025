import numpy as np


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


class LogReg:
    def __init__(self, method: str):
        self.learning_rate = None
        self.weights = None
        self.Q = None
        self.method = method


    def _fit_nr(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.2, n_iter: int = 10) -> list[float]:
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        Q = []
        for i in range(n_iter):
            sigma = self._sigmoid(self.weights @ X.T * y)

            grad = - np.linalg.inv(X.T @ np.diag((1 - sigma) * sigma) @ X + np.eye(X.shape[1]) * 1e-4) @ X.T @ (y / sigma)


            self.weights -= learning_rate * grad

            new_Q = self._get_Q(X, y)
            print(f"Iteration {i + 1}: Q = {new_Q}")
            Q.append(new_Q)

        return Q
    

    def _fit_irls(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.2, n_iter: int = 10) -> list[float]:
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        Q = []
        for i in range(n_iter):
            sigma = self._sigmoid(self.weights @ X.T * y)
            gamma = np.sqrt((1 - sigma) * sigma)

            F_hat = np.diag(gamma) @ X
            y_hat = y * np.sqrt((1 - sigma) / sigma)

            grad = np.linalg.inv(F_hat.T @ F_hat) @ F_hat.T @ y_hat


            self.weights += learning_rate * grad

            new_Q = self._get_Q(X, y)
            print(f"Iteration {i + 1}: Q = {new_Q}")
            Q.append(new_Q)

        return Q


    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.2, n_iter: int = 10) -> None:
        if self.method == "nr":
            return self._fit_nr(X, y, learning_rate, n_iter)

        elif self.method == "irls":
            return self._fit_irls(X, y, learning_rate, n_iter)


    def _get_Q(self, X: np.ndarray, y_true: np.ndarray) -> float:
        margins = self.weights @ X.T * y_true
        return np.sum(np.log(1 + np.exp(-margins)))
    

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
        

    def predict(self, X: np.ndarray, raw: bool = False) -> np.ndarray:
        if not raw:
            return np.sign(self.weights @ X.T)
        
        return self.weights @ X.T
    
