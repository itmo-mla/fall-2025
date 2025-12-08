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
