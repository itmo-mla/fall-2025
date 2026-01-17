from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsEstimator:

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> "MetricsEstimator":
        self.accuracy = self.get_accuracy(y_true, y_pred)
        self.precision = self.get_precision(y_true, y_pred)
        self.recall = self.get_recall(y_true, y_pred)
        self.f1_score = self.get_f1_score(y_true, y_pred)
        return self

    @staticmethod
    def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float((y_true == y_pred).mean())

    @staticmethod
    def get_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == -1) & (y_pred == 1)).sum()
        denom = tp + fp
        return float(tp / denom) if denom > 0 else 0.0

    @staticmethod
    def get_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == -1)).sum()
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0

    @classmethod
    def get_f1_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = cls.get_precision(y_true, y_pred)
        recall = cls.get_recall(y_true, y_pred)
        denom = precision + recall
        return float(2 * precision * recall / denom) if denom > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"accuracy  = {self.accuracy}\n"
            f"precision = {self.precision}\n"
            f"recall    = {self.recall}\n"
            f"f1_score  = {self.f1_score}"
        )
