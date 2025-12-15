import numpy as np


class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        classes = np.unique(y_true)
        precisions = []
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            if tp + fp == 0: precisions.append(0.0)
            else: precisions.append(tp / (tp + fp))
        return np.mean(precisions) if average == 'macro' else precisions

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        classes = np.unique(y_true)
        recalls = []
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            if tp + fn == 0: recalls.append(0.0)
            else: recalls.append(tp / (tp + fn))
        return np.mean(recalls) if average == 'macro' else recalls

    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
        precisions = ClassificationMetrics.precision(y_true, y_pred, average=None)
        recalls = ClassificationMetrics.recall(y_true, y_pred, average=None)
        f1s = []
        for p, r in zip(precisions, recalls):
            if p + r == 0: f1s.append(0.0)
            else: f1s.append(2 * p * r / (p + r))
        return np.mean(f1s) if average == 'macro' else f1s

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        classes = np.unique(y_true)
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for t, p in zip(y_true, y_pred):
            cm[class_to_idx[t], class_to_idx[p]] += 1
        return cm

    @staticmethod
    def get_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            'accuracy': ClassificationMetrics.accuracy(y_true, y_pred),
            'precision': ClassificationMetrics.precision(y_true, y_pred),
            'recall': ClassificationMetrics.recall(y_true, y_pred),
            'f1': ClassificationMetrics.f1(y_true, y_pred),
            'confusion_matrix': ClassificationMetrics.confusion_matrix(y_true, y_pred)
        }
