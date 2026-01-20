import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float((y_true == y_pred).mean())


def confusion_pm_one(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == -1) & (y_pred == -1)).sum())
    fp = int(((y_true == -1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == -1)).sum())
    return tp, tn, fp, fn


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp, tn, fp, fn = confusion_pm_one(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)
