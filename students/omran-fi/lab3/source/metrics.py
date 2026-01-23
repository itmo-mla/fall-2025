from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class Metrics:
    accuracy: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    precision_neg: float
    recall_neg: float
    f1_neg: float
    f1_macro: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """
    y_true, y_pred in {-1, +1}
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)

    # per-class
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    prec_neg = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    rec_neg = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    f1_neg = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return Metrics(
        accuracy=float(acc),
        precision_pos=float(prec_pos),
        recall_pos=float(rec_pos),
        f1_pos=float(f1_pos),
        precision_neg=float(prec_neg),
        recall_neg=float(rec_neg),
        f1_neg=float(f1_neg),
        f1_macro=float(f1_macro),
    )


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """
    Returns confusion matrix as nested list with fixed label order [-1, +1]
    """
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    return cm.tolist()
