from __future__ import annotations

import numpy as np


def to01(y_pm: np.ndarray) -> np.ndarray:
    y_pm = np.asarray(y_pm).reshape(-1).astype(int)
    return (y_pm == 1).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def confusion_matrix_pm(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    For y in {-1,+1}. Returns [[TN, FP],[FN, TP]] w.r.t. positive class = +1.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def roc_curve_from_scores(y_true_pm: np.ndarray, scores_pos: np.ndarray, num_thresholds: int = 200):
    """
    y_true in {-1,+1}, scores_pos in [0,1] (probability of +1).
    """
    y01 = to01(y_true_pm)
    scores_pos = np.asarray(scores_pos, dtype=float).reshape(-1)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    tpr = []
    fpr = []

    P = max(int(np.sum(y01 == 1)), 1)
    N = max(int(np.sum(y01 == 0)), 1)

    for thr in thresholds:
        pred01 = (scores_pos >= thr).astype(int)
        tp = int(np.sum((pred01 == 1) & (y01 == 1)))
        fp = int(np.sum((pred01 == 1) & (y01 == 0)))
        fn = int(np.sum((pred01 == 0) & (y01 == 1)))
        tn = int(np.sum((pred01 == 0) & (y01 == 0)))

        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)

    return np.array(fpr), np.array(tpr), thresholds


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    # NumPy >= 2.0
    return float(np.trapezoid(y[order], x[order]))

