import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)

    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': np.array([[tn, fp],
                                      [fn, tp]])
    }

    return metrics


def print_metrics(metrics: dict, name: str = "Model"):
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")

    cm = metrics['confusion_matrix']
    labels = [0, 1]

    print("Confusion Matrix:")
    print("        Predicted")
    print("Actual  " + "  ".join(f"{l:>5}" for l in labels))
    for i, row in enumerate(cm): print(f"{labels[i]:>6}  " + "  ".join(f"{x:>5}" for x in row))