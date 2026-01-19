from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, degree=3):
    return (np.dot(x, y) + 0.0) ** degree

def rbf_kernel(x, y, gamma):
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))

def calculate_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    split: str,
    file_path: str
):
    """Метод, вычисляющий качество предсказания моделей"""
    f1 = f1_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)

    metrics_str = (
        f"split: {split}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"\n"
    )

    # Write to file
    with open(file_path, "a") as file:
        file.write(metrics_str)
