import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(z)

def convert_labels(arr: np.ndarray) -> np.ndarray:
    unique = np.unique(arr)

    if set(unique).issubset({0, 1}):
        return arr.astype(int)

    if set(unique).issubset({-1, 1}):
        return (arr == 1).astype(int)

def calculate_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    file_path: str
):
    """Вычисляет качество предсказания моделей"""
    preds = convert_labels(preds)
    labels = convert_labels(labels)

    f1 = f1_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)

    metrics_str = (
        f"Accuracy: {accuracy:.5f}\n"
        f"Precision: {precision:.5f}\n"
        f"Recall: {recall:.5f}\n"
        f"F1 Score: {f1:.5f}\n\n"
    )

    with open(file_path, "a") as file:
        file.write(metrics_str)