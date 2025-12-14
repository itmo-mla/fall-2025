import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, load_iris


def load_dataset(name):
    if name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "iris":
        data = load_iris()
    else:
        raise ValueError("Неизвестный датасет")
    mask = (data.target == 0) | (data.target == 1)
    X = data.data[mask, :2]
    y = data.target[mask]
    y = np.where(y == 0, -1, 1)
    return X, y
