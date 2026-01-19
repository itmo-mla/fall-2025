import numpy as np
from sklearn.datasets import fetch_california_housing, make_regression


def generate_dataset(n_samples=500, n_features=10, effective_rank=5, noise=2.0, random_state=42):

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=effective_rank,
        noise=noise,
        random_state=random_state
    )
    data = X.astype(np.float64)
    target = y.astype(np.float64)
    return data, target

def load_dataset(name):
    if name == "housing":
        dataset = fetch_california_housing()
    else:
        raise ValueError("Неизвестное название")
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.float64)
    return X, y
