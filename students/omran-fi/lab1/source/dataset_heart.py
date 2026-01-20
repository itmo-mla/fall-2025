import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


FEATURES = [
    "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh",
    "exng", "oldpeak", "slp", "caa", "thall"
]
TARGET = "output"


def add_bias(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


def load_heart_csv(path: str, test_size: float = 0.2, seed: int = 42):
    """
    Returns:
        X_train, X_test, y_train, y_test
    Notes:
        - y is in {-1, +1}
        - X includes bias column x0=1
        - Features scaled to (0,1)
    """
    df = pd.read_csv(path)

    # labels: {0,1} -> {-1,+1}
    y = df[TARGET].replace({0: -1, 1: 1}).to_numpy(dtype=float).reshape(-1, 1)
    X = df[FEATURES].to_numpy(dtype=float)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = add_bias(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test
