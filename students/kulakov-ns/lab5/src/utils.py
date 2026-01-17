import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def neg_log_likelihood(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    z = X @ w
    p = sigmoid(z)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grad_nll(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    p = sigmoid(X @ w)
    return X.T @ (p - y)


def hess_nll(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    p = sigmoid(X @ w)
    W = p * (1 - p)
    return X.T @ (X * W[:, None])


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def load_dataset(seed: float):
    data = load_breast_cancer()
    X_raw = data.data
    y = data.target.astype(float)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train_i = add_intercept(X_train)
    X_test_i = add_intercept(X_test)

    return (X_train, y_train), (X_test, y_test), (X_train_i, y_train), (X_test_i, y_test)