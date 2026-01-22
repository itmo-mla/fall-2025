from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_binary_dataset(test_size: float = 0.25, random_state: int = 42):
    """
    Loads a binary dataset from sklearn at runtime (no large files in repo).
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    Labels are returned as {-1, +1}.
    """
    data = load_breast_cancer()
    X = data.data
    y01 = data.target  # 0/1
    feature_names = list(data.feature_names)

    # convert to {-1, +1}
    y = np.where(y01 == 1, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names
