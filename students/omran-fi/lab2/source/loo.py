import numpy as np
from .knn import my_KNN


def loo_risk_for_k(X, y, k, mode="parzen_variable"):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    knn = my_KNN(neighbours=k, mode=mode)
    n = len(X)
    errors = 0

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        knn.fit(X_train, y_train)
        pred = knn.predict_single(X[i])
        if pred != y[i]:
            errors += 1

    return errors / n


def select_k_by_loo(X, y, k_values, mode="parzen_variable"):
    risks = np.zeros(len(k_values), dtype=float)
    for idx, k in enumerate(k_values):
        risks[idx] = loo_risk_for_k(X, y, k, mode=mode)

    best_idx = int(np.argmin(risks))
    return int(k_values[best_idx]), risks
