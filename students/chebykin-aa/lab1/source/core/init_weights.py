from typing import Any

import numpy as np

def init_weights_using_correlation(
    X: np.ndarray, 
    y: np.ndarray,
    eps = 1e-8
)-> np.ndarray:
    """
    Инициализируем веса по формуле: 
    w_j := <y, f_j> / <f_j, f_j>, где f_j - вектор значений признака
    """
    return np.dot(X.T, y) / (np.diag(np.dot(X.T, X)) + eps)

def init_weights_using_multistart(
    X: np.ndarray, 
    loss: Any, 
    n_iters: int = 25
)-> np.ndarray:
    """
    Инициализируем веса по данной стратегии(мультистарт): 
    многократные запуски из разных случайных начальных 
    приближений и выбор лучшего решения.
    """
    best_loss = np.inf 
    best_weights = np.nan 
    for _ in range(n_iters): 
        rand_w = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        loss_value = loss(np.dot(X, rand_w))
        if loss_value < best_loss: 
            best_loss = loss_value 
            best_weights = rand_w

    return best_weights