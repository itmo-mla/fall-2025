import numpy as np
from typing import Callable

# from source.models import ABCModel


def correlation_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Инициализация весов линейного слоя на основе корреляции признаков с целевой переменной.

    Args:
        X: numpy.ndarray, shape (n_samples, n_features)
        y: numpy.ndarray, shape (n_samples,)

    Returns:
        W: numpy.ndarray, shape (1, n_features)
    """
    n_samples, n_features = X.shape

    # Вычисляем корреляцию каждого признака с y
    W = np.zeros(n_features)
    for i in range(n_features):
        xi = X[:, i]
        # Корреляция Пирсона
        if np.std(xi) == 0:
            W[i] = 0  # если признак постоянный, корреляция = 0
        else:
            W[i] = np.corrcoef(xi, y)[0, 1]
    
    # Преобразуем в форму (1, n_features)
    W = W.reshape(1, n_features)
    return W


def multistart_init(
        model,
        X: np.ndarray, y: np.ndarray,
        loss_func: Callable,
        n_starts=10
    ):
    """
    Мультистарт-инициализация: многократные случайные запуски и выбор лучшего результата.

    :param model: Модель с методами get_weights_layers() и __call__(X) для предсказания.
    :param X: Входные данные (n_samples, n_features).
    :param y: Целевая переменная (n_samples, ...).
    :param loss_func: Функция потерь: loss = loss_func(y_true, y_pred).
    :param n_starts: Количество случайных стартов.
    
    :return: best_W, best_b — списки весов и смещений для всех слоев модели. Для слоев без смещения b будет None.
    """
    best_loss = None
    best_W = None
    best_b = None

    model.eval()
    for _ in range(n_starts):
        new_W, new_b = [], []
        for layer in model.get_weights_layers():
            W, b = layer.get_weights()
            out_features, in_features = W.shape
            weights = np.array([random_numbers_init(in_features if b is None else in_features + 1) for _ in range(out_features)])
            W = weights[:, :-1] if b else weights
            b = weights[:, -1] if b else None

            layer.update_weights(W, b)
            new_W.append(W)
            new_b.append(b)

        y_pred = model(X)
        loss = loss_func(y, y_pred)

        if (best_loss is None) or (loss < best_loss):
            best_loss = loss
            best_W = new_W.copy()
            if b is not None:
                best_b = new_b.copy()

    for layer, W, b in zip(model.get_weights_layers(), best_W, best_b):
        layer.update_weights(W, b)

    return model


def random_numbers_init(n_in, sigma=None):
    """
    Эвристическая инициализация весов случайными числами из Var(w).
    
    :param n_in: Количество входов слоя.
    :type n_in: int
    :param sigma: Стандартное отклонение (если None, можно выбрать 1/√n_in).
    :type sigma: float | None
    
    :return: W — матрица весов размерности (n_out, n_in).
    :rtype: np.ndarray
    """
    if sigma is None:
        sigma = 1.0 / np.sqrt(n_in)
    
    W = np.random.normal(0.0, sigma, n_in)
    return W


def xavier_init(n_in, n_out):
    """
    Инициализация весов по методу Ксавье.

    :param n_in: Количество входов слоя.
    :param n_out: Количество выходов слоя.

    :return: W — матрица весов размерности (n_out, n_in).
    :rtype: np.ndarray
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    W = np.random.uniform(-limit, limit, n_in)
    return W


def he_init(n_in):
    """
    Инициализация весов по методу He.
    
    :param n_in: Количество входов слоя.
    :type n_in: int

    :return: W — матрица весов размерности (n_out, n_in).
    :rtype: np.ndarray
    """
    stddev = np.sqrt(2.0 / n_in)
    W = np.random.normal(0.0, stddev, n_in)
    return W
