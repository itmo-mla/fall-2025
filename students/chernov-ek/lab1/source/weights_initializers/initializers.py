import numpy as np


def correlated_init():
    pass


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
