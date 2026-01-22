import numpy as np


def euclidean_distance(a, b):
    """
    Вычисляем евклидово расстояние между двумя векторами
    """
    return np.sqrt(np.sum((a - b) ** 2))


def gaussian_kernel(u):
    """
    Гауссово ядро K(u) = exp(-0.5 * u^2) / sqrt(2*pi)
    """
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
