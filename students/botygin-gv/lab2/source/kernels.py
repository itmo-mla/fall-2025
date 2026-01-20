import numpy as np


def gaussian_kernel(x):
    return np.exp(-2 * x ** 2)
