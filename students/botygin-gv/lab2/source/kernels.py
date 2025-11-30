import numpy as np


def gaussian_kernel(x):
    return np.exp(-0.5 * x ** 2)
