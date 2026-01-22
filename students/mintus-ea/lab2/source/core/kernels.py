import numpy as np

def gaussian_kernel(u):
    return np.exp(-0.5 * u**2)

def rectangular_kernel(u):
    return (np.abs(u) <= 1).astype(float)

def triangular_kernel(u):
    return np.maximum(1 - np.abs(u), 0)

def epanechnikov_kernel(u):
    return np.maximum(0.75 * (1 - u**2), 0)

def quartic_kernel(u):
    return np.maximum((15/16) * (1 - u**2)**2, 0)
