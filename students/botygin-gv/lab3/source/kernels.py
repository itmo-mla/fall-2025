import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def polynomial_kernel(x1, x2, degree=3, coef0=1.0):
    return (np.dot(x1, x2) + coef0) ** degree

KERNELS = {
    'linear': linear_kernel,
    'rbf': rbf_kernel,
    'poly': polynomial_kernel
}