import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def polynomial_kernel(x1, x2, degree=3, c=1):
    return (np.dot(x1, x2.T) + c) ** degree

def rbf_kernel(x1, x2, gamma=0.5):
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    if x1.ndim == 1: x1 = x1.reshape(1, -1)
    if x2.ndim == 1: x2 = x2.reshape(1, -1)
    
    x1_sq = np.sum(x1**2, axis=1, keepdims=True)
    x2_sq = np.sum(x2**2, axis=1, keepdims=True)
    dists = x1_sq + x2_sq.T - 2 * np.dot(x1, x2.T)
    return np.exp(-gamma * dists)
