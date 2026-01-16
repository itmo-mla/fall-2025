# функции ядра
import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0)**degree

def rbf_kernel(x, y, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)