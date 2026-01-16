import numpy as np


# ACTIVATION FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------------

def sigmoid(x):
    # stable sigmoid
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def linear(x):
    return x

def softmax(x):
    """
    Stable softmax.
    Supports shapes: (n_classes, 1) or (n_samples, n_classes).
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def d_sigmoid(s):
    # derivative given sigmoid(s) value
    return s * (1 - s)

def d_relu(r):
    return np.where(r > 0, 1, 0)

def d_tanh(t):
    return 1 - t ** 2

def d_leaky_relu(x):
    return np.where(x > 0, 1, 0.01)

def d_linear(x):
    return np.ones_like(x)

def d_softmax(x):
    # Full Jacobian is not used in our training loops; placeholder for API compatibility.
    return np.ones_like(x)

activation_functions = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'leaky_relu': leaky_relu,
    'linear': linear,
    'softmax': softmax,
}

derivative_functions = {
    'sigmoid': d_sigmoid,
    'relu': d_relu,
    'tanh': d_tanh,
    'leaky_relu': d_leaky_relu,
    'linear': d_linear,
    'softmax': d_softmax,
}



# LOSS FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------------

def MSE_1n(y, y_pred):
    return 0.5*(y-y_pred)**2

def d_MSE_1n(y, y_pred):
    return -2*(y - y_pred)

def categorical_cross_entropy(y, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    # y can be one-hot vector (n_classes,) or (n_classes,1)
    return -np.sum(y * np.log(y_pred))

def d_categorical_cross_entropy(y, y_pred, eps=1e-12):
    # Rarely used directly; for softmax+CE we use (y_pred - y).
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -(y / y_pred)

losses = {
    'mse': MSE_1n,
    'categorical_cross_entropy': categorical_cross_entropy,
}

d_losses = {
    'mse': d_MSE_1n,
    'categorical_cross_entropy': d_categorical_cross_entropy,
}