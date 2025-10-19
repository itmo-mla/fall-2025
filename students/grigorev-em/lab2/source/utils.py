import numpy as np
def accuracy_score(y_pred, y_true):
    return np.sum(y_pred == y_true) / y_pred.shape[0]