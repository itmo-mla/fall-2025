def accuracy_score(y_pred, y_true):
    return (y_pred == y_true) / y_pred.shape[0]