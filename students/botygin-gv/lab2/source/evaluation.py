from sklearn.model_selection import LeaveOneOut


def loo_cross_validation(X, y, k_range, model_class, **model_kwargs):
    errors = []
    for k in k_range:
        knn = model_class(k=k, **model_kwargs)
        loo = LeaveOneOut()
        correct = 0
        for train_index, test_index in loo.split(X):
            X_train_loo, X_test_loo = X[train_index], X[test_index]
            y_train_loo, y_test_loo = y[train_index], y[test_index]
            knn.fit(X_train_loo, y_train_loo)
            pred = knn.predict(X_test_loo)
            if pred[0] == y_test_loo[0]:
                correct += 1
        error = 1 - correct / len(X)
        errors.append(error)
    return errors
