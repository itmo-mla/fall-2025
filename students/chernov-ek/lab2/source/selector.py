import numpy as np
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier

from source.model import KNN


def loo_optimal_k(
        model_class: KNN | KNeighborsClassifier,
        X: np.ndarray,
        y: np.ndarray,
        k_values: list[int] | np.ndarray
    ):
    """
    Подбор оптимального k с помощью Leave-One-Out кросс-валидации
    для KNN с окном Парзена.
    
    :param model_class: Класс модели KNN.
    :type model_class: KNN | KNeighborsClassifier
    :param X: Матрица признаков.
    :type X: np.ndarray формы (n_samples, n_features)
    :param y: Вектор целевых значений.
    :type y: np.ndarray формы (n_samples,)
    :param k_values: Набор возможных значений гиперпараметра *k* для KNN.
    :type k_values: list[int] | np.ndarray
    
    :return: Возвращает k с минимальной ошибкой LOO.
    :rtype: tuple[int, list]
    """
    n_samples = X.shape[0]
    loo_errors = []

    for k in k_values:
        model = model_class(k)
        errors = 0

        for i in range(n_samples):
            # Формируем тренировочные данные без i-й точки
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            X_test = X[i].reshape(1, -1)
            y_test = y[i]

            # Обучаем KNN
            model.fit(X_train, y_train)

            # Предсказываем
            y_pred = model.predict(X_test)[0]

            if y_pred != y_test:
                errors += 1

        loo_error_rate = errors / n_samples
        loo_errors.append(loo_error_rate)
        print(f"k={k}, LOO ошибка={loo_error_rate:.4f}")

    # Выбираем k с минимальной LOO ошибкой
    best_k = k_values[np.argmin(loo_errors)]
    print(f"\nОптимальное k: {best_k} с ошибкой LOO={min(loo_errors):.4f}")
    return best_k, loo_errors

def reduce_training_set_knn(model_class, X_train, y_train, X_val, y_val, k=3, n_clusters=None):
    """
    Итеративное удаление объектов из тренировочной выборки для KNN.
    Пока точность на валидации не падает, объекты удаляются.

    Параметры:
        model_class - класс KNN
        X_train, y_train - тренировочные данные
        X_val, y_val - валидационные данные
        k - число соседей
        n_clusters - если хотим использовать cluster_fit

    Возвращает:
        X_reduced, y_reduced - уменьшенные тренировочные данные
    """

    # Приведение признаков
    X_curr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    X_val_num = X_val.to_numpy() if hasattr(X_val, "to_numpy") else X_val
    y_curr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    y_val = y_val.to_numpy() if hasattr(y_val, "to_numpy") else y_val

    # Начальная точность
    model = model_class(k)
    if n_clusters is not None:
        model.cluster_fit(X_curr.to, y_curr, n_clusters=n_clusters)
    else:
        model.fit(X_curr, y_curr)

    y_pred = model.predict(X_val_num)
    best_acc = np.mean(y_pred == y_val)

    changed = True

    while changed:
        changed = False
        # Проходим по объектам по одному
        for i in range(len(X_curr)):
            X_try = np.delete(X_curr, i, axis=0)
            y_try = np.delete(y_curr, i, axis=0)

            model = model_class(k)
            if n_clusters is not None:
                model.cluster_fit(X_try, y_try, n_clusters=n_clusters)
            else:
                model.fit(X_try, y_try)

            y_pred = model.predict(X_val_num)
            acc = np.mean(y_pred == y_val)

            # Если точность не уменьшилась — удаляем объект
            if acc >= best_acc:
                X_curr = X_try
                y_curr = y_try
                best_acc = acc
                changed = True
                break  # начинаем цикл заново

    return X_curr, y_curr
