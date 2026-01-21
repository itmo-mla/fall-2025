import numpy as np
from sklearn.metrics import f1_score


def loo_optimal_k(
        model_class,
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


def greedy_prototype_selection(model, X, y, max_prototypes, e=0.1):
    selected_indices = []
    current_best_score = 0
    
    for _ in range(max_prototypes):
        scores = []
        for i in range(len(X)):
            if i not in selected_indices:
                trial_indices = selected_indices + [i]
                model.fit(X[trial_indices], y[trial_indices])
                # CCV: оцениваем точность на ВСЕЙ исходной выборке
                y_pred = model.predict(X)
                scores.append((f1_score(y, y_pred, average='micro'), i))
        
        best_step_score, best_idx = max(scores)
        if best_step_score > (current_best_score - e):
            current_best_score = best_step_score
            selected_indices.append(best_idx)
        else:
            break # Улучшений больше нет
    
    return selected_indices
