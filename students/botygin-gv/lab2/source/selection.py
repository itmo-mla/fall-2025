import numpy as np
from models import ParzenWindowKNN


class PrototypeSelection:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        keep = np.ones(n, dtype=bool)

        changed = True
        while changed:
            changed = False
            current_indices = np.where(keep)[0]
            if len(current_indices) <= 1:
                break

            for idx in reversed(current_indices):
                # Временное множество без idx
                temp_keep = keep.copy()
                temp_keep[idx] = False
                X_temp = X[temp_keep]
                y_temp = y[temp_keep]

                if len(X_temp) == 0:
                    continue

                # Обучаем модель множестве без элемента idx
                knn = ParzenWindowKNN(k=1)
                knn.fit(X_temp, y_temp)

                # Проверяем: правильно ли классифицируется объект idx
                pred = knn.predict([X[idx]])
                if pred[0] == y[idx]:
                    # Удаляем idx
                    keep[idx] = False
                    changed = True
                    break

        return X[keep], y[keep]
