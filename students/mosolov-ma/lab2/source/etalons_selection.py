from knn import KNN
from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
import numpy as np
from metrics import Metrics

df = load_and_prepare_data()

X = scale_features(df.drop(columns=['target']))
y = df['target']

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split_data(X, y)

k = 15

class PrototypeSelection:
    def __init__(self, k=1):
        self.k = k
        self.history = []

    def accuracy_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def fit(self, X, y):
        n = X.shape[0]
        mask = np.ones(n, dtype=bool)
        model = KNN(k=self.k)

        def compute_empirical_error(mask):
            x_train = X[mask]
            y_train = y[mask]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_train)
            return 1 - self.accuracy_score(y_true=y_train, y_pred=y_pred)

        flag = True
        while flag and np.sum(mask) > 1:
            min_error = float('inf')
            min_index = -1
            for idx in range(n):
                if not mask[idx]:
                    continue
                mask[idx] = False
                error = compute_empirical_error(mask)
                if error < min_error:
                    min_error = error
                    min_index = idx
                mask[idx] = True

            flag = self._should_continue(min_error)
            if flag:
                mask[min_index] = False
                self.history.append(min_error)
            else:
                break

        return mask, self.history

    def _should_continue(self, new_error):
        if not self.history:
            return True
        if new_error < 1e-10:
            print(f"Small new_error: {new_error}")
            return False
        recent_mean = np.mean(self.history[-5:])
        print(f"Current error: {new_error}, Recent mean error: {recent_mean}, Diff: {new_error - recent_mean}")
        return (new_error - recent_mean) < 1e-5

# Пример использования:
selector = PrototypeSelection(k=15)
mask, history = selector.fit(X_train, y_train)
X_selected = X_train[mask]
y_selected = y_train[mask]


etalon_model = KNN(k=17)
etalon_model.fit(X_selected, y_selected)

model = KNN(k=17)
model.fit(X_train, y_train)

y_pred_etalon = etalon_model.predict(X_test)
y_pred_model = model.predict(X_test)

print(f"Accuracy etalon: {Metrics.accuracy(y_test, y_pred_etalon)}")
print(f"Accuracy model: {Metrics.accuracy(y_test, y_pred_model)}")