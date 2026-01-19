from knn import KNN
from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
import numpy as np
from metrics import Metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


if __name__ == "__main__":

    df = load_and_prepare_data()

    X = scale_features(df.drop(columns=['target']))
    y = df['target']

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    k = 15

    selector = PrototypeSelection(k=17)
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

    print(f"Number of objects: {len(X_train)}")
    print(f"Number of prototypes: {len(X_selected)}")


    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_selected_pca = pca.transform(X_selected)


    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}


    plt.figure(figsize=(12, 10))


    train_mask = np.ones(len(X_train), dtype=bool)
    train_mask[mask] = False 
    if np.any(train_mask):
        plt.scatter(X_train_pca[train_mask, 0], X_train_pca[train_mask, 1],
                c=[class_to_color[label] for label, m in zip(y_train, train_mask) if m],
                marker='o', s=30, alpha=0.6, edgecolors='black', linewidth=0.4,
                label='Обычные тренировочные')

    plt.scatter(X_selected_pca[:, 0], X_selected_pca[:, 1],
            c=[class_to_color[label] for label in y_selected],
            marker='^', s=180, alpha=1.0, edgecolors='black', linewidth=2,
            label=f'Прототипы ({len(X_selected)})')

    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
            c=[class_to_color[label] for label in y_test],
            marker='s', s=60, alpha=0.8, edgecolors='darkred', linewidth=1.2,
            label='Тестовые данные')

    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title(f'KNN: Прототипы vs Все данные (k={k})\n'
            f'Прототипов: {len(X_selected)} / {len(X_train)} ({100*len(X_selected)/len(X_train):.1f}%)', 
            fontsize=14, pad=20)

    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.95)
    plt.tight_layout()
    plt.show()
