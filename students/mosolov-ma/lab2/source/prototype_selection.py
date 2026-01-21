from knn import KNN
from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
import numpy as np
from metrics import Metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PrototypeSelection:
    def __init__(self, min_prototypes=1):
        self.min_prototypes = min_prototypes
        self.history_ccv = []
        self.removal_order = []

    def fit(self, X, y):
        n = X.shape[0]
        if n < 2:
            raise ValueError("Нужно хотя бы 2 объекта для отбора эталонов")

        knn = KNN()
        knn.fit(X, y)

        mask = np.ones(n, dtype=bool)

        while np.sum(mask) > self.min_prototypes:
            best_idx = None
            best_ccv = float('inf')

            for idx in np.where(mask)[0]:
                mask[idx] = False
                if np.sum(mask) == 0:
                    mask[idx] = True
                    continue

                new_ccv = knn.compute_ccv(proto_idx=np.where(mask)[0])
                if new_ccv < best_ccv:
                    best_ccv = new_ccv
                    best_idx = idx

                mask[idx] = True

            if best_idx is None:
                break
            
            mask[best_idx] = False
            self.removal_order.append(best_idx)
            self.history_ccv.append(best_ccv)

            print(f"Удалён объект {best_idx}, CCV = {best_ccv:.4f}, эталонов: {np.sum(mask)}")
            if np.sum(mask) == 120:
                self.best_mask = np.where(mask)[0]


        final_prototypes = np.where(mask)[0]
        return final_prototypes, self.history_ccv, self.removal_order


if __name__ == "__main__":

    df = load_and_prepare_data()

    X = scale_features(df.drop(columns=['target']))
    y = df['target']

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    k = 15

    selector = PrototypeSelection(min_prototypes=20)
    final_mask, history_ccv, removal_order = selector.fit(X_train, y_train)

    n_removed = np.arange(len(history_ccv))

    plt.figure(figsize=(12, 6))

    plt.plot(n_removed, history_ccv, 'b-o', linewidth=2, markersize=4, label='CCV')


    plt.xlabel('Количество удалённых объектов', fontsize=12)
    plt.ylabel('CCV', fontsize=12)
    plt.title('Зависимость CCV от числа удалённых эталонов\n'
              f'(всего объектов: {len(X_train)})', fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    X_selected = X_train[final_mask, :]
    y_selected = y_train[final_mask]

    etalon_model = KNN(k=17)
    etalon_model.fit(X_selected, y_selected)

    model = KNN(k=17)
    model.fit(X_train, y_train)

    y_pred_etalon = etalon_model.predict(X_test)
    y_pred_model = model.predict(X_test)

    print(f"Accuracy prototypes: {Metrics.accuracy(y_test, y_pred_etalon)}")
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
    train_mask[selector.best_mask] = False 
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
