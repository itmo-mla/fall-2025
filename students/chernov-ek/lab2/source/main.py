from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from source.model import KNN
from source.dataset import load_iris_dataset
from source.selector import loo_optimal_k
from source.visualizer import *


if __name__ == '__main__':
    # Загрузка датасета
    X_train, X_val, X_test, y_train, y_val, y_test = load_iris_dataset()

    # Подбор параметра k
    k_values = range(1, 80)
    best_k, loo_errors = loo_optimal_k(KNN, X_train, y_train, k_values)
    vis_LOO_errors(loo_errors, 15)

    # Подбор параметра k для sklearn модели
    best_k, loo_errors = loo_optimal_k(KNeighborsClassifier, X_train, y_train, k_values)
    vis_LOO_errors(loo_errors, 15)

    # Визуализация отбора эталонов
    visualize_knn_predictions(
        X_train, y_train,
        k_list=[3, 17, 30],
        n_clusters_list=[1, 3, 10],
        knn_class=KNN
    )

    # Сравнение качества KNN с и без отбора эталонов
    model = KNN(17)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    model = KNN(5)
    model.ccv_fit(X_train, y_train, 20)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
