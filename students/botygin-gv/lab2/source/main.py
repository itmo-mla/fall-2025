import numpy as np
import os
from sklearn.model_selection import train_test_split
from models import ParzenWindowKNN
from selection import PrototypeSelection
from evaluation import loo_cross_validation
from visualization import plot_risk, visualize_prototype_selection
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def compare_with_sklearn(X_train, y_train, X_test, y_test, k=3):
    sk_knn = KNeighborsClassifier(n_neighbors=k)
    sk_knn.fit(X_train, y_train)
    sk_pred = sk_knn.predict(X_test)
    sk_accuracy = np.mean(sk_pred == y_test)
    return sk_accuracy


def main():
    data = load_wine()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Подбор параметра k методом LOO
    k_range = range(1, min(40, len(X)))
    print("Подбор оптимального k с помощью LOO")
    errors = loo_cross_validation(X, y, k_range, ParzenWindowKNN)
    optimal_k = k_range[np.argmin(errors)]
    print(f"Оптимальное k: {optimal_k}")

    plot_risk(errors, k_range)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    knn = ParzenWindowKNN(k=optimal_k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy = np.mean(pred == y_test)
    print(f"\nТочность ParzenWindowKNN (k={optimal_k}): {accuracy:.4f}")

    # Сравнение с sklearn
    sk_accuracy = compare_with_sklearn(X_train, y_train, X_test, y_test, k=optimal_k)
    print(f"Точность sklearn KNeighborsClassifier: {sk_accuracy:.4f}")

    # Отбор эталонов
    print("\nЗапуск алгоритма отбора эталонов...")
    ps = PrototypeSelection()
    X_reduced, y_reduced = ps.fit(X_train, y_train)
    print(f"Размер исходного набора: {len(X_train)}")
    print(f"Размер отобранного набора: {len(X_reduced)}")

    # Оценка KNN на отобранных эталонах
    knn_reduced = ParzenWindowKNN(k=optimal_k)
    knn_reduced.fit(X_reduced, y_reduced)
    pred_reduced = knn_reduced.predict(X_test)
    acc_reduced = np.mean(pred_reduced == y_test)
    print(f"Точность KNN с отбором эталонов: {acc_reduced:.4f}")
    print(f"Точность KNN без отбора эталонов: {accuracy:.4f}")

    # Визуализация отобранных эталонов через LDA
    visualize_prototype_selection(X, y, X_reduced, y_reduced)


if __name__ == "__main__":
    main()
