import os
import numpy as np
import matplotlib.pyplot as plt

from model import KNNParzen
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from util import *

def main():
    os.makedirs("artifacts", exist_ok=True)

    wine = datasets.load_wine()
    X = wine.data
    y = wine.target

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ks = list(range(1, 30))
    loo_results = loo_select_k(Xs, y, ks)

    plt.figure()
    plt.plot(list(loo_results.keys()), list(loo_results.values()), marker='o')
    plt.xlabel("k (расстояние до k-го соседа)")
    plt.ylabel("LOO accuracy")
    plt.title("LOO-подбор k для KNN-Parzen (Wine)")
    plt.grid(True)
    plt.savefig("artifacts/loo_k_selection.png", dpi=200)
    plt.close()

    best_k = max(loo_results, key=loo_results.get)
    print("Best k =", best_k)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.3, random_state=42
    )

    clf_parzen = KNNParzen(k=2)
    clf_parzen.fit(X_train, y_train)
    preds_parzen = clf_parzen.predict(X_test)
    parzen_acc = accuracy_score(y_test, preds_parzen)
    print("Parzen KNN acc:", parzen_acc)

    for w in ["uniform", "distance"]:
        kn = KNeighborsClassifier(n_neighbors=best_k, weights=w)
        kn.fit(X_train, y_train)
        acc = accuracy_score(y_test, kn.predict(X_test))
        print(f"sklearn KNN (weights={w}) acc:", acc)

    idxs_cnn = condensed_nn(Xs, y)
    idxs_enn = edited_nn(Xs, y, k=3)

    print("CNN prototypes:", len(idxs_cnn), "/", len(y))
    print("ENN prototypes:", len(idxs_enn), "/", len(y))

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(X2[mask, 0], X2[mask, 1], label=f"class {cls}", alpha=0.6)
    plt.scatter(X2[idxs_cnn, 0], X2[idxs_cnn, 1], c="black", marker="x", label="CNN prototypes")
    plt.title("PCA проекция Wine + CNN прототипы")
    plt.legend()
    plt.grid(True)
    plt.savefig("artifacts/pca_cnn.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(X2[mask, 0], X2[mask, 1], label=f"class {cls}", alpha=0.6)
    plt.scatter(X2[idxs_enn, 0], X2[idxs_enn, 1], c="black", marker="x", label="ENN prototypes")
    plt.title("PCA проекция Wine + ENN прототипы")
    plt.legend()
    plt.grid(True)
    plt.savefig("artifacts/pca_enn.png", dpi=200)
    plt.close()

    clf_parzen_cnn = KNNParzen(k=1)
    clf_parzen_cnn.fit(Xs[idxs_cnn], y[idxs_cnn])
    acc_cnn = accuracy_score(y, clf_parzen_cnn.predict(Xs))
    print("Parzen KNN on CNN prototypes acc:", acc_cnn)

    clf_parzen_enn = KNNParzen(k=3)
    clf_parzen_enn.fit(Xs[idxs_enn], y[idxs_enn])
    acc_enn = accuracy_score(y, clf_parzen_enn.predict(Xs))
    print("Parzen KNN on ENN prototypes acc:", acc_enn)


if __name__ == "__main__":
    main()
