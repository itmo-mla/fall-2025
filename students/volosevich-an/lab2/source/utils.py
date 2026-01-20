import matplotlib.pyplot as plt
import numpy as np


def visualize_prototypes(X, y, X_proto, y_proto):
    X_centered = X - X.mean(axis=0)

    cov = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:2]]
    X_2d = X_centered @ eigvecs
    X_proto_2d = (X_proto - X.mean(axis=0)) @ eigvecs
    mask_proto = np.array([tuple(x) in set(map(tuple, X_proto)) for x in X])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_2d[~mask_proto, 0],
        X_2d[~mask_proto, 1],
        c=y[~mask_proto],
        cmap="coolwarm",
        s=30,
        label="Original points",
        alpha=0.6
    )
    plt.scatter(
        X_proto_2d[:, 0],
        X_proto_2d[:, 1],
        c=y_proto,
        cmap="coolwarm",
        marker='*',
        s=200,
        edgecolor='black',
        label="Prototypes"
    )
    plt.legend()
    plt.title("PCA 2D Visualization of Prototypes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_empirical_risk_k(model, X_train, y_train, X_test, y_test, k_values):
    risks = []                  

    for k in k_values:
        model.k = k
        model.fit(X_train, y_train)

        y_pred = np.array([model.predict(x) for x in X_test])
        risk = 1 - np.mean(y_pred == y_test)
        risks.append(risk)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, risks, marker='o')
    plt.xlabel("k")
    plt.ylabel("Empirical Risk Value")
    plt.title("Empirical Risk per k")
    plt.grid(True)
    plt.show()

    return np.array(risks)

def plot_empirical_risk_h(model, X_train, y_train, X_test, y_test, h_values):

    risks = []

    for h in h_values:
        model.h = h 
        model.fit(X_train, y_train)

        y_pred = np.array([model.predict(x) for x in X_test])

        risk = 1 - np.mean(y_pred == y_test)
        risks.append(risk)

    plt.figure(figsize=(8, 5))
    plt.plot(h_values, risks, marker='o')
    plt.xlabel("h")
    plt.ylabel("Empirical Risk Value")
    plt.title("Empirical Risk per h")
    plt.grid(True)
    plt.show()

    return np.array(risks)
