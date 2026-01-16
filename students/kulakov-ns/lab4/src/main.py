from pca import CustomPCA

import numpy as np
import matplotlib.pyplot as plt


def make_synthetic(seed: int = 0, n_samples: int = 2000, n_features: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_regression

    X, y = make_regression(
    n_samples=2000,
    n_features=40,
    n_informative=5,
    effective_rank=5,
    tail_strength=0.0,
    noise=5.0,
    random_state=0,
)
    return X.astype(float), y.astype(float)


def draw_scatter(model, model_name: str, X: np.ndarray, y: np.ndarray) -> None:
    Z = model.transform(X)
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", alpha=0.7, s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="y")
    plt.title(model_name)
    plt.grid(True, alpha=0.3)

def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got shape={x.shape}")
    return x


def plot_ratio(model, model_name: str):
    plt.figure()
    plt.plot(model.explained_variance_ratio_)
    plt.grid(True)
    plt.xlabel("component index")
    plt.ylabel("explained variance ratio")
    plt.title(model_name)

if __name__ == '__main__':
    X, y = make_synthetic(seed=42)

    X = _as_2d(X)
    Xc = X - X.mean(axis=0)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    from sklearn.decomposition import PCA as SkPCA

    custom_pca = CustomPCA(n_components=20).fit(X)
    sk_pca = SkPCA(n_components=20, svd_solver="full").fit(X)

    print(f"Эффективная размерность выборки: {custom_pca.eff_size}")

    print(f"Первые {custom_pca.eff_size} компонентов CustomPCA: {custom_pca.transform(X)}")
    print(f"Первые {custom_pca.eff_size} компонентов SklearnPCA: {sk_pca.transform(X)}")


    plot_ratio(custom_pca, "CustomPCA")
    plot_ratio(sk_pca, "Sklearn PCA")

    draw_scatter(CustomPCA(n_components=2).fit(X), "CustomPCA", X, y)
    draw_scatter(SkPCA(n_components=2, svd_solver="full").fit(X),"Sklearn PCA", X, y)

    plt.show()
