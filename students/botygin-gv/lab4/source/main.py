from pca import CustomPCA
from loader import load_dataset, generate_dataset
from evaluate import evaluate
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def main():
    X, y = load_dataset("housing")
    print(f"Original data shape: {X.shape}")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    custom_pca = CustomPCA()
    custom_pca.fit(X)

    eff_dim = custom_pca.effective_dimension(threshold=0.95)
    print(f"Effective dimension (95% variance): {eff_dim}")

    custom_pca_ed = CustomPCA(n_components=eff_dim)
    X_reduced = custom_pca_ed.fit_transform(X)
    print(f"Reduced data shape: {X_reduced.shape}")

    metrics = evaluate(X, X_reduced, y)
    print("\nLinear Regression Performance:")
    print(f"Original data: MSE: {metrics['original']['MSE']:.4f}, R2: {metrics['original']['R2']:.4f}")
    print(f"PCA-reduced data: MSE: {metrics['pca']['MSE']:.4f}, R2: {metrics['pca']['R2']:.4f}")

    sklearn_pca = PCA(n_components=eff_dim)
    X_sklearn = sklearn_pca.fit_transform(X)
    metrics = evaluate(X, X_sklearn, y)
    print(f"PCA-reduced data (sklearn): MSE: {metrics['pca']['MSE']:.4f}, R2: {metrics['pca']['R2']:.4f}")


if __name__ == "__main__":
    main()
