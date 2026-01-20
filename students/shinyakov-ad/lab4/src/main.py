import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from model.pca import PCA
from data_load.data_load import load_raw_dataframe

def find_effective_dimension(pca_model, threshold=0.95):
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    effective_dim = np.argmax(cumulative_variance >= threshold) + 1
    return effective_dim

def compare_with_sklearn(X_train, X_test):
    n_components = min(X_train.shape[0], X_train.shape[1])
    
    custom_pca = PCA(n_components=n_components)
    X_train_custom = custom_pca.fit_transform(X_train.values)
    X_test_custom = custom_pca.transform(X_test.values)
    
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_train_sklearn = sklearn_pca.fit_transform(X_train.values)
    X_test_sklearn = sklearn_pca.transform(X_test.values)
    
    diff_train = np.abs(X_train_custom - X_train_sklearn)
    diff_test = np.abs(X_test_custom - X_test_sklearn)
    
    print(f"Mean difference on train: {np.mean(diff_train):.10f}")
    print(f"Mean difference on test: {np.mean(diff_test):.10f}")
    
    explained_var_diff = np.abs(custom_pca.explained_variance_ratio_ - sklearn_pca.explained_variance_ratio_)
    print(f"Max difference in explained variance ratio: {np.max(explained_var_diff):.10f}")
    
    return custom_pca, sklearn_pca

def visualize_results(custom_pca, sklearn_pca, X_train, y_train, X_test, y_test):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    n_components = len(custom_pca.explained_variance_ratio_)
    components_range = np.arange(1, n_components + 1)
    
    axes[0, 0].plot(components_range, custom_pca.explained_variance_ratio_, 'o-', label='Custom PCA', linewidth=2)
    axes[0, 0].plot(components_range, sklearn_pca.explained_variance_ratio_, 's--', label='Sklearn PCA', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Explained Variance Ratio by Component')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    cumulative_custom = np.cumsum(custom_pca.explained_variance_ratio_)
    cumulative_sklearn = np.cumsum(sklearn_pca.explained_variance_ratio_)
    axes[0, 1].plot(components_range, cumulative_custom, 'o-', label='Custom PCA', linewidth=2)
    axes[0, 1].plot(components_range, cumulative_sklearn, 's--', label='Sklearn PCA', linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    effective_dim = find_effective_dimension(custom_pca, threshold=0.95)
    print(f"\nEffective dimension (95% variance): {effective_dim}")
    
    mse_scores = []
    mse_scoress = []
    n_comp_range = range(1, min(11, n_components + 1))
    for n_comp in n_comp_range:
        custom_pca_n = PCA(n_components=n_comp)
        X_train_reduced = custom_pca_n.fit_transform(X_train.values)
        X_test_reduced = custom_pca_n.transform(X_test.values)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_reduced, y_train.values)
        y_pred = ridge.predict(X_test_reduced)
        mse = mean_squared_error(y_test.values, y_pred)
        mse_scores.append(mse)

        custom_pca_n = SklearnPCA(n_components=n_comp)
        X_train_reduced = custom_pca_n.fit_transform(X_train.values)
        X_test_reduced = custom_pca_n.transform(X_test.values)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_reduced, y_train.values)
        y_pred = ridge.predict(X_test_reduced)
        mse = mean_squared_error(y_test.values, y_pred)
        mse_scoress.append(mse)
    
    axes[1, 0].plot(list(n_comp_range), mse_scores, 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Ridge Regression MSE with custom PCA')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(components_range[:10], custom_pca.explained_variance_[:10], alpha=0.7, label='Custom PCA')
    axes[1, 1].bar(components_range[:10], sklearn_pca.explained_variance_[:10], alpha=0.7, label='Sklearn PCA')
    axes[1, 1].set_xlabel('Component')
    axes[1, 1].set_ylabel('Explained Variance')
    axes[1, 1].set_title('Explained Variance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    axes[2, 0].plot(list(n_comp_range), mse_scores, 'o-', linewidth=2, markersize=8)
    axes[2, 0].set_xlabel('Number of Components')
    axes[2, 0].set_ylabel('MSE')
    axes[2, 0].set_title('Ridge Regression MSE with sklearn PCA')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].bar(components_range[:10], custom_pca.explained_variance_[:10], alpha=0.7, label='Custom PCA')
    axes[2, 1].bar(components_range[:10], sklearn_pca.explained_variance_[:10], alpha=0.7, label='Sklearn PCA')
    axes[2, 1].set_xlabel('Component')
    axes[2, 1].set_ylabel('Explained Variance')
    axes[2, 1].set_title('Explained Variance')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('artifacts/pca_results.png', dpi=300, bbox_inches='tight')
    print("\nResults saved to pca_results.png")

if __name__ == '__main__':
    X, y = load_raw_dataframe()
    
    print("Dataset shape:", X.shape)
    print("Target shape:", y.shape)
    print("\n" + "="*50)
    print("Comparing Custom PCA with Sklearn PCA")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    custom_pca, sklearn_pca = compare_with_sklearn(X_train, X_test)
    
    print("\n" + "="*50)
    print("Visualization")
    print("="*50)
    
    visualize_results(custom_pca, sklearn_pca, X_train, y_train, X_test, y_test)