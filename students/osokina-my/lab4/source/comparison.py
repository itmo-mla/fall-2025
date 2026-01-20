from sklearn.decomposition import PCA as sklearn_PCA


def compare_with_sklearn(pca_svd, X_scaled, X_pca_svd):
    pca_sklearn = sklearn_PCA()
    X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)

    print("Сравнение реализаций:")
    print("=" * 60)

    print("\n1. Главные компоненты:")
    print(f"   Наша реализация: {pca_svd.components_.shape}")
    print(f"   sklearn: {pca_sklearn.components_.shape}")

    print("\n2. Объясненная дисперсия:")
    print(f"   Наша реализация: {pca_svd.explained_variance_[:5]}")
    print(f"   sklearn: {pca_sklearn.explained_variance_[:5]}")

    print("\n3. Отношение объясненной дисперсии:")
    print(f"   Наша реализация: {pca_svd.explained_variance_ratio_[:5]}")
    print(f"   sklearn: {pca_sklearn.explained_variance_ratio_[:5]}")

    print("\n4. Преобразованные данные:")
    print(f"   Наша реализация: {X_pca_svd.shape}")
    print(f"   sklearn: {X_pca_sklearn.shape}")
    
    return pca_sklearn, X_pca_sklearn

