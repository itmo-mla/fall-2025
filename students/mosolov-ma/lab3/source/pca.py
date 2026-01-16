from sklearn.decomposition import PCA

def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    return pca, principal_components
