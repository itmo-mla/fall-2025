import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X):
        X = np.array(X)
        self.n_samples_, self.n_features_ = X.shape
        
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. SVD
        # X_centered = U * S * Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Explained Variance
        # variance = S^2 / (n_samples - 1)
        explained_variance = (S ** 2) / (self.n_samples_ - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        
        # Store all components initially
        self.components_ = Vt
        self.singular_values_ = S
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        
        # 4. Truncate if needed
        if self.n_components is not None:
            if isinstance(self.n_components, int):
                k = self.n_components
            elif isinstance(self.n_components, float) and 0 < self.n_components < 1.0:
                # Select components to reach explained variance ratio
                cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
                k = np.searchsorted(cumulative_variance_ratio, self.n_components) + 1
            else:
                k = min(self.n_samples_, self.n_features_)

            self.components_ = self.components_[:k]
            self.singular_values_ = self.singular_values_[:k]
            self.explained_variance_ = self.explained_variance_[:k]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:k]
            
        return self

    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components_) + self.mean_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
