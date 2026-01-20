import numpy as np


class PCA_SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components  # главные компоненты
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        
    def fit(self, X):
        X = np.array(X)  # (n_samples, n_features)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.singular_values_ = s
        self.components_ = Vt
        
        # λ = s² / (n-1)
        n_samples = X.shape[0]
        eigenvalues = s ** 2 / (n_samples - 1)
        self.explained_variance_ = eigenvalues
        
        # Отношение объясненной дисперсии
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            self.singular_values_ = self.singular_values_[:self.n_components]
        
        return self
    
    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_
        
        if self.n_components is not None:
            components = self.components_[:self.n_components]
        else:
            components = self.components_
            
        # Проецируем на главные компоненты
        X_transformed = X_centered @ components.T
        
        return X_transformed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

