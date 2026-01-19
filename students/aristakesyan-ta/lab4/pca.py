import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.singular_values_ = S
        self.components_ = Vt
        
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        total_var = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        if self.n_components is not None:
            return X_centered @ self.components_[:self.n_components].T
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        if self.n_components is not None:
            return X_transformed @ self.components_[:self.n_components] + self.mean_
        return X_transformed @ self.components_ + self.mean_
