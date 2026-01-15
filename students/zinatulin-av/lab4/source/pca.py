import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        
    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        F = X - self.mean_
        
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        
        m = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:m]
        self.explained_variance_ = (s[:m] ** 2) / (X.shape[0] - 1)
        
        return self
    
    def transform(self, X):
        X = np.array(X)
        F = X - self.mean_
        return F @ self.components_.T
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
