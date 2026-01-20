
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import comb

class Compactness:
    def __init__(self, X, y):
        
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.L = len(y)

    def _nearest_neighbors(self, Omega):
        X_Omega = self.X[Omega]
        distances = cdist(self.X, X_Omega)
        sorted_idx = np.argsort(distances, axis=1)
        return Omega[sorted_idx] 

    def compactness_profile(self, Omega):
        
        neighbors = self._nearest_neighbors(Omega)
        L = self.L
        Pi = np.zeros(len(Omega))

        for m in range(len(Omega)):
            mismatch = self.y != self.y[neighbors[:, m]]
            Pi[m] = np.mean(mismatch)
        return Pi

    def CCV(self, Omega, l, k=None):
        if k is None:
            k = len(Omega)

        neighbors = self._nearest_neighbors(Omega)[:, :k]

       
        mismatches = self.y[:, None] != self.y[neighbors]
        mismatches = mismatches.astype(float)

        m = np.arange(1, k+1)

        l = self.L - len(Omega)

       
        weights = comb(self.L - 1 - m, l-1) / comb(self.L - 1,l)

        return np.mean(mismatches @ weights)
