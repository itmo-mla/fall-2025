import numpy as np
from collections import Counter
from .distances import euclidean_distance
from .kernels import gaussian_kernel

class KNN:
    def __init__(self, k=5, kernel=None, method='variable_window'):
        self.k = k
        self.kernel = kernel
        self.method = method
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=self.y_train.dtype)
        
        # Compute distances to all training points
        dists = euclidean_distance(X, self.X_train) # (N_test, N_train)
        
        for i in range(n_samples):
            # Sort distances
            sorted_indices = np.argsort(dists[i])
            sorted_dists = dists[i][sorted_indices]
            
            # Select k neighbors
            neighbors_indices = sorted_indices[:self.k]
            neighbors_dists = sorted_dists[:self.k]
            neighbors_labels = self.y_train[neighbors_indices]
            
            if self.kernel is None:
                # Standard KNN: Majority vote
                counter = Counter(neighbors_labels)
                y_pred[i] = counter.most_common(1)[0][0]
            else:
                # Weighted KNN (Parzen)
                weights = np.zeros(self.k)
                
                if self.method == 'variable_window':
                    if self.k < len(self.X_train):
                        h = dists[i][sorted_indices[self.k]]
                    else:
                        h = sorted_dists[-1]
                    
                    if h == 0:
                        # Fallback for duplicates: assign weight 1 to distance 0
                        weights = (neighbors_dists == 0).astype(float)
                    else:
                        weights = self.kernel(neighbors_dists / h)
                        
                elif self.method == 'fixed_window':
                    h = 1.0 # Placeholder
                    weights = self.kernel(neighbors_dists / h)
                
                # Weighted voting
                class_scores = {c: 0.0 for c in self.classes_}
                for idx, label in enumerate(neighbors_labels):
                    class_scores[label] += weights[idx]
                
                # Select max score
                y_pred[i] = max(class_scores, key=class_scores.get)
                
        return y_pred

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
