import numpy as np
from .knn import KNN
from .evaluation import margin

class STOLP:
    def __init__(self, k=5, kernel=None, margin_threshold=-0.1, error_tolerance=0):
        self.k = k
        self.kernel = kernel
        self.margin_threshold = margin_threshold
        self.selected_indices = []
        self.model = KNN(k=k, kernel=kernel, method='variable_window')

    def fit(self, X, y):
        n_samples = len(X)
        self.classes_ = np.unique(y)
        
        # 1. Compute LOO margins
        margins = []
        for i in range(n_samples):
            # We use a temporary model trained on X \ {x_i} inside margin function
            m = margin(X, y, self.model, i)
            margins.append(m)
        margins = np.array(margins)
        
        clean_indices = np.where(margins > self.margin_threshold)[0]
        
        if len(clean_indices) == 0:
            # Fallback if all margins are bad
            clean_indices = np.arange(n_samples)
            
        X_clean = X[clean_indices]
        y_clean = y[clean_indices]
        
        sorted_clean_indices = clean_indices[np.argsort(margins[clean_indices])[::-1]]
        
        # Initialize Omega (selected set)
        omega_indices = []
        
        # Add one representative for each class (max margin)
        for c in self.classes_:
            class_indices = [idx for idx in sorted_clean_indices if y[idx] == c]
            if class_indices:
                omega_indices.append(class_indices[0])
                
        # Iteratively add objects that are misclassified by current Omega
        while True:
            # Train model on current Omega
            X_omega = X[omega_indices]
            y_omega = y[omega_indices]
            self.model.fit(X_omega, y_omega)
            
            # Check classification on all clean objects
            misclassified = []
            preds = self.model.predict(X_clean)
            
            for i, idx in enumerate(clean_indices):
                if idx in omega_indices:
                    continue
                
                if preds[i] != y[idx]:
                    misclassified.append(idx)
            
            if not misclassified:
                break
                
            # Add the first misclassified
            omega_indices.append(misclassified[0])
            
            # Safety break to avoid infinite loops
            if len(omega_indices) == len(clean_indices):
                break
                
        self.selected_indices = np.array(omega_indices)
        return self

    def get_prototypes(self):
        return self.selected_indices
