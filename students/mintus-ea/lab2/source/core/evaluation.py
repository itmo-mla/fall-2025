import numpy as np
from .knn import KNN

def leave_one_out_error(X, y, k_values, kernel=None):
    n_samples = len(X)
    errors = {k: 0 for k in k_values}
    
    for i in range(n_samples):
        # Hold out i-th sample
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_val = X[i:i+1]
        y_val = y[i]
        
        for k in k_values:
            # Skip if k > n_train
            if k > len(X_train):
                continue
                
            model = KNN(k=k, kernel=kernel, method='variable_window')
            model.fit(X_train, y_train)
            pred = model.predict(X_val)[0]
            
            if pred != y_val:
                errors[k] += 1
                
    # Normalize
    for k in k_values:
        errors[k] /= n_samples
        
    return errors

def margin(X, y, model, i):
    # Let's extract weights from KNN manually for flexibility
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    x_target = X[i:i+1]
    y_target = y[i]
    
    model.fit(X_train, y_train)
    # Re-implement minimal logic here for margin
    dists = model.X_train - x_target
    dists_sq = np.sum(dists**2, axis=1)
    dists = np.sqrt(dists_sq)
    
    sorted_indices = np.argsort(dists)
    
    # Variable window
    if model.k < len(X_train):
        h = dists[sorted_indices[model.k]]
    else:
        h = dists[sorted_indices[-1]]
    
    neighbors_indices = sorted_indices[:model.k]
    neighbors_dists = dists[neighbors_indices]
    neighbors_labels = model.y_train[neighbors_indices]
    
    weights = model.kernel(neighbors_dists / h) if h > 0 else (neighbors_dists == 0).astype(float)
    
    scores = {c: 0.0 for c in model.classes_}
    for idx, label in enumerate(neighbors_labels):
        scores[label] += weights[idx]
        
    score_true = scores.get(y_target, 0.0)
    score_other = max([s for l, s in scores.items() if l != y_target], default=0.0)
    
    return score_true - score_other
