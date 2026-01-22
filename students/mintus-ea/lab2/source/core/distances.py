import numpy as np

def euclidean_distance(x1, x2):
    # Optimized using broadcasting: (a-b)^2 = a^2 + b^2 - 2ab
    x1_sq = np.sum(x1**2, axis=1, keepdims=True)
    x2_sq = np.sum(x2**2, axis=1, keepdims=True)
    
    # dist_sq = x1^2 + x2^2 - 2*x1*x2
    dist_sq = x1_sq + x2_sq.T - 2 * np.dot(x1, x2.T)
    
    # Numerical stability
    dist_sq = np.maximum(dist_sq, 0)
    
    return np.sqrt(dist_sq)
