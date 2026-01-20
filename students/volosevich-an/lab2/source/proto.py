import numpy as np


class PrototypeSelector:
    def __init__(self, n_centroids=1, n_border=2):
        self.n_centroids = n_centroids
        self.n_border = n_border
    
    def select(self, X, y):
        X_proto = []
        y_proto = []
        
        classes = np.unique(y)
        
        for cls in classes:
            X_c = X[y == cls]
            
            selected = []
            remaining = list(range(len(X_c)))
            
            centroid_all = X_c.mean(axis=0)
            dists_to_mean = np.linalg.norm(X_c - centroid_all, axis=1)
            first_idx = np.argmin(dists_to_mean)
            selected.append(first_idx)
            remaining.remove(first_idx)
            
            # Жадно перебираем остальные центроиды
            while len(selected) < self.n_centroids and remaining:
                max_min_dist = -1
                best_idx = remaining[0]
                
                for idx in remaining:
                    min_dist = np.min(np.linalg.norm(X_c[idx] - X_c[selected], axis=1))
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = idx
                
                selected.append(best_idx)
                remaining.remove(best_idx)
            
            # Жадно выбираем граничные точки
            if self.n_border > 0:
                X_other = X[y != cls]
                if len(X_other) > 0:
                    dist_to_other = np.min(np.linalg.norm(
                        X_c[:, None, :] - X_other[None, :, :], axis=2), axis=1)
                    
                    while remaining and len(selected) < self.n_centroids + self.n_border:
                        border_idx = remaining[np.argmin(dist_to_other[remaining])]
                        selected.append(border_idx)
                        
                        to_remove = [border_idx]
                        for j in remaining:
                            if j == border_idx:
                                continue
                            if dist_to_other[j] > np.linalg.norm(X_c[j] - X_c[border_idx]):
                                to_remove.append(j)
                        
                        remaining = [r for r in remaining if r not in to_remove]
            
            X_proto.append(X_c[selected])
            y_proto.append(np.full(len(selected), cls))
        
        X_proto = np.vstack(X_proto) if X_proto else np.array([]).reshape(0, X.shape[1])
        y_proto = np.concatenate(y_proto) if y_proto else np.array([])
        
        return X_proto, y_proto
    