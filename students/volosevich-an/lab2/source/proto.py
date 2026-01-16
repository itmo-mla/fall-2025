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
            
            centroid = X_c.mean(axis=0)
            dists = np.linalg.norm(X_c - centroid, axis=1)
            
            # Выбираем n_centroids ближайших к центру
            idx_sorted = np.argsort(dists)
            selected = list(idx_sorted[:self.n_centroids])
            
            if self.n_border > 0:
                X_other = X[y != cls]
                if len(X_other) > 0:
                    dist_to_other = np.min(np.linalg.norm(X_c[:, None, :] - X_other[None, :, :], axis=2), axis=1)
                    
                    # Ищем объекты с минимальным расстоянием к другим классам
                    remaining = [i for i in range(len(X_c)) if i not in selected]
                    
                    while remaining and len(selected) < self.n_centroids + self.n_border:
                        # Берем самый граничный из оставшихся
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
