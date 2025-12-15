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
            d = np.linalg.norm(X_c - centroid, axis=1)
            idx_centroid = np.argsort(d)[:self.n_centroids]

            X_proto.append(X_c[idx_centroid])
            y_proto.append(np.full(len(idx_centroid), cls))

            X_other = X[y != cls]
            if X_other.shape[0] == 0 or self.n_border == 0:
                continue

            dist_to_other = np.min(np.linalg.norm(X_c[:, None, :] - X_other[None, :, :], axis=2), axis=1)
            idx_border = np.argsort(dist_to_other)[:self.n_border]

            X_proto.append(X_c[idx_border])
            y_proto.append(np.full(len(idx_border), cls))

        X_proto = np.vstack(X_proto)
        y_proto = np.concatenate(y_proto)

        return X_proto, y_proto
