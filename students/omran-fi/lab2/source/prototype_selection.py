import numpy as np
from scipy.spatial.distance import cdist
from .knn import my_KNN


def stolp_select_prototypes(X, y, max_prototypes=None, remove_noise=True, noise_threshold=0.0):
    """
    STOLP-like prototype selection (simplified).
    Returns indices of selected prototypes in ORIGINAL X.

    - remove_noise: remove samples with margin < noise_threshold
    - then iteratively add worst misclassified samples under 1-NN
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n = len(X)
    all_idx = np.arange(n)

    keep_mask = np.ones(n, dtype=bool)

    if remove_noise:
        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)

        for i in range(n):
            same = (y == y[i])
            diff = ~same
            df = np.min(D[i, same]) if np.any(same) else np.inf
            de = np.min(D[i, diff]) if np.any(diff) else np.inf
            margin = de - df
            if margin < noise_threshold:
                keep_mask[i] = False

    Xk = X[keep_mask]
    yk = y[keep_mask]
    idxk = all_idx[keep_mask]

    classes = np.unique(yk)


    Dk = cdist(Xk, Xk)
    np.fill_diagonal(Dk, np.inf)

    proto_local = []
    for c in classes:
        inds = np.where(yk == c)[0]
        best_i = inds[0]
        best_margin = -np.inf
        for i in inds:
            same = (yk == c)
            diff = ~same
            df = np.min(Dk[i, same]) if np.any(same) else np.inf
            de = np.min(Dk[i, diff]) if np.any(diff) else np.inf
            margin = de - df
            if margin > best_margin:
                best_margin = margin
                best_i = i
        proto_local.append(int(best_i))

    proto_local = list(sorted(set(proto_local)))

    knn = my_KNN(neighbours=1, mode="simple")

    max_iters = len(Xk) if max_prototypes is None else int(max_prototypes)
    for _ in range(max_iters):
        Xp = Xk[proto_local]
        yp = yk[proto_local]
        knn.fit(Xp, yp)

        preds = knn.predict(Xk)
        mis = np.where(preds != yk)[0]
        if len(mis) == 0:
            break


        worst_i = int(mis[0])
        worst_score = -np.inf

        Xp = Xk[proto_local]
        yp = yk[proto_local]

        for i in mis:
            i = int(i)
            true_c = yk[i]
            pred_c = preds[i]
            d = cdist(Xp, Xk[i].reshape(1, -1))[:, 0]
            dt = np.min(d[yp == true_c]) if np.any(yp == true_c) else np.inf
            dp = np.min(d[yp == pred_c]) if np.any(yp == pred_c) else np.inf
            score = dt - dp
            if score > worst_score:
                worst_score = score
                worst_i = i

        if worst_i not in proto_local:
            proto_local.append(worst_i)

        if max_prototypes is not None and len(proto_local) >= max_prototypes:
            break

    proto_idx = idxk[np.array(proto_local, dtype=int)]
    return np.unique(proto_idx)
