from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def euclidean_distances_squared(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Pairwise squared Euclidean distances between rows of X and rows of Y.
    Returns shape (X.shape[0], Y.shape[0]).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.maximum(D2, 0.0)


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """
    Gaussian kernel K(u) = exp(-0.5 u^2). Normalizing const is unnecessary for argmax.
    """
    return np.exp(-0.5 * (u * u))


@dataclass
class KNNParzenClassifier:
    """
    KNN with Parzen window of variable width (h = distance to k-th neighbor).
    Prediction uses weighted vote with Gaussian kernel.
    """

    k: int = 5
    eps: float = 1e-12

    X_: np.ndarray | None = None
    y_: np.ndarray | None = None
    classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNParzenClassifier":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        self.classes_ = np.unique(self.y_)
        return self

    def predict_one(self, x: np.ndarray, X: np.ndarray | None = None, y: np.ndarray | None = None) -> int:
        Xtr = self.X_ if X is None else np.asarray(X, dtype=float)
        ytr = self.y_ if y is None else np.asarray(y, dtype=int)
        classes = self.classes_ if self.classes_ is not None else np.unique(ytr)

        x = np.asarray(x, dtype=float).reshape(1, -1)
        d2 = euclidean_distances_squared(x, Xtr).reshape(-1)
        idx = np.argsort(d2)
        k = int(self.k)
        k = min(k, len(idx))
        neigh = idx[:k]
        d = np.sqrt(d2[neigh])
        h = float(d[-1]) if k > 0 else 0.0
        h = max(h, self.eps)

        u = d / h
        w = gaussian_kernel(u)  # (k,)

        scores = {int(c): 0.0 for c in classes}
        for wi, yi in zip(w, ytr[neigh]):
            scores[int(yi)] += float(wi)

        # fallback: if all weights are ~0 (can happen when h is tiny)
        best_c = max(scores.keys(), key=lambda c: scores[c])
        return int(best_c)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x) for x in X], dtype=int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=int)
        yp = self.predict(X)
        return float(np.mean(yp == y))


def loo_cv_risk_knn_parzen(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    show_progress: bool = True,
) -> dict:
    """
    Leave-One-Out cross-validation empirical risk for each k.
    Returns dict: {"k": [...], "risk": [...], "acc": [...]}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = X.shape[0]
    risks = []
    accs = []

    for k in k_values:
        clf = KNNParzenClassifier(k=int(k)).fit(X, y)
        errors = 0
        it = range(n)
        if show_progress and tqdm is not None:
            it = tqdm(it, total=n, desc=f"LOO k={k}")
        for i in it:
            # predict xi using all points except i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            yi = clf.predict_one(X[i], X=X[mask], y=y[mask])
            if yi != y[i]:
                errors += 1
        risk = errors / n
        risks.append(float(risk))
        accs.append(float(1.0 - risk))

    return {"k": list(map(int, k_values)), "risk": risks, "acc": accs}


def condensed_nearest_neighbor(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    max_iters: int = 50,
) -> np.ndarray:
    """
    Prototype selection: Hart's Condensed Nearest Neighbor (CNN).
    Returns indices of selected prototypes.
    Uses 1-NN (unweighted) criterion for adding misclassified points.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    classes = np.unique(y)

    # start with one random sample per class
    S = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        S.append(int(rng.choice(idx_c)))
    S = list(dict.fromkeys(S))  # unique

    def nn_predict_one(x: np.ndarray, Xs: np.ndarray, ys: np.ndarray) -> int:
        d2 = euclidean_distances_squared(x.reshape(1, -1), Xs).reshape(-1)
        j = int(np.argmin(d2))
        return int(ys[j])

    for _ in range(int(max_iters)):
        changed = False
        order = rng.permutation(n)
        for i in order:
            if i in S:
                continue
            yi_pred = nn_predict_one(X[i], X[np.array(S)], y[np.array(S)])
            if yi_pred != y[i]:
                S.append(int(i))
                changed = True
        if not changed:
            break

    return np.array(sorted(set(S)), dtype=int)


def condensed_parzen_knn(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int = 42,
    max_iters: int = 50,
) -> np.ndarray:
    """
    Condensation (CNN-style) but with Parzen-KNN decision rule (variable bandwidth) instead of pure 1-NN.

    Motivation:
    - Classic Hart CNN is tailored to preserve 1-NN behaviour.
    - For Parzen-KNN with variable bandwidth (h=d_(k)), condensing changes local density,
      so using Hart CNN prototypes directly may degrade accuracy.

    This procedure iteratively adds points misclassified by the *current Parzen-KNN on prototypes*,
    aiming to preserve the Parzen-KNN decision rule at fixed k.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    classes = np.unique(y)

    # start with one random sample per class
    S = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        S.append(int(rng.choice(idx_c)))
    S = list(dict.fromkeys(S))

    for _ in range(int(max_iters)):
        changed = False
        order = rng.permutation(n)
        clf = KNNParzenClassifier(k=int(k)).fit(X[np.array(S)], y[np.array(S)])
        for i in order:
            if i in S:
                continue
            yi_pred = clf.predict_one(X[i])
            if yi_pred != y[i]:
                S.append(int(i))
                changed = True
                # update classifier incrementally (cheap enough for Iris-size)
                clf = KNNParzenClassifier(k=int(k)).fit(X[np.array(S)], y[np.array(S)])
        if not changed:
            break

    return np.array(sorted(set(S)), dtype=int)


def stolp_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    remove_bad: bool = True,
    r_scale: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    STOLP-like prototype selection (metric condensation).

    Intuition:
    - for each object i compute nearest "friend" and nearest "enemy"
    - radius r_i is proportional to distance to nearest enemy (how far we can "own" space)
    - iteratively pick objects that cover the largest number of same-class points within r_i

    remove_bad=True removes points with negative margin (enemy closer than friend), acting as noise filter.
    r_scale allows shrinking radii (e.g. 0.5) to be more conservative.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    D2 = euclidean_distances_squared(X, X)
    # ignore self-distances
    np.fill_diagonal(D2, np.inf)
    D = np.sqrt(D2)

    classes = np.unique(y)

    friend = np.full(n, np.inf, dtype=float)
    enemy = np.full(n, np.inf, dtype=float)
    for i in range(n):
        yi = y[i]
        friend[i] = float(np.min(D[i, (y == yi)]))  # self is inf, so ok
        enemy[i] = float(np.min(D[i, (y != yi)]))

    margin = enemy - friend
    keep = np.ones(n, dtype=bool)
    if remove_bad:
        keep = margin >= 0.0

    kept_idx = np.where(keep)[0]
    if len(kept_idx) == 0:
        return np.array([], dtype=int)

    # per-point radius
    r = enemy * float(r_scale)

    prototypes = []
    # cover per-class
    for c in classes:
        remaining = set(map(int, kept_idx[y[kept_idx] == c]))
        if not remaining:
            # ensure at least one prototype per class
            fallback = int(rng.choice(np.where(y == c)[0]))
            prototypes.append(fallback)
            continue

        while remaining:
            rem_list = np.array(sorted(remaining), dtype=int)
            best_i = None
            best_cov = -1

            # choose point that covers max remaining points of same class
            for i in rem_list:
                cov = np.sum((y == c) & keep & (D[i] <= r[i]) & np.isin(np.arange(n), rem_list))
                if cov > best_cov:
                    best_cov = int(cov)
                    best_i = int(i)
            if best_i is None:
                break

            prototypes.append(best_i)
            # remove covered points
            covered = set(map(int, rem_list[(D[best_i, rem_list] <= r[best_i])]))
            remaining -= covered if covered else {best_i}

    return np.array(sorted(set(prototypes)), dtype=int)


def pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Simple PCA to 2D using SVD (numpy-only).
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T  # (d, 2)
    return Xc @ W
