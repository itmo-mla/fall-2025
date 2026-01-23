import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
from tqdm import tqdm

KernelName = Literal["gaussian", "rectangular", "triangular"]

MIN_LEFT_SAMPLES = 20


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")
    return X.astype(float, copy=False)


def _as_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape={y.shape}")
    return y


@dataclass
class SimpleKNNClassifier:
    """
    kNN классификатор с ядровыми весами.
    Bandwidth h = расстояние до k-го соседа (в LOO и в predict).
    """

    k: Optional[int] = None
    ord: Optional[int] = None
    ker: KernelName = "gaussian"

    # train
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None

    # optional "shrunk" train (после отбора эталонов)
    shrinked_X_train: Optional[np.ndarray] = None
    shrinked_y_train: Optional[np.ndarray] = None

    # precomputed for LOO k-selection + CCV
    distance_matrix: Optional[np.ndarray] = None
    _sorted_indices: Optional[np.ndarray] = None

    # reference elements selection
    ref_elements_indices: Optional[np.ndarray] = None
    ref_elements_indices_history: List[List[int]] = None

    def __post_init__(self) -> None:
        if self.ref_elements_indices_history is None:
            self.ref_elements_indices_history = []

    # -------------------------
    # core: distance + kernel
    # -------------------------
    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return float(np.linalg.norm(x1 - x2, ord=self.ord))

    def _kernel(self, u: float) -> float:
        """
        u = dist / h
        """
        au = abs(u)
        if self.ker == "gaussian":
            return float(np.exp(-2.0 * (u**2)))
        if self.ker == "rectangular":
            return 1.0 if au <= 1.0 else 0.0
        if self.ker == "triangular":
            return float(max(0.0, 1.0 - au))
        raise ValueError(f"Incorrect kernel: {self.ker}")

    # -------------------------
    # fit + choose k by LOO
    # -------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_selection_callback: Optional[Callable[[List[float]], None]] = None,
        max_k: int = 50,
    ) -> "SimpleKNNClassifier":
        X = _as_2d_float(X_train)
        y = _as_1d(y_train)

        if len(X) != len(y):
            raise ValueError(f"X and y lengths differ: {len(X)} vs {len(y)}")
        if len(X) < 2:
            raise ValueError("Need at least 2 samples for kNN")

        self.X_train = X
        self.y_train = y
        self.shrinked_X_train = None
        self.shrinked_y_train = None

        n = len(X)
        self.ref_elements_indices = np.arange(n, dtype=int)

        # precompute distances (for LOO selection + CCV)
        # (n,n) matrix; diag = 0
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            # small vectorization inside row (still O(n^2), но чище)
            diffs = X - X[i]
            if self.ord is None:
                D[i] = np.sqrt(np.sum(diffs * diffs, axis=1))
            else:
                D[i] = np.linalg.norm(diffs, ord=self.ord, axis=1)
        self.distance_matrix = D

        # sorted neighbor indices for each i
        self._sorted_indices = np.argsort(D, axis=1).astype(int)

        # pick k by LOO if needed
        if self.k is None:
            k_max = min(max_k, n - 1)  # k neighbors excluding itself
            scores: List[float] = []
            for k in range(1, k_max + 1):
                scores.append(self._loo_score_for_k(k))
            best_idx = int(np.argmax(scores))
            self.k = best_idx + 1
            if k_selection_callback is not None:
                k_selection_callback(scores)

        # validate chosen k
        if self.k < 1 or self.k > n - 1:
            raise ValueError(f"Invalid k={self.k} for n={n}")
        return self

    def _loo_score_for_k(self, k: int) -> float:
        """
        LOO accuracy for fixed k, using precomputed distances and sorting.
        Weighted voting with bandwidth h = distance to k-th neighbor (excluding self).
        Voting is done over k nearest neighbors (excluding self).
        """
        assert self.X_train is not None and self.y_train is not None
        assert self.distance_matrix is not None and self._sorted_indices is not None

        n = len(self.X_train)
        correct = 0

        for i in range(n):
            order = self._sorted_indices[i]
            # order[0] is i itself (distance 0)
            neigh = order[1 : k + 1]  # k neighbors, excluding self
            # bandwidth = distance to k-th neighbor
            h = self.distance_matrix[i, order[k]]  # index k => k-th neighbor after self
            if h <= 0:
                # в вырожденном случае (дубликаты) — делаем h=1, чтобы не делить на 0
                h = 1.0

            votes: Dict[float, float] = {}
            for j in neigh:
                u = self.distance_matrix[i, j] / h
                w = self._kernel(float(u))
                cls = self.y_train[j]
                votes[cls] = votes.get(cls, 0.0) + w

            # If all weights ended up zero (e.g. compact kernels + weird h), fallback to unweighted majority
            if not votes or max(votes.values()) <= 0.0:
                # unweighted majority among neigh
                labels = self.y_train[neigh]
                vals, cnts = np.unique(labels, return_counts=True)
                pred = vals[int(np.argmax(cnts))]
            else:
                pred = max(votes, key=votes.get)

            correct += int(pred == self.y_train[i])

        return correct / n

    # -------------------------
    # predict
    # -------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.k is None:
            raise ValueError("Model is not fitted (k is None). Call fit() first.")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        Xq = _as_2d_float(X_test)

        Xtr = (
            self.shrinked_X_train if self.shrinked_X_train is not None else self.X_train
        )
        ytr = (
            self.shrinked_y_train if self.shrinked_y_train is not None else self.y_train
        )

        n = len(Xtr)
        k = int(self.k)
        if k < 1 or k > n:
            raise ValueError(f"Invalid k={k} for train size n={n}")

        preds: List[float] = []

        for x in Xq:
            # compute distances to all train points
            diffs = Xtr - x
            if self.ord is None:
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            else:
                dists = np.linalg.norm(diffs, ord=self.ord, axis=1)

            order = np.argsort(dists)
            neigh = order[:k]  # k nearest neighbors
            # bandwidth = distance to k-th neighbor in this test query
            h = float(dists[order[k - 1]])
            if h <= 0:
                h = 1.0

            votes: Dict[float, float] = {}
            for j in order:
                u = float(dists[j]) / h
                w = self._kernel(u)
                cls = ytr[j]
                votes[cls] = votes.get(cls, 0.0) + w

            if not votes or max(votes.values()) <= 0.0:
                labels = ytr[neigh]
                vals, cnts = np.unique(labels, return_counts=True)
                pred = vals[int(np.argmax(cnts))]
            else:
                pred = max(votes, key=votes.get)

            preds.append(pred)

        return np.asarray(preds)

    # -------------------------
    # CCV reference elements logic (as in your code, but cleaned)
    # -------------------------
    @staticmethod
    def _comb_factor(L: int, l: int, m: int) -> float:
        # guard for invalid comb arguments
        if L - 1 - m < l - 1:
            return 0.0
        return math.comb(L - 1 - m, l - 1) / math.comb(L - 1, l - 1)

    def _ccv_on_ref_elements(self, ref_indices: List[int]) -> float:
        """
        Complete cross-validation score computed on a set of reference elements.
        """
        if self.X_train is None or self.y_train is None or self._sorted_indices is None:
            raise ValueError("Model must be fitted before CCV.")

        MAX_CONTROL_SIZE = 3
        m_size = min(MAX_CONTROL_SIZE, len(ref_indices))
        if m_size <= 1:
            return 0.0

        n = len(self.X_train)
        ref_mask = np.zeros(n, dtype=bool)
        ref_mask[np.asarray(ref_indices, dtype=int)] = True

        p = np.zeros(m_size, dtype=float)

        for i in range(n):
            sorted_idx = self._sorted_indices[i]
            is_ref = ref_mask[sorted_idx]
            ref_positions = np.where(is_ref)[0]

            for m in range(1, m_size):
                if m <= len(ref_positions):
                    pos = ref_positions[m - 1]
                    # ensure we don't take self (position 0) if it happens to be ref
                    if pos == 0 and m < len(ref_positions):
                        pos = ref_positions[m]
                    if pos < len(sorted_idx):
                        ref_j = sorted_idx[pos]
                        p[m] += float(self.y_train[ref_j] != self.y_train[i])

        for m in range(1, m_size):
            p[m] *= self._comb_factor(n, n - m_size, m)

        return float(np.sum(p))

    def adjust_ref_by_ccv(
        self, ccv_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Iteratively remove elements from reference list by minimizing CCV score.
        Stores history of reference sets.
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before adjust_ref_by_ccv().")
        if self.ref_elements_indices is None:
            self.ref_elements_indices = np.arange(len(self.X_train), dtype=int)

        self.ref_elements_indices_history = []

        n = len(self.X_train)
        max_iter = max(0, n - MIN_LEFT_SAMPLES)

        current = list(map(int, self.ref_elements_indices.tolist()))

        for _ in tqdm(range(max_iter)):
            best_score = float("inf")
            best_set: Optional[List[int]] = None

            # try removing each element once
            for i in range(len(current)):
                candidate = current[:i] + current[i + 1 :]
                score = self._ccv_on_ref_elements(candidate)
                if score < best_score:
                    best_score = score
                    best_set = candidate

            if best_set is None:
                break

            current = best_set
            self.ref_elements_indices = np.asarray(current, dtype=int)
            self.ref_elements_indices_history.append(current)

            if ccv_callback is not None:
                ccv_callback(best_score)

    def shrink_x_by_ref_elements(self, removed_samples: int) -> None:
        """
        Use the reference set at step `removed_samples` from history
        and build shrinked_X_train/shrinked_y_train.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted first.")
        if not self.ref_elements_indices_history:
            raise ValueError(
                "No reference history found. Call adjust_ref_by_ccv() first."
            )
        if removed_samples < 0 or removed_samples >= len(
            self.ref_elements_indices_history
        ):
            raise ValueError(
                f"removed_samples out of range: {removed_samples}, "
                f"history size={len(self.ref_elements_indices_history)}"
            )

        idx = np.asarray(self.ref_elements_indices_history[removed_samples], dtype=int)
        self.shrinked_X_train = self.X_train[idx]
        self.shrinked_y_train = self.y_train[idx]
