from typing import Literal, Iterable, Optional
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import plotly.graph_objects as go

#================================================================================================================#

__all__ = [
    "uniform",
    "inverse",
    "triangular",
    "epanechnikov",
    "gaussian",
    "tricube",
    "Kernel",
    "return_kernel",
    "KNearestNeighbors",
    "SklearnParzenKNN"
]

RANDOM_SEED = 4012025

#========== Kernels ==========#

def uniform(r):
    return (r <= 1).astype(float)

# TODO: modify to take power to make more universal (p=0 == uniform, p=1 == inverse, p=2 == squared inverse (too local))
def inverse(r):
    return (1. / (r + 1e-15))

def triangular(r):
    return np.maximum(0.0, 1.0 - r)

def epanechnikov(r):
    return np.maximum(0.0, 1.0 - r*r)

def gaussian(r):
    return np.exp(-0.5 * r * r)

def tricube(r):
    return np.maximum(0.0, (1.0 - np.abs(r)**3)**3)


class Kernel(Enum):
    UNIFORM = 0
    INVERSE = 1
    TRIANGULAR = 2
    EPANECHNIKOV = 3
    GAUSSIAN = 4
    TRICUBE = 5

def return_kernel(k: Kernel) -> callable:
    match k:
        case Kernel.UNIFORM:
            return uniform
        case Kernel.INVERSE:
            return inverse
        case Kernel.TRIANGULAR:
            return triangular
        case Kernel.EPANECHNIKOV:
            return epanechnikov
        case Kernel.GAUSSIAN:
            return gaussian
        case Kernel.TRICUBE:
            return tricube
        case _:
            raise ValueError(f"kernels are selected from range [0,5]. Got {k} instead.")

#===================================================#

#========== Helpers ==========#

@dataclass
class _ENNParams:
    k: int | tuple[int, int] = 1
    n_iter: int = 10
    min_size: int = 10 # min prototype set size
    remove_fraction: float = 1.0

_ENNMethod = Literal["enn", "stolp", "oss"]
_UpdateMode = Literal["replace", "intersect", "union", "none"]

#==========================================#

#========== Custom kNN Model ==========#

class KNearestNeighbors:

    def __init__(self, kernel: callable = uniform, p: int = 2, eps: float = 1e-15):
        self.kernel = kernel
        self.p = p
        self.eps = eps

        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

        # absolute indices to X_train_
        self.proto_idx: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Basic utils
    # ---------------------------------------------------------------------

    # analogue of clone from sklearn - returns unfitted estimator
    def copy(self):
        copy_estimator = self.__class__(self.kernel)
        return copy_estimator

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self.proto_idx = None
        return self
    
    @property
    def n_train_(self) -> int:
        if self.X_train_ is None:
            return 0
        return int(self.X_train_.shape[0])
    
    def set_prototypes(self, proto_idx: Optional[Iterable[int]]):
        """Set current prototypes (absolute indices). Pass None to reset to full set."""
        if proto_idx is None:
            self.proto_idx = None
        else:
            self.proto_idx = self._normalize_abs_idx(proto_idx)

    def _normalize_abs_idx(self, idx: Iterable[int]) -> np.ndarray:
        """Normalize indices to a sorted, unique, int64 array; validate bounds."""
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before using indices.")

        arr = np.asarray(list(idx), dtype=np.int64).ravel()
        if arr.size == 0:
            return arr

        n = self.n_train_
        if arr.min() < 0 or arr.max() >= n:
            raise IndexError(f"indices out of bounds: valid range is [0, {n-1}]")

        return np.unique(arr) # sort + unique
    
    def _resolve_ref_idx(self, ref_idx: Optional[Iterable[int]]) -> np.ndarray:
        """
        Resolve which reference set to use for prediction/distances:
          - if ref_idx is provided => use that (absolute)
          - else if self.proto_idx is set => use it
          - else => use full training set
        """
        if self.X_train_ is None:
            raise RuntimeError("Call fit() first.")

        if ref_idx is not None:
            return self._normalize_abs_idx(ref_idx)

        if self.proto_idx is not None:
            return self.proto_idx

        return np.arange(self.n_train_, dtype=np.int64)

    # ---------------------------------------------------------------------
    # Distances
    # ---------------------------------------------------------------------

    def _cdist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Pairwise distances between rows of A and rows of B under Lp norm."""
        A = np.asarray(A)
        B = np.asarray(B)

        if self.p == 2:
            # Faster Euclidean
            dists = np.sqrt(
                np.maximum(
                    0.0, 
                    (A * A).sum(axis=1, keepdims=True) +   # (num_test, 1) (A * A) is slightly faster than np.power(A, 2)
                    (B * B).sum(axis=1, keepdims=True).T + # (1, num_train)
                    -2 * (A @ B.T)
                )
            )
        else:
            diff = A[:, None, :] - B[None, :, :]                # (num_test, num_train, D)
            dists = np.linalg.norm(diff, ord=self.p, axis=2)
        
        return dists

    def compute_distances(self, X, ref_idx: Optional[Iterable[int]] = None) -> np.ndarray:
        """Distances from X to reference set (columns correspond to ref_idx order)."""
        if self.X_train_ is None:
            raise RuntimeError("Call fit() first.")
        ref = self._resolve_ref_idx(ref_idx)
        X_ref = self.X_train_[ref]
        return self._cdist(np.asarray(X), X_ref)

    # ---------------------------------------------------------------------
    # Distances
    # ---------------------------------------------------------------------   

    def predict(self, X, k=1, vect: bool = True, ref_idx: Optional[Iterable[int]] = None):
        dists, y_ref = self._prepare_pred(np.asarray(X), ref_idx=ref_idx)
        if vect:
            y_pred, _ = self.predict_labels_vect(dists, y_ref=y_ref, k=k)
        else:
            y_pred, _ = self.predict_labels(dists, y_ref=y_ref, k=k)
        return y_pred
    
    def predict_proba(self, X, k: int = 1, ref_idx: Optional[Iterable[int]] = None):
        dists, y_ref = self._prepare_pred(np.asarray(X), ref_idx=ref_idx)
        _, probs = self.predict_labels(dists, y_ref=y_ref, k=k)
        return probs
    
    def _prepare_pred(self, X: np.ndarray, ref_idx: Optional[Iterable[int]]):
        ref = self._resolve_ref_idx(ref_idx)
        dists = self._cdist(X, self.X_train_[ref])
        y_ref = self.y_train_[ref]
        return dists, y_ref
    
    def _n_classes(self) -> int:
        if self.y_train_ is None:
            raise RuntimeError("Call fit() first.")
        return int(np.max(self.y_train_)) + 1
        
    def predict_labels(self, dists: np.ndarray, y_ref: np.ndarray, k: int = 1):
        """Non-vectorized version."""
        num_test = dists.shape[0]
        y_pred = np.full(num_test, -1, dtype=float)

        n_ref = dists.shape[1]
        if k < 1 or (k + 1) > n_ref:
            raise ValueError(f"Need at least k+1 reference points. Got k={k}, n_ref={n_ref}.")
        
        n_classes = self._n_classes()
        class_probs = np.empty((num_test, n_classes), dtype=float)

        for i in range(num_test):
            row = dists[i]
            top_k1 = np.argpartition(row, k)[: k + 1]
            d = row[top_k1]

            # If everything was masked / inf -- can't predict
            if not np.isfinite(d).any():
                class_probs[i] = 0.0
                y_pred[i] = -1
                continue

            h = np.max(d) + self.eps          # bandwidth
            non_max_mask = np.isfinite(d) & (d < (h - self.eps)) # self.eps решает (k=12/13)
            r = d[non_max_mask] / h           # normalized distances
            w = self.kernel(r)                # kernel weights

            nbr_cls = y_ref[top_k1[non_max_mask]]
            row_class_probs = np.bincount(nbr_cls, weights=w, minlength=n_classes)
            y_pred[i] = np.argmax(row_class_probs)
            class_probs[i] = row_class_probs

        return y_pred, class_probs
    
    def predict_labels_vect(self, dists: np.ndarray, y_ref: np.ndarray, k: int = 1, tie_tol: float = 0.0):
        """Vectorized kNN with your (k+1)-bandwidth trick and tie-breaking."""
        n_ref = dists.shape[1]
        if k < 1 or (k + 1) > n_ref:
            raise ValueError(f"Need at least k+1 reference points. Got k={k}, n_ref={n_ref}.")
        
        n_classes = self._n_classes()

        top_k1 = np.argpartition(dists, k, axis=1)[:, : k + 1]
        d = np.take_along_axis(dists, top_k1, axis=1)

        # 1) argpartition doesn't guarantee sorted order 2) for tie-breaking by index (primary = distance, secondary = index) 
        order = np.lexsort((top_k1, d), axis=1)
        top_k1_sorted = np.take_along_axis(top_k1, order, axis=1)
        d_sorted = np.take_along_axis(d, order, axis=1)

        h = d_sorted[:, k] + self.eps # bandwidth
        nbr_idx = top_k1_sorted[:, :k]
        nbr_d   = d_sorted[:, :k]

        r = nbr_d / h[:, None]  # normalized distances
        w = self.kernel(r)      # kernel weights

        nbr_cls = y_ref[nbr_idx]
        cls_probs = self._fast_weighted_bincount(nbr_cls, weights=w, max_value=n_classes)
        y_pred = np.argmax(cls_probs, axis=1)
        y_pred = self._break_ties_by_nearest_neighbor(cls_probs, y_pred, nbr_cls, nbr_d, tie_tol=tie_tol)

        return y_pred, cls_probs

    @staticmethod
    def _fast_weighted_bincount(classes: np.ndarray, weights: np.ndarray, max_value: int):
        n_rows = classes.shape[0]
        # 1. Create row offsets: [0, n_classes, 2*n_classes, ...]
        offset = np.arange(n_rows)[:, None] * max_value
        # 2. Flatten both and apply offset to indices
        flat_indices = (classes + offset).ravel()
        flat_weights = weights.ravel()
        # 3. Single bincount on the flattened data
        counts = np.bincount(flat_indices, weights=flat_weights, minlength=n_rows * max_value)
        # 4. Reshape back to (Rows, Classes)
        return counts.reshape(n_rows, max_value)

    @staticmethod
    def _break_ties_by_nearest_neighbor(cls_probs, y_pred, nbr_cls, nbr_d, tie_tol: float = 0.0):
        """
        If cls_probs has a tie for the maximum in a row, replace y_pred[row] with
        the class of the nearest neighbor among the k contributing neighbors.
        """
        maxv = cls_probs.max(axis=1)
        if tie_tol > 0.0:
            is_max = np.abs(cls_probs - maxv[:, None]) <= tie_tol
        else:
            is_max = (cls_probs == maxv[:, None])

        tie_rows = is_max.sum(axis=1) > 1
        if not np.any(tie_rows):
            return y_pred

        # already sorted by distance; nearest neighbor is position 0
        nn_class = nbr_cls[:, 0]
        out = y_pred.copy()
        out[tie_rows] = nn_class[tie_rows]
        return out
    
    # ---------------------------------------------------------------------
    # LOO (works on any chosen reference subset)
    # ---------------------------------------------------------------------
    def leave_one_out(
        self,
        k_min: int,
        k_max: int,
        plot: bool = True,
        show_plot: bool = False,
        save_dir: Path | None = None,
        filename: str | None = None,
        ref_idx: Optional[Iterable[int]] = None,
    ):

        ref = self._resolve_ref_idx(ref_idx)
        X_ref = self.X_train_[ref]
        y_ref = self.y_train_[ref]

        assert k_min > 0, "k_min should be at least 1"
        n = X_ref.shape[0]
        assert k_max <= n - 2, "Need k+1 neighbors after removing self => k_max <= n-2."

        D = self._cdist(X_ref, X_ref)
        np.fill_diagonal(D, np.inf)

        ks = np.arange(k_min, k_max + 1)
        errors = np.zeros((ks.size,), dtype=float)

        for i, k in enumerate(ks):
            y_pred, _ = self.predict_labels_vect(D, y_ref=y_ref, k=int(k))
            errors[i] = 1.0 - np.mean(y_pred == y_ref) # rate of mistakes (0-1 loss)

        best_idx = int(np.argmin(errors))
        opt_k = int(ks[best_idx])
        opt_err = float(errors[best_idx])

        if plot:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=ks,
                y=errors,
                mode="lines",
                name="LOO empirical risk",
            ))

            # Highlight optimum
            fig.add_trace(go.Scatter(
                x=[opt_k],
                y=[opt_err],
                mode="markers",
                name=f"min @ k={opt_k}",
                marker=dict(size=12, symbol="x"),
            ))

            fig.update_layout(
                title="LOO empirical risk vs k",
                xaxis_title="k (number of neighbors)",
                yaxis_title="Empirical risk (LOO error rate)",
                hovermode="x unified",
            )

            # Optional: show a vertical dashed line at the optimum
            fig.add_vline(
                x=opt_k,
                line_dash="dash",
                annotation_text=f"k* = {opt_k}",
                annotation_position="top",
            )

            if show_plot:
                fig.show()

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                if filename is None:
                    filename = f"loo_optimal_{opt_k}NN.png"
                save_path = save_dir / filename
                fig.write_image(save_path, scale=2, width=1200, height=600)

        return opt_k, errors
    
    # ---------------------------------------------------------------------
    # Prototype selection (sequential by default)
    # ---------------------------------------------------------------------
    def select_prototypes(
        self,
        method: _ENNMethod = "stolp",
        enn_params: Optional[_ENNParams] = None,
        seed: int = RANDOM_SEED,
        candidates: Optional[Iterable[int]] = None,
        update: _UpdateMode = "replace",
        inplace: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        candidates:
          - None => run on current proto_idx if set, else on full training set (sequential-friendly)
          - otherwise => absolute indices into the training set

        update:
          - "replace": self.proto_idx = selected
          - "intersect": self.proto_idx = intersection(self.proto_idx, selected) (rarely needed)
          - "union": self.proto_idx = union(self.proto_idx, selected) (rarely needed)
          - "none": do not touch self.proto_idx
        """
        if self.X_train_ is None:
            raise RuntimeError("Call fit() first.")
        
        if enn_params is None:
            enn_params = _ENNParams()

        rng = np.random.default_rng(seed)

        cand = self._resolve_ref_idx(candidates)
        X_cand = self.X_train_[cand]
        y_cand = self.y_train_[cand]

        if verbose:
            base = "current proto set" if (candidates is None and self.proto_idx is not None) else "given candidates/full"
            print(f"[select_prototypes] method={method}, pool_size={cand.size} ({base})")

        if method == 'enn':
            selected_abs = self._select_enn(cand, X_cand, y_cand, enn_params, rng, verbose)
        elif method == "stolp":
            selected_abs = self._select_stolp(cand, X_cand, y_cand, verbose)
        elif method == 'oss':
            selected_abs = self._select_oss(cand, X_cand, y_cand, rng, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply update strategy
        if update != "none":
            if update == 'replace':
                self.proto_idx = selected_abs.copy()
            elif update in ("intersect", "union"):
                # For union/intersect, the natural "base" is the previous proto_idx,
                # not "proto_idx or full training set".
                prev = self.proto_idx

                if prev is None:
                    prev = np.array([], dtype=np.int64)

                if update == "intersect":
                    self.proto_idx = np.intersect1d(prev, selected_abs, assume_unique=False)
                elif update == "union":
                    self.proto_idx = np.union1d(prev, selected_abs)
                    
            else:
                raise ValueError(f"Unknown update mode: {update}")
        
        if inplace:
            self.X_train_ = self.X_train_[selected_abs]
            self.y_train_ = self.y_train_[selected_abs]
            self.proto_idx = None
            return np.arange(self.X_train_.shape[0], dtype=np.int64)

        return selected_abs

    # ---------------- ENN ----------------
    def _select_enn(
        self,
        cand_abs: np.ndarray,
        X_cand: np.ndarray,
        y_cand: np.ndarray,
        params: _ENNParams,
        rng: np.random.Generator,
        verbose: bool,
    ) -> np.ndarray:
        k = params.k
        if isinstance(k, (int, np.integer)):
            k_min, k_max = int(k), int(k)
        else:
            k_min, k_max = int(k[0]), int(k[1])

        assert k_min >= 1 and k_max >= k_min

        n = X_cand.shape[0]
        if n < k_max + 2:
            raise ValueError(f"ENN needs at least k_max+2 points. Got n={n}, k_max={k_max}.")

        D_full = self._cdist(X_cand, X_cand)
        np.fill_diagonal(D_full, np.inf)

        alive = np.ones(n, dtype=bool)

        for it in range(params.n_iter):
            pos = np.flatnonzero(alive)  # local positions in candidate pool
            if pos.size <= params.min_size:
                if verbose:
                    print(f"[ENN] stop: |S|={pos.size} <= min_size={params.min_size}")
                break

            D_sub = D_full[np.ix_(pos, pos)]
            y_sub = y_cand[pos]

            mis_any = np.zeros(pos.size, dtype=bool)

            for kk in range(k_min, k_max + 1):
                y_pred, _ = self.predict_labels_vect(D_sub, y_ref=y_sub, k=int(kk))
                mis = (y_pred != y_sub)
                mis_any |= mis
                if verbose:
                    loo_err = 1.0 - np.mean(y_pred == y_sub)
                    print(f"[ENN] it {it+1}/{params.n_iter}  k={kk}  |S|={pos.size}  LOO={loo_err:.4f}  mis={mis.sum()}")

            if not np.any(mis_any):
                break

            mis_pos_local = np.flatnonzero(mis_any)    # positions inside pos[]
            mis_pos = pos[mis_pos_local]               # positions inside candidate pool
            mis_abs = cand_abs[mis_pos]                # absolute indices

            if params.remove_fraction < 1.0:
                m = max(1, int(np.ceil(params.remove_fraction * mis_abs.size)))
                chosen_abs = rng.choice(mis_abs, size=m, replace=False)
            else:
                chosen_abs = mis_abs

            # cand_abs is sorted (by _normalize_abs_idx), so searchsorted works
            chosen_pos = np.searchsorted(cand_abs, chosen_abs)
            alive[chosen_pos] = False

        final_pos = np.flatnonzero(alive)
        final_abs = cand_abs[final_pos]
        if verbose:
            print(f"[ENN] final |S|={final_abs.size}")
        return final_abs
    
    # ---------------- STOLP ----------------
    def _select_stolp(self, cand_abs: np.ndarray, X_cand: np.ndarray, y_cand: np.ndarray, verbose: bool) -> np.ndarray:
        if verbose:
            print("[STOLP] starting...")

        # 1. Precompute full distance matrix for the entire training set
        D_full = self._cdist(X_cand, X_cand)

        # 2. Phase 1: Cleaning (Remove points with non-positive margins)
        margins = self._compute_margins(D_full, y_cand)
        cleaned_mask = margins > 0
        cleaned_pos = np.flatnonzero(cleaned_mask)

        if cleaned_pos.size == 0:
            if verbose:
                print("[STOLP] cleaning removed everything; returning empty set.")
            return np.array([], dtype=np.int64)

        if verbose:
            print(f"[STOLP] cleaning removed {X_cand.shape[0] - cleaned_pos.size}, remaining={cleaned_pos.size}")

        # Phase 2: Seed Selection (Pick most typical object per class)
        proto_pos = []
        for c in np.unique(y_cand[cleaned_pos]):
            class_pos = cleaned_pos[y_cand[cleaned_pos] == c]
            best = class_pos[np.argmax(margins[class_pos])]
            proto_pos.append(int(best))
        proto_pos = np.array(proto_pos, dtype=np.int64)

        # 4. Phase 3: Iterative Expansion
        # We only need to correctly classify the CLEANED subset
        X_clean = X_cand[cleaned_pos]
        y_clean = y_cand[cleaned_pos]

        iteration = 0
        while True:
            iteration += 1

            X_proto = X_cand[proto_pos]
            y_proto = y_cand[proto_pos]

            D_to_proto = self._cdist(X_clean, X_proto)
            y_pred, _ = self.predict_labels_vect(D_to_proto, y_ref=y_proto, k=1)

            mis = (y_pred != y_clean)
            n_mis = int(mis.sum())

            if verbose:
                print(f"[STOLP] iter {iteration}: |P|={proto_pos.size}, misclassified={n_mis}")

            if n_mis == 0:
                break

            # Find the "most misclassified" point among the misclassified ones
            # This is the point with the smallest (most negative) local margin
            mis_pos_in_clean = np.flatnonzero(mis)
            local_margins = self._get_local_margins(
                D_subset=D_to_proto[mis_pos_in_clean],
                y_subset=y_clean[mis_pos_in_clean],
                y_proto=y_proto,
            )

            # Pick worst point among cleaned (map back to candidate positions)
            worst_clean_local = mis_pos_in_clean[np.argmin(local_margins)]
            worst_cand_pos = cleaned_pos[worst_clean_local]

            proto_pos = np.append(proto_pos, worst_cand_pos)

        selected_abs = cand_abs[proto_pos]
        selected_abs = np.unique(selected_abs)
        return selected_abs
    
    # ---------------- OSS ----------------
    def _select_oss(
        self,
        cand_abs: np.ndarray,
        X_cand: np.ndarray,
        y_cand: np.ndarray,
        rng: np.random.Generator,
        verbose: bool,
    ) -> np.ndarray:
        classes, counts = np.unique(y_cand, return_counts=True)
        if classes.size != 2:
            raise ValueError(f"OSS expects binary classes; got classes={classes}.")

        # 1. Identify Minority and Majority classes
        min_class = classes[np.argmin(counts)]
        maj_class = classes[np.argmax(counts)]

        min_pos = np.flatnonzero(y_cand == min_class)
        maj_pos = np.flatnonzero(y_cand == maj_class)

        # Shuffle majority indices (CNN is sensitive to order)
        rng.shuffle(maj_pos)

        D_full = self._cdist(X_cand, X_cand)

        # 2. Step 1: Initialize S with all minority and 1 random majority
        current = min_pos.tolist()
        current.append(int(maj_pos[0]))
        maj_pos = maj_pos[1:]

        # 3. Step 2: ITERATIVE CNN Phase
        if verbose:
            print(f"[OSS] CNN phase over {maj_pos.size} majority points...")

        for pos in maj_pos:
            # Fast slice: distances from candidate 'pos' to all points in current proto idx candidates set'
            d = D_full[pos, current]
            # Identify 1-NN class by indexing y_train with the proto list
            nn_local = int(np.argmin(d))
            nn_pos = current[nn_local]
            # If nearest neighbor is minority class, the candidate is a boundary point
            if y_cand[nn_pos] != maj_class:
                current.append(int(pos))

        proto_pos = np.array(current, dtype=np.int64)
        if verbose:
            print(f"[OSS] after CNN: |S|={proto_pos.size}")

        # 4. Step 3: Tomek Link Removal (Cleaning)
        # Reuse D_full to avoid any extra distance calculations
        tomek_mask = self._find_tomek_links(indices_pos=proto_pos, D_full=D_full, y_all=y_cand)
        # Remove only the Majority class member of a Tomek Link
        is_maj = (y_cand[proto_pos] == maj_class)
        to_remove = tomek_mask & is_maj

        proto_pos = proto_pos[~to_remove]
        if verbose:
            print(f"[OSS] Tomek removed {int(to_remove.sum())} majority points; final |S|={proto_pos.size}")

        selected_abs = cand_abs[proto_pos]
        selected_abs = np.unique(selected_abs)
        return selected_abs
    
    # ---------------------------------------------------------------------
    # Prototype selection (sequential by default)
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_margins(D: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Global margins: dist(nearest_enemy) - dist(nearest_friend)."""
        mask_same = (y.reshape(-1, 1) == y)
        np.fill_diagonal(mask_same, False)

        # nearest friend
        masked_friend = np.ma.masked_array(D, ~mask_same)
        dist_friend = np.min(masked_friend, axis=1)
        dist_friend[~np.any(mask_same, axis=1)] = 0

        # nearest enemy
        masked_enemy = np.ma.masked_array(D, mask_same)
        np.fill_diagonal(masked_enemy.mask, True)
        dist_enemy = np.min(masked_enemy, axis=1)

        margins = dist_enemy - dist_friend
        return margins.data
    
    @staticmethod
    def _get_local_margins(D_subset: np.ndarray, y_subset: np.ndarray, y_proto: np.ndarray) -> np.ndarray:
        """
        Margins relative to current prototypes.
        D_subset: distances from points to current prototypes (N_mis, N_proto)
        y_subset: true labels of misclassified points (N_mis,)
        y_proto:  labels of the current prototypes (N_proto,)
        """
        # friend = same class, enemy = different class
        friend_mask = (y_subset.reshape(-1, 1) != y_proto) # (N_min, 1) == (N_proto, ) -> (N_min, N_proto)
        enemy_mask = ~friend_mask                          # (N_min, 1) != (N_proto, ) -> (N_min, N_proto)

        # friend distance: min over same-class prototypes
        dist_friend = np.min(np.ma.masked_array(D_subset, mask=~friend_mask), axis=1).filled(np.inf) # inverse for correct masked array logic

        # enemy distance: min over different-class prototypes
        dist_enemy = np.min(np.ma.masked_array(D_subset, mask=~enemy_mask), axis=1).filled(np.inf)

        local_margins = dist_enemy - dist_friend
        return local_margins.data
    
    @staticmethod
    def _find_tomek_links(indices_pos: np.ndarray, D_full: np.ndarray, y_all: np.ndarray) -> np.ndarray:
        """Vectorized Tomek link detection inside a subset given by positions into candidate arrays."""
        n = indices_pos.size
        if n < 2:
            return np.zeros(n, dtype=bool)
            
        # Slice D_full to get distances between prototypes
        D_sub = D_full[np.ix_(indices_pos, indices_pos)].copy()
        np.fill_diagonal(D_sub, np.inf)

        # 1-NN for everyone in prototype subset
        nn = np.argmin(D_sub, axis=1)

        # Check mutual nearest neighbors
        # point i's neighbor is j, and j's neighbor is i
        j = nn
        is_mutual = (nn[j] == np.arange(n))

        # Check class difference
        y_S = y_all[indices_pos]
        is_diff = (y_S != y_S[j])
        return is_mutual & is_diff

#=================================================================================================#

#========== Sklearn Analogue ==========#

class SklearnParzenKNN:

    def __init__(self, k: int, kernel, metric="minkowski", p=2, eps=1e-12):
        assert k >= 1
        self.k = k
        self.kernel = kernel
        self.metric = metric
        self.p = p
        self.eps = eps

    def copy(self):
        copy_estimator = self.__class__(self.k, self.kernel, self.metric, self.p, self.eps)
        return copy_estimator

    def _weights(self, distances: np.ndarray) -> np.ndarray:

        d = np.asarray(distances)

        # Single row case
        if d.ndim == 1:
            h = np.max(d) + self.eps
            r = d / h
            w = self.kernel(r)

            # Exclude the farthest neighbor(s): those at distance == max(d)
            w[d >= (h - self.eps)] = 0.0
            return w

        # Batch case
        h = np.max(d, axis=1, keepdims=True) + self.eps
        r = d / h
        w = self.kernel(r)

        # Zero out farthest neighbor(s) per row
        w[d >= (h - self.eps)] = 0.0
        return w

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)

        # sklearn will use k+1 neighbors so weights() sees k+1 distances.
        self.model = KNeighborsClassifier(
            n_neighbors=self.k + 1,
            weights=self._weights,
            metric=self.metric,
            p=self.p if self.metric == "minkowski" else None
        )
        self.model.fit(X, y)

        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # kernel weights, normalized by sum of weights internally
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
    def leave_one_out(
        self, 
        k_min: int, 
        k_max: int, 
        plot: bool = True,
        show_plot: bool = False,
        save_dir: Path | None = None,
        filename: str | None = None,
    ):
        assert k_min > 0, "k_min should be at least 1"
        n = self.X_train_.shape[0]
        assert k_max <= n - 2, "k_max should be <= n-2 for Parzen-kNN LOO."

        # We will compute neighbors for each point among the other n-1 points.
        # Request k+2 because (1) the nearest neighbor is the point itself at distance 0.
        # (2) Then drop that self neighbor => we get k+1 neighbors needed for Parzen bandwidth.
        ks = np.arange(k_min, k_max + 1)
        errors = np.zeros(len(ks), dtype=float)

        nn = NearestNeighbors(metric=self.metric, p=self.p if self.metric == "minkowski" else None)
        nn.fit(self.X_train_)

        for idx_k, k in enumerate(ks):
            distances, indices = nn.kneighbors(self.X_train_, n_neighbors=k+2, return_distance=True)

            # Drop self neighbor (distance 0, index = i) at column 0
            d = distances[:, 1:]     # shape (n, k+1)
            ind = indices[:, 1:]     # shape (n, k+1)

            # bandwidth h(x) = max distance among these k+1
            h = np.max(d, axis=1, keepdims=True) + self.eps

            # weights for all k+1 according to kernel
            r = d / h
            w = self.kernel(r)

            # exclude the farthest neighbor(s) (distance == h - eps)
            w[d >= (h - self.eps)] = 0.0

            # Now compute weighted vote per row for binary/multiclass labels:
            # We'll accumulate weights into class bins.
            # This expects non-negative integer labels (like your np.bincount approach).
            y_pred = np.empty(n, dtype=int)

            num_classes = int(np.max(self.y_train_) + 1)
            for i in range(n):
                labels_i = self.y_train_[ind[i]]
                weights_i = w[i]
                class_scores = np.bincount(labels_i, weights=weights_i, minlength=num_classes)
                y_pred[i] = int(np.argmax(class_scores))

            errors[idx_k] = 1.0 - np.mean(y_pred == self.y_train_)

        best_idx = int(np.argmin(errors))
        opt_k = int(ks[best_idx])
        opt_err = float(errors[best_idx])

        if plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ks, y=errors, mode="lines", name="LOO empirical risk"))
            fig.add_trace(go.Scatter(
                x=[opt_k], y=[opt_err],
                mode="markers",
                name=f"min @ k={opt_k}",
                marker=dict(size=12, symbol="x")
            ))
            fig.add_vline(x=opt_k, line_dash="dash",
                            annotation_text=f"k* = {opt_k}",
                            annotation_position="top")
            fig.update_layout(
                title="LOO empirical risk vs k (sklearn-based Parzen-kNN)",
                xaxis_title="k",
                yaxis_title="Empirical risk (LOO error rate)",
                hovermode="x unified",
            )

            if show_plot:
                fig.show()
            
            save_path = None
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                if filename is None:
                    filename = f"loo_optimal_{k}NN.png"

                save_path = save_dir / filename
                fig.write_image(save_path, scale=2, width=1200, height=600)

        return opt_k, errors
    
#=================================================================================================#