from typing import Optional, Any, Iterable, Mapping
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

# import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import StratifiedKFold

from . import KNearestNeighbors

#================================================================================================================#

__all__ = [
    "optimize_k_with_cv",
    "optimize_k_with_loo"
]

RANDOM_SEED = 4012025

#================================================================================================================#

#========== Helpers ==========#

def _extract_method(p: Any) -> str:
    if p is None:
        return "none"
    if isinstance(p, Mapping):
        return str(p.get("method", "none"))
    return "none"

def _get_method_string(proto_params: Optional[dict[str, Any] | Iterable[dict[str, Any]]] = None):
    if proto_params is None:
        methods_str = ""
    elif isinstance(proto_params, Mapping):
        methods_str = _extract_method(proto_params).upper()
    elif isinstance(proto_params, Iterable):
        methods_str = " + ".join(_extract_method(p) for p in proto_params).upper()
    else:
        methods_str = ""
    return methods_str

#=============================#

#========== K Parameter Optimization Process Functions ==========#

def optimize_k_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: callable,
    preprocessing,
    k_min: int,
    k_max: int,
    proto_params: Optional[dict[str, Any] | Iterable[dict[str, Any]]] = None,
    n_splits: int = 20,
    seed: int = RANDOM_SEED,
    plot: bool = True,
    show_plot: bool = False,
    save_dir: Path | None = None,
    filename: str | None = None,
):
    """
    Optimized CV-based k selection:
      - Each fold preprocessing is fit once.
      - Prototype selection is run once per fold.
      - Distances test->ref are computed once per fold.
      - For each fold we keep only the nearest (k_max+1) neighbors, sorted.
      - Loop over k reuses these precomputed neighbors and only does voting.

    proto_params:
      - None -> no prototype selection
      - dict -> one selection call per fold
      - list[dict] -> sequential prototype selection per fold
    """
    assert k_min > 0, "k_min should be at least 1"
    y_all = np.asarray(y_train)

    ks = np.arange(k_min, k_max + 1)
    errors = np.zeros(ks.size, dtype=np.float64)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X_train, y_all))

    # Normalize proto params into steps (or None)
    proto_steps = None
    if proto_params is not None:
        if isinstance(proto_params, dict):
            proto_steps = [dict(proto_params)]
        else:
            proto_steps = [dict(step) for step in proto_params]

        for step in proto_steps:
            step.setdefault("seed", seed)
            if "method" not in step or step["method"] is None:
                raise ValueError("Each proto_params step must include 'method'.")

    # We'll reuse this instance during precompute
    model = KNearestNeighbors(kernel)

    # We need a stable n_classes across folds (same as your classifier expects)
    n_classes = int(np.max(y_all)) + 1

    # Per-fold cache: (top_sorted, d_sorted, y_ref, y_te)
    fold_cache = []

    # ---- 1) Precompute folds once ----
    for train_idx, test_idx in tqdm(splits, desc="precompute folds"):
        X_tr = preprocessing.fit_transform(X_train.iloc[train_idx])
        y_tr = y_all[train_idx]

        X_te = preprocessing.transform(X_train.iloc[test_idx])
        y_te = y_all[test_idx]

        model.fit(X_tr, y_tr)

        # Prototype selection (once per fold)
        if proto_steps is not None:
            model.proto_idx = None
            for step in proto_steps:
                model.select_prototypes(**step)
            ref_idx = model.proto_idx  # abs indices into fold-train
        else:
            ref_idx = None

        # Distances once per fold (not per k)
        dists = model.compute_distances(X_te, ref_idx=ref_idx)

        # predict_labels_vect logic replicated here
        n_ref = dists.shape[1]
        if (k_max + 1) > n_ref:
            raise ValueError(
                f"k_max={k_max} requires at least k_max+1={k_max+1} reference points, "
                f"but fold has only n_ref={n_ref} (after prototype selection). "
                f"Reduce k_max or prototype aggressiveness."
            )

        # y_ref must align with columns of dists
        y_ref = model.y_train_ if ref_idx is None else model.y_train_[ref_idx]

        # Precompute nearest (k_max+1) neighbors, sorted by (distance, index)
        top = np.argpartition(dists, k_max, axis=1)[:, : (k_max + 1)]
        d_top = np.take_along_axis(dists, top, axis=1)

        order = np.lexsort((top, d_top), axis=1)
        top_sorted = np.take_along_axis(top, order, axis=1)
        d_sorted = np.take_along_axis(d_top, order, axis=1)

        fold_cache.append((top_sorted, d_sorted, y_ref, y_te))


    # ---- 2) Evaluate all k using cached neighbors ----
    eps = model.eps
    for i, k in enumerate(tqdm(ks, desc="k")):
        fold_err = np.zeros(len(fold_cache), dtype=np.float64)

        for j, (top_sorted, d_sorted, y_ref, y_te) in enumerate(fold_cache):
            # bandwidth from (k+1)-th neighbor distance (position k in the k_max+1 array)
            h = d_sorted[:, k] + eps

            nbr_idx = top_sorted[:, :k]  # (n_test_fold, k)
            nbr_d = d_sorted[:, :k]

            r = nbr_d / h[:, None]
            w = kernel(r)

            nbr_cls = y_ref[nbr_idx]
            cls_probs = KNearestNeighbors._fast_weighted_bincount(nbr_cls, weights=w, max_value=n_classes)

            y_pred = np.argmax(cls_probs, axis=1)
            y_pred = KNearestNeighbors._break_ties_by_nearest_neighbor(
                cls_probs, y_pred, nbr_cls, nbr_d, tie_tol=0.0
            )

            fold_err[j] = 1.0 - np.mean(y_pred == y_te)

        errors[i] = fold_err.mean()

    # ---- 3) Pick best k, plot ----
    best_idx = int(np.argmin(errors))
    opt_k = int(ks[best_idx])
    opt_err = float(errors[best_idx])

    if plot:

        methods_str = _get_method_string(proto_params)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=errors, mode="lines", name="CV empirical risk"))
        fig.add_trace(
            go.Scatter(
                x=[opt_k],
                y=[opt_err],
                mode="markers",
                name=f"min @ k={opt_k}",
                marker=dict(size=12, symbol="x"),
            )
        )  

        title = "CV empirical risk vs k"
        if methods_str:
            title += f" [{methods_str}]"

        fig.update_layout(
            title=title,
            xaxis_title="k (number of neighbors)",
            yaxis_title="Empirical risk (CV error rate)",
            hovermode="x unified",
        )
        fig.add_vline(x=opt_k, line_dash="dash", annotation_text=f"k* = {opt_k}", annotation_position="top")

        if show_plot:
            fig.show()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if filename is None:
                filename = f"CV_fair_optimal_{opt_k}NN.png"
            fig.write_image(save_dir / filename, scale=2, width=1200, height=600)

    return opt_k, opt_err


# Currently not used, but presented as a concept
def optimize_k_with_loo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: callable,
    preprocessing,
    k_min: int,
    k_max: int,
    proto_params: Optional[dict[str, Any] | Iterable[dict[str, Any]]] = None,
    block_size: int = 256,
    seed: int = RANDOM_SEED,
    plot: bool = True,
    show_plot: bool = False,
    save_dir: Path | None = None,
    filename: str | None = None,
):
    """
    Fair Leave-One-Out k optimization, memory-efficient via block processing.

    For each LOO fold:
      - fit preprocessing on train (N-1 points)
      - transform train/test
      - fit kNN
      - (optional) prototype selection on fold-train only
      - compute distances from the held-out point to ref set
      - keep only top (k_max+1) neighbors sorted by (distance, index)

    Then for each block we evaluate all k in [k_min, k_max] using only cached neighbors,
    accumulate total misclassifications, and discard block cache.

    proto_params:
      - None -> no prototype selection
      - dict -> one selection call per fold
      - list[dict] -> sequential prototype selection per fold (each step sees updated proto_idx)

    block_size:
      Number of LOO folds to process per block (controls memory use).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    assert k_min > 0, "k_min should be at least 1"
    y_all = np.asarray(y_train)
    n = int(X_train.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 samples for LOO with k+1 logic.")
    assert k_max <= n - 2, "k_max must be <= n-2 (need k+1 neighbors after removing self)."

    ks = np.arange(k_min, k_max + 1)
    mis_counts = np.zeros(ks.size, dtype=np.int64)

    # Normalize proto params into steps (or None)
    proto_steps = None
    if proto_params is not None:
        if isinstance(proto_params, dict):
            proto_steps = [dict(proto_params)]
        else:
            proto_steps = [dict(step) for step in proto_params]

        for step in proto_steps:
            step.setdefault("seed", seed)
            if "method" not in step or step["method"] is None:
                raise ValueError("Each proto_params step must include 'method'.")

    # Reuse model object for fitting; methods only depend on its current train set
    model = KNearestNeighbors(kernel)

    # assume labels are 0..C-1
    n_classes = int(np.max(y_all)) + 1
    eps = model.eps

    # Iterate over LOO folds in blocks
    for start in tqdm(range(0, n, block_size), desc="LOO blocks"):
        end = min(n, start + block_size)
        B = end - start

        # Block cache: fixed-size because we store only k_max+1 neighbors per fold
        d_sorted_block = np.empty((B, k_max + 1), dtype=np.float64)
        cls_sorted_block = np.empty((B, k_max + 1), dtype=np.int64)
        y_te_block = np.empty(B, dtype=np.int64)

        # ---- Precompute neighbor cache for this block ----
        for b, i_test in enumerate(range(start, end)):
            # train indices are all except i_test
            train_idx = np.concatenate([np.arange(0, i_test), np.arange(i_test + 1, n)])
            y_te_block[b] = y_all[i_test]

            # fair preprocessing per fold
            X_tr = preprocessing.fit_transform(X_train.iloc[train_idx])
            y_tr = y_all[train_idx]
            X_te = preprocessing.transform(X_train.iloc[[i_test]])  # shape (1, d)

            model.fit(X_tr, y_tr)

            # fair prototype selection per fold (sequential if steps provided)
            if proto_steps is not None:
                model.proto_idx = None
                for step in proto_steps:
                    model.select_prototypes(**step)
                ref_idx = model.proto_idx
            else:
                ref_idx = None

            # distances from held-out point to ref set
            dists = model.compute_distances(X_te, ref_idx=ref_idx)  # (1, n_ref)
            row = dists[0]
            n_ref = row.size
            if (k_max + 1) > n_ref:
                raise ValueError(
                    f"k_max={k_max} requires at least k_max+1={k_max+1} reference points, "
                    f"but LOO fold has n_ref={n_ref} (after prototype selection). "
                    f"Reduce k_max or prototype aggressiveness."
                )

            # y_ref must correspond to ref columns
            y_ref = model.y_train_ if ref_idx is None else model.y_train_[ref_idx]

            # top (k_max+1) neighbors, sorted by (distance, index)
            top = np.argpartition(row, k_max)[: (k_max + 1)]
            d_top = row[top]
            order = np.lexsort((top, d_top))
            top_sorted = top[order]
            d_sorted = d_top[order]

            d_sorted_block[b, :] = d_sorted
            cls_sorted_block[b, :] = y_ref[top_sorted]

        # ---- Evaluate all k for this block using cached neighbors ----
        for ik, k in enumerate(ks):
            # bandwidth from (k+1)-th neighbor distance (position k)
            h = d_sorted_block[:, k] + eps

            nbr_d = d_sorted_block[:, :k]
            nbr_cls = cls_sorted_block[:, :k]

            r = nbr_d / h[:, None]
            w = kernel(r)

            cls_probs = KNearestNeighbors._fast_weighted_bincount(
                nbr_cls, weights=w, max_value=n_classes
            )
            y_pred = np.argmax(cls_probs, axis=1)
            y_pred = KNearestNeighbors._break_ties_by_nearest_neighbor(
                cls_probs, y_pred, nbr_cls, nbr_d, tie_tol=0.0
            )

            mis_counts[ik] += int(np.sum(y_pred != y_te_block))

    errors = mis_counts / float(n)

    best_idx = int(np.argmin(errors))
    opt_k = int(ks[best_idx])
    opt_err = float(errors[best_idx])

    if plot:

        methods_str = _get_method_string(proto_params)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=errors, mode="lines", name="LOO empirical risk"))
        fig.add_trace(
            go.Scatter(
                x=[opt_k],
                y=[opt_err],
                mode="markers",
                name=f"min @ k={opt_k}",
                marker=dict(size=12, symbol="x"),
            )
        )

        title = "LOO empirical risk vs k"
        if methods_str:
            title += f" [{methods_str}]"

        fig.update_layout(
            title=title,
            xaxis_title="k (number of neighbors)",
            yaxis_title="Empirical risk (LOO error rate)",
            hovermode="x unified",
        )
        fig.add_vline(x=opt_k, line_dash="dash", annotation_text=f"k* = {opt_k}", annotation_position="top")

        if show_plot:
            fig.show()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            if filename is None:
                filename = f"LOO_optimal_{opt_k}NN.png"
            fig.write_image(save_dir / filename, scale=2, width=1200, height=600)

    return opt_k, opt_err

#================================================================================================================#