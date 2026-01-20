from typing import Optional
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from sklearn.svm import SVC

import umap

from ..svc import MySVC

#==============================================================================#

__all__ = [
    'plot_svc_solution_lda_pca2',
    'plot_svc_solution_umap2_fit'
]

RANDOM_SEED = 18012026

#==============================================================================#

#========== Visualization ==========#

def plot_svc_solution_lda_pca2(
    estimator: SVC | MySVC,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    grid_step: int = 200,
    use_std: bool = False,
    plot_confidence: bool = False,
    plot_all_points: bool = True,
    highlight_support_vectors: bool = True,
    plot_margins: bool = True,
    title: str = "",
    save_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = False,
):
    """
    2D visualization for binary (or multiclass) SVC using:
      - axis 1: LDA direction (canonical variate; for K=2 it's 1D)
      - axis 2: top PCA direction orthogonal to the LDA direction (in the same space)

    This produces a supervised+informative 2D plane for binary problems,
    avoiding the "LDA only 1D" limitation.

    Pipeline:
      1) (optional) standardize
      2) fit LDA (1D if binary, 2D if multiclass and requested)
      3) build a 2D orthonormal basis B = [w1, w2]
      4) project points: Z = X @ B
      5) grid in Z, reconstruct plane points: X_grid = Z_grid @ B.T
      6) inverse-standardize (if needed), then evaluate original estimator
    """

    check_is_fitted(estimator)

    # ---- resolve data ----
    if X is None:
        if hasattr(estimator, "support_vectors_"):
            X = estimator.support_vectors_
        else:
            raise ValueError("Provide X (training data). Could not infer data from estimator.")
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if y is None and plot_all_points:
        y = estimator.predict(X)
    if y is None:
        raise ValueError("Need y (or enable plot_all_points fallback) to compute LDA direction.")
    y = np.asarray(y)
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")

    classes = estimator.classes_
    n_classes = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    def to_idx(labels: np.ndarray) -> np.ndarray:
        return np.vectorize(cls_to_idx.get, otypes=[int])(labels)

    y_idx = to_idx(y)

    # ---- standardize (recommended for LDA/PCA) ----
    scaler = None
    X_work = X
    if use_std:
        scaler = StandardScaler()
        X_work = scaler.fit_transform(X)

    # ---- get LDA direction w1 ----
    # For multiclass, LDA can produce >=2 dims, but for binary it is 1D.
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_work, y)

    # sklearn exposes scalings_ for 'svd' solver, coef_ for 'lsqr'/'eigen'.
    if hasattr(lda, "scalings_") and lda.scalings_ is not None:
        w1 = lda.scalings_[:, 0]
    else:
        # coef_ shape: (n_classes, n_features); binary => (1, n_features) or (2, n_features) depending
        w1 = np.ravel(lda.coef_[0])

    # normalize
    w1 = w1 / (np.linalg.norm(w1) + 1e-12)

    # ---- get w2: first PCA direction in subspace orthogonal to w1 ----
    # remove projection along w1
    proj = X_work @ w1  # (n,)
    X_res = X_work - np.outer(proj, w1)  # residuals orthogonal to w1

    pca = PCA(n_components=1)
    pca.fit(X_res)
    w2 = pca.components_[0]
    # ensure orthonormal (numerical safety)
    w2 = w2 - (w2 @ w1) * w1
    w2 = w2 / (np.linalg.norm(w2) + 1e-12)

    # Basis matrix: columns are basis vectors in feature space
    B = np.column_stack([w1, w2])  # (n_features, 2)

    # ---- project points into 2D ----
    Z = X_work @ B  # (n,2)

    def inverse_transform(z2d: np.ndarray) -> np.ndarray:
        # reconstruct points on the 2D plane in X_work space
        X_back = z2d @ B.T
        if scaler is not None:
            X_back = scaler.inverse_transform(X_back)
        return X_back

    # ---- grid in Z space ----
    mins, maxs = Z.min(axis=0), Z.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-12)
    l, r = mins - pad, maxs + pad

    x1 = np.linspace(l[0], r[0], num=grid_step)
    x2 = np.linspace(l[1], r[1], num=grid_step)
    xx1, xx2 = np.meshgrid(x1, x2)

    Z2d = np.c_[xx1.ravel(), xx2.ravel()]
    X_grid = inverse_transform(Z2d)

    # ---- predictions on grid ----
    y_grid = estimator.predict(X_grid)
    y_grid_idx = to_idx(y_grid)
    Z_labels = y_grid_idx.reshape(xx1.shape)

    # ---- colormap ----
    base_cmap = plt.get_cmap("Set1")
    colors = base_cmap(np.arange(max(n_classes, 3)))[:n_classes]
    cmap = ListedColormap(colors)
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, ncolors=n_classes)

    # ---- plot ----
    fig = plt.figure(figsize=(10, 8))

    if plot_confidence:
        probs = None
        if hasattr(estimator, "predict_proba"):
            try:
                probs = estimator.predict_proba(X_grid)
            except Exception:
                probs = None

        if probs is None:
            scores = estimator.decision_function(X_grid)
            if n_classes == 2:
                s = scores.reshape(-1, 1)
                p1 = 1.0 / (1.0 + np.exp(-s))
                conf = p1.reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]}) (approx)")
            else:
                # fallback for multiclass
                if np.ndim(scores) == 2 and scores.shape[1] == n_classes:
                    s = scores - scores.max(axis=1, keepdims=True)
                    exp_s = np.exp(s)
                    probs_approx = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)
                    max_conf = probs_approx.max(axis=1).reshape(xx1.shape)
                    plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
                    plt.colorbar(label="max class probability (approx)")
                else:
                    max_conf = np.max(np.abs(scores), axis=1) if np.ndim(scores) == 2 else np.abs(scores)
                    max_conf = max_conf.reshape(xx1.shape)
                    plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
                    plt.colorbar(label="confidence proxy from decision_function")
        else:
            if n_classes == 2:
                p1 = probs[:, 1].reshape(xx1.shape)
                plt.contourf(xx1, xx2, p1, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]})")
            else:
                max_conf = probs.max(axis=1).reshape(xx1.shape)
                plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
                plt.colorbar(label="max class probability")
    else:
        plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)

    # ---- boundary & margins ----
    if n_classes == 2:
        df = estimator.decision_function(X_grid).reshape(xx1.shape)
        plt.contour(xx1, xx2, df, levels=[0.0], linewidths=2.0)
        if plot_margins:
            plt.contour(xx1, xx2, df, levels=[-1.0, 1.0], linestyles="--", linewidths=1.5)
    else:
        plt.contour(xx1, xx2, Z_labels, levels=bounds, linewidths=1.0, alpha=0.7)

    # ---- points ----
    if plot_all_points:
        plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y_idx,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.8,
            s=18,
        )

    # ---- highlight support vectors ----
    if highlight_support_vectors and hasattr(estimator, "support_"):
        sv_idx = estimator.support_
        if sv_idx is not None and sv_idx.size > 0 and sv_idx.max(initial=-1) < n_samples:
            plt.scatter(
                Z[sv_idx, 0],
                Z[sv_idx, 1],
                facecolors="none",
                edgecolors="black",
                linewidth=1.3,
                s=80,
                marker="o",
            )

    # ---- legend ----
    legend_handles = []
    for i, cls in enumerate(classes):
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   label=f"class {cls}",
                   markerfacecolor=colors[i],
                   markeredgecolor="black",
                   markersize=7, linewidth=0)
        )
    if highlight_support_vectors and hasattr(estimator, "support_"):
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   label="support vectors",
                   markerfacecolor="none",
                   markeredgecolor="black",
                   markersize=8, linewidth=0)
        )

    plt.legend(handles=legend_handles, title="Legend", loc="upper right")
    plt.title(title, fontsize=15)
    plt.xlabel("LDA direction (canonical variate)")
    plt.ylabel("Orthogonal PCA direction")

    # ---- save/show ----
    saved_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            conf_tag = "confidence" if plot_confidence else "labels"
            std_tag = "std" if use_std else "raw"
            margin_tag = "margins" if (plot_margins and n_classes == 2) else "nomargins"
            filename = f"svc_lda_pca2_{conf_tag}_{std_tag}_{margin_tag}.png"

        saved_path = save_dir / filename
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def plot_svc_solution_umap2_fit(
    estimator: SVC | MySVC,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    grid_step: int = 200,
    use_std: bool = False,
    plot_confidence: bool = False,
    plot_all_points: bool = True,
    highlight_support_vectors: bool = True,
    plot_margins: bool = True,
    title: str = "",
    save_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = False,
    # reasonable defaults for UMAP
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.01,
    umap_metric: str = "euclidean",
    umap_random_state: int = 42, # !!!!!!!!!!!!!!!!!!!!!!!!!!! Если поменять на RANDOM_SEED, то linear C=1.0 сойдется к какой-то фигне.
    supervised_umap: bool = True,
):
    """
    Fit UMAP->2D on (X,y) (optionally supervised), then fit a fresh copy of `estimator`
    on the 2D embedding, and plot the *2D model's* decision regions / boundary / margins.

    This answers: "What does an SVC trained on the UMAP 2D embedding look like?"
    It does NOT visualize the original high-D SVC boundary.

    Requirements:
      pip install umap-learn
    """

    # ---- get data ----
    if X is None or y is None:
        raise ValueError("For UMAP 2D training visualization, please provide both X and y.")

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")

    # ---- optional standardization before UMAP ----
    scaler = None
    X_work = X
    if use_std:
        scaler = StandardScaler()
        X_work = scaler.fit_transform(X)

    # ---- UMAP 2D embedding ----

    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric=umap_metric,
        n_jobs=1,
        random_state=umap_random_state,
    )
    if supervised_umap:
        Z = reducer.fit_transform(X_work, y).astype(np.float64)
    else:
        Z = reducer.fit_transform(X_work).astype(np.float64)

    # ---- fit a fresh SVC on the 2D embedding ----
    # (So support vectors, margins etc. are meaningful in 2D.)
    svc2d = clone(estimator)
    svc2d.fit(Z, y)
    check_is_fitted(svc2d)

    classes = svc2d.classes_
    n_classes = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    def to_idx(labels: np.ndarray) -> np.ndarray:
        return np.vectorize(cls_to_idx.get, otypes=[int])(labels)

    y_idx = to_idx(y)

    # ---- grid in UMAP space ----
    mins, maxs = Z.min(axis=0), Z.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-12)
    l, r = mins - pad, maxs + pad

    x1 = np.linspace(l[0], r[0], num=grid_step)
    x2 = np.linspace(l[1], r[1], num=grid_step)
    xx1, xx2 = np.meshgrid(x1, x2)
    Z_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # ---- predictions on grid (2D model) ----
    y_grid = svc2d.predict(Z_grid)
    y_grid_idx = to_idx(y_grid)
    Z_labels = y_grid_idx.reshape(xx1.shape)

    # ---- colormap ----
    base_cmap = plt.get_cmap("Set1")
    colors = base_cmap(np.arange(max(n_classes, 3)))[:n_classes]
    cmap = ListedColormap(colors)
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, ncolors=n_classes)

    # ---- plot ----
    fig = plt.figure(figsize=(10, 8))

    if plot_confidence:
        probs = None
        if hasattr(svc2d, "predict_proba"):
            try:
                probs = svc2d.predict_proba(Z_grid)
            except Exception:
                probs = None

        if probs is None:
            scores = svc2d.decision_function(Z_grid)

            if n_classes == 2:
                s = scores.reshape(-1, 1)
                p1 = 1.0 / (1.0 + np.exp(-s))
                conf = p1.reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]}) (approx)")
            else:
                if np.ndim(scores) == 2 and scores.shape[1] == n_classes:
                    s = scores - scores.max(axis=1, keepdims=True)
                    exp_s = np.exp(s)
                    probs_approx = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)
                    max_conf = probs_approx.max(axis=1).reshape(xx1.shape)
                    plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
                    plt.colorbar(label="max class probability (approx)")
                else:
                    max_conf = np.max(np.abs(scores), axis=1) if np.ndim(scores) == 2 else np.abs(scores)
                    max_conf = max_conf.reshape(xx1.shape)
                    plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
                    plt.colorbar(label="confidence proxy from decision_function")
        else:
            if n_classes == 2:
                p1 = probs[:, 1].reshape(xx1.shape)
                plt.contourf(xx1, xx2, p1, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]})")
            else:
                max_conf = probs.max(axis=1).reshape(xx1.shape)
                plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
                plt.colorbar(label="max class probability")
    else:
        plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)

    # ---- boundaries & margins (2D model) ----
    if n_classes == 2:
        df = svc2d.decision_function(Z_grid).reshape(xx1.shape)
        plt.contour(xx1, xx2, df, levels=[0.0], linewidths=2.0)
        if plot_margins:
            plt.contour(xx1, xx2, df, levels=[-1.0, 1.0], linestyles="--", linewidths=1.5)
    else:
        plt.contour(xx1, xx2, Z_labels, levels=bounds, linewidths=1.0, alpha=0.7)

    # ---- points ----
    if plot_all_points:
        plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y_idx,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.8,
            s=18,
        )

    # ---- support vectors (of the 2D-fitted SVC) ----
    if highlight_support_vectors and hasattr(svc2d, "support_"):
        sv_idx = svc2d.support_
        if sv_idx is not None and sv_idx.size > 0 and sv_idx.max(initial=-1) < n_samples:
            plt.scatter(
                Z[sv_idx, 0],
                Z[sv_idx, 1],
                facecolors="none",
                edgecolors="black",
                linewidth=1.3,
                s=80,
                marker="o",
            )

    # ---- legend ----
    legend_handles = []
    for i, cls in enumerate(classes):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                label=f"class {cls}",
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markersize=7,
                linewidth=0,
            )
        )
    if highlight_support_vectors and hasattr(svc2d, "support_"):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                label="support vectors (2D)",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
        )

    plt.legend(handles=legend_handles, title="Legend", loc="upper right")
    plt.title(title, fontsize=15)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # ---- save/show ----
    saved_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            conf_tag = "confidence" if plot_confidence else "labels"
            std_tag = "std" if use_std else "raw"
            margin_tag = "margins" if (plot_margins and n_classes == 2) else "nomargins"
            sup_tag = "sup" if supervised_umap else "unsup"
            filename = f"svc_umap2_fit_{sup_tag}_{conf_tag}_{std_tag}_{margin_tag}.png"

        saved_path = save_dir / filename
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path, svc2d, reducer


#==============================================================================#
