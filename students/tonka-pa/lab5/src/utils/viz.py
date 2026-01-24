from typing import Optional
from pathlib import Path

import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from sklearn.linear_model import LogisticRegression

import umap

from ..logreg import MyLogisticRegression, SMGLMLogitClassifier

#==============================================================================#

__all__ = [
    'plot_logreg_solution_lda_pca2',
    'plot_clf_solution_umap2_fit'
]

RANDOM_SEED = 18012026

#==============================================================================#

#========== Visualization ==========#

def plot_logreg_solution_lda_pca2(
    estimator,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    grid_step: int = 200,
    use_std: bool = False,
    plot_confidence: bool = False,
    plot_all_points: bool = True,
    plot_boundary: bool = True,
    title: str = "",
    save_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = False,
):
    """
    Universal 2D visualization for *classifiers* (binary or multiclass) using:
      - axis 1: LDA direction (1D for binary)
      - axis 2: top PCA direction orthogonal to the LDA direction

    Works with:
      - MyLogisticRegression
      - sklearn LogisticRegression
      - statsmodels wrapper SMGLMLogitClassifier
      - and most sklearn-like estimators with predict + (decision_function or predict_proba)

    Notes:
      - Logistic regression has no "margins" like SVM; for binary we show boundary at score=0
        (logit=0 => p=0.5).
    """
    check_is_fitted(estimator)

    # ---- resolve data ----
    if X is None:
        raise ValueError("Provide X (training data) for LDA/PCA projection and grid plotting.")
    X = np.asarray(X)
    n_samples, n_features = X.shape

    # y is required to compute LDA direction
    if y is None:
        if plot_all_points:
            y = estimator.predict(X)
        else:
            raise ValueError("Need y to compute LDA direction.")
    y = np.asarray(y)
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")

    if not hasattr(estimator, "classes_"):
        raise ValueError("Estimator must have classes_ attribute (sklearn-like).")

    classes = np.asarray(estimator.classes_)
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
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_work, y)

    if hasattr(lda, "scalings_") and lda.scalings_ is not None:
        w1 = lda.scalings_[:, 0]
    else:
        w1 = np.ravel(lda.coef_[0])

    w1 = w1 / (np.linalg.norm(w1) + 1e-12)

    # ---- get w2: first PCA direction orthogonal to w1 ----
    proj = X_work @ w1
    X_res = X_work - np.outer(proj, w1)

    pca = PCA(n_components=1)
    pca.fit(X_res)
    w2 = pca.components_[0]
    w2 = w2 - (w2 @ w1) * w1
    w2 = w2 / (np.linalg.norm(w2) + 1e-12)

    B = np.column_stack([w1, w2])  # (n_features, 2)

    # ---- project points into 2D ----
    Z = X_work @ B  # (n,2)

    def inverse_transform(z2d: np.ndarray) -> np.ndarray:
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

    # ---- model outputs on grid ----
    y_grid = estimator.predict(X_grid)
    y_grid_idx = to_idx(y_grid)
    Z_labels = y_grid_idx.reshape(xx1.shape)

    # decision scores (preferred for boundary); else approximate via proba
    df = None
    if hasattr(estimator, "decision_function"):
        try:
            df = estimator.decision_function(X_grid)
        except Exception:
            df = None

    probs = None
    if hasattr(estimator, "predict_proba"):
        try:
            probs = estimator.predict_proba(X_grid)
        except Exception:
            probs = None

    # ---- colormap ----
    base_cmap = plt.get_cmap("Set1")
    colors = base_cmap(np.arange(max(n_classes, 3)))[:n_classes]
    cmap = ListedColormap(colors)
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, ncolors=n_classes)

    # ---- plot ----
    fig = plt.figure(figsize=(8, 7))

    if plot_confidence:
        # for binary: show P(class=classes[1]) if available, else approximate
        if probs is not None:
            if n_classes == 2:
                conf = probs[:, 1].reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]})")
            else:
                conf = probs.max(axis=1).reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=30, cmap="viridis", alpha=0.7)
                plt.colorbar(label="max class probability")
        else:
            # fallback if predict_proba missing
            if df is None:
                raise ValueError("plot_confidence=True requires predict_proba or decision_function.")
            if n_classes == 2:
                s = np.asarray(df).reshape(-1)
                conf = expit(s).reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]}) (from sigmoid(decision_function))")
            else:
                # multiclass fallback: show label regions if confidence is not available
                plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)
    else:
        plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)

    # ---- boundary ----
    # For logistic regression binary: boundary is df=0 (logit=0 => p=0.5)
    if plot_boundary and n_classes == 2:
        if df is None:
            # approximate df via proba if possible: logit(p)=log(p/(1-p))
            if probs is not None:
                p1 = np.clip(probs[:, 1], 1e-12, 1.0 - 1e-12)
                df = np.log(p1 / (1.0 - p1))
            else:
                df = None

        if df is not None:
            df2 = np.asarray(df).reshape(xx1.shape)
            plt.contour(xx1, xx2, df2, levels=[0.0], linewidths=2.0)

    # ---- points ----
    if plot_all_points:
        plt.scatter(
            Z[:, 0], Z[:, 1],
            c=y_idx, cmap=cmap, norm=norm,
            edgecolor="black", linewidth=0.8, s=18
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
            filename = f"logreg_lda_pca2_{conf_tag}_{std_tag}.png"

        saved_path = save_dir / filename
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def plot_clf_solution_umap2_fit(
    estimator,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    grid_step: int = 200,
    use_std: bool = False,
    plot_confidence: bool = False,
    plot_all_points: bool = True,
    plot_boundary: bool = True,
    title: str = "",
    save_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = False,
    # UMAP params
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.01,
    umap_metric: str = "euclidean",
    umap_random_state: int = 42,
    supervised_umap: bool = True,
):
    """
    Fit UMAP->2D on (X,y) (optionally supervised), then fit a fresh copy of `estimator`
    on the 2D embedding, and plot the *2D model's* decision regions / boundary.

    This answers: "What does a classifier trained on the UMAP 2D embedding look like?"
    It does NOT visualize the original high-D boundary.

    Works with:
      - MyLogisticRegression
      - sklearn LogisticRegression
      - SMGLMLogitClassifier
      - most sklearn-like classifiers with predict + (predict_proba or decision_function)
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
        random_state=umap_random_state,
    )
    if supervised_umap:
        Z = reducer.fit_transform(X_work, y).astype(np.float64)
    else:
        Z = reducer.fit_transform(X_work).astype(np.float64)

    # ---- fit a fresh classifier on the 2D embedding ----
    clf2d = clone(estimator)
    clf2d.fit(Z, y)
    check_is_fitted(clf2d)

    if not hasattr(clf2d, "classes_"):
        raise ValueError("Fitted estimator must have classes_ attribute.")

    classes = np.asarray(clf2d.classes_)
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
    y_grid = clf2d.predict(Z_grid)
    y_grid_idx = to_idx(y_grid)
    Z_labels = y_grid_idx.reshape(xx1.shape)

    # ---- optional scores / probs on grid ----
    df = None
    if hasattr(clf2d, "decision_function"):
        try:
            df = clf2d.decision_function(Z_grid)
        except Exception:
            df = None

    probs = None
    if hasattr(clf2d, "predict_proba"):
        try:
            probs = clf2d.predict_proba(Z_grid)
        except Exception:
            probs = None

    # ---- colormap ----
    base_cmap = plt.get_cmap("Set1")
    colors = base_cmap(np.arange(max(n_classes, 3)))[:n_classes]
    cmap = ListedColormap(colors)
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, ncolors=n_classes)

    # ---- plot ----
    fig = plt.figure(figsize=(8, 6))

    if plot_confidence:
        if probs is not None:
            if n_classes == 2:
                conf = probs[:, 1].reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]})")
            else:
                conf = probs.max(axis=1).reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=30, cmap="viridis", alpha=0.7)
                plt.colorbar(label="max class probability")
        else:
            if df is None:
                raise ValueError("plot_confidence=True requires predict_proba or decision_function.")
            if n_classes == 2:
                s = np.asarray(df).reshape(-1)
                conf = expit(s).reshape(xx1.shape)
                plt.contourf(xx1, xx2, conf, levels=50, cmap="RdBu", alpha=0.7)
                plt.colorbar(label=f"P(class={classes[1]}) (from sigmoid(decision_function))")
            else:
                # if no proba for multiclass, fall back to labels
                plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)
    else:
        plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)

    # ---- boundary (2D model) ----
    if plot_boundary and n_classes == 2:
        if df is None:
            # try to reconstruct a "logit-like" score from proba
            if probs is not None:
                p1 = np.clip(probs[:, 1], 1e-12, 1.0 - 1e-12)
                df = np.log(p1 / (1.0 - p1))
        if df is not None:
            df2 = np.asarray(df).reshape(xx1.shape)
            plt.contour(xx1, xx2, df2, levels=[0.0], linewidths=2.0)

    # ---- points ----
    if plot_all_points:
        plt.scatter(
            Z[:, 0], Z[:, 1],
            c=y_idx, cmap=cmap, norm=norm,
            edgecolor="black", linewidth=0.8, s=18
        )

    # ---- legend ----
    legend_handles = []
    for i, cls in enumerate(classes):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o", color="none",
                label=f"class {cls}",
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markersize=7, linewidth=0,
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
            sup_tag = "sup" if supervised_umap else "unsup"
            filename = f"clf_umap2_fit_{sup_tag}_{conf_tag}_{std_tag}.png"

        saved_path = save_dir / filename
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path, clf2d, reducer


#==============================================================================#
