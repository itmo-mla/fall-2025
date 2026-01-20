from time import perf_counter_ns
from typing import Union, Optional, Any, Literal, Mapping, Iterable
from pathlib import Path
from collections import defaultdict

import zipfile
import requests

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..knn_model import KNearestNeighbors, SklearnParzenKNN

#================================================================================================================#

__all__ = [
    "check_for_alzheimers_dataset",
    "score_knn",
    "plot_decision_boundaries",
    "compare_models"
]

RANDOM_SEED = 4012025

#========== Download Dataset ==========#

def check_for_alzheimers_dataset(
    *,
    filename: str = "alzheimers_disease_data.csv",
    force_download: bool = False,
    timeout_seconds: int = 120,
) -> Path:
    # Resolve project root from this file location:
    # utils.py -> utils -> src -> project-root
    project_root = Path(__file__).resolve().parents[2]
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    csv_path = datasets_dir / filename
    if csv_path.exists() and not force_download:
        return csv_path

    url = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "rabieelkharoua/alzheimers-disease-dataset"
    )
    zip_path = datasets_dir / "alzheimers-disease-dataset.zip"

    # Download ZIP (streaming)
    try:
        with requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout_seconds,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download dataset from Kaggle endpoint: {e}") from e

    # Extract ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(datasets_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            "Downloaded file is not a valid ZIP. "
            "Kaggle may have returned an HTML page or an auth/consent response."
        ) from e
    finally:
        # Clean up zip if it exists
        if zip_path.exists():
            zip_path.unlink()

    if not csv_path.exists():
        # If Kaggle changes the CSV name, show what's in datasets_dir to help debug.
        candidates = sorted(
            p.name for p in datasets_dir.glob("*.csv") if p.is_file()
        )
        raise FileNotFoundError(
            f"Expected CSV '{filename}' not found in {datasets_dir} after extraction. "
            f"CSV files present: {candidates or 'none'}"
        )

    return csv_path

#========== Scorer ==========#

def score_knn(
    clf: Union[KNearestNeighbors, "SklearnParzenKNN"],
    X: pd.DataFrame,
    y: pd.Series,
    preprocessing,
    proto_params: Optional[dict[str, Any] | Iterable[dict[str, Any]]] = None,
    k: int = 5,
    n_splits: int = 20,
    repeat_preds: int = 100,
    seed: int = RANDOM_SEED,
):
    """
    Cross-validated scoring for either:
      - KNearestNeighbors
      - SklearnParzenKNN

    proto_params (KNearestNeighbors only):
      - None: no prototype selection
      - dict: one selection call per fold (passed to select_prototypes)
      - list[dict]: sequential prototype selection per fold

    Timing:
      - times only the voting step (predict_labels_vect),
        distances are computed once per fold (not timed).
    """
    metrics = defaultdict(list)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if isinstance(clf, KNearestNeighbors):
        model = clf.copy()

        # Normalize proto_params into steps (or None)
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

        y_np = y.to_numpy()

        for _, (train_idx, test_idx) in enumerate(cv.split(X, y_np)):
            # Fold preprocessing (no leakage)
            X_train_cv = preprocessing.fit_transform(X.iloc[train_idx])
            y_train_cv = y_np[train_idx]

            X_test_cv = preprocessing.transform(X.iloc[test_idx])
            y_test_cv = y_np[test_idx]

            model.fit(X_train_cv, y_train_cv)

            # Sequential proto selection on fold-train (if enabled)
            if proto_steps is not None:
                model.proto_idx = None
                for step in proto_steps:
                    model.select_prototypes(**step)
                ref_idx = model.proto_idx  # abs indices into fold-train
            else:
                ref_idx = None

            # Distances once (not timed); columns correspond to ref_idx (or full train if None)
            dists = model.compute_distances(X_test_cv, ref_idx=ref_idx)

            # y_ref must match dists columns
            y_train_used = model.y_train_  # supports inplace=True selection too
            y_ref = y_train_used if ref_idx is None else y_train_used[ref_idx]

            # Time voting only (repeat)
            timings = []
            y_pred = None
            for _ in range(repeat_preds):
                start = perf_counter_ns()
                y_pred, _ = model.predict_labels_vect(dists, y_ref=y_ref, k=k)
                timings.append((perf_counter_ns() - start) / len(y_train_used))

            acc = accuracy_score(y_test_cv, y_pred)
            f1 = f1_score(y_test_cv, y_pred)
            precision = precision_score(y_test_cv, y_pred)
            recall = recall_score(y_test_cv, y_pred)

            metrics["accuracy"].append(acc)
            metrics["f1"].append(f1)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["prediction time"].append(float(np.mean(timings)))

    elif isinstance(clf, SklearnParzenKNN):
        model = clf.copy()
        model.k = k

        for _, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_cv = preprocessing.fit_transform(X.iloc[train_idx])
            y_train_cv = y.iloc[train_idx]
            model.fit(X_train_cv, y_train_cv)

            X_test_cv = preprocessing.transform(X.iloc[test_idx])
            y_test_cv = y.iloc[test_idx]

            timings = []
            for _ in range(repeat_preds):
                start = perf_counter_ns()
                y_pred = model.predict(X_test_cv)
                timings.append((perf_counter_ns() - start) / len(y_train_cv))

            acc = accuracy_score(y_test_cv, y_pred)
            f1 = f1_score(y_test_cv, y_pred)
            precision = precision_score(y_test_cv, y_pred)
            recall = recall_score(y_test_cv, y_pred)

            metrics["accuracy"].append(acc)
            metrics["f1"].append(f1)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["prediction time"].append(float(np.mean(timings)))

    # Aggregate over folds
    for kk, v in metrics.items():
        metrics[kk] = float(np.mean(v))

    if proto_params is None:
        methods_str = "NONE"
    elif isinstance(proto_params, Mapping):
        methods_str = _extract_method(proto_params).upper()
    elif isinstance(proto_params, Iterable):
        methods_str = " + ".join(_extract_method(p) for p in proto_params).upper()
    else:
        methods_str = "NONE" # fallback case
    print(f"Model cross-validation performance estimation on {n_splits} folds.\n",
          "=" * 60, '\n',
          "Model:".ljust(34), clf.__class__.__qualname__, '\n',
          "Kernel:".ljust(34), clf.kernel.__name__, '\n',
          "Parameter k:".ljust(34), k, '\n',
          "Prototype selection algorithm:".ljust(34), methods_str, '\n',
          "-" * 60,
          sep="")

    for kk, v in metrics.items():
        print((kk + ":").ljust(33), f"{v*100:.2f}", "%" if kk != "prediction time" else "ns")

    print("=" * 60, '\n')
    return metrics


def _extract_method(p: Any) -> str:
    if p is None:
        return "none"
    if isinstance(p, Mapping):
        return str(p.get("method", "none"))
    return "none"

#================================================================================================================#

#========== Plots & Visualizations ==========#

def plot_decision_boundaries(
    estimator: KNearestNeighbors,
    k: int,
    proto_idx: np.ndarray | None = None,
    grid_step: int = 100,
    use_std: bool = False,
    plot_confidence: bool = False,
    plot_all_points: bool = True,
    title: str = "",
    save_dir: Path | None = None,
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = False,
):
    """
    Decision boundary plot in 2D PCA space, evaluated by mapping the grid back to original space.

    proto_idx:
      - None: use estimator.proto_idx (if present), else full train set
      - array: absolute indices into estimator.X_train_ / y_train_
    """

    if estimator.X_train_ is None or estimator.y_train_ is None:
        raise RuntimeError("Estimator must be fitted before plotting.")

    X = estimator.X_train_.copy()
    y = estimator.y_train_.copy()

    # Resolve which reference indices to use for classification and prototype highlighting
    ref_idx = estimator._resolve_ref_idx(proto_idx) if proto_idx is not None else estimator._resolve_ref_idx(None)

    # ----- PCA projection (optionally in standardized space) -----
    pca = PCA(n_components=2)
    scaler = None
    if use_std:
        scaler = StandardScaler()
        X_for_pca = scaler.fit_transform(X)
    else:
        X_for_pca = X

    pca_slice = pca.fit_transform(X_for_pca)

    def inverse_transform(z2d: np.ndarray) -> np.ndarray:
        x_back = pca.inverse_transform(z2d)
        if scaler is not None:
            x_back = scaler.inverse_transform(x_back)
        return x_back

    # ----- grid in PCA space -----
    mins, maxs = pca_slice.min(axis=0), pca_slice.max(axis=0)
    pad = 0.05 * (maxs - mins)
    l, r = mins - pad, maxs + pad

    x1 = np.linspace(l[0], r[0], num=grid_step)
    x2 = np.linspace(l[1], r[1], num=grid_step)
    xx1, xx2 = np.meshgrid(x1, x2)

    Z2d = np.c_[xx1.ravel(), xx2.ravel()]
    X_grid = inverse_transform(Z2d)

    # ----- predict on grid using the chosen reference set -----
    # probs returned are "class scores" from predictions; normalize for plotting
    probs = estimator.predict_proba(X_grid, k=k, ref_idx=ref_idx)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)

    y_grid = np.argmax(probs, axis=1)
    Z_labels = y_grid.reshape(xx1.shape)

    # ----- Plot prep -----
    classes = np.unique(y)
    n_classes = len(classes)

    full_cmap = plt.get_cmap("Set1")
    colors = full_cmap(np.arange(n_classes))
    cmap = ListedColormap(colors)

    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, ncolors=n_classes)

    # ----- Plot -----
    fig = plt.figure(figsize=(8, 6))

    if plot_confidence:
        if n_classes == 2:
            p1 = probs[:, 1].reshape(xx1.shape)
            plt.contourf(xx1, xx2, p1, levels=50, cmap="RdBu", alpha=0.7)
            plt.colorbar(label="P(class=1)")
        else:
            max_conf = probs.max(axis=1).reshape(xx1.shape)
            plt.contourf(xx1, xx2, max_conf, levels=30, cmap="viridis", alpha=0.7)
            plt.colorbar(label="max class probability")
    else:
        plt.contourf(xx1, xx2, Z_labels, levels=bounds, cmap=cmap, norm=norm, alpha=0.65)

    if plot_all_points:
        plt.scatter(
            pca_slice[:, 0],
            pca_slice[:, 1],
            c=y,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.8,
            s=10,
        )

    # Highlight prototypes (the ref set we used), only if it's not the full set
    # Note: ref_idx is absolute indices into training data
    if ref_idx is not None and ref_idx.size > 0:
        # If ref_idx == full set, drawing stars over everything is noisy; skip
        if ref_idx.size < X.shape[0]:
            plt.scatter(
                pca_slice[ref_idx, 0],
                pca_slice[ref_idx, 1],
                c=y[ref_idx],
                cmap=cmap,
                norm=norm,
                edgecolor="black",
                marker="*",
                linewidth=1.0,
                s=75,
            )

    # Legend
    legend_handles = []
    for idx, cls in enumerate(classes):
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                label=f"class {cls}",
                markerfacecolor=colors[idx],
                markeredgecolor="black",
                markersize=7,
                linewidth=0,
            )
        )

    if ref_idx is not None and ref_idx.size > 0 and ref_idx.size < X.shape[0]:
        for idx, cls in enumerate(classes):
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="*",
                    color="none",
                    label=f"prototype class {cls}",
                    markerfacecolor=colors[idx],
                    markeredgecolor="black",
                    markersize=7,
                    linewidth=0,
                )
            )

    plt.legend(handles=legend_handles, title="Classes", loc="upper right")
    plt.title(title, fontsize=15)

    # ----- Save if requested -----
    saved_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            conf_tag = "confidence" if plot_confidence else "labels"
            std_tag = "std" if use_std else "raw"
            filename = f"decision_boundary_k{k}_{conf_tag}_{std_tag}.png"

        saved_path = save_dir / filename
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return saved_path

#================================================================================================================#

#========== Models comparison based on results dict ===========#

def compare_models(
    results: dict,
    sort_by: Literal['f1', 'accuracy', 'precision', 'recall', 'prediction time'] = 'f1',
    save_dir: Path | None = None,
    filename: str = "model_comparison.md",
):
    key_map = {
        "f1": "F1",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "prediction time": "Prediction Time (ns)",
        "number of prototypes": "Number of Prototypes"
    }

    rows = []
    for model_name, metrics in results.items():
        # metrics might be a defaultdict(list, {...}); coerce to plain dict
        metrics_dict = dict(metrics)
        # normalize keys (lowercase, trimmed)
        normalized = {str(k).strip().lower(): v for k, v in metrics_dict.items()}

        row = {"Model": model_name}
        for in_key, out_col in key_map.items():
            if in_key in normalized:
                row[out_col] = normalized[in_key]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist (even if missing in some models)
    cols_order = [
        "Model",
        "F1",
        "Accuracy",
        "Precision",
        "Recall",
        "Prediction Time (ns)",
        "Number of Prototypes"
    ]
    for c in cols_order:
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce numeric columns for sorting/styling
    numeric_cols = [c for c in cols_order if c != "Model"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Sort by F1 (descending)
    df = df[cols_order].sort_values(by=key_map[sort_by], ascending=False, na_position="last").reset_index(drop=True)

    # Save to Markdown if directory is provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        md_path = save_dir / filename
        df.to_markdown(md_path, index=False)

    # Metrics groups
    higher_is_better = ["F1", "Accuracy", "Precision", "Recall"]
    lower_is_better = ["Prediction Time (ns)", "Number of Prototypes"]

    styled_df = (
        df.style
        # Best (green) / worst (red)
        .highlight_max(subset=higher_is_better, color="#c6efce")
        .highlight_min(subset=lower_is_better, color="#c6efce")
        .highlight_min(subset=higher_is_better, color="#ffc7ce")
        .highlight_max(subset=lower_is_better, color="#ffc7ce")
        # Gradients
        .background_gradient(subset=higher_is_better, cmap="RdYlGn")
        .background_gradient(subset=lower_is_better, cmap="RdYlGn_r")
        # Formatting
        .format({
            "F1": "{:.3f}",
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "Prediction Time (ns)": "{:.4f}",
            "Number of Prototypes": "{}"
        }, na_rep="—")
        .set_caption("Model Performance Comparison (sorted by F1)")
    )

    return styled_df

#======================================================================================#