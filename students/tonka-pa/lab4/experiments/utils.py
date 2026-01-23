from typing import Union, Optional, Any, Literal, Type
from pathlib import Path
from time import perf_counter

import zipfile
import requests

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.pca import MyPCA
from src.intrinsic_dim_mle import estimate_intrinsic_dim_skidim


#==================================================================

__all__ = [
    "check_for_dataset",
    "read_dataset",
    "baseline_case",
    "score_model",
    "cross_validate_with_pca",
    "plot_2d_reduction"
]

RANDOM_SEED = 23012026

#==================================================================

#========== Download Dataset ==========#

def check_for_dataset(
    *,
    filename: str = "instagram_usage_lifestyle.csv",
    force_download: bool = False,
    timeout_seconds: int = 120,
) -> Path:
    # Resolve project root from this file location:
    # utils.py -> experiments -> project-root
    project_root = Path(__file__).resolve().parents[1]
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    csv_path = datasets_dir / filename
    if csv_path.exists() and not force_download:
        return csv_path

    url = "https://www.kaggle.com/api/v1/datasets/download/rockyt07/social-media-user-analysis"
    zip_path = datasets_dir / "instagram.zip"

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

def read_dataset(
    input_dir: str | Path
) -> pd.DataFrame:
    df = pd.read_csv(input_dir, header=0)
    df = df.drop(columns=['user_id', 'app_name', 'last_login_date'])
    return df

#==================================================================

#========== Model tests ==========

def baseline_case(
    X_train, y_train, # preprocessed data
    X_test,  y_test,  # preprocessed data
    *,
    pca_class: Type[PCA | MyPCA] = PCA,
    n_components: int = 10,
    svd_solver: Literal['full', 'covariance_eigh'] = 'full',
    include_baselines: bool = True,
    model_label: Optional[str] = None,
):
    results: list[dict[str, Any]] = []

    if include_baselines:
        preds = np.repeat(y_train.mean(), y_test.size)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        print("----Mean prediciton model----")
        print(f'RMSE: {rmse:.4f}')
        print(f'R^2:  {r2:.4f}')
        print(f'MAE:  {mae:.4f}')
        results.append(
            {
                "model": "Mean prediction",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pca_fit_time": None,
                "n_components": None,
            }
        )

        linreg_baseline = LinearRegression()
        linreg_baseline.fit(X_train, y_train)
        rmse, mae, r2 = score_model(linreg_baseline, X_test, y_test)
        results.append(
            {
                "model": "Linear regression (full features)",
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pca_fit_time": None,
                "n_components": None,
            }
        )

    pca_label = model_label or pca_class.__name__
    pca = pca_class(n_components=n_components, svd_solver=svd_solver)
    pca_start = perf_counter()
    X_train_reduced = pca.fit_transform(X_train)
    pca_fit_time = perf_counter() - pca_start
    linreg_pca = LinearRegression()
    linreg_pca.fit(X_train_reduced, y_train)
    X_test_reduced = pca.transform(X_test)
    rmse, mae, r2 = score_model(
        linreg_pca,
        X_test_reduced,
        y_test,
        model_name=f'Linear regression PCA ({pca_label}, n_components={n_components})',
    )
    results.append(
        {
            "model": f"Linear regression + {pca_label}",
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pca_fit_time": pca_fit_time,
            "n_components": getattr(pca, "n_components_", n_components),
        }
    )

    return results


def score_model(model, X_test, y_test, model_name: str = "Linear regression"):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    print(f"----{model_name}----")
    print(f'RMSE: {rmse:.4f}')
    print(f'R^2:  {r2:.4f}')
    print(f'MAE:  {mae:.4f}')
    return rmse, mae, r2


def cross_validate_with_pca(
    X, y,
    preprocessor: Pipeline | ColumnTransformer,
    pca_class: PCA | MyPCA,
    n_components: int | float | Literal['profile_likelihood'] | None = None,
    use_mle_estimator: bool = False,
    mle_estimator_comb: Literal['mean', 'mle'] = 'mean',
    svd_solver: Literal['full', 'covariance_eigh'] = 'full',
    n_splits: int = 5,
    plot_profile_likelihood_dir: Path | None = None,
    plot_mle_dir: Path | None = None,
):
    results = {
        "rmse": [],
        "mae": [],
        "r2": [],
        "intrinsic_dim": [],
        "pca_fit_time": [],
    }

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=RANDOM_SEED)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test,  y_test  = X.iloc[test_idx],  y[test_idx]
        X_train = preprocessor.fit_transform(X_train)
        X_test  = preprocessor.transform(X_test)
        
        if use_mle_estimator:
            mle_plot_path = None
            if plot_mle_dir is not None:
                plot_mle_dir.mkdir(parents=True, exist_ok=True)
                mle_plot_path = plot_mle_dir / (
                    f"mle_intrinsic_dim_fold_{i + 1}_comb_{mle_estimator_comb}.png"
                )
            mle_intrinsic_dim = estimate_intrinsic_dim_skidim(
                X_train,
                2,
                70,
                comb=mle_estimator_comb,
                plot=plot_mle_dir is not None,
                save_path=mle_plot_path,
            )
            n_components_fold = mle_intrinsic_dim
        else:
            n_components_fold = n_components

        reducer = pca_class(n_components=n_components_fold, svd_solver=svd_solver)
        pca_start = perf_counter()
        reducer.fit(X_train)
        pca_fit_time = perf_counter() - pca_start
        X_train_reduced = reducer.transform(X_train)

        if plot_profile_likelihood_dir is not None and hasattr(reducer, "likelihood_profile_"):
            plot_profile_likelihood_dir.mkdir(parents=True, exist_ok=True)
            save_profile_likelihood_plot(
                reducer.likelihood_profile_,
                plot_profile_likelihood_dir / f"profile_likelihood_fold_{i + 1}.png",
                title=f"Profile likelihood (fold {i + 1})",
            )

        linreg = LinearRegression()
        linreg.fit(X_train_reduced, y_train)
        X_test_reduced = reducer.transform(X_test)
        preds = linreg.predict(X_test_reduced)
        rmse = root_mean_squared_error(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['r2'].append(r2)
        results['intrinsic_dim'].append(getattr(reducer, "n_components_", n_components_fold))
        results['pca_fit_time'].append(pca_fit_time)

    for name, values in results.items():
        print(f"{name:<16} {np.mean(values):.4f}")
    
    return results

#==================================================================

#========== Profile Likelihood Plot ==========#

def save_profile_likelihood_plot(
    likelihood_profile: np.ndarray,
    save_path: Path | str,
    *,
    title: str | None = None,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(likelihood_profile)
    xs = np.arange(1, values.size + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    ax.plot(xs, values, linestyle="--", marker="o", markersize=3)
    ax.set_xlabel("Components")
    ax.set_ylabel("Log-likelihood")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

#==================================================================

#========== Compare implementations ==========#

def principal_angles(Va, Vb):
    # Va, Vb: p x k orthonormal
    M = Va.T @ Vb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return angles

def projector_fro_error(Va, Vb):
    Pa = Va @ Va.T
    Pb = Vb @ Vb.T
    return np.linalg.norm(Pa - Pb, ord='fro')

def procrustes_align(Z_ref, Z):
    # find R s.t. Z ≈ Z_ref @ R, with R orthogonal
    M = Z_ref.T @ Z
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return Z_ref @ R, R

#==================================================================

#========== Plots ==========#

def plot_2d_reduction(
    data: np.ndarray | Any,
    reducer_cls: Type[Any],
    *,
    n_components: int = 2,
    svd_solver: Optional[str] = None,
    labels: Optional[Union[np.ndarray, list]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    s: float = 18,
    alpha: float = 0.85,
    random_state: Optional[int] = None,
    save_path: Path | str | None = None,
    save_cumulative_path: Path | str | None = None,
    cumulative_title: Optional[str] = None,
) -> plt.Axes:

    X = np.asarray(data)
    if X.ndim != 2:
        raise ValueError(f"`data` must be 2D (n_samples, n_features). Got shape {X.shape}.")
    if n_components != 2:
        raise ValueError("This plotting function is for 2D. Use n_components=2.")

    # Build kwargs carefully (only pass what is explicitly allowed / supported).
    init_kwargs: dict[str, Any] = {"n_components": n_components}
    if svd_solver is not None:
        init_kwargs["svd_solver"] = svd_solver

    # Best-effort: some PCA-like reducers accept random_state; your custom may ignore it.
    # Only pass if the constructor appears to accept it (avoid surprising TypeError).
    if random_state is not None:
        try:
            import inspect

            sig = inspect.signature(reducer_cls.__init__)
            if "random_state" in sig.parameters:
                init_kwargs["random_state"] = random_state
        except Exception:
            # If signature inspection fails, don't pass random_state.
            pass

    reducer = reducer_cls(**init_kwargs)

    # Reduce using fit_transform if present; otherwise fit + transform.
    if hasattr(reducer, "fit_transform"):
        Z = reducer.fit_transform(X)
    else:
        reducer.fit(X)
        if not hasattr(reducer, "transform"):
            raise TypeError(
                "`reducer_cls` must implement fit_transform(X) or fit(X) + transform(X)."
            )
        Z = reducer.transform(X)

    Z = np.asarray(Z)
    if Z.shape[1] != 2:
        raise ValueError(f"Reducer output must be 2D. Got shape {Z.shape}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)

    # Scatter (with optional coloring)
    if labels is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=s, alpha=alpha, linewidths=0)
    else:
        c = np.asarray(labels)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, s=s, alpha=alpha, linewidths=0)
        # Colorbar only if labels look continuous/numeric
        if np.issubdtype(c.dtype, np.number):
            plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)

    # Light dotted grid
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    # "Scientific-like borders": open top/right, subtle left/bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(direction="out", length=4, width=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Clean layout
    ax.figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)

    if save_cumulative_path is not None:
        save_cumulative_path = Path(save_cumulative_path)
        save_cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        full_components = min(X.shape[0], X.shape[1])
        full_kwargs = dict(init_kwargs)
        full_kwargs["n_components"] = full_components
        reducer_full = reducer_cls(**full_kwargs)
        reducer_full.fit(X)
        if not hasattr(reducer_full, "explained_variance_ratio_"):
            raise AttributeError("Reducer does not expose explained_variance_ratio_.")
        cum = np.cumsum(reducer_full.explained_variance_ratio_)
        fig, ax_full = plt.subplots(figsize=(6.4, 4.2), dpi=120)
        ax_full.plot(
            np.arange(1, cum.size + 1),
            cum,
            linestyle="--",
            # marker="o",
            markersize=3,
        )
        ax_full.set_xlabel("Number of components")
        ax_full.set_ylabel("Cumulative explained variance ratio")
        ax_full.set_ylim(0.0, 1.02)
        if cumulative_title:
            ax_full.set_title(cumulative_title)
        ax_full.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.35)
        fig.tight_layout()
        fig.savefig(save_cumulative_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return ax
