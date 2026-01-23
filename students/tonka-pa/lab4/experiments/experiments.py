from pathlib import Path
from numbers import Real
from typing import Iterable

import numpy as np

from sklearn.decomposition import PCA

from .preprocess import preprocess_dataset
from .utils import (
    baseline_case,
    cross_validate_with_pca,
    plot_2d_reduction,
    read_dataset,
    principal_angles,
    projector_fro_error,
    procrustes_align,
)
from src.pca import MyPCA


#==================================================================

__all__ = [
    "run_tests"
]

#==================================================================

def _format_value(value: object, float_format: str = "{:.4f}") -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return float_format.format(float(value))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, Real):
        return float_format.format(float(value))
    return str(value)


def _write_markdown_table(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    headers: list[str] | None = None,
) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        formatted = [_format_value(row.get(h)) for h in headers]
        lines.append("| " + " | ".join(formatted) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cv_results_rows(model_name: str, results: dict[str, list[object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_folds = len(results.get("rmse", []))
    for i in range(n_folds):
        rows.append(
            {
                "model": model_name,
                "fold": i + 1,
                "rmse": results["rmse"][i],
                "mae": results["mae"][i],
                "r2": results["r2"][i],
                "n_components": results["intrinsic_dim"][i],
                "pca_fit_time": results["pca_fit_time"][i],
            }
        )
    if n_folds:
        rows.append(
            {
                "model": model_name,
                "fold": "mean",
                "rmse": float(np.mean(results["rmse"])),
                "mae": float(np.mean(results["mae"])),
                "r2": float(np.mean(results["r2"])),
                "n_components": float(np.mean(results["intrinsic_dim"])),
                "pca_fit_time": float(np.mean(results["pca_fit_time"])),
            }
        )
        rows.append(
            {
                "model": model_name,
                "fold": "std",
                "rmse": float(np.std(results["rmse"])),
                "mae": float(np.std(results["mae"])),
                "r2": float(np.std(results["r2"])),
                "n_components": float(np.std(results["intrinsic_dim"])),
                "pca_fit_time": float(np.std(results["pca_fit_time"])),
            }
        )
    return rows


def _ensure_solver_list(solvers: Iterable[str] | None) -> list[str]:
    available = ["full", "covariance_eigh"]
    if solvers is None:
        return available
    normalized: list[str] = []
    for solver in solvers:
        if solver not in available:
            raise ValueError(f"Unknown solver '{solver}'. Available: {available}")
        if solver not in normalized:
            normalized.append(solver)
    return normalized


def _compare_pca_models(
    X_train: np.ndarray,
    *,
    svd_solver: str,
) -> list[dict[str, object]]:
    my_pca = MyPCA(n_components=0.9, svd_solver=svd_solver)
    sk_pca = PCA(n_components=0.9, svd_solver=svd_solver)
    my_pca.fit(X_train)
    sk_pca.fit(X_train)

    Va = my_pca.components_
    Vb = sk_pca.components_.T
    k = min(Va.shape[1], Vb.shape[1])
    Va = Va[:, :k]
    Vb = Vb[:, :k]

    angles = principal_angles(Va, Vb)
    angles_deg = np.degrees(angles)
    proj_error = projector_fro_error(Va, Vb)

    Z_my = my_pca.transform(X_train)[:, :k]
    Z_sk = sk_pca.transform(X_train)[:, :k]
    Z_sk_aligned, _ = procrustes_align(Z_sk, Z_my)
    alignment_error = np.linalg.norm(Z_my - Z_sk_aligned, ord='fro') / np.linalg.norm(Z_my, ord='fro')

    return [
        {"metric": "n_components_my", "value": my_pca.n_components_},
        {"metric": "n_components_sklearn", "value": sk_pca.n_components_},
        {"metric": "subspace_angle_mean_deg", "value": float(np.mean(angles_deg))},
        {"metric": "subspace_angle_max_deg", "value": float(np.max(angles_deg))},
        {"metric": "projector_fro_error", "value": float(proj_error)},
        {"metric": "procrustes_relative_error", "value": float(alignment_error)},
    ]


def run_tests(
    *,
    input_dir: str | Path,
    results_dir: str | Path = "./results",
    solvers: Iterable[str] | None = None,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_root = Path(results_dir)
    if not results_root.is_absolute():
        results_root = project_root / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    print("== PCA test run ==")
    print("Loading dataset and preprocessing...")
    df = read_dataset(input_dir)
    X, y, X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        df,
        results_dir=results_root,
        correlation_plot_name="correlation_matrix_pearson.png",
    )

    print("Fitting preprocessor on train split...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    solver_list = _ensure_solver_list(solvers)
    for solver in solver_list:
        print(f"== Solver: {solver} ==")
        solver_dir = results_root / solver
        solver_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = solver_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        print("Running baseline cases...")
        baseline_rows = []
        baseline_rows += baseline_case(
            X_train_proc,
            y_train,
            X_test_proc,
            y_test,
            pca_class=MyPCA,
            n_components=10,
            svd_solver=solver,
            include_baselines=True,
            model_label="MyPCA",
        )
        baseline_rows += baseline_case(
            X_train_proc,
            y_train,
            X_test_proc,
            y_test,
            pca_class=PCA,
            n_components=10,
            svd_solver=solver,
            include_baselines=False,
            model_label="sklearn PCA",
        )
        _write_markdown_table(
            baseline_rows,
            solver_dir / f"baseline_results_svd_solver_{solver}.md",
            headers=["model", "rmse", "mae", "r2", "n_components", "pca_fit_time"],
        )

        cv_cases = [
            {
                "name": "n_components_0.95",
                "n_components_my": 0.95,
                "n_components_sk": 0.95,
                "use_mle_estimator": False,
                "plot_profile_likelihood": False,
                "plot_mle": False,
            },
            {
                "name": "profile_likelihood",
                "n_components_my": "profile_likelihood",
                "n_components_sk": "mle",
                "use_mle_estimator": False,
                "plot_profile_likelihood": True,
                "plot_mle": False,
            },
            {
                "name": "use_mle_estimator_true",
                "n_components_my": None,
                "n_components_sk": None,
                "use_mle_estimator": True,
                "plot_profile_likelihood": False,
                "plot_mle": True,
            },
        ]

        for case in cv_cases:
            print(f"Running CV case: {case['name']}...")
            profile_dir = None
            mle_dir = None
            if case["plot_profile_likelihood"]:
                profile_dir = plot_dir / "profile_likelihood"
            if case["plot_mle"]:
                mle_dir = plot_dir / f"mle_intrinsic_dim_{case['name']}"

            my_results = cross_validate_with_pca(
                X,
                y,
                preprocessor,
                MyPCA,
                n_components=case["n_components_my"],
                use_mle_estimator=case["use_mle_estimator"],
                mle_estimator_comb="mean",
                svd_solver=solver,
                plot_profile_likelihood_dir=profile_dir,
                plot_mle_dir=mle_dir,
            )
            sk_results = cross_validate_with_pca(
                X,
                y,
                preprocessor,
                PCA,
                n_components=case["n_components_sk"],
                use_mle_estimator=case["use_mle_estimator"],
                mle_estimator_comb="mean",
                svd_solver=solver,
                plot_profile_likelihood_dir=None,
                plot_mle_dir=None,
            )

            cv_rows = _cv_results_rows("MyPCA", my_results) + _cv_results_rows("sklearn PCA", sk_results)
            _write_markdown_table(
                cv_rows,
                solver_dir / f"cv_results_{case['name']}_svd_solver_{solver}.md",
                headers=["model", "fold", "rmse", "mae", "r2", "n_components", "pca_fit_time"],
            )

        print("Comparing fitted PCA models...")
        comparison_rows = _compare_pca_models(X_train_proc, svd_solver=solver)
        _write_markdown_table(
            comparison_rows,
            solver_dir / f"pca_model_comparison_svd_solver_{solver}.md",
            headers=["metric", "value"],
        )

    print("Generating 2D projection and cumulative variance plots...")
    full_dir = results_root / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    plot_2d_reduction(
        X_train_proc,
        MyPCA,
        n_components=2,
        svd_solver="full",
        labels=y_train.to_numpy(),
        title="MyPCA 2D projection (svd_solver=full)",
        save_path=full_dir / "pca_2d_projection_mypca_svd_solver_full.png",
        save_cumulative_path=full_dir / "pca_cumulative_explained_variance_mypca_svd_solver_full.png",
        cumulative_title="Cumulative explained variance (MyPCA, svd_solver=full)",
    )
