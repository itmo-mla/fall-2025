from pathlib import Path
from typing import Sequence

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.knn_model import Kernel, KNearestNeighbors, SklearnParzenKNN, _ENNParams
from src.k_param_optim import optimize_k_with_cv, optimize_k_with_loo
from src.utils import utils

#=========================================================================#

__all__ = [
    "read_alzheimer_dataset",
    "classify_cat_cols",
    "build_preprocessor",
    "normalize_kernels",
    "format_proto_tag",
    "make_cv_plot_filename",
    "make_loo_plot_filename",
    "make_decision_boundary_filename",
    "make_comparison_filename",
    "record_result",
    "run_default_case",
    "run_enn_case",
    "run_oss_case",
    "run_stolp_case",
    "run_enn_oss_case",
    "run_sklearn_case",
    "plot_decision_boundary",
]

#=========================================================================#

def read_alzheimer_dataset(
    input_dir: str | Path
) -> pd.DataFrame:
    df = pd.read_csv(input_dir, header=0)
    df = df.drop(columns=["PatientID", "DoctorInCharge"])
    return df


def classify_cat_cols(X: pd.DataFrame, int_cols: list) -> tuple[list, list]:
    bin_cols = []
    cat_cols = []
    for col in int_cols:
        n_unique = X[col].nunique()
        if n_unique == 2:
            bin_cols.append(col)
        elif n_unique > 2 and n_unique < 10:
            cat_cols.append(col)

    for col in (bin_cols + cat_cols):
        int_cols.remove(col)

    print("int cols:         ", len(int_cols))
    print("catecorical cols: ", len(cat_cols))
    print("binary cols:      ", len(bin_cols), "\n")

    return bin_cols, cat_cols


def build_preprocessor(
    int_cols: list,
    float_cols: list,
    bin_cols: list,
    cat_cols: list
) -> ColumnTransformer:
    preprocessing = ColumnTransformer(
        transformers=[
            ("float_cols", StandardScaler(), float_cols + int_cols + [cat_cols[1]]),
            ("cat_cols", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), [cat_cols[0]]),
            # ("ord_cols", OrdinalEncoder(), [cat_cols[1]]),
            ("bin_cols", "passthrough", bin_cols),
        ],
        remainder="passthrough"
    )
    return preprocessing


def normalize_kernels(
    kernels: Sequence[str] | Sequence[Kernel] | Kernel | str | None
) -> list[Kernel]:
    if kernels is None:
        return list(Kernel)

    if isinstance(kernels, Kernel):
        return [kernels]

    if isinstance(kernels, str):
        kernels = [kernels]

    if not kernels:
        return list(Kernel)

    out = []
    for item in kernels:
        if isinstance(item, Kernel):
            out.append(item)
            continue
        if not isinstance(item, str):
            raise TypeError(f"Unsupported kernel type: {type(item)}")

        token = item.strip()
        if not token:
            continue

        if token.isdigit():
            try:
                out.append(Kernel(int(token)))
            except ValueError as exc:
                raise ValueError(f"Unknown kernel id: {token}") from exc
        else:
            try:
                out.append(Kernel[token.upper()])
            except KeyError as exc:
                raise ValueError(f"Unknown kernel name: {token}") from exc

    if not out:
        return list(Kernel)

    deduped = []
    seen = set()
    for kern in out:
        if kern not in seen:
            deduped.append(kern)
            seen.add(kern)

    return deduped


def _slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "-")
        .replace("+", "-")
        .replace("_", "-")
    )


def format_proto_tag(proto_params) -> str:
    if proto_params is None:
        return "no-proto"

    if isinstance(proto_params, dict):
        methods = [proto_params.get("method")]
    else:
        methods = [step.get("method") for step in proto_params]

    methods = [m for m in methods if m]
    if not methods:
        return "no-proto"

    return "-".join(_slugify(m) for m in methods)


def make_cv_plot_filename(
    kernel_name: str,
    k_min: int,
    k_max: int,
    proto_params
) -> str:
    proto_tag = format_proto_tag(proto_params)
    return f"cv_risk_kernel-{_slugify(kernel_name)}_k-{k_min}-{k_max}_proto-{proto_tag}.png"


def make_loo_plot_filename(
    kernel_name: str,
    k_min: int,
    k_max: int,
    variant: str,
    proto_params=None
) -> str:
    proto_tag = format_proto_tag(proto_params)
    return (
        f"loo_{_slugify(variant)}_kernel-{_slugify(kernel_name)}"
        f"_k-{k_min}-{k_max}_proto-{proto_tag}.png"
    )


def make_decision_boundary_filename(
    kernel_name: str,
    k: int,
    proto_label: str,
    use_std: bool,
    plot_confidence: bool,
    plot_all_points: bool
) -> str:
    std_tag = "standardized" if use_std else "raw-space"
    confidence_tag = "confidence" if plot_confidence else "class-labels"
    points_tag = "all-points" if plot_all_points else "prototypes-only"
    proto_tag = _slugify(proto_label)
    return (
        f"decision_boundary_kernel-{_slugify(kernel_name)}_k-{k}"
        f"_proto-{proto_tag}_{confidence_tag}_{std_tag}_{points_tag}.png"
    )


def make_comparison_filename(kernel_name: str) -> str:
    return f"model_comparison_kernel-{_slugify(kernel_name)}.md"

#=========================================================================#
# Experiments helpers
#=========================================================================#

def record_result(results: dict, name: str, metrics: dict, n_prototypes: int) -> None:
    metrics["number of prototypes"] = n_prototypes
    results[name] = metrics


def run_default_case(
    knn_clf: KNearestNeighbors,
    X,
    y,
    preprocessing,
    kernel_name: str,
    kernel_results_dir: Path,
    results: dict,
    k_min: int,
    k_max: int,
    n_splits: int,
    loo_block_size: int,
) -> tuple[int, int, int]:
    print("------ Full train dataset case \n")

    opt_k_loo_fast, _ = knn_clf.leave_one_out(
        k_min, k_max, plot=True,
        save_dir=kernel_results_dir,
        filename=make_loo_plot_filename(kernel_name, k_min, k_max, "fast"),
    )
    opt_k_cv, _ = optimize_k_with_cv(
        X, y,
        kernel=knn_clf.kernel,
        preprocessing=preprocessing,
        k_min=k_min, k_max=k_max,
        proto_params=None,
        n_splits=n_splits,
        plot=True,
        save_dir=kernel_results_dir,
        filename=make_cv_plot_filename(kernel_name, k_min, k_max, proto_params=None),
    )
    opt_k_loo_fair, _ = optimize_k_with_loo(
        X, y,
        kernel=knn_clf.kernel,
        preprocessing=preprocessing,
        k_min=k_min, k_max=k_max,
        proto_params=None,
        block_size=loo_block_size,
        plot=True,
        save_dir=kernel_results_dir,
        filename=make_loo_plot_filename(kernel_name, k_min, k_max, "fair"),
    )

    print(f"Optimal k by fast LOO: {opt_k_loo_fast}")
    print(f"Optimal k by CV:       {opt_k_cv}")
    print(f"Optimal k by fair LOO: {opt_k_loo_fair}")

    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing, proto_params=None, k=5
    )
    record_result(results, "default_5nn", metrics, len(X))

    print("Fast LOO estimation results")
    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing, proto_params=None, k=opt_k_loo_fast
    )
    record_result(results, f"default_fast_loo_{opt_k_loo_fast}nn", metrics, len(X))

    print("CV estimation results")
    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing, proto_params=None, k=opt_k_cv
    )
    record_result(results, f"default_cv_{opt_k_cv}nn", metrics, len(X))

    print("LOO with no leakage estimation results")
    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing, proto_params=None, k=opt_k_loo_fair
    )
    record_result(results, f"default_fair_loo_{opt_k_loo_fair}nn", metrics, len(X))

    return opt_k_cv, opt_k_loo_fast, opt_k_loo_fair


def run_enn_case(
    knn_clf: KNearestNeighbors,
    X,
    y,
    preprocessing,
    kernel_name: str,
    kernel_results_dir: Path,
    results: dict,
    k_min: int,
    k_max: int,
    n_splits: int,
) -> tuple[int, object]:
    print("------ ENN prototype selection \n")

    enn_params = _ENNParams(k=(1, 1), n_iter=100, min_size=10, remove_fraction=0.8)
    proto_params = dict(
        method="enn",
        enn_params=enn_params,
        update="replace",
        verbose=False,
        inplace=False,
    )

    opt_k_cv_enn, _ = optimize_k_with_cv(
        X, y,
        kernel=knn_clf.kernel,
        preprocessing=preprocessing,
        k_min=k_min, k_max=k_max,
        proto_params=proto_params,
        n_splits=n_splits,
        plot=True,
        save_dir=kernel_results_dir,
        filename=make_cv_plot_filename(kernel_name, k_min, k_max, proto_params=proto_params),
    )
    print(f"Optimal k by fair CV with ENN proto selection: {opt_k_cv_enn}\n")

    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing,
        proto_params=proto_params,
        k=opt_k_cv_enn
    )
    enn_proto_idx = knn_clf.select_prototypes(
        method="enn",
        enn_params=enn_params,
        candidates=None,
        update="none",
        verbose=True
    )
    record_result(results, f"enn_opt_{opt_k_cv_enn}nn", metrics, len(enn_proto_idx))

    print(f"Minority class proportion: {knn_clf.y_train_[enn_proto_idx].mean() * 100:.2f}%\n")
    return opt_k_cv_enn, enn_proto_idx


def run_oss_case(
    knn_clf: KNearestNeighbors,
    X,
    y,
    preprocessing,
    kernel_name: str,
    kernel_results_dir: Path,
    results: dict,
    k_min: int,
    k_max: int,
    n_splits: int,
) -> tuple[int, object]:
    print("------ OSS prototype selection \n")

    proto_params = dict(method="oss", verbose=False)

    opt_k_cv_oss, _ = optimize_k_with_cv(
        X, y,
        kernel=knn_clf.kernel,
        preprocessing=preprocessing,
        k_min=k_min, k_max=k_max,
        proto_params=proto_params,
        n_splits=n_splits,
        plot=True,
        save_dir=kernel_results_dir,
        filename=make_cv_plot_filename(kernel_name, k_min, k_max, proto_params=proto_params),
    )
    print(f"Optimal k by fair CV with OSS proto selection: {opt_k_cv_oss}\n")

    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing,
        proto_params=proto_params,
        k=opt_k_cv_oss
    )
    oss_proto_idx = knn_clf.select_prototypes(
        method="oss",
        candidates=None,
        update="none",
        verbose=True
    )
    record_result(results, f"oss_opt_{opt_k_cv_oss}nn", metrics, len(oss_proto_idx))

    print(f"Minority class proportion: {knn_clf.y_train_[oss_proto_idx].mean() * 100:.2f}%\n")
    return opt_k_cv_oss, oss_proto_idx


def run_stolp_case(
    knn_clf: KNearestNeighbors,
    X,
    y,
    preprocessing,
    kernel_name: str,
    kernel_results_dir: Path,
    results: dict,
    k_min: int,
    k_max: int,
    n_splits: int,
) -> tuple[int, object]:
    print("------ STOLP prototype selection \n")

    proto_params = dict(method="stolp", verbose=False)

    opt_k_cv_stolp, _ = optimize_k_with_cv(
        X, y,
        kernel=knn_clf.kernel,
        preprocessing=preprocessing,
        k_min=k_min, k_max=k_max,
        proto_params=proto_params,
        n_splits=n_splits,
        plot=True,
        save_dir=kernel_results_dir,
        filename=make_cv_plot_filename(kernel_name, k_min, k_max, proto_params=proto_params),
    )
    print(f"Optimal k by fair CV with STOLP proto selection: {opt_k_cv_stolp}\n")

    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing,
        proto_params=proto_params,
        k=opt_k_cv_stolp
    )
    stolp_proto_idx = knn_clf.select_prototypes(
        method="stolp",
        candidates=None,
        update="none",
        verbose=True
    )
    record_result(results, f"stolp_opt_{opt_k_cv_stolp}nn", metrics, len(stolp_proto_idx))

    print(f"Minority class proportion: {knn_clf.y_train_[stolp_proto_idx].mean() * 100:.2f}%\n")
    return opt_k_cv_stolp, stolp_proto_idx


def run_enn_oss_case(
    knn_clf: KNearestNeighbors,
    X,
    y,
    preprocessing,
    kernel_name: str,
    kernel_results_dir: Path,
    results: dict,
    k_min: int,
    k_max: int,
    n_splits: int,
) -> tuple[int, object]:
    print("------ ENN + OSS prototype selection \n")

    enn_params = _ENNParams(k=(1, 1), n_iter=50, remove_fraction=0.8)
    proto_params = [
        dict(method="enn", enn_params=enn_params, update="replace", verbose=False),
        dict(method="oss", verbose=False),
    ]

    opt_k_cv_enn_oss, _ = optimize_k_with_cv(
        X, y,
        kernel=knn_clf.kernel,
        preprocessing=preprocessing,
        k_min=k_min, k_max=k_max,
        proto_params=proto_params,
        n_splits=n_splits,
        plot=True,
        save_dir=kernel_results_dir,
        filename=make_cv_plot_filename(kernel_name, k_min, k_max, proto_params=proto_params),
    )
    print(f"Optimal k by fair CV with ENN+OSS proto selection: {opt_k_cv_enn_oss}\n")

    metrics = utils.score_knn(
        knn_clf, X, y, preprocessing=preprocessing,
        proto_params=proto_params,
        k=opt_k_cv_enn_oss
    )

    knn_clf.set_prototypes(None)
    intermediate_proto_idx = knn_clf.select_prototypes(
        method="enn",
        enn_params=enn_params,
        candidates=None,
        update="none",
        verbose=True
    )
    enn_oss_proto_idx = knn_clf.select_prototypes(
        method="oss",
        candidates=intermediate_proto_idx,
        update="none",
        verbose=True
    )
    record_result(results, f"enn_oss_opt_{opt_k_cv_enn_oss}nn", metrics, len(enn_oss_proto_idx))

    print(f"Minority class proportion: {knn_clf.y_train_[enn_oss_proto_idx].mean() * 100:.2f}%\n")
    return opt_k_cv_enn_oss, enn_oss_proto_idx


def run_sklearn_case(
    X_prep,
    y,
    X,
    preprocessing,
    kernel_fn,
    kernel_name: str,
    kernel_results_dir: Path,
    results: dict,
    k_min: int,
    k_max: int,
) -> int:
    ref = SklearnParzenKNN(k=5, kernel=kernel_fn, metric="minkowski", p=2)
    ref.fit(X_prep, y)

    metrics = utils.score_knn(
        ref, X, y, preprocessing=preprocessing, proto_params=None, k=5
    )
    record_result(results, "sklearn_5nn", metrics, len(X))

    opt_k_loo_sk, _ = ref.leave_one_out(
        k_min, k_max, plot=True,
        save_dir=kernel_results_dir,
        filename=make_loo_plot_filename(kernel_name, k_min, k_max, "sklearn"),
    )
    print(f"Optimal k by loo: {opt_k_loo_sk}\n")

    ref = SklearnParzenKNN(k=opt_k_loo_sk, kernel=kernel_fn, metric="minkowski", p=2)
    ref.fit(X_prep, y)
    metrics = utils.score_knn(
        ref, X, y, preprocessing=preprocessing, proto_params=None, k=opt_k_loo_sk
    )
    record_result(results, f"sklearn_{opt_k_loo_sk}nn", metrics, len(X))

    return opt_k_loo_sk


def plot_decision_boundary(
    knn_clf: KNearestNeighbors,
    kernel_name: str,
    kernel_results_dir: Path,
    k: int,
    proto_idx,
    proto_label: str,
    use_std: bool,
    plot_confidence: bool,
    plot_all_points: bool,
    title: str = ""
) -> None:
    filename = make_decision_boundary_filename(
        kernel_name=kernel_name,
        k=k,
        proto_label=proto_label,
        use_std=use_std,
        plot_confidence=plot_confidence,
        plot_all_points=plot_all_points,
    )
    utils.plot_decision_boundaries(
        knn_clf, k, proto_idx,
        use_std=use_std,
        plot_confidence=plot_confidence,
        plot_all_points=plot_all_points,
        grid_step=300,
        title=title,
        save_dir=kernel_results_dir,
        filename=filename,
    )

#=========================================================================#
# End experiments helpers
#=========================================================================#
