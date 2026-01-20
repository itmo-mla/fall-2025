from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from src.svc import MySVC, Kernel
from src.utils.preprocess import (
    read_alzheimer_dataset,
    classify_cat_cols,
    build_preprocessor,
    score_model_cv,
    compare_models,
)

from . import helpers as exp_helpers

#=========================================================================#

__all__ = [
    "run_tests"
]

RANDOM_SEED = 18012026

#=========================================================================#

def run_tests(
    input_dir: str | Path,
    results_dir: str | Path,
    kernels: list[str] | list[Kernel] | Kernel | str | None = None,
):
    df_alzh = read_alzheimer_dataset(input_dir)

    X, y = df_alzh.drop(columns=["Diagnosis"]), df_alzh["Diagnosis"]
    print(f"Dataset shape: {df_alzh.shape}")
    print(f"Number of classes: {df_alzh['Diagnosis'].nunique()}\n")
    print("Classes distribution: ")
    print(y.value_counts(normalize=True), "\n")

    int_cols = X.select_dtypes(include=["int"]).columns.tolist()
    float_cols = X.select_dtypes(include=["float"]).columns.tolist()

    print("Int cols: ", len(int_cols))
    print("Float cols: ", len(float_cols))

    bin_cols, cat_cols = classify_cat_cols(X, int_cols)

    preprocessor = build_preprocessor(int_cols, float_cols, bin_cols, cat_cols)
    X_prep = preprocessor.fit_transform(X)

    # ------------------------------------

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    kernel_names = normalize_kernel_names(kernels)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    total_variants = sum(len(build_kernel_variants(name)) for name in kernel_names)
    variant_idx = 0

    for kernel_name in kernel_names:
        variants = build_kernel_variants(kernel_name)
        for params in variants:
            variant_idx += 1
            variant_tag = exp_helpers.build_variant_tag(kernel_name, params)
            kernel_results_dir = results_dir / variant_tag
            kernel_results_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 78)
            print(f"Experiment {variant_idx}/{total_variants} | {variant_tag}")
            print("=" * 78)

            print(params)
            my_svc = MySVC(**params)
            sk_svc = SVC(**params)

            my_pipe = Pipeline([("preprocess", preprocessor), ("model", my_svc)])
            sk_pipe = Pipeline([("preprocess", preprocessor), ("model", sk_svc)])

            print("[Stage] Cross-validation scoring")
            my_metrics = score_model_cv(my_pipe, X, y, cv=cv, include_timing=True)
            sk_metrics = score_model_cv(sk_pipe, X, y, cv=cv, include_timing=True)

            print("[Stage] Fit full models on preprocessed data")
            my_svc.fit(X_prep, y)
            sk_svc.fit(X_prep, y)

            print("[Stage] Decision boundary plots")
            exp_helpers.plot_decision_boundary(
                my_svc,
                X_prep,
                y,
                kernel_name,
                kernel_results_dir,
                use_std=False,
                plot_confidence=False,
                plot_all_points=True,
                title=f"MySVC | {variant_tag}",
                variant_tag=variant_tag,
            )
            exp_helpers.plot_decision_boundary(
                sk_svc,
                X_prep,
                y,
                kernel_name,
                kernel_results_dir,
                use_std=False,
                plot_confidence=False,
                plot_all_points=True,
                title=f"sklearn SVC | {variant_tag}",
                variant_tag=variant_tag,
            )

            kernel_results: dict = {}
            my_support_arr = getattr(my_svc, "support_", None)
            sk_support_arr = getattr(sk_svc, "support_", None)
            my_support = int(my_support_arr.size) if my_support_arr is not None else 0
            sk_support = int(sk_support_arr.size) if sk_support_arr is not None else 0
            exp_helpers.record_result(kernel_results, "MySVC", my_metrics, my_support)
            exp_helpers.record_result(kernel_results, "sklearn SVC", sk_metrics, sk_support)

            print("[Stage] Metrics comparison report")
            compare_models(
                kernel_results,
                sort_by="f1",
                save_dir=kernel_results_dir,
                filename=exp_helpers.make_comparison_filename(variant_tag),
            )

            print(f"[Done] Results saved to: {kernel_results_dir}")

        # === END of single kernel tests


def normalize_kernel_names(
    kernels_input: list[str] | list[Kernel] | Kernel | str | None
) -> list[str]:
    allowed = {"linear", "rbf", "poly"}
    id_map = {"0": "linear", "1": "rbf", "2": "poly"}

    if kernels_input is None:
        return ["linear", "rbf", "poly"]

    if isinstance(kernels_input, Kernel):
        kernels_list = [kernels_input.kernel]
    elif isinstance(kernels_input, str):
        kernels_list = [kernels_input]
    else:
        kernels_list = list(kernels_input)

    names: list[str] = []
    for item in kernels_list:
        if isinstance(item, Kernel):
            raw_names = [item.kernel]
        else:
            raw_names = [str(item)]

        for raw in raw_names:
            parts = [p for p in raw.replace(",", " ").split() if p]
            for part in parts:
                key = part.strip().lower()
                if key in id_map:
                    key = id_map[key]
                if key not in allowed:
                    raise ValueError(f"Unknown kernel: {part}. Allowed: {sorted(allowed)}")
                if key not in names:
                    names.append(key)
    return names


def build_kernel_variants(kernel_name: str) -> list[dict]:
    if kernel_name == "poly":
        return [
            {"kernel": "poly", "degree": 2, "tol": 1e-6},
            {"kernel": "poly", "degree": 3, "tol": 1e-6},
        ]
    if kernel_name == "linear":
        return [
            # {"kernel": "linear", "C": 1.0,   "tol": 1e-6},
            {"kernel": "linear", "C": 100.0, "tol": 1e-6},
            # {"kernel": "linear", "C": 0.01,  "tol": 1e-6},
        ]
    return [{"kernel": kernel_name}]
