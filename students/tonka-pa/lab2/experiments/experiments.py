from pathlib import Path
from collections import defaultdict

from src.knn_model import KNearestNeighbors, Kernel, return_kernel
from src.utils import utils

from . import helpers as exp_helpers

#=========================================================================#

__all__ = [
    "run_tests"
]

K_MIN = 1
K_MAX = 50 # 100 for tricube?
CV_SPLITS = 30
LOO_BLOCK_SIZE = 256

#=========================================================================#

def run_tests(
    input_dir: str | Path,
    results_dir: str | Path,
    kernels: list[str] | list[Kernel] | Kernel | str | None = None,
):
    df_alzh = exp_helpers.read_alzheimer_dataset(input_dir)

    X, y = df_alzh.drop(columns=["Diagnosis"]), df_alzh["Diagnosis"]
    print(f"Dataset shape: {df_alzh.shape}")
    print(f"Number of classes: {df_alzh['Diagnosis'].nunique()}\n")
    print("Classes distribution: ")
    print(y.value_counts(normalize=True), "\n")

    int_cols = X.select_dtypes(include=["int"]).columns.tolist()
    float_cols = X.select_dtypes(include=["float"]).columns.tolist()

    print("Int cols: ", len(int_cols))
    print("Float cols: ", len(float_cols))

    bin_cols, cat_cols = exp_helpers.classify_cat_cols(X, int_cols)

    preprocessor = exp_helpers.build_preprocessor(int_cols, float_cols, bin_cols, cat_cols)
    X_prep = preprocessor.fit_transform(X)

    # ------------------------------------

    per_kernel_results = defaultdict(dict)
    kernels_to_test = exp_helpers.normalize_kernels(kernels)

    for kern_id in kernels_to_test:

        print("--- Current kernel", kern_id, "\n")

        kernel_fn = return_kernel(kern_id)
        knn_clf = KNearestNeighbors(kernel_fn)
        knn_clf.fit(X_prep, y)

        kernel_name = kernel_fn.__name__
        kernel_results_dir = Path(results_dir) / kernel_name

        kernel_results = per_kernel_results[kernel_name]

        opt_k_cv, _, _ = exp_helpers.run_default_case(
            knn_clf,
            X, y,
            preprocessor,
            kernel_name,
            kernel_results_dir,
            kernel_results,
            k_min=K_MIN,
            k_max=K_MAX,
            n_splits=CV_SPLITS,
            loo_block_size=LOO_BLOCK_SIZE,
        )

        opt_k_cv_enn, enn_proto_idx = exp_helpers.run_enn_case(
            knn_clf,
            X, y,
            preprocessor,
            kernel_name,
            kernel_results_dir,
            kernel_results,
            k_min=K_MIN,
            k_max=K_MAX,
            n_splits=CV_SPLITS,
        )

        opt_k_cv_oss, oss_proto_idx = exp_helpers.run_oss_case(
            knn_clf,
            X, y,
            preprocessor,
            kernel_name,
            kernel_results_dir,
            kernel_results,
            k_min=K_MIN,
            k_max=K_MAX,
            n_splits=CV_SPLITS,
        )

        opt_k_cv_stolp, stolp_proto_idx = exp_helpers.run_stolp_case(
            knn_clf,
            X, y,
            preprocessor,
            kernel_name,
            kernel_results_dir,
            kernel_results,
            k_min=K_MIN,
            k_max=K_MAX,
            n_splits=CV_SPLITS,
        )

        opt_k_cv_enn_oss, enn_oss_proto_idx = exp_helpers.run_enn_oss_case(
            knn_clf,
            X, y,
            preprocessor,
            kernel_name,
            kernel_results_dir,
            kernel_results,
            k_min=K_MIN,
            k_max=K_MAX,
            n_splits=CV_SPLITS,
        )

        exp_helpers.run_sklearn_case(
            X_prep,
            y,
            X,
            preprocessor,
            kernel_fn,
            kernel_name,
            kernel_results_dir,
            kernel_results,
            k_min=K_MIN,
            k_max=K_MAX,
        )

        # ---------- Decision boundaries visualization ----------

        exp_helpers.plot_decision_boundary(
            knn_clf,
            kernel_name,
            kernel_results_dir,
            k=opt_k_cv,
            proto_idx=None,
            proto_label="full-train",
            use_std=False,
            plot_confidence=False,
            plot_all_points=True,
        )
        exp_helpers.plot_decision_boundary(
            knn_clf,
            kernel_name,
            kernel_results_dir,
            k=opt_k_cv_enn,
            proto_idx=enn_proto_idx,
            proto_label="enn",
            use_std=False,
            plot_confidence=False,
            plot_all_points=False,
        )
        exp_helpers.plot_decision_boundary(
            knn_clf,
            kernel_name,
            kernel_results_dir,
            k=opt_k_cv_oss,
            proto_idx=oss_proto_idx,
            proto_label="oss",
            use_std=False,
            plot_confidence=False,
            plot_all_points=False,
        )
        exp_helpers.plot_decision_boundary(
            knn_clf,
            kernel_name,
            kernel_results_dir,
            k=opt_k_cv_stolp,
            proto_idx=stolp_proto_idx,
            proto_label="stolp",
            use_std=False,
            plot_confidence=False,
            plot_all_points=False,
        )
        exp_helpers.plot_decision_boundary(
            knn_clf,
            kernel_name,
            kernel_results_dir,
            k=opt_k_cv_enn_oss,
            proto_idx=enn_oss_proto_idx,
            proto_label="enn-oss",
            use_std=False,
            plot_confidence=False,
            plot_all_points=False,
        )

        # ---------- Metrics comparison ----------

        utils.compare_models(
            kernel_results,
            sort_by="f1",
            save_dir=kernel_results_dir,
            filename=exp_helpers.make_comparison_filename(kernel_name),
        )

        # === END of single kernel tests
