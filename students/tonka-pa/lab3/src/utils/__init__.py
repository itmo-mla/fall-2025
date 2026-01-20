from .preprocess import (check_for_alzheimers_dataset, read_alzheimer_dataset, classify_cat_cols,
                         build_preprocessor, score_model_cv, compare_models)

from .viz import (plot_svc_solution_lda_pca2, plot_svc_solution_umap2_fit)

__all__ = [
    "check_for_alzheimers_dataset",
    "read_alzheimer_dataset",
    "classify_cat_cols",
    "build_preprocessor",
    "score_model_cv",
    "compare_models",
    "plot_svc_solution_lda_pca2",
    "plot_svc_solution_umap2_fit",
]
