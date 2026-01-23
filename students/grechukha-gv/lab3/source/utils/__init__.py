"""Utilities for comparison, metrics and visualization"""

from .comparison import (
    evaluate_model,
    compare_with_sklearn,
    print_confusion_matrix,
    format_metrics_table,
    save_comparison_results,
    analyze_support_vectors
)
from .visualization import (
    visualize_svm_2d,
    visualize_linear_svm_exact,
    visualize_support_vectors,
    visualize_with_tsne,
    plot_kernel_comparison,
    plot_pca_data
)

__all__ = [
    'evaluate_model',
    'compare_with_sklearn',
    'print_confusion_matrix',
    'format_metrics_table',
    'save_comparison_results',
    'analyze_support_vectors',
    'visualize_svm_2d',
    'visualize_linear_svm_exact',
    'visualize_support_vectors',
    'visualize_with_tsne',
    'plot_kernel_comparison',
    'plot_pca_data'
]
