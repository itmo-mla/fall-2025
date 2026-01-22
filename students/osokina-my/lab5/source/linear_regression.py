import numpy as np


def solve_system(A, b):
    regularized = A.T @ A + 1e-8 * np.eye(A.shape[1])
    try:
        U, singular, Vt = np.linalg.svd(
            regularized, full_matrices=False, compute_uv=True
        )
        max_singular = np.max(singular)
        threshold = max(max_singular * 1e-10, 1e-10)
        inv_singular = np.where(singular > threshold, 1.0 / singular, 0.0)
        return Vt.T @ np.diag(inv_singular) @ U.T @ A.T @ b
    except (np.linalg.LinAlgError, ValueError):
        print("Ошибка всё-таки есть!")
        return np.linalg.solve(regularized, A.T @ b)
