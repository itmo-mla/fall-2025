import numpy as np


def solve_linear_system(A, b):
    A_reg = A.T @ A + 1e-8 * np.eye(A.shape[1])
    try:
        U, s, Vt = np.linalg.svd(A_reg, full_matrices=False, compute_uv=True)
        s_max = np.max(s)
        threshold = max(s_max * 1e-10, 1e-10)
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        return Vt.T @ np.diag(s_inv) @ U.T @ A.T @ b
    except (np.linalg.LinAlgError, ValueError):
        return np.linalg.solve(A_reg, A.T @ b)


def matrix_inverse(A):
    A_reg = A + 1e-8 * np.eye(A.shape[0])
    try:
        U, s, Vt = np.linalg.svd(A_reg, full_matrices=False, compute_uv=True)
        s_max = np.max(s)
        threshold = max(s_max * 1e-10, 1e-10)
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        return Vt.T @ np.diag(s_inv) @ U.T
    except (np.linalg.LinAlgError, ValueError):
        return np.linalg.inv(A_reg)
