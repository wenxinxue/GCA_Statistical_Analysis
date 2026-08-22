from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    matrix = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    tolerance = np.finfo(float).eps * max(matrix.shape) * max(float(np.max(np.abs(values))), 1.0)
    if float(np.min(values)) < -100 * tolerance:
        raise np.linalg.LinAlgError("CR2 adjustment matrix is not positive semidefinite")
    powered = np.zeros_like(values)
    positive = values > tolerance
    powered[positive] = values[positive] ** power
    return (vectors * powered) @ vectors.T


def cr2_ols(
    outcome,
    design,
    clusters,
    contrast,
) -> dict:
    """OLS contrast with physician-clustered CR2/Satterthwaite inference.

    The implementation is for unweighted least squares with an identity working
    covariance model. The Satterthwaite degrees of freedom are specific to the
    supplied one-dimensional contrast.
    """
    y = np.asarray(outcome, float).reshape(-1)
    x = np.asarray(design, float)
    c = np.asarray(contrast, float).reshape(-1)
    cluster = np.asarray(clusters).astype(str).reshape(-1)

    if x.ndim != 2 or len(y) != x.shape[0] or len(cluster) != len(y):
        raise ValueError("Outcome, design matrix and clusters must have matching rows")
    if len(c) != x.shape[1]:
        raise ValueError("Contrast length must match the number of model coefficients")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("CR2 analysis requires complete finite outcomes and predictors")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise np.linalg.LinAlgError("CR2 design matrix is not full rank")

    cluster_levels, cluster_index = np.unique(cluster, return_inverse=True)
    if len(cluster_levels) <= x.shape[1]:
        raise ValueError("The number of clusters must exceed the number of coefficients")

    bread = np.linalg.inv(x.T @ x)
    bread_cholesky = np.linalg.cholesky((bread + bread.T) / 2)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    cluster_energy = []
    h_rows = []
    cluster_sizes = []

    for index in range(len(cluster_levels)):
        rows = cluster_index == index
        x_i = x[rows]
        e_i = residual[rows]
        cluster_sizes.append(int(rows.sum()))
        leverage_complement = np.eye(rows.sum()) - x_i @ bread @ x_i.T
        adjustment = _symmetric_matrix_power(leverage_complement, -0.5)
        adjusted_score = x_i.T @ adjustment @ e_i
        meat += np.outer(adjusted_score, adjusted_score)

        g_i = c @ bread @ x_i.T @ adjustment
        h_i = g_i @ x_i @ bread_cholesky
        cluster_energy.append(float(g_i @ g_i.T))
        h_rows.append(np.asarray(h_i, float))

    covariance = bread @ meat @ bread
    variance = float(c @ covariance @ c)
    if variance < -1e-10:
        raise np.linalg.LinAlgError("CR2 contrast variance is negative")
    standard_error = math.sqrt(max(variance, 0.0))
    if standard_error == 0:
        raise ValueError("CR2 contrast standard error is zero")

    h_matrix = np.vstack(h_rows)
    p_matrix = np.diag(cluster_energy) - h_matrix @ h_matrix.T
    expected_variance = float(np.trace(p_matrix))
    variance_of_variance = float(np.sum(p_matrix**2))
    degrees_of_freedom = expected_variance**2 / variance_of_variance
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError("Invalid Satterthwaite degrees of freedom")

    estimate = float(c @ beta)
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    t_statistic = estimate / standard_error
    return {
        "estimate": estimate,
        "se": standard_error,
        "df": float(degrees_of_freedom),
        "ci_low": float(estimate - critical * standard_error),
        "ci_high": float(estimate + critical * standard_error),
        "p": float(2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)),
        "n": int(len(y)),
        "n_clusters": int(len(cluster_levels)),
        "cluster_size_min": int(min(cluster_sizes)),
        "cluster_size_median": float(np.median(cluster_sizes)),
        "cluster_size_max": int(max(cluster_sizes)),
        "variance_estimator": "CR2",
        "working_covariance": "Identity",
        "reference_distribution": "Satterthwaite t",
        "software_implementation": "Native Python (NumPy/SciPy; src/cluster_robust.py)",
    }
