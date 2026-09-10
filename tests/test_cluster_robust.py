from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from src.cluster_robust import cr2_ols


class ClusterRobustTests(unittest.TestCase):
    def test_balanced_intercept_only_model(self) -> None:
        clusters = np.repeat(np.arange(8), 3)
        design = np.ones((24, 1))
        outcome = np.arange(24, dtype=float) % 5
        result = cr2_ols(outcome, design, clusters, [1])
        cluster_means = np.array([outcome[clusters == cluster].mean() for cluster in np.unique(clusters)])
        expected_se = cluster_means.std(ddof=1) / np.sqrt(len(cluster_means))
        critical = stats.t.ppf(0.975, 7)
        self.assertAlmostEqual(result["estimate"], float(outcome.mean()), places=12)
        self.assertAlmostEqual(result["se"], float(expected_se), places=12)
        self.assertAlmostEqual(result["df"], 7.0, places=12)
        self.assertAlmostEqual(result["ci_low"], result["estimate"] - critical * expected_se, places=12)
        self.assertAlmostEqual(result["ci_high"], result["estimate"] + critical * expected_se, places=12)
        self.assertEqual(result["software_implementation"], "Native Python (NumPy/SciPy; src/cluster_robust.py)")

    def test_balanced_cluster_level_treatment(self) -> None:
        clusters = np.repeat(np.arange(8), 3)
        treatment = np.repeat(np.r_[np.zeros(4), np.ones(4)], 3)
        design = np.column_stack([np.ones(24), treatment])
        outcome = (
            2 + 3 * treatment
            + np.repeat(np.array([-1, 0.2, 0.4, 0.4, -0.7, 0.1, 0.2, 0.4]), 3)
            + np.tile(np.array([-1, 0, 1]), 8)
        )
        result = cr2_ols(outcome, design, clusters, [0, 1])
        expected = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
        cluster_means = np.array([outcome[clusters == cluster].mean() for cluster in np.unique(clusters)])
        expected_se = np.sqrt(
            cluster_means[:4].var(ddof=1) / 4
            + cluster_means[4:].var(ddof=1) / 4
        )
        critical = stats.t.ppf(0.975, 6)
        self.assertAlmostEqual(result["estimate"], float(expected), places=12)
        self.assertAlmostEqual(result["se"], float(expected_se), places=12)
        self.assertAlmostEqual(result["df"], 6.0, places=12)
        self.assertAlmostEqual(result["ci_low"], result["estimate"] - critical * expected_se, places=12)
        self.assertAlmostEqual(result["ci_high"], result["estimate"] + critical * expected_se, places=12)

    def test_unbalanced_multivariable_model_matches_direct_cr2_algebra(self) -> None:
        cluster_sizes = np.array([2, 3, 4, 2, 5, 3, 4, 2, 3])
        clusters = np.repeat(np.arange(cluster_sizes.size), cluster_sizes)
        treatment = (np.arange(clusters.size) % 2).astype(float)
        seniority = (clusters >= 5).astype(float)
        covariate = np.linspace(-1.5, 1.8, clusters.size)
        design = np.column_stack([np.ones(clusters.size), treatment, seniority, covariate])
        outcome = (
            1.7 + 2.2 * treatment - 0.8 * seniority + 0.5 * covariate
            + np.repeat(np.linspace(-0.6, 0.7, cluster_sizes.size), cluster_sizes)
            + 0.15 * np.sin(np.arange(clusters.size))
        )
        contrast = np.array([0.0, 1.0, 0.0, 0.0])
        result = cr2_ols(outcome, design, clusters, contrast)

        bread = np.linalg.inv(design.T @ design)
        beta = bread @ design.T @ outcome
        residual = outcome - design @ beta
        residual_maker = np.eye(design.shape[0]) - design @ bread @ design.T
        covariance = np.zeros((design.shape[1], design.shape[1]))
        p_vectors = []
        for cluster in np.unique(clusters):
            rows = np.flatnonzero(clusters == cluster)
            x_i = design[rows]
            complement = np.eye(rows.size) - x_i @ bread @ x_i.T
            eigenvalues, eigenvectors = np.linalg.eigh((complement + complement.T) / 2)
            powered = np.zeros_like(eigenvalues)
            powered[eigenvalues > 1e-12] = eigenvalues[eigenvalues > 1e-12] ** -0.5
            inverse_sqrt = (eigenvectors * powered) @ eigenvectors.T
            adjusted_score = x_i.T @ inverse_sqrt @ residual[rows]
            covariance += bread @ np.outer(adjusted_score, adjusted_score) @ bread
            selector = np.zeros((design.shape[0], rows.size))
            selector[rows, np.arange(rows.size)] = 1.0
            p_vectors.append(
                residual_maker @ selector @ inverse_sqrt @ x_i @ bread @ contrast
            )
        expected_se = np.sqrt(contrast @ covariance @ contrast)
        inner_products = np.array(
            [[left @ right for right in p_vectors] for left in p_vectors]
        )
        expected_df = np.trace(inner_products) ** 2 / np.sum(inner_products**2)

        self.assertAlmostEqual(result["estimate"], float(contrast @ beta), places=12)
        self.assertAlmostEqual(result["se"], float(expected_se), places=12)
        self.assertAlmostEqual(result["df"], float(expected_df), places=11)
        self.assertEqual(result["n_clusters"], cluster_sizes.size)


if __name__ == "__main__":
    unittest.main()
