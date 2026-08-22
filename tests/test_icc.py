from __future__ import annotations

import unittest

import numpy as np

from src.icc import physician_random_intercept_icc


class PhysicianIccTests(unittest.TestCase):
    def test_random_intercept_icc_is_bounded_and_recovers_cluster_signal(self) -> None:
        physician = np.repeat(np.arange(12), 8)
        treatment = np.repeat(np.r_[np.zeros(6), np.ones(6)], 8)
        seniority = np.repeat(np.tile([0, 1], 6), 8)
        physician_effect = np.repeat(np.linspace(-2.5, 2.5, 12), 8)
        residual = np.tile(np.linspace(-0.15, 0.15, 8), 12)
        outcome = (
            4.0 + 1.2 * treatment - 0.4 * seniority
            + 0.3 * treatment * seniority + physician_effect + residual
        )
        result = physician_random_intercept_icc(
            outcome, treatment, seniority, physician
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["n"], 96)
        self.assertEqual(result["n_clusters"], 12)
        self.assertGreater(result["icc"], 0.80)
        self.assertLessEqual(result["icc"], 1.0)
        self.assertEqual(result["fixed_effects"], "treatment * seniority")
        self.assertEqual(result["scale"], "Gaussian continuous")


if __name__ == "__main__":
    unittest.main()
