from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def physician_random_intercept_icc(
    outcome,
    treatment,
    seniority,
    physician_id,
    *,
    observed_binary_scale: bool = False,
) -> dict:
    frame = pd.DataFrame({
        "outcome": np.asarray(outcome, float),
        "treatment": np.asarray(treatment, float),
        "seniority": np.asarray(seniority, float),
        "physician_id": np.asarray(physician_id).astype(str),
    })
    if not np.isfinite(frame[["outcome", "treatment", "seniority"]].to_numpy()).all():
        raise ValueError("ICC analysis requires complete finite outcomes and predictors")
    if frame["physician_id"].eq("").any():
        raise ValueError("ICC analysis requires complete physician identifiers")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(
            "outcome ~ treatment * seniority",
            frame,
            groups=frame["physician_id"],
            re_formula="1",
        ).fit(reml=True, method="powell", maxiter=5000, disp=False)
    if not fit.converged:
        raise RuntimeError("Physician random-intercept model did not converge")

    physician_variance = max(float(fit.cov_re.iloc[0, 0]), 0.0)
    residual_variance = max(float(fit.scale), 0.0)
    total_variance = physician_variance + residual_variance
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("ICC variance components are invalid")

    cluster_sizes = frame.groupby("physician_id", sort=True).size().to_numpy(int)
    icc = physician_variance / total_variance
    return {
        "icc": float(icc),
        "physician_variance": physician_variance,
        "residual_variance": residual_variance,
        "n": int(len(frame)),
        "n_clusters": int(len(cluster_sizes)),
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_median": float(np.median(cluster_sizes)),
        "cluster_size_max": int(cluster_sizes.max()),
        "converged": True,
        "boundary_estimate": bool(icc < 1e-8),
        "estimation": "REML",
        "optimizer": "Powell",
        "fixed_effects": "treatment * seniority",
        "random_effect": "physician intercept",
        "scale": "Observed binary" if observed_binary_scale else "Gaussian continuous",
    }
