from __future__ import annotations

import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from .cluster_robust import cr2_ols
from .icc import physician_random_intercept_icc


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Source Data"
RESULTS = ROOT / "Analysis_Results"
SEED = 20260722
BOOTSTRAP_REPLICATES = 5000
PDQI_RCT = [
    "Up-to-date", "Accurate", "Thorough", "Useful", "Organized",
    "Comprehensible", "Succinct", "Synthesized", "Internally Consistent",
]
PDQI_EXTERNAL = [
    "Up-to-date", "Accurate", "Thorough", "Useful", "Organized",
    "Comprehensible", "Succinct", "Synthesized", "Internally consistent",
]
PATIENT_ITEMS = [
    "Communication Convenience", "Perceived Physician Attention",
    "Perceived Empathy", "Overall Satisfaction",
]
CENTRES = [
    ("Centre 1", "HM.xlsx"),
    ("Centre 2", "JJ.xlsx"),
    ("Centre 3", "JX.xlsx"),
    ("Centre 4", "WHRM.xlsx"),
    ("Centre 5", "WHZX.xlsx"),
]

ERROR_SEVERITY_COLUMN = "AI Draft Clinical Error Severity Level (0/1/2)"
ERROR_SEVERITY_LABELS = {
    0: "Level 0: no medically substantive error",
    1: "Level 1: non-serious medically substantive error",
    2: "Level 2: potentially serious clinical error",
}


def validated_error_severity(frame: pd.DataFrame, context: str) -> pd.Series:
    if ERROR_SEVERITY_COLUMN not in frame.columns:
        raise KeyError(f"Missing {ERROR_SEVERITY_COLUMN!r} in {context}")
    values = pd.to_numeric(frame[ERROR_SEVERITY_COLUMN], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Clinical error severity contains missing or non-numeric values in {context}")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"Clinical error severity must contain integer levels in {context}")
    values = values.astype(int)
    invalid = sorted(set(values.unique()) - set(ERROR_SEVERITY_LABELS))
    if invalid:
        raise ValueError(f"Clinical error severity contains invalid levels in {context}: {invalid}")
    return values


def clinical_error_severity_summary(values) -> dict:
    series = pd.Series(values, dtype="int64")
    nobs = int(len(series))
    levels = {}
    for level, label in ERROR_SEVERITY_LABELS.items():
        count = int((series == level).sum())
        levels[str(level)] = {
            "label": label,
            "count": count,
            "proportion": count / nobs,
        }
    if sum(item["count"] for item in levels.values()) != nobs:
        raise AssertionError("Clinical error severity counts do not sum to the number of drafts")
    return {"n": nobs, "levels": levels}


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def bh(p_values):
    values = np.asarray(p_values, float)
    if not np.isfinite(values).all():
        raise ValueError("BH adjustment requires a complete family of finite P values")
    return [float(x) for x in multipletests(values, method="fdr_bh")[1]]


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def clopper_pearson(count, nobs):
    low, high = proportion_confint(int(count), int(nobs), alpha=0.05, method="beta")
    return [float(low), float(high)]


def describe(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return {
        "n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
        "median": float(np.median(x)), "q1": float(np.quantile(x, 0.25)),
        "q3": float(np.quantile(x, 0.75)), "minimum": float(x.min()),
        "maximum": float(x.max()),
    }


def hedges_g(gca, scw):
    """Small-sample-corrected pooled-SD standardised mean difference.

    The direction is GCA minus SCW. The 95% CI uses
    g +/- z(0.975) * sqrt((n_gca+n_scw)/(n_gca*n_scw)
    + g^2/(2*(n_gca+n_scw-2))).
    """
    a = pd.to_numeric(pd.Series(gca), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(scw), errors="coerce").dropna().to_numpy(float)
    df = a.size + b.size - 2
    pooled_sd = math.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / df
    )
    correction = 1 - 3 / (4 * df - 1)
    cohen_d = (a.mean() - b.mean()) / pooled_sd
    estimate = correction * cohen_d
    se = math.sqrt(
        (a.size + b.size) / (a.size * b.size) + estimate**2 / (2 * df)
    )
    critical = stats.norm.ppf(0.975)
    return {
        "n_gca": int(a.size),
        "n_scw": int(b.size),
        "mean_gca": float(a.mean()),
        "mean_scw": float(b.mean()),
        "pooled_sd": float(pooled_sd),
        "small_sample_correction": float(correction),
        "estimate": float(estimate),
        "se": float(se),
        "ci_low": float(estimate - critical * se),
        "ci_high": float(estimate + critical * se),
    }


def mean_inference(values, use_t=True):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    estimate = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    if use_t:
        critical = stats.t.ppf(0.975, x.size - 1)
        p = 2 * stats.t.sf(abs(estimate / se), x.size - 1)
    else:
        critical = stats.norm.ppf(0.975)
        p = 2 * stats.norm.sf(abs(estimate / se))
    return {
        "n": int(x.size), "estimate": estimate, "se": se,
        "ci_low": float(estimate - critical * se),
        "ci_high": float(estimate + critical * se), "p": float(p),
    }


def welch(gca, scw):
    a = pd.to_numeric(pd.Series(gca), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(scw), errors="coerce").dropna().to_numpy(float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / a.size + vb / b.size)
    df = (va / a.size + vb / b.size) ** 2 / (
        (va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1)
    )
    estimate = float(a.mean() - b.mean())
    critical = stats.t.ppf(0.975, df)
    return {
        "gca": describe(a), "scw": describe(b), "estimate": estimate,
        "se": float(se), "df": float(df),
        "ci_low": float(estimate - critical * se),
        "ci_high": float(estimate + critical * se),
        "p": float(2 * stats.t.sf(abs(estimate / se), df)),
        "relative_change": float(estimate / b.mean()),
    }


def paired(gca, scw):
    frame = pd.DataFrame({"gca": gca, "scw": scw}).dropna()
    diff = frame["gca"].to_numpy(float) - frame["scw"].to_numpy(float)
    out = mean_inference(diff, use_t=True)
    out.update({
        "gca": describe(frame["gca"]), "scw": describe(frame["scw"]),
        "relative_change": float(diff.mean() / frame["scw"].mean()),
        "wilcoxon_p": float(stats.wilcoxon(diff, alternative="two-sided").pvalue),
    })
    return out


def linear_contrast(fit, vector, use_t):
    vector = np.asarray(vector, float)
    estimate = float(vector @ fit.params.to_numpy())
    se = math.sqrt(max(float(vector @ fit.cov_params().to_numpy() @ vector), 0.0))
    if use_t:
        critical = stats.t.ppf(0.975, fit.df_resid)
        p = 2 * stats.t.sf(abs(estimate / se), fit.df_resid)
    else:
        critical = stats.norm.ppf(0.975)
        p = 2 * stats.norm.sf(abs(estimate / se))
    return {
        "estimate": estimate, "se": se,
        "ci_low": float(estimate - critical * se),
        "ci_high": float(estimate + critical * se), "p": float(p),
    }


def continuous_interaction(data, outcome):
    fit = smf.ols(f"Q({outcome!r}) ~ treatment * senior", data=data).fit(
        cov_type="HC3", use_t=True
    )
    names = list(fit.params.index)
    specifications = {
        "junior": {"treatment": 1},
        "senior": {"treatment": 1, "treatment:senior": 1},
        "interaction": {"treatment:senior": 1},
    }
    results = {}
    for label, terms in specifications.items():
        vector = np.zeros(len(names))
        for term, value in terms.items():
            vector[names.index(term)] = value
        results[label] = linear_contrast(fit, vector, use_t=True)
    return results


def standardised_logistic_rct(data, outcome):
    fit = smf.glm(
        f"Q({outcome!r}) ~ treatment * senior", data=data,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3", use_t=True)
    names = list(fit.params.index)
    beta, covariance = fit.params.to_numpy(), fit.cov_params().to_numpy()

    def design(treatment, senior):
        values = {
            "Intercept": 1.0, "treatment": treatment, "senior": senior,
            "treatment:senior": treatment * senior,
        }
        return np.asarray([values[name] for name in names])

    def probability_and_gradient(vector):
        probability = scipy.special.expit(float(vector @ beta))
        return probability, probability * (1 - probability) * vector

    j1, gj1 = probability_and_gradient(design(1, 0))
    j0, gj0 = probability_and_gradient(design(0, 0))
    s1, gs1 = probability_and_gradient(design(1, 1))
    s0, gs0 = probability_and_gradient(design(0, 1))
    estimands = {
        "junior": (j1 - j0, gj1 - gj0),
        "senior": (s1 - s0, gs1 - gs0),
        "interaction": ((s1 - s0) - (j1 - j0), (gs1 - gs0) - (gj1 - gj0)),
    }
    critical = stats.t.ppf(0.975, fit.df_resid)
    output = {}
    for label, (estimate, gradient) in estimands.items():
        se = math.sqrt(float(gradient @ covariance @ gradient))
        output[label] = {
            "estimate": float(estimate), "se": se,
            "ci_low": float(estimate - critical * se),
            "ci_high": float(estimate + critical * se),
            "p": float(2 * stats.t.sf(abs(estimate / se), fit.df_resid)),
            "df": float(fit.df_resid),
        }
    return output


def load_rct():
    frames = {}
    for treatment_name in ("GCA", "SCW"):
        frame = pd.read_excel(SOURCE / "RCT.xlsx", sheet_name=treatment_name)
        frame["treatment"] = int(treatment_name == "GCA")
        frame["physician_id"] = frame["Physician ID"].astype("string").str.strip()
        frame["senior"] = frame["Physician Seniority"].astype(str).str.contains("Senior").astype(int)
        frame["female"] = frame["Patient Sex"].astype(str).str.lower().eq("female").astype(int)
        frame["pdqi_total"] = frame[PDQI_RCT].sum(axis=1)
        frame["patient_experience"] = frame[PATIENT_ITEMS].sum(axis=1)
        frame["final_record_length"] = (
            frame["Total Record Word Count After Editing"]
            if treatment_name == "GCA" else frame["Total Record Word Count"]
        )
        if treatment_name == "GCA":
            frame["clinical_error_severity"] = validated_error_severity(
                frame, "RCT GCA sheet"
            )
        frames[treatment_name] = frame
    combined = pd.concat([frames["GCA"], frames["SCW"]], ignore_index=True)
    return frames["GCA"], frames["SCW"], combined


def rct_physician_cluster_sensitivity(data, continuous, diagnosis_column, patient_reported):
    if data["physician_id"].isna().any() or data["physician_id"].eq("").any():
        raise ValueError("RCT physician identifiers must be complete")
    physician = data.groupby("physician_id", sort=True).agg(
        n=("ID", "size"),
        n_treatments=("treatment", "nunique"),
        n_seniority=("senior", "nunique"),
        treatment=("treatment", "first"),
        senior=("senior", "first"),
    )
    if int(physician["n_seniority"].max()) != 1:
        raise ValueError("Physician seniority must be constant within RCT physician")

    treatment = data["treatment"].to_numpy(float)
    senior = data["senior"].to_numpy(float)
    clusters = data["physician_id"].astype(str).to_numpy()
    overall_design = np.column_stack([np.ones(len(data)), treatment])
    interaction_design = np.column_stack([
        np.ones(len(data)), treatment, senior, treatment * senior,
    ])
    outcomes = dict(continuous)
    outcomes["Accurate preliminary diagnosis"] = diagnosis_column
    overall = {
        label: cr2_ols(data[column], overall_design, clusters, [0, 1])
        for label, column in outcomes.items()
    }
    interaction = {
        label: cr2_ols(data[column], interaction_design, clusters, [0, 0, 0, 1])
        for label, column in outcomes.items()
    }
    patient_reported_results = {
        label: cr2_ols(data[column], overall_design, clusters, [0, 1])
        for label, column in patient_reported.items()
    }
    cell_counts = {
        f"{'GCA' if treatment_value else 'SCW'}_{'senior' if senior_value else 'junior'}": int(count)
        for (treatment_value, senior_value), count in physician.groupby(
            ["treatment", "senior"]
        ).size().items()
    }
    return {
        "overall": overall,
        "seniority_interaction": interaction,
        "patient_reported": patient_reported_results,
        "cluster_diagnostics": {
            "n_physicians": int(len(physician)),
            "n_junior_physicians": int((physician["senior"] == 0).sum()),
            "n_senior_physicians": int((physician["senior"] == 1).sum()),
            "physicians_in_both_treatment_groups": int((physician["n_treatments"] > 1).sum()),
            "cluster_size_min": int(physician["n"].min()),
            "cluster_size_median": float(physician["n"].median()),
            "cluster_size_max": int(physician["n"].max()),
            "treatment_by_seniority_physician_counts": cell_counts,
        },
        "overall_model": "outcome ~ treatment",
        "interaction_model": "outcome ~ treatment * seniority",
        "effect_direction": "GCA minus SCW",
        "interaction_direction": "senior minus junior difference in treatment effects",
    }


def rct_physician_icc(data, continuous, diagnosis_column):
    treatment = data["treatment"].to_numpy(float)
    seniority = data["senior"].to_numpy(float)
    physician_id = data["physician_id"].astype(str).to_numpy()
    continuous_outcomes = dict(continuous)
    continuous_outcomes["Patient experience score"] = "patient_experience"
    continuous_results = {
        label: physician_random_intercept_icc(
            data[column], treatment, seniority, physician_id
        )
        for label, column in continuous_outcomes.items()
    }
    binary_result = physician_random_intercept_icc(
        data[diagnosis_column], treatment, seniority, physician_id,
        observed_binary_scale=True,
    )
    return {
        "continuous": continuous_results,
        "binary": {"Accurate preliminary diagnosis": binary_result},
        "model": "outcome ~ treatment * seniority + (1 | physician)",
    }


def ordinal_logistic_rct(data, outcomes):
    output = {}
    exog = data[["treatment", "senior"]].copy()
    exog["treatment_x_senior"] = exog["treatment"] * exog["senior"]
    for outcome in outcomes:
        try:
            model = OrderedModel(data[outcome].astype(int), exog, distr="logit")
            fit = model.fit(method="bfgs", disp=False, maxiter=1000)
            term = "treatment_x_senior"
            estimate = float(fit.params[term])
            ci = fit.conf_int().loc[term]
            output[outcome] = {
                "converged": bool(fit.mle_retvals.get("converged", False)),
                "interaction_or": float(np.exp(estimate)),
                "ci_low": float(np.exp(ci.iloc[0])),
                "ci_high": float(np.exp(ci.iloc[1])),
                "p": float(fit.pvalues[term]),
            }
            output[outcome]["valid"] = bool(
                output[outcome]["converged"] and all_finite(output[outcome])
            )
        except Exception as exc:
            output[outcome] = {"converged": False, "valid": False, "error": str(exc)}
    return output


def rct_implementation_outcomes(gca):
    definitions = [
        ("Useful top-5 diagnostic recommendations", "Usefulness of Recommended Top 5 Diagnoses", "pearson_chi2"),
        ("Useful test recommendations", "Usefulness of Recommended Top 5 Tests", "pearson_chi2"),
        ("Useful graph-based prompts", "Useful Graph-based Prompts", "fisher_exact"),
        ("Perceived workload reduction", "Reduced Workload", "pearson_chi2"),
        ("Reported workflow interference", "Interfered with Normal Workflow", "pearson_chi2"),
        ("Willingness to use AI assistance again", "Willingness to Use AI Assistance Again", "pearson_chi2"),
    ]
    output = {}
    p_values = []
    for label, column, test_name in definitions:
        junior = gca.loc[gca["senior"] == 0, column].astype(int)
        senior = gca.loc[gca["senior"] == 1, column].astype(int)
        count_j, count_s = int(junior.sum()), int(senior.sum())
        n_j, n_s = len(junior), len(senior)
        difference = count_s / n_s - count_j / n_j
        ci_low, ci_high = confint_proportions_2indep(
            count_s, n_s, count_j, n_j, method="newcomb", compare="diff"
        )
        table = np.asarray([[count_s, n_s - count_s], [count_j, n_j - count_j]])
        if test_name == "fisher_exact":
            p_value = float(stats.fisher_exact(table, alternative="two-sided").pvalue)
        else:
            p_value = float(stats.chi2_contingency(table, correction=False).pvalue)
        p_values.append(p_value)
        output[label] = {
            "column": column,
            "junior_events": count_j,
            "junior_n": n_j,
            "senior_events": count_s,
            "senior_n": n_s,
            "risk_difference": float(difference),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p": p_value,
            "test": test_name,
        }
    for result, q_value in zip(output.values(), bh(p_values)):
        result["q"] = q_value
    return output


def rct_gca_overall_proportions(gca):
    columns = {
        "Drafts containing at least 1 factually conflicting character": (
            (gca["Factually Conflicting Characters"] > 0).astype(int)
        ),
        "Useful top-5 diagnostic recommendations": gca["Usefulness of Recommended Top 5 Diagnoses"],
        "Useful test recommendations": gca["Usefulness of Recommended Top 5 Tests"],
        "Useful graph-based prompts": gca["Useful Graph-based Prompts"],
        "Perceived workload reduction": gca["Reduced Workload"],
        "Reported workflow interference": gca["Interfered with Normal Workflow"],
        "Willingness to use AI assistance again": gca["Willingness to Use AI Assistance Again"],
    }
    output = {}
    for label, values in columns.items():
        numeric = pd.to_numeric(values, errors="raise").astype(int)
        count, nobs = int(numeric.sum()), int(len(numeric))
        output[label] = {
            "events": count,
            "n": nobs,
            "proportion": float(count / nobs),
            "exact_ci": clopper_pearson(count, nobs),
        }
    return output


def rct_bootstrap_subgroups(data, outcomes):
    rng = np.random.default_rng(SEED)
    strata = {}
    for treatment in (0, 1):
        for senior in (0, 1):
            subset = data[(data["treatment"] == treatment) & (data["senior"] == senior)]
            strata[(treatment, senior)] = (
                subset,
                rng.integers(0, len(subset), size=(BOOTSTRAP_REPLICATES, len(subset))),
            )
    output = {}
    for outcome in outcomes:
        means = {}
        for key, (subset, indices) in strata.items():
            values = subset[outcome].to_numpy(float)
            means[key] = values[indices].mean(axis=1)
        junior = means[(1, 0)] - means[(0, 0)]
        senior = means[(1, 1)] - means[(0, 1)]
        interaction = senior - junior
        output[outcome] = {
            "junior_ci": np.quantile(junior, [0.025, 0.975]).tolist(),
            "senior_ci": np.quantile(senior, [0.025, 0.975]).tolist(),
            "interaction_ci": np.quantile(interaction, [0.025, 0.975]).tolist(),
        }
    return output


def rct_analysis():
    gca, scw, data = load_rct()
    continuous = {
        "Total admission workflow time": "Total Time",
        "Interview time": "Interview Time",
        "Physician follow-up work time": "Physician Follow-up Work Time",
        "Text editing workload": "Text Editing Workload",
        "PDQI-9 total score": "pdqi_total",
    }
    core = {label: welch(gca[column], scw[column]) for label, column in continuous.items()}
    standardised_effects = {
        label: hedges_g(gca[column], scw[column]) for label, column in continuous.items()
    }

    event_column = "Accuracy of the initial diagnosis"
    events_g, events_s = int(gca[event_column].sum()), int(scw[event_column].sum())
    table = np.asarray([[events_g, len(gca) - events_g], [events_s, len(scw) - events_s]])
    chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
    pg, ps = events_g / len(gca), events_s / len(scw)
    rd, rd_se = pg - ps, math.sqrt(pg * (1 - pg) / len(gca) + ps * (1 - ps) / len(scw))
    diagnosis = {
        "gca": f"{events_g}/{len(gca)}", "scw": f"{events_s}/{len(scw)}",
        "risk_difference": rd, "ci_low": rd - 1.96 * rd_se,
        "ci_high": rd + 1.96 * rd_se, "pearson_chi2": float(chi2), "p": float(p_chi),
    }
    core_secondary_p = [
        core["Interview time"]["p"], core["Physician follow-up work time"]["p"],
        core["Text editing workload"]["p"], core["PDQI-9 total score"]["p"], p_chi,
    ]

    patient_outcomes = ["patient_experience"] + PATIENT_ITEMS
    patient_reported_columns = {
        "Patient experience score": "patient_experience",
        "Communication convenience": "Communication Convenience",
        "Perceived physician attention": "Perceived Physician Attention",
        "Perceived empathy": "Perceived Empathy",
        "Overall satisfaction": "Overall Satisfaction",
    }
    patient = {outcome: welch(gca[outcome], scw[outcome]) for outcome in patient_outcomes}
    pdqi = {outcome: welch(gca[outcome], scw[outcome]) for outcome in PDQI_RCT}
    interview = {
        outcome: welch(gca[outcome], scw[outcome])
        for outcome in ["Interview Fluency", "Interview Completeness"]
    }
    exploratory = {
        "Patient Attention Time": welch(gca["Patient Attention Time"], scw["Patient Attention Time"]),
        "Final record length": welch(gca["final_record_length"], scw["final_record_length"]),
    }

    correlations = {}
    for label, frame in [("overall", data), ("SCW", scw), ("GCA", gca)]:
        rho, p = stats.spearmanr(frame["Patient Attention Time"], frame["patient_experience"])
        correlations[label] = {"rho": float(rho), "p": float(p)}
    attention = smf.ols(
        "patient_experience ~ attention_per_60 + treatment + senior + Q('Patient Age') + female",
        data=data.assign(attention_per_60=data["Patient Attention Time"] / 60),
    ).fit(cov_type="HC3", use_t=False)
    attention_ci = attention.conf_int().loc["attention_per_60"]
    attention_result = {
        "estimate_per_60_s": float(attention.params["attention_per_60"]),
        "ci_low": float(attention_ci.iloc[0]), "ci_high": float(attention_ci.iloc[1]),
        "p": float(attention.pvalues["attention_per_60"]),
    }

    subgroup_continuous = {
        label: continuous_interaction(data, column) for label, column in continuous.items()
    }
    subgroup_diagnosis = standardised_logistic_rct(data, event_column)
    subgroup_core_p = [
        subgroup_continuous[label]["interaction"]["p"]
        for label in ["Interview time", "Physician follow-up work time", "Text editing workload", "PDQI-9 total score"]
    ] + [subgroup_diagnosis["interaction"]["p"]]
    subgroup_pdqi = {outcome: continuous_interaction(data, outcome) for outcome in PDQI_RCT}
    subgroup_interview = {
        outcome: continuous_interaction(data, outcome)
        for outcome in ["Interview Fluency", "Interview Completeness"]
    }
    subgroup_patient = {outcome: continuous_interaction(data, outcome) for outcome in patient_outcomes}

    ordinal_outcomes = PDQI_RCT + ["Interview Fluency", "Interview Completeness"] + PATIENT_ITEMS
    ordinal = ordinal_logistic_rct(data, ordinal_outcomes)
    ordinal_families = {
        "PDQI-9 domains": PDQI_RCT,
        "Interview quality": ["Interview Fluency", "Interview Completeness"],
        "Patient-reported outcomes": PATIENT_ITEMS,
    }
    for family, outcomes in ordinal_families.items():
        family_q = bh([ordinal[outcome]["p"] for outcome in outcomes])
        for outcome, q_value in zip(outcomes, family_q):
            ordinal[outcome]["family"] = family
            ordinal[outcome]["q"] = q_value
    bootstrap_outcomes = list(continuous.values()) + [event_column] + PDQI_RCT + [
        "Interview Fluency", "Interview Completeness", "patient_experience",
    ] + PATIENT_ITEMS

    validation = {
        "n_gca": int(len(gca)), "n_scw": int(len(scw)),
        "no_duplicate_gca_id": bool(not gca["ID"].duplicated().any()),
        "no_duplicate_scw_id": bool(not scw["ID"].duplicated().any()),
        "no_missing_gca_fields": bool(not gca.isna().any().any()),
        "no_missing_scw_fields": bool(not scw.isna().any().any()),
        "total_time_identity_gca": bool(np.allclose(gca["Total Time"], gca["Interview Time"] + gca["Physician Follow-up Work Time"])),
        "total_time_identity_scw": bool(np.allclose(scw["Total Time"], scw["Interview Time"] + scw["Physician Follow-up Work Time"])),
        "editing_identity_gca": bool(np.allclose(gca["Text Editing Workload"], gca["Deleted Characters"] + gca["Added Characters"])),
        "editing_identity_scw": bool(np.allclose(scw["Text Editing Workload"], scw["Total Record Word Count"])),
        "clinical_error_severity_complete": bool(not gca["clinical_error_severity"].isna().any()),
        "clinical_error_severity_levels_valid": bool(
            set(gca["clinical_error_severity"].unique()).issubset(ERROR_SEVERITY_LABELS)
        ),
    }
    physician_cluster_sensitivity = rct_physician_cluster_sensitivity(
        data, continuous, event_column, patient_reported_columns
    )
    physician_icc = rct_physician_icc(data, continuous, event_column)
    return {
        "validation": validation, "core": core, "hedges_g": standardised_effects,
        "diagnosis": diagnosis,
        "core_secondary_q": bh(core_secondary_p),
        "patient_outcomes": patient, "patient_outcome_q": bh([patient[x]["p"] for x in patient_outcomes]),
        "pdqi_domains": pdqi, "pdqi_domain_q": bh([pdqi[x]["p"] for x in PDQI_RCT]),
        "interview_quality": interview, "interview_quality_q": bh([interview[x]["p"] for x in interview]),
        "other_exploratory": exploratory, "other_exploratory_q": bh([exploratory[x]["p"] for x in exploratory]),
        "attention_correlations": correlations, "attention_hc3_regression": attention_result,
        "subgroup_continuous": subgroup_continuous,
        "subgroup_diagnostic_accuracy": subgroup_diagnosis,
        "subgroup_core_q": bh(subgroup_core_p),
        "subgroup_pdqi": subgroup_pdqi,
        "subgroup_pdqi_q": bh([subgroup_pdqi[x]["interaction"]["p"] for x in PDQI_RCT]),
        "subgroup_interview": subgroup_interview,
        "subgroup_interview_q": bh([subgroup_interview[x]["interaction"]["p"] for x in subgroup_interview]),
        "subgroup_patient": subgroup_patient,
        "subgroup_patient_q": bh([subgroup_patient[x]["interaction"]["p"] for x in subgroup_patient]),
        "subgroup_bootstrap": rct_bootstrap_subgroups(data, bootstrap_outcomes),
        "ordinal_logistic_sensitivity": ordinal,
        "ordinal_sensitivity_families": {
            family: {"n_tests": len(outcomes), "outcomes": outcomes}
            for family, outcomes in ordinal_families.items()
        },
        "gca_implementation_by_seniority": rct_implementation_outcomes(gca),
        "gca_overall_exact_proportions": rct_gca_overall_proportions(gca),
        "physician_cluster_sensitivity": physician_cluster_sensitivity,
        "physician_icc": physician_icc,
        "gca_draft_safety": {
            "clinical_error_severity": clinical_error_severity_summary(
                gca["clinical_error_severity"]
            ),
            "conflicting_characters": describe(gca["Factually Conflicting Characters"]),
            "conflicting_characters_mean_inference": mean_inference(gca["Factually Conflicting Characters"]),
            "drafts_with_any_detected_conflict": int((gca["Factually Conflicting Characters"] > 0).sum()),
            "conflicts_per_1000_final_characters": describe(
                1000 * gca["Factually Conflicting Characters"] / gca["final_record_length"]
            ),
            "conflicts_per_1000_mean_inference": mean_inference(
                1000 * gca["Factually Conflicting Characters"] / gca["final_record_length"]
            ),
        },
    }


def load_external():
    frames, validation_rows = [], []
    for centre, filename in CENTRES:
        gca = pd.read_excel(SOURCE / filename, sheet_name="GCA")
        scw = pd.read_excel(SOURCE / filename, sheet_name="SCW")
        gca_physician = gca["Physician ID"].astype("string").str.strip()
        scw_physician = scw["Physician ID"].astype("string").str.strip()
        severity = validated_error_severity(gca, f"{filename} GCA sheet")
        checks = {
            "centre": centre, "file": filename, "n_gca": len(gca), "n_scw": len(scw),
            "ids_identical_and_ordered": bool(gca["ID"].equals(scw["ID"])),
            "no_duplicate_gca_id": bool(not gca["ID"].duplicated().any()),
            "no_duplicate_scw_id": bool(not scw["ID"].duplicated().any()),
            "no_missing_gca_fields": bool(not gca.isna().any().any()),
            "no_missing_scw_fields": bool(not scw.isna().any().any()),
            "physician_ids_identical_and_ordered": bool(gca_physician.equals(scw_physician)),
            "physician_seniority_identical_and_ordered": bool(
                gca["Physician Seniority"].astype(str).equals(scw["Physician Seniority"].astype(str))
            ),
            "gca_editing_identity": bool(np.allclose(gca["Text Editing Workload"], gca["Deleted Characters"] + gca["Added Characters"])),
            "gca_length_identity": bool(np.allclose(gca["Total Record Word Count After Editing"], gca["Total Record Word Count Before Editing"] - gca["Deleted Characters"] + gca["Added Characters"])),
            "scw_workload_identity": bool(np.allclose(scw["Text Editing Workload"], scw["Total Record Word Count"])),
            "clinical_error_severity_complete": bool(not severity.isna().any()),
            "clinical_error_severity_levels_valid": bool(
                set(severity.unique()).issubset(ERROR_SEVERITY_LABELS)
            ),
        }
        if not all(value for key, value in checks.items() if isinstance(value, bool)) or len(gca) != len(scw):
            raise AssertionError(f"External-validation check failed for {filename}: {checks}")
        validation_rows.append(checks)
        frame = pd.DataFrame({
            "centre": centre, "pair_id": centre + "_" + gca["ID"].astype(str),
            "physician_id": centre + "::" + gca_physician.astype(str),
            "senior": gca["Physician Seniority"].astype(str).str.contains("Senior").astype(int),
            "age": pd.to_numeric(gca["Patient Age"]),
            "female": gca["Patient Sex"].astype(str).str.lower().eq("female").astype(int),
            "clinical_error_severity": severity,
        })
        for domain in PDQI_EXTERNAL:
            frame[f"gca_{domain}"] = gca[domain].to_numpy()
            frame[f"scw_{domain}"] = scw[domain].to_numpy()
            frame[f"diff_{domain}"] = frame[f"gca_{domain}"] - frame[f"scw_{domain}"]
        frame["gca_pdqi_total"] = gca[PDQI_EXTERNAL].sum(axis=1).to_numpy()
        frame["scw_pdqi_total"] = scw[PDQI_EXTERNAL].sum(axis=1).to_numpy()
        frame["diff_pdqi_total"] = frame["gca_pdqi_total"] - frame["scw_pdqi_total"]
        frame["gca_workload"] = gca["Text Editing Workload"].to_numpy()
        frame["scw_workload"] = scw["Text Editing Workload"].to_numpy()
        frame["diff_workload"] = frame["gca_workload"] - frame["scw_workload"]
        frame["gca_final_length"] = gca["Total Record Word Count After Editing"].to_numpy()
        frame["scw_final_length"] = scw["Total Record Word Count"].to_numpy()
        frame["diff_final_length"] = frame["gca_final_length"] - frame["scw_final_length"]
        frame["conflicts"] = gca["Factually Conflicting Characters"].to_numpy()
        frame["deleted"] = gca["Deleted Characters"].to_numpy()
        frame["any_conflict"] = (frame["conflicts"] > 0).astype(int)
        frame["conflicts_per_1000"] = 1000 * frame["conflicts"] / frame["gca_final_length"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), validation_rows


def centre_heterogeneity(data, outcome):
    fit = smf.ols(f"Q({outcome!r}) ~ C(centre)", data=data).fit(cov_type="HC3", use_t=False)
    names = list(fit.params.index)
    terms = [name for name in names if name.startswith("C(centre)")]
    restriction = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        restriction[row, names.index(term)] = 1
    test = fit.wald_test(restriction, use_f=True, scalar=True)
    return {"f_statistic": float(test.statistic), "df_num": len(terms), "p": float(test.pvalue)}


def external_seniority_model(data, outcome, add_age_sex=False):
    formula = f"Q({outcome!r}) ~ senior + C(centre)"
    if add_age_sex:
        formula += " + age + female"
    fit = smf.ols(formula, data=data).fit(cov_type="HC3", use_t=True)
    design_info = fit.model.data.design_info
    results = {}
    for senior, label in [(0, "junior"), (1, "senior")]:
        prediction_data = data.copy()
        prediction_data["senior"] = senior
        if add_age_sex:
            prediction_data["age"] = data["age"].mean()
            prediction_data["female"] = data["female"].mean()
        matrix = np.asarray(patsy.build_design_matrices([design_info], prediction_data)[0])
        results[label] = linear_contrast(fit, matrix.mean(axis=0), use_t=True)
    vector = [1.0 if name == "senior" else 0.0 for name in fit.params.index]
    results["difference"] = linear_contrast(fit, vector, use_t=True)
    return results


def external_physician_cluster_sensitivity(data, key_map):
    if data["physician_id"].isna().any() or data["physician_id"].eq("").any():
        raise ValueError("Multicentre physician identifiers must be complete")
    physician = data.groupby("physician_id", sort=True).agg(
        n=("pair_id", "size"),
        n_centres=("centre", "nunique"),
        n_seniority=("senior", "nunique"),
        centre=("centre", "first"),
        senior=("senior", "first"),
    )
    if int(physician["n_centres"].max()) != 1:
        raise ValueError("Multicentre physician cluster identifiers must be centre-specific")
    if int(physician["n_seniority"].max()) != 1:
        raise ValueError("Physician seniority must be constant within multicentre physician")

    clusters = data["physician_id"].astype(str).to_numpy()
    overall_design = np.ones((len(data), 1))
    seniority_design = patsy.dmatrix(
        "senior + C(centre)", data=data, return_type="dataframe"
    )
    seniority_contrast = np.zeros(seniority_design.shape[1])
    seniority_contrast[list(seniority_design.columns).index("senior")] = 1
    overall = {
        label: cr2_ols(data[diff], overall_design, clusters, [1])
        for label, (_, _, diff) in key_map.items()
    }
    seniority = {
        label: cr2_ols(data[diff], seniority_design, clusters, seniority_contrast)
        for label, (_, _, diff) in key_map.items()
    }
    cell_counts = {
        f"{centre}_{'senior' if senior else 'junior'}": int(count)
        for (centre, senior), count in physician.groupby(["centre", "senior"]).size().items()
    }
    return {
        "overall": overall,
        "seniority_difference": seniority,
        "cluster_diagnostics": {
            "n_physicians": int(len(physician)),
            "n_junior_physicians": int((physician["senior"] == 0).sum()),
            "n_senior_physicians": int((physician["senior"] == 1).sum()),
            "singleton_physicians": int((physician["n"] == 1).sum()),
            "cluster_size_min": int(physician["n"].min()),
            "cluster_size_median": float(physician["n"].median()),
            "cluster_size_max": int(physician["n"].max()),
            "centre_by_seniority_physician_counts": cell_counts,
        },
        "overall_model": "paired difference ~ 1",
        "seniority_model": "paired difference ~ seniority + centre",
        "effect_direction": "GCA minus SCW",
        "seniority_direction": "senior minus junior difference in paired effects",
    }


def centre_stratified_bootstrap(data, outcomes):
    rng = np.random.default_rng(SEED)
    groups = []
    total = len(data)
    for _, group in data.groupby("centre", sort=True):
        groups.append((group, rng.integers(0, len(group), size=(BOOTSTRAP_REPLICATES, len(group)))))
    output = {}
    for outcome in outcomes:
        estimates = np.zeros(BOOTSTRAP_REPLICATES)
        for group, indices in groups:
            estimates += (len(group) / total) * group[outcome].to_numpy(float)[indices].mean(axis=1)
        output[outcome] = {
            "ci_low": float(np.quantile(estimates, 0.025)),
            "ci_high": float(np.quantile(estimates, 0.975)),
        }
    return output


def centre_equal_estimates(data, outcomes):
    centre_names = sorted(data["centre"].unique())
    rng = np.random.default_rng(SEED)
    groups = []
    for centre in centre_names:
        group = data.loc[data["centre"] == centre]
        groups.append((group, rng.integers(0, len(group), size=(BOOTSTRAP_REPLICATES, len(group)))))
    output = {}
    for outcome in outcomes:
        centre_means = np.asarray(
            [data.loc[data["centre"] == centre, outcome].mean() for centre in centre_names],
            dtype=float,
        )
        bootstrap_estimates = np.mean(
            np.vstack([
                group[outcome].to_numpy(float)[indices].mean(axis=1)
                for group, indices in groups
            ]),
            axis=0,
        )
        output[outcome] = {
            "estimate": float(centre_means.mean()),
            "ci_low": float(np.quantile(bootstrap_estimates, 0.025)),
            "ci_high": float(np.quantile(bootstrap_estimates, 0.975)),
        }
    return output


def seniority_stratified_bootstrap(data, outcomes):
    """Bootstrap centre-adjusted senior-minus-junior coefficients.

    Sampling is independent within each centre-by-seniority stratum. Because
    stratum sizes are fixed in every replicate, the OLS seniority coefficient
    in D ~ senior + C(centre) is the weighted mean of within-centre senior-minus-
    junior contrasts, with weights n_j*n_s/(n_j+n_s).
    """
    rng = np.random.default_rng(SEED)
    strata = {}
    weights = {}
    for centre in sorted(data["centre"].unique()):
        sizes = {}
        for senior in (0, 1):
            group = data[(data["centre"] == centre) & (data["senior"] == senior)]
            if group.empty:
                raise ValueError(f"Missing centre-by-seniority stratum: {centre}, {senior}")
            strata[(centre, senior)] = (
                group,
                rng.integers(0, len(group), size=(BOOTSTRAP_REPLICATES, len(group))),
            )
            sizes[senior] = len(group)
        weights[centre] = sizes[0] * sizes[1] / (sizes[0] + sizes[1])
    denominator = sum(weights.values())
    output = {}
    for outcome in outcomes:
        bootstrap_difference = np.zeros(BOOTSTRAP_REPLICATES)
        for centre in sorted(weights):
            group_j, index_j = strata[(centre, 0)]
            group_s, index_s = strata[(centre, 1)]
            mean_j = group_j[outcome].to_numpy(float)[index_j].mean(axis=1)
            mean_s = group_s[outcome].to_numpy(float)[index_s].mean(axis=1)
            bootstrap_difference += weights[centre] * (mean_s - mean_j) / denominator
        output[outcome] = {
            "ci_low": float(np.quantile(bootstrap_difference, 0.025)),
            "ci_high": float(np.quantile(bootstrap_difference, 0.975)),
        }
    return output


def standardised_external_logistic(data):
    fit = smf.glm(
        "any_conflict ~ senior + C(centre)", data=data,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3", use_t=False)
    design_info = fit.model.data.design_info
    beta, covariance = fit.params.to_numpy(), fit.cov_params().to_numpy()
    outputs = {}
    probabilities, gradients = {}, {}
    for senior, label in [(0, "junior"), (1, "senior")]:
        prediction_data = data.copy()
        prediction_data["senior"] = senior
        matrix = np.asarray(patsy.build_design_matrices([design_info], prediction_data)[0])
        p = scipy.special.expit(matrix @ beta)
        probabilities[label] = float(p.mean())
        gradients[label] = (p * (1 - p)) @ matrix / len(matrix)
    gradient = gradients["senior"] - gradients["junior"]
    estimate = probabilities["senior"] - probabilities["junior"]
    se = math.sqrt(float(gradient @ covariance @ gradient))
    outputs.update({
        "junior_probability": probabilities["junior"],
        "senior_probability": probabilities["senior"],
        "risk_difference": estimate, "se": se,
        "ci_low": float(estimate - 1.96 * se), "ci_high": float(estimate + 1.96 * se),
        "p": float(2 * stats.norm.sf(abs(estimate / se))),
    })
    return outputs


def safety_descriptive_group(frame):
    events = int(frame["any_conflict"].sum())
    nobs = int(len(frame))
    return {
        "n": nobs,
        "clinical_error_severity": clinical_error_severity_summary(
            frame["clinical_error_severity"]
        ),
        "conflicting_characters": describe(frame["conflicts"]),
        "drafts_with_any_detected_conflict": events,
        "proportion_with_any_detected_conflict": float(events / nobs),
        "exact_ci": clopper_pearson(events, nobs),
        "conflicts_per_1000_final_characters": describe(frame["conflicts_per_1000"]),
        "aggregate_conflicts_as_fraction_of_deleted": float(frame["conflicts"].sum() / frame["deleted"].sum()),
    }


def ordinal_gee_external(data):
    long_parts = []
    for condition, prefix in [(1, "gca"), (0, "scw")]:
        part = data[["pair_id", "centre", "senior"]].copy()
        part["condition"] = condition
        for domain in PDQI_EXTERNAL:
            part[domain] = data[f"{prefix}_{domain}"].to_numpy()
        long_parts.append(part)
    long_data = pd.concat(long_parts, ignore_index=True)
    output = {}
    for domain in PDQI_EXTERNAL:
        domain_data = long_data[["pair_id", "centre", "senior", "condition", domain]].rename(columns={domain: "score"})
        output[domain] = {}
        centre_indicators = pd.get_dummies(
            domain_data["centre"], prefix="centre", drop_first=True, dtype=float
        )
        overall_exog = pd.concat(
            [domain_data[["condition"]].astype(float).reset_index(drop=True), centre_indicators.reset_index(drop=True)],
            axis=1,
        )
        interaction_exog = overall_exog.copy()
        interaction_exog.insert(1, "senior", domain_data["senior"].to_numpy(float))
        interaction_exog.insert(
            2,
            "condition_x_senior",
            (domain_data["condition"] * domain_data["senior"]).to_numpy(float),
        )
        specifications = [
            ("overall", overall_exog, "condition"),
            ("interaction", interaction_exog, "condition_x_senior"),
        ]
        for label, exog, term in specifications:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = OrdinalGEE(
                    domain_data["score"].astype(int).reset_index(drop=True),
                    exog,
                    groups=domain_data["pair_id"].reset_index(drop=True),
                    cov_struct=Independence(),
                ).fit(maxiter=100, ctol=1e-6)
            ci = fit.conf_int().loc[term]
            result = {
                "converged": bool(fit.converged),
                "iterations": len(fit.fit_history.get("params", [])),
                "warnings": [str(item.message) for item in caught],
                "ordinary_intercept_in_exog": False,
                "centre_reference": "Centre 1",
                "n_centre_indicators": int(centre_indicators.shape[1]),
            }
            result.update({
                "or": float(np.exp(fit.params[term])),
                "ci_low": float(np.exp(ci.iloc[0])), "ci_high": float(np.exp(ci.iloc[1])),
                "p": float(fit.pvalues[term]),
            })
            result["valid"] = bool(result["converged"] and all_finite(result))
            if not result["valid"]:
                raise RuntimeError(f"Invalid Ordinal GEE fit for {domain}, {label}: {result}")
            output[domain][label] = result
    overall_q = bh([output[domain]["overall"]["p"] for domain in PDQI_EXTERNAL])
    interaction_q = bh([output[domain]["interaction"]["p"] for domain in PDQI_EXTERNAL])
    for domain, q_overall, q_interaction in zip(PDQI_EXTERNAL, overall_q, interaction_q):
        output[domain]["overall"]["q"] = q_overall
        output[domain]["interaction"]["q"] = q_interaction
    return output


def external_analysis():
    data, validation_rows = load_external()
    key_map = {
        "Text editing workload": ("gca_workload", "scw_workload", "diff_workload"),
        "PDQI-9 total score": ("gca_pdqi_total", "scw_pdqi_total", "diff_pdqi_total"),
        "Final record length": ("gca_final_length", "scw_final_length", "diff_final_length"),
    }
    key = {label: paired(data[g], data[s]) for label, (g, s, _) in key_map.items()}
    domains = {domain: paired(data[f"gca_{domain}"], data[f"scw_{domain}"]) for domain in PDQI_EXTERNAL}
    heterogeneity = {label: centre_heterogeneity(data, diff) for label, (_, _, diff) in key_map.items()}
    domain_heterogeneity = {domain: centre_heterogeneity(data, f"diff_{domain}") for domain in PDQI_EXTERNAL}
    centre_specific = {
        label: {
            centre: paired(group[g], group[s])
            for centre, group in data.groupby("centre", sort=True)
        }
        for label, (g, s, _) in key_map.items()
    }
    seniority = {label: external_seniority_model(data, diff) for label, (_, _, diff) in key_map.items()}
    seniority_adjusted = {label: external_seniority_model(data, diff, True) for label, (_, _, diff) in key_map.items()}
    condition_gaps = {
        "SCW": external_seniority_model(data, "scw_pdqi_total"),
        "GCA": external_seniority_model(data, "gca_pdqi_total"),
    }
    bootstrap_outcomes = [diff for _, _, diff in key_map.values()] + [f"diff_{d}" for d in PDQI_EXTERNAL]
    bootstrap = centre_stratified_bootstrap(data, bootstrap_outcomes)
    centre_equal = centre_equal_estimates(data, bootstrap_outcomes)
    seniority_bootstrap = seniority_stratified_bootstrap(
        data, [diff for _, _, diff in key_map.values()]
    )
    physician_cluster_sensitivity = external_physician_cluster_sensitivity(data, key_map)
    leave_one_out = {
        label: {
            excluded: mean_inference(data.loc[data["centre"] != excluded, diff], use_t=True)
            for excluded in sorted(data["centre"].unique())
        }
        for label, (_, _, diff) in key_map.items()
    }

    safety_logistic = standardised_external_logistic(data)
    nb = smf.negativebinomial(
        "conflicts ~ senior + C(centre)", data=data,
        offset=np.log(data["gca_final_length"] / 1000),
    ).fit(disp=False, cov_type="HC0")
    nb_ci = nb.conf_int().loc["senior"]
    centre_fit = smf.glm(
        "any_conflict ~ C(centre)", data=data,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3", use_t=False)
    names = list(centre_fit.params.index)
    terms = [name for name in names if name.startswith("C(centre)")]
    restriction = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        restriction[row, names.index(term)] = 1
    centre_test = centre_fit.wald_test(restriction, use_f=False, scalar=True)
    safety_p = [safety_logistic["p"], float(nb.pvalues["senior"])]
    safety = {
        **safety_descriptive_group(data),
        "by_centre": {
            centre: safety_descriptive_group(group)
            for centre, group in data.groupby("centre", sort=True)
        },
        "by_seniority": {
            "Junior physicians": safety_descriptive_group(data.loc[data["senior"] == 0]),
            "Senior physicians": safety_descriptive_group(data.loc[data["senior"] == 1]),
        },
        "seniority_any_conflict": safety_logistic,
        "seniority_count_rate": {
            "irr": float(np.exp(nb.params["senior"])),
            "ci_low": float(np.exp(nb_ci.iloc[0])), "ci_high": float(np.exp(nb_ci.iloc[1])),
            "p": float(nb.pvalues["senior"]), "dispersion_alpha": float(nb.params["alpha"]),
        },
        "seniority_bh_q": bh(safety_p),
        "centre_logistic_omnibus": {
            "chi2": float(centre_test.statistic), "df": len(terms), "p": float(centre_test.pvalue),
        },
    }

    return {
        "validation": {"n_pairs": int(len(data)), "centres": validation_rows},
        "key_outcomes": key,
        "key_outcome_q": bh([key["Text editing workload"]["p"], key["PDQI-9 total score"]["p"]]),
        "pdqi_domains": domains,
        "pdqi_domain_q": bh([domains[d]["p"] for d in PDQI_EXTERNAL]),
        "centre_heterogeneity": heterogeneity,
        "key_heterogeneity_q": bh([heterogeneity["Text editing workload"]["p"], heterogeneity["PDQI-9 total score"]["p"]]),
        "domain_heterogeneity": domain_heterogeneity,
        "domain_heterogeneity_q": bh([domain_heterogeneity[d]["p"] for d in PDQI_EXTERNAL]),
        "centre_specific": centre_specific,
        "seniority": seniority,
        "seniority_q": bh([seniority["Text editing workload"]["difference"]["p"], seniority["PDQI-9 total score"]["difference"]["p"]]),
        "seniority_age_sex_adjusted": seniority_adjusted,
        "condition_pdqi_seniority_gaps": condition_gaps,
        "centre_stratified_bootstrap": bootstrap,
        "centre_equal_effects": centre_equal,
        "seniority_stratified_bootstrap": seniority_bootstrap,
        "physician_cluster_sensitivity": physician_cluster_sensitivity,
        "leave_one_centre_out": leave_one_out,
        "ordinal_gee_sensitivity": ordinal_gee_external(data),
        "draft_safety": safety,
    }


def flatten_summary(results):
    rows = []
    for label, result in results["rct"]["core"].items():
        rows.append({"study": "RCT", "analysis": "overall", "outcome": label, **{k: result[k] for k in ["estimate", "ci_low", "ci_high", "p"]}})
    result = results["rct"]["diagnosis"]
    rows.append({"study": "RCT", "analysis": "overall", "outcome": "Accurate preliminary diagnosis", "estimate": result["risk_difference"], "ci_low": result["ci_low"], "ci_high": result["ci_high"], "p": result["p"]})
    for label, result in results["rct"]["subgroup_continuous"].items():
        rows.append({"study": "RCT", "analysis": "seniority interaction", "outcome": label, **{k: result["interaction"][k] for k in ["estimate", "ci_low", "ci_high", "p"]}})
    result = results["rct"]["subgroup_diagnostic_accuracy"]["interaction"]
    rows.append({"study": "RCT", "analysis": "seniority interaction", "outcome": "Accurate preliminary diagnosis", **{k: result[k] for k in ["estimate", "ci_low", "ci_high", "p"]}})
    for label, result in results["external"]["key_outcomes"].items():
        rows.append({"study": "Multicentre", "analysis": "paired overall", "outcome": label, **{k: result[k] for k in ["estimate", "ci_low", "ci_high", "p"]}})
    for label, result in results["external"]["seniority"].items():
        rows.append({"study": "Multicentre", "analysis": "seniority interaction", "outcome": label, **{k: result["difference"][k] for k in ["estimate", "ci_low", "ci_high", "p"]}})
    return pd.DataFrame(rows)


def flatten_multiplicity(results):
    rows = []

    def add(study, family, outcomes, p_values, q_values):
        if not (len(outcomes) == len(p_values) == len(q_values)):
            raise ValueError(f"Multiplicity family length mismatch: {study}, {family}")
        for outcome, p_value, q_value in zip(outcomes, p_values, q_values):
            rows.append({
                "study": study,
                "family": family,
                "family_size": len(outcomes),
                "outcome": outcome,
                "p": float(p_value),
                "q": float(q_value),
            })

    rct = results["rct"]
    core_outcomes = [
        "Interview time", "Physician follow-up work time", "Text editing workload",
        "PDQI-9 total score", "Accurate preliminary diagnosis",
    ]
    core_p = [rct["core"][outcome]["p"] for outcome in core_outcomes[:-1]] + [rct["diagnosis"]["p"]]
    add("RCT", "overall_core_secondary", core_outcomes, core_p, rct["core_secondary_q"])
    patient_outcomes = ["patient_experience"] + PATIENT_ITEMS
    add("RCT", "overall_patient_reported", patient_outcomes,
        [rct["patient_outcomes"][x]["p"] for x in patient_outcomes], rct["patient_outcome_q"])
    add("RCT", "overall_pdqi_domains", PDQI_RCT,
        [rct["pdqi_domains"][x]["p"] for x in PDQI_RCT], rct["pdqi_domain_q"])
    interview_outcomes = ["Interview Fluency", "Interview Completeness"]
    add("RCT", "overall_interview_quality", interview_outcomes,
        [rct["interview_quality"][x]["p"] for x in interview_outcomes], rct["interview_quality_q"])
    exploratory_outcomes = ["Patient Attention Time", "Final record length"]
    add("RCT", "overall_other_exploratory", exploratory_outcomes,
        [rct["other_exploratory"][x]["p"] for x in exploratory_outcomes], rct["other_exploratory_q"])
    add("RCT", "seniority_core_secondary_interactions", core_outcomes,
        [rct["subgroup_continuous"][x]["interaction"]["p"] for x in core_outcomes[:-1]]
        + [rct["subgroup_diagnostic_accuracy"]["interaction"]["p"]], rct["subgroup_core_q"])
    add("RCT", "seniority_pdqi_interactions", PDQI_RCT,
        [rct["subgroup_pdqi"][x]["interaction"]["p"] for x in PDQI_RCT], rct["subgroup_pdqi_q"])
    add("RCT", "seniority_interview_interactions", interview_outcomes,
        [rct["subgroup_interview"][x]["interaction"]["p"] for x in interview_outcomes], rct["subgroup_interview_q"])
    add("RCT", "seniority_patient_reported_interactions", patient_outcomes,
        [rct["subgroup_patient"][x]["interaction"]["p"] for x in patient_outcomes], rct["subgroup_patient_q"])
    for family, specification in rct["ordinal_sensitivity_families"].items():
        outcomes = specification["outcomes"]
        add("RCT", f"ordinal_sensitivity_{family}", outcomes,
            [rct["ordinal_logistic_sensitivity"][x]["p"] for x in outcomes],
            [rct["ordinal_logistic_sensitivity"][x]["q"] for x in outcomes])
    implementation = list(rct["gca_implementation_by_seniority"])
    add("RCT", "gca_implementation_seniority", implementation,
        [rct["gca_implementation_by_seniority"][x]["p"] for x in implementation],
        [rct["gca_implementation_by_seniority"][x]["q"] for x in implementation])

    external = results["external"]
    key_outcomes = ["Text editing workload", "PDQI-9 total score"]
    add("Multicentre", "key_paired_effects", key_outcomes,
        [external["key_outcomes"][x]["p"] for x in key_outcomes], external["key_outcome_q"])
    add("Multicentre", "key_centre_heterogeneity", key_outcomes,
        [external["centre_heterogeneity"][x]["p"] for x in key_outcomes], external["key_heterogeneity_q"])
    add("Multicentre", "pdqi_domain_paired_effects", PDQI_EXTERNAL,
        [external["pdqi_domains"][x]["p"] for x in PDQI_EXTERNAL], external["pdqi_domain_q"])
    add("Multicentre", "pdqi_domain_centre_heterogeneity", PDQI_EXTERNAL,
        [external["domain_heterogeneity"][x]["p"] for x in PDQI_EXTERNAL], external["domain_heterogeneity_q"])
    add("Multicentre", "key_seniority_interactions", key_outcomes,
        [external["seniority"][x]["difference"]["p"] for x in key_outcomes], external["seniority_q"])
    safety_outcomes = ["Any factual conflict", "Conflict count rate"]
    safety_p = [external["draft_safety"]["seniority_any_conflict"]["p"], external["draft_safety"]["seniority_count_rate"]["p"]]
    add("Multicentre", "safety_seniority", safety_outcomes, safety_p, external["draft_safety"]["seniority_bh_q"])
    add("Multicentre", "ordinal_overall", PDQI_EXTERNAL,
        [external["ordinal_gee_sensitivity"][x]["overall"]["p"] for x in PDQI_EXTERNAL],
        [external["ordinal_gee_sensitivity"][x]["overall"]["q"] for x in PDQI_EXTERNAL])
    add("Multicentre", "ordinal_seniority_interaction", PDQI_EXTERNAL,
        [external["ordinal_gee_sensitivity"][x]["interaction"]["p"] for x in PDQI_EXTERNAL],
        [external["ordinal_gee_sensitivity"][x]["interaction"]["q"] for x in PDQI_EXTERNAL])
    return pd.DataFrame(rows)


def flatten_rct_implementation(results):
    return pd.DataFrame([
        {"outcome": outcome, **values}
        for outcome, values in results["rct"]["gca_implementation_by_seniority"].items()
    ])


def flatten_rct_hedges_g(results):
    return pd.DataFrame([
        {"outcome": outcome, **values}
        for outcome, values in results["rct"]["hedges_g"].items()
    ])


def flatten_physician_cluster_sensitivity(results):
    rows = []
    specifications = [
        ("RCT", "Overall treatment effect", results["rct"]["physician_cluster_sensitivity"]["overall"]),
        ("RCT", "Treatment-by-seniority interaction", results["rct"]["physician_cluster_sensitivity"]["seniority_interaction"]),
        ("RCT", "Patient-reported outcome", results["rct"]["physician_cluster_sensitivity"]["patient_reported"]),
        ("Multicentre", "Overall paired effect", results["external"]["physician_cluster_sensitivity"]["overall"]),
        ("Multicentre", "Senior-minus-junior difference in paired effects", results["external"]["physician_cluster_sensitivity"]["seniority_difference"]),
    ]
    for study, target, outcomes in specifications:
        for outcome, result in outcomes.items():
            rows.append({
                "study": study,
                "analysis_target": target,
                "outcome": outcome,
                **{key: result[key] for key in [
                    "estimate", "se", "df", "ci_low", "ci_high", "p", "n", "n_clusters",
                    "cluster_size_min", "cluster_size_median", "cluster_size_max",
                    "variance_estimator", "working_covariance", "reference_distribution",
                    "software_implementation",
                ]},
            })
    return pd.DataFrame(rows)


def flatten_physician_icc(results):
    rows = []
    specifications = [
        ("Continuous", results["rct"]["physician_icc"]["continuous"]),
        ("Observed binary", results["rct"]["physician_icc"]["binary"]),
    ]
    for outcome_type, outcomes in specifications:
        for outcome, result in outcomes.items():
            rows.append({"outcome_type": outcome_type, "outcome": outcome, **result})
    return pd.DataFrame(rows)


def run_regression_checks(results):
    ordinal = results["external"]["ordinal_gee_sensitivity"]
    ordinal_fits = [ordinal[domain][analysis] for domain in PDQI_EXTERNAL for analysis in ("overall", "interaction")]
    rct_ordinal = results["rct"]["ordinal_logistic_sensitivity"]
    implementation = results["rct"]["gca_implementation_by_seniority"]
    rct_cluster = results["rct"]["physician_cluster_sensitivity"]
    rct_icc = results["rct"]["physician_icc"]
    external_cluster = results["external"]["physician_cluster_sensitivity"]
    cluster_groups = [
        rct_cluster["overall"], rct_cluster["seniority_interaction"], rct_cluster["patient_reported"],
        external_cluster["overall"], external_cluster["seniority_difference"],
    ]
    checks = {
        "all_machine_results_finite": all_finite(results),
        "rct_n": results["rct"]["validation"]["n_gca"] == 200 and results["rct"]["validation"]["n_scw"] == 200,
        "external_n": results["external"]["validation"]["n_pairs"] == 761,
        "rct_data_validation": all(
            value for key, value in results["rct"]["validation"].items()
            if key not in {"n_gca", "n_scw"}
        ),
        "external_data_validation": all(
            value
            for row in results["external"]["validation"]["centres"]
            for key, value in row.items()
            if isinstance(value, bool)
        ),
        "rct_ordinal_all_valid": len(rct_ordinal) == 15 and all(item.get("valid") for item in rct_ordinal.values()),
        "rct_ordinal_three_complete_bh_families": sorted(
            item["n_tests"] for item in results["rct"]["ordinal_sensitivity_families"].values()
        ) == [2, 4, 9],
        "rct_implementation_six_outcomes": len(implementation) == 6 and all(all_finite(item) for item in implementation.values()),
        "external_ordinal_18_valid": len(ordinal_fits) == 18 and all(item["valid"] for item in ordinal_fits),
        "external_ordinal_no_ordinary_intercept": all(
            item["ordinary_intercept_in_exog"] is False and item["n_centre_indicators"] == 4
            for item in ordinal_fits
        ),
        "external_centre_equal_present": all(
            outcome in results["external"]["centre_equal_effects"]
            for outcome in ["diff_workload", "diff_pdqi_total", "diff_final_length"]
        ),
        "external_seniority_bootstrap_present": all(
            outcome in results["external"]["seniority_stratified_bootstrap"]
            for outcome in ["diff_workload", "diff_pdqi_total", "diff_final_length"]
        ),
        "rct_cluster_sensitivity_complete": (
            rct_cluster["cluster_diagnostics"]["n_physicians"] > 1
            and len(rct_cluster["overall"]) == 6
            and len(rct_cluster["seniority_interaction"]) == 6
            and len(rct_cluster["patient_reported"]) == 5
        ),
        "external_cluster_sensitivity_complete": (
            external_cluster["cluster_diagnostics"]["n_physicians"] > 1
            and len(external_cluster["overall"]) == 3
            and len(external_cluster["seniority_difference"]) == 3
        ),
        "clustered_point_estimates_match_primary_estimands": (
            all(abs(rct_cluster["overall"][outcome]["estimate"] - results["rct"]["core"][outcome]["estimate"]) < 1e-9
                for outcome in results["rct"]["core"])
            and abs(rct_cluster["overall"]["Accurate preliminary diagnosis"]["estimate"] - results["rct"]["diagnosis"]["risk_difference"]) < 1e-9
            and all(
                abs(rct_cluster["patient_reported"][label]["estimate"] - results["rct"]["patient_outcomes"][key]["estimate"]) < 1e-9
                for label, key in {
                    "Patient experience score": "patient_experience",
                    "Communication convenience": "Communication Convenience",
                    "Perceived physician attention": "Perceived Physician Attention",
                    "Perceived empathy": "Perceived Empathy",
                    "Overall satisfaction": "Overall Satisfaction",
                }.items()
            )
            and all(abs(external_cluster["overall"][outcome]["estimate"] - results["external"]["key_outcomes"][outcome]["estimate"]) < 1e-9
                    for outcome in results["external"]["key_outcomes"])
            and all(abs(external_cluster["seniority_difference"][outcome]["estimate"] - results["external"]["seniority"][outcome]["difference"]["estimate"]) < 1e-9
                    for outcome in results["external"]["seniority"])
        ),
        "clustered_satterthwaite_df_valid": all(
            0 < item["df"] <= item["n_clusters"] - 1 + 1e-9
            for group in cluster_groups
            for item in group.values()
        ),
        "clustered_cr2_qc_fields_complete": all(
            item["variance_estimator"] == "CR2"
            and item["working_covariance"] == "Identity"
            and item["reference_distribution"] == "Satterthwaite t"
            and item["software_implementation"] == "Native Python (NumPy/SciPy; src/cluster_robust.py)"
            and all(np.isfinite(item[field]) for field in ["estimate", "se", "df", "ci_low", "ci_high"])
            for group in cluster_groups
            for item in group.values()
        ),
        "clustered_ci_reconstructs_from_se_and_df": all(
            abs(item["ci_low"] - (
                item["estimate"] - stats.t.ppf(0.975, item["df"]) * item["se"]
            )) < 1e-10
            and abs(item["ci_high"] - (
                item["estimate"] + stats.t.ppf(0.975, item["df"]) * item["se"]
            )) < 1e-10
            for group in cluster_groups
            for item in group.values()
        ),
        "physician_icc_complete_and_valid": (
            len(rct_icc["continuous"]) == 6
            and len(rct_icc["binary"]) == 1
            and all(
                item["converged"]
                and item["n"] == 400
                and item["n_clusters"] == rct_cluster["cluster_diagnostics"]["n_physicians"]
                and 0 <= item["icc"] <= 1
                and item["physician_variance"] >= 0
                and item["residual_variance"] > 0
                for group in [rct_icc["continuous"], rct_icc["binary"]]
                for item in group.values()
            )
        ),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = {
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__, "patsy": patsy.__version__,
            "random_seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "rct": rct_analysis(),
        "external": external_analysis(),
    }
    checks = run_regression_checks(results)
    (RESULTS / "reproducibility_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=json_default, allow_nan=False), encoding="utf-8"
    )
    (RESULTS / "regression_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flatten_summary(results).to_csv(RESULTS / "analysis_summary.csv", index=False, encoding="utf-8-sig")
    flatten_multiplicity(results).to_csv(RESULTS / "multiplicity_families.csv", index=False, encoding="utf-8-sig")
    flatten_rct_implementation(results).to_csv(RESULTS / "rct_implementation_by_seniority.csv", index=False, encoding="utf-8-sig")
    flatten_rct_hedges_g(results).to_csv(RESULTS / "rct_hedges_g.csv", index=False, encoding="utf-8-sig")
    flatten_physician_cluster_sensitivity(results).to_csv(
        RESULTS / "physician_cluster_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    flatten_physician_icc(results).to_csv(
        RESULTS / "physician_level_icc.csv", index=False, encoding="utf-8-sig"
    )
    validation_rows = results["external"]["validation"]["centres"]
    pd.DataFrame(validation_rows).to_csv(RESULTS / "data_validation.csv", index=False, encoding="utf-8-sig")
    ordinal_rows = []
    for domain, fits in results["external"]["ordinal_gee_sensitivity"].items():
        for analysis, fit in fits.items():
            ordinal_rows.append({"domain": domain, "analysis": analysis, **fit})
    pd.DataFrame(ordinal_rows).to_csv(RESULTS / "ordinal_gee_status.csv", index=False, encoding="utf-8-sig")
    if not checks["all_passed"]:
        raise RuntimeError(f"Regression check failed: {checks}")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
