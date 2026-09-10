from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis
from .terminology import (
    CHARACTER_LEVEL_FACTUAL_CONFLICT_HEADING,
    CLINICAL_ERROR_SEVERITY_LEVELS,
    CLINICAL_ERROR_SEVERITY_MULTICENTRE_HEADING,
    CLINICAL_ERROR_SEVERITY_RCT_HEADING,
    IMPLEMENTATION_OUTCOMES,
    PATIENT_REPORTED_ITEMS,
    RCT_PDQI_DOMAINS,
)


def _write(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _mean_sd(values) -> str:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return f"{x.mean():.1f} ({x.std(ddof=1):.1f})"


def _count_percent(values) -> str:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return f"{int(x.sum())}/{len(x)} ({100 * x.mean():.1f}%)"


def _n_percent(values) -> str:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return f"{int(x.sum())} ({100 * x.mean():.1f}%)"


def _effect(result: dict, scale: float = 1.0, decimals: int = 2) -> str:
    return (
        f"{scale * result['estimate']:.{decimals}f} "
        f"({scale * result['ci_low']:.{decimals}f} to {scale * result['ci_high']:.{decimals}f})"
    )


def _smd_continuous(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(abs(a.mean() - b.mean()) / pooled)


def _smd_binary(a, b) -> float:
    pa, pb = float(np.mean(a)), float(np.mean(b))
    denominator = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return 0.0 if denominator == 0 else float(abs(pa - pb) / denominator)


def rct_baseline_characteristics(output: Path) -> None:
    gca, scw, combined = analysis.load_rct()
    junior = combined["senior"].eq(0)
    senior = combined["senior"].eq(1)
    empty = {"Overall (N=400)": "", "GCA (n=200)": "", "SCW (n=200)": "", "Absolute SMD": ""}

    def admissions_per_physician(frame: pd.DataFrame) -> str:
        counts = frame.groupby("physician_id", sort=True).size()
        return f"{counts.median():.0f} ({counts.min()}–{counts.max()})"

    rows = [
        {"Characteristic": "Patient characteristics", **empty},
        {
            "Characteristic": "Age, years, mean (s.d.)",
            "Overall (N=400)": _mean_sd(combined["Patient Age"]),
            "GCA (n=200)": _mean_sd(gca["Patient Age"]),
            "SCW (n=200)": _mean_sd(scw["Patient Age"]),
            "Absolute SMD": f"{_smd_continuous(gca['Patient Age'], scw['Patient Age']):.2f}",
        },
        {
            "Characteristic": "Female sex, n (%)",
            "Overall (N=400)": _n_percent(combined["female"]),
            "GCA (n=200)": _n_percent(gca["female"]),
            "SCW (n=200)": _n_percent(scw["female"]),
            "Absolute SMD": f"{_smd_binary(gca['female'], scw['female']):.2f}",
        },
        {
            "Characteristic": "Han ethnicity, n (%)",
            "Overall (N=400)": _n_percent(combined["Han Ethnicity"]),
            "GCA (n=200)": _n_percent(gca["Han Ethnicity"]),
            "SCW (n=200)": _n_percent(scw["Han Ethnicity"]),
            "Absolute SMD": f"{_smd_binary(gca['Han Ethnicity'], scw['Han Ethnicity']):.2f}",
        },
        {
            "Characteristic": "Hubei permanent residence, n (%)",
            "Overall (N=400)": _n_percent(combined["Permanent Residence"].eq("Hubei")),
            "GCA (n=200)": _n_percent(gca["Permanent Residence"].eq("Hubei")),
            "SCW (n=200)": _n_percent(scw["Permanent Residence"].eq("Hubei")),
            "Absolute SMD": f"{_smd_binary(
                gca["Permanent Residence"].eq("Hubei"),
                scw["Permanent Residence"].eq("Hubei"),
            ):.2f}",
        },
        {"Characteristic": "Physician seniority", **empty},
        {
            "Characteristic": "Junior physician-managed admissions, n (%)",
            "Overall (N=400)": _n_percent(junior), "GCA (n=200)": _n_percent(gca["senior"].eq(0)),
            "SCW (n=200)": _n_percent(scw["senior"].eq(0)), "Absolute SMD": "—",
        },
        {
            "Characteristic": "Senior physician-managed admissions, n (%)",
            "Overall (N=400)": _n_percent(senior), "GCA (n=200)": _n_percent(gca["senior"].eq(1)),
            "SCW (n=200)": _n_percent(scw["senior"].eq(1)), "Absolute SMD": "—",
        },
        {"Characteristic": "Participating physicians", **empty},
        {
            "Characteristic": "Unique junior physicians, n",
            "Overall (N=400)": int(combined.loc[combined["senior"].eq(0), "physician_id"].nunique()),
            "GCA (n=200)": int(gca.loc[gca["senior"].eq(0), "physician_id"].nunique()),
            "SCW (n=200)": int(scw.loc[scw["senior"].eq(0), "physician_id"].nunique()), "Absolute SMD": "—",
        },
        {
            "Characteristic": "Unique senior physicians, n",
            "Overall (N=400)": int(combined.loc[combined["senior"].eq(1), "physician_id"].nunique()),
            "GCA (n=200)": int(gca.loc[gca["senior"].eq(1), "physician_id"].nunique()),
            "SCW (n=200)": int(scw.loc[scw["senior"].eq(1), "physician_id"].nunique()), "Absolute SMD": "—",
        },
        {
            "Characteristic": "Admissions per physician, median (range)",
            "Overall (N=400)": admissions_per_physician(combined),
            "GCA (n=200)": admissions_per_physician(gca),
            "SCW (n=200)": admissions_per_physician(scw),
            "Absolute SMD": "—",
        },
    ]
    _write(rows, output / "rct_baseline_characteristics.csv")


def rct_outcomes(results: dict, output: Path) -> None:
    rct = results["rct"]
    empty = {
        "GCA (n=200)": "", "SCW (n=200)": "", "Difference (95% CI)": "",
        "Relative change": "", "P value": "", "BH-adjusted P value": "",
    }
    rows = [{"Outcome": "Primary outcome", **empty}]
    secondary = [
        "Interview time", "Physician follow-up work time", "Text editing workload",
        "PDQI-9 total score", "Accurate preliminary diagnosis",
    ]
    adjusted = dict(zip(secondary, rct["core_secondary_q"]))
    display = {
        "Total admission workflow time": "Total admission workflow time, s",
        "Interview time": "Interview time, s",
        "Physician follow-up work time": "Physician follow-up work time, s",
        "Text editing workload": "Text editing workload, characters",
        "PDQI-9 total score": "PDQI-9 total score",
    }
    for outcome, result in rct["core"].items():
        if outcome == "Interview time":
            rows.append({"Outcome": "Secondary outcomes", **empty})
        rows.append({
            "Outcome": display[outcome],
            "GCA (n=200)": f"{result['gca']['mean']:.1f} ({result['gca']['sd']:.1f})",
            "SCW (n=200)": f"{result['scw']['mean']:.1f} ({result['scw']['sd']:.1f})",
            "Difference (95% CI)": _effect(result, decimals=1),
            "Relative change": f"{100 * result['relative_change']:.1f}%",
            "P value": _p(result["p"]),
            "BH-adjusted P value": "" if outcome not in adjusted else _p(adjusted[outcome]),
        })
    diagnosis = rct["diagnosis"]
    rows.append({
        "Outcome": "Accurate preliminary diagnosis",
        "GCA (n=200)": f"{diagnosis['gca']} (93.0%)", "SCW (n=200)": f"{diagnosis['scw']} (78.0%)",
        "Difference (95% CI)": (
            f"{100 * diagnosis['risk_difference']:.1f} pp "
            f"({100 * diagnosis['ci_low']:.1f} to {100 * diagnosis['ci_high']:.1f})"
        ),
        "Relative change": "19.2%", "P value": _p(diagnosis["p"]),
        "BH-adjusted P value": _p(adjusted["Accurate preliminary diagnosis"]),
    })
    rows.append({"Outcome": "Patient-reported domains", **empty})
    names = ["patient_experience"] + list(analysis.PATIENT_ITEMS)
    display = ["Patient experience score"] + PATIENT_REPORTED_ITEMS
    patient_rows = []
    for name, label, adjusted in zip(names, display, rct["patient_outcome_q"]):
        result = rct["patient_outcomes"][name]
        patient_rows.append({
            "Outcome": label,
            "GCA (n=200)": f"{result['gca']['mean']:.1f} ({result['gca']['sd']:.1f})",
            "SCW (n=200)": f"{result['scw']['mean']:.1f} ({result['scw']['sd']:.1f})",
            "Difference (95% CI)": _effect(result, decimals=1),
            "Relative change": f"{100 * result['relative_change']:.1f}%",
            "P value": _p(result["p"]),
            "BH-adjusted P value": _p(adjusted),
        })
    rows.extend(patient_rows[1:] + patient_rows[:1])
    _write(rows, output / "rct_outcomes.csv")


def rct_ai_safety_and_implementation(results: dict, output: Path) -> None:
    safety = results["rct"]["gca_draft_safety"]
    overall = results["rct"]["gca_overall_exact_proportions"]
    characters = safety["conflicting_characters"]
    mean_ci = safety["conflicting_characters_mean_inference"]
    rates = safety["conflicts_per_1000_final_characters"]
    rate_ci = safety["conflicts_per_1000_mean_inference"]
    severity = safety["clinical_error_severity"]
    rows = [
        {"Outcome": "AI-specific safety outcomes", "Overall GCA (n=200)": "", "95% CI": ""},
        {"Outcome": CLINICAL_ERROR_SEVERITY_RCT_HEADING, "Overall GCA (n=200)": "", "95% CI": ""},
    ]
    for level, label in enumerate(CLINICAL_ERROR_SEVERITY_LEVELS):
        item = severity["levels"][str(level)]
        rows.append({
            "Outcome": label,
            "Overall GCA (n=200)": f"{item['count']} ({100 * item['proportion']:.1f}%)",
            "95% CI": "",
        })
    rows.extend([
        {"Outcome": CHARACTER_LEVEL_FACTUAL_CONFLICT_HEADING, "Overall GCA (n=200)": "", "95% CI": ""},
        {"Outcome": "Factually conflicting characters per GCA-generated draft, median (IQR)",
         "Overall GCA (n=200)": f"{characters['median']:.0f} ({characters['q1']:.0f} to {characters['q3']:.0f})", "95% CI": ""},
        {"Outcome": "Factually conflicting characters per GCA-generated draft, mean (s.d.)",
         "Overall GCA (n=200)": f"{characters['mean']:.1f} ({characters['sd']:.1f})", "95% CI": f"{mean_ci['ci_low']:.1f} to {mean_ci['ci_high']:.1f}"},
        {"Outcome": "Drafts containing at least one factually conflicting character",
         "Overall GCA (n=200)": f"{overall['Drafts containing at least 1 factually conflicting character']['events']}/200 "
         f"({100 * overall['Drafts containing at least 1 factually conflicting character']['proportion']:.1f}%)",
         "95% CI": f"{100 * overall['Drafts containing at least 1 factually conflicting character']['exact_ci'][0]:.1f}% to "
         f"{100 * overall['Drafts containing at least 1 factually conflicting character']['exact_ci'][1]:.1f}%"},
        {"Outcome": "Factually conflicting characters in GCA-generated drafts, standardised per 1,000 final-record characters, mean (s.d.)",
         "Overall GCA (n=200)": f"{rates['mean']:.1f} ({rates['sd']:.1f})", "95% CI": f"{rate_ci['ci_low']:.1f} to {rate_ci['ci_high']:.1f}"},
        {"Outcome": "Clinician-perceived utility", "Overall GCA (n=200)": "", "95% CI": ""},
    ])
    for outcome in IMPLEMENTATION_OUTCOMES:
        if outcome == "Perceived workload reduction":
            rows.append({"Outcome": "Workflow integration and adoption", "Overall GCA (n=200)": "", "95% CI": ""})
        item = overall[outcome]
        rows.append({
            "Outcome": outcome,
            "Overall GCA (n=200)": f"{item['events']}/{item['n']} ({100 * item['proportion']:.1f}%)",
            "95% CI": f"{100 * item['exact_ci'][0]:.1f}% to {100 * item['exact_ci'][1]:.1f}%",
        })
    _write(rows, output / "rct_ai_safety_and_implementation.csv")


def multicentre_characteristics(output: Path) -> None:
    data, _ = analysis.load_external()
    rows = []
    for label, frame in [("Overall", data)] + list(data.groupby("centre", sort=True)):
        rows.append({
            "Cohort": label,
            "Patient age, years": _mean_sd(frame["age"]),
            "Female sex": _count_percent(frame["female"]),
            "Han ethnicity": f"{len(frame)}/{len(frame)} (100.0%)",
            "Admissions managed by junior physicians": _n_percent(frame["senior"].eq(0)),
            "Admissions managed by senior physicians": _n_percent(frame["senior"].eq(1)),
            "Unique junior physicians": int(frame.loc[frame["senior"].eq(0), "physician_id"].nunique()),
            "Unique senior physicians": int(frame.loc[frame["senior"].eq(1), "physician_id"].nunique()),
        })
    _write(rows, output / "multicentre_characteristics.csv")


def rct_seniority_analysis(results: dict, output: Path) -> None:
    rct = results["rct"]
    gca, scw, _ = analysis.load_rct()
    rows = []

    def add(outcome, family, result, column, adjusted="", scale=1.0):
        junior_gca = gca.loc[gca["senior"].eq(0), column]
        junior_scw = scw.loc[scw["senior"].eq(0), column]
        senior_gca = gca.loc[gca["senior"].eq(1), column]
        senior_scw = scw.loc[scw["senior"].eq(1), column]
        rows.append({
            "Outcome": outcome, "Multiplicity family": family,
            "Junior GCA": _mean_sd(junior_gca), "Junior SCW": _mean_sd(junior_scw),
            "Senior GCA": _mean_sd(senior_gca), "Senior SCW": _mean_sd(senior_scw),
            "Seniority interaction (95% CI)": _effect(result["interaction"], scale),
            "Unadjusted interaction P value": _p(result["interaction"]["p"]),
            "BH-adjusted interaction P value": "" if adjusted == "" else _p(adjusted),
            "Model": "HC3-robust OLS" if scale == 1 else "HC3-robust logistic regression with model-standardised risk differences",
        })

    core_columns = {
        "Total admission workflow time": "Total Time", "Interview time": "Interview Time",
        "Physician follow-up work time": "Physician Follow-up Work Time",
        "Text editing workload": "Text Editing Workload", "PDQI-9 total score": "pdqi_total",
    }
    add("Total admission workflow time", "Primary interaction", rct["subgroup_continuous"]["Total admission workflow time"], "Total Time")
    core_names = list(core_columns)[1:]
    for name, q in zip(core_names, rct["subgroup_core_q"][:4]):
        add(name, "Core secondary interactions", rct["subgroup_continuous"][name], core_columns[name], q)
    diagnosis = rct["subgroup_diagnostic_accuracy"]
    event = "Accuracy of the initial diagnosis"
    rows.append({
        "Outcome": "Accurate preliminary diagnosis", "Multiplicity family": "Core secondary interactions",
        "Junior GCA": _count_percent(gca.loc[gca["senior"].eq(0), event]),
        "Junior SCW": _count_percent(scw.loc[scw["senior"].eq(0), event]),
        "Senior GCA": _count_percent(gca.loc[gca["senior"].eq(1), event]),
        "Senior SCW": _count_percent(scw.loc[scw["senior"].eq(1), event]),
        "Seniority interaction (95% CI)": _effect(diagnosis["interaction"], 100.0),
        "Unadjusted interaction P value": _p(diagnosis["interaction"]["p"]),
        "BH-adjusted interaction P value": _p(rct["subgroup_core_q"][4]),
        "Model": "HC3-robust logistic regression with model-standardised risk differences",
    })
    for raw_name, label, q in zip(analysis.PDQI_RCT, RCT_PDQI_DOMAINS, rct["subgroup_pdqi_q"]):
        add(label, "PDQI-9 domain interactions", rct["subgroup_pdqi"][raw_name], raw_name, q)
    for raw_name, q in zip(["Interview Fluency", "Interview Completeness"], rct["subgroup_interview_q"]):
        add(raw_name.replace("Interview ", "Interview ").lower().capitalize(),
            "Interview-quality interactions", rct["subgroup_interview"][raw_name], raw_name, q)
    patient_names = ["patient_experience"] + list(analysis.PATIENT_ITEMS)
    patient_labels = ["Patient experience score"] + PATIENT_REPORTED_ITEMS
    for name, label, q in zip(patient_names, patient_labels, rct["subgroup_patient_q"]):
        add(label, "Patient-reported interactions", rct["subgroup_patient"][name], name, q)
    _write(rows, output / "rct_seniority_effects.csv")


def multicentre_paired_analyses(results: dict, output: Path) -> None:
    ext = results["external"]
    key_names = ["Text editing workload", "PDQI-9 total score", "Final record length"]
    adjusted = dict(zip(key_names[:2], ext["key_outcome_q"]))
    rows = []
    for outcome in key_names:
        item = ext["key_outcomes"][outcome]
        decimals = 2 if outcome == "PDQI-9 total score" else 1
        rows.append({
            "Outcome": outcome, "Pairs": item["n"],
            "GCA": f"{item['gca']['mean']:.{decimals}f} ({item['gca']['sd']:.{decimals}f})",
            "SCW": f"{item['scw']['mean']:.{decimals}f} ({item['scw']['sd']:.{decimals}f})",
            "Paired GCA minus SCW (95% CI)": _effect(item),
            "Relative change": f"{100 * item['relative_change']:.1f}%",
            "Unadjusted P value": _p(item["p"]),
            "BH-adjusted P value": "" if outcome not in adjusted else _p(adjusted[outcome]),
        })
    _write(rows, output / "multicentre_overall_paired_effects.csv")

    rows = []
    for outcome in key_names:
        for centre, item in ext["centre_specific"][outcome].items():
            rows.append({"Outcome": outcome, "Centre": centre, "Pairs": item["n"],
                         "Paired GCA minus SCW (95% CI)": _effect(item),
                         "Relative change": f"{100 * item['relative_change']:.1f}%",
                         "Unadjusted P value": _p(item["p"])})
    _write(rows, output / "multicentre_centre_specific_effects.csv")

    rows = []
    key_map = {"Text editing workload": "diff_workload", "PDQI-9 total score": "diff_pdqi_total", "Final record length": "diff_final_length"}
    for outcome in key_names[:2]:
        primary = ext["key_outcomes"][outcome]
        boot = ext["centre_stratified_bootstrap"][key_map[outcome]]
        equal = ext["centre_equal_effects"][key_map[outcome]]
        leave = ext["leave_one_centre_out"][outcome]
        leave_estimates = [item["estimate"] for item in leave.values()]
        rows.append({
            "Outcome": outcome, "Primary paired effect (95% CI)": _effect(primary),
            "Centre-stratified bootstrap 95% CI": f"{boot['ci_low']:.2f} to {boot['ci_high']:.2f}",
            "Wilcoxon signed-rank P value": _p(primary["wilcoxon_p"]),
            "Centre-equal effect (95% bootstrap CI)": _effect(equal),
            "Leave-one-centre-out effect range": f"{min(leave_estimates):.2f} to {max(leave_estimates):.2f}",
        })
    _write(rows, output / "multicentre_paired_effect_sensitivity.csv")

    rows = []
    key_q = dict(zip(key_names[:2], ext["key_heterogeneity_q"]))
    for outcome in key_names:
        h = ext["centre_heterogeneity"][outcome]
        rows.append({
            "Outcome": outcome,
            "Centre heterogeneity F": h["f_statistic"],
            "Heterogeneity df": h["df_num"],
            "Unadjusted heterogeneity P value": _p(h["p"]),
            "BH-adjusted heterogeneity P value": "" if outcome not in key_q else _p(key_q[outcome]),
        })
    _write(rows, output / "multicentre_centre_heterogeneity.csv")

    rows = []
    for domain, q, hq in zip(RCT_PDQI_DOMAINS, ext["pdqi_domain_q"], ext["domain_heterogeneity_q"]):
        item = ext["pdqi_domains"][domain]
        heterogeneity = ext["domain_heterogeneity"][domain]
        rows.append({
            "PDQI-9 domain": domain, "Pairs": item["n"], "GCA": f"{item['gca']['mean']:.2f} ({item['gca']['sd']:.2f})",
            "SCW": f"{item['scw']['mean']:.2f} ({item['scw']['sd']:.2f})", "Paired GCA minus SCW (95% CI)": _effect(item),
            "Unadjusted paired P value": _p(item["p"]), "BH-adjusted paired P value": _p(q),
            "Centre heterogeneity F": heterogeneity["f_statistic"],
            "Unadjusted heterogeneity P value": _p(heterogeneity["p"]),
            "BH-adjusted heterogeneity P value": _p(hq),
        })
    _write(rows, output / "multicentre_pdqi_domains.csv")


def multicentre_seniority_analyses(results: dict, output: Path) -> None:
    ext = results["external"]
    data, _ = analysis.load_external()
    key_names = ["Text editing workload", "PDQI-9 total score", "Final record length"]
    adjusted = dict(zip(key_names[:2], ext["seniority_q"]))
    rows = []
    for outcome in key_names:
        item = ext["seniority"][outcome]
        sensitivity = ext["seniority_age_sex_adjusted"][outcome]
        rows.append({
            "Outcome": outcome,
            "Junior admissions": int(data["senior"].eq(0).sum()),
            "Junior paired effect (95% CI)": _effect(item["junior"]),
            "Senior admissions": int(data["senior"].eq(1).sum()),
            "Senior paired effect (95% CI)": _effect(item["senior"]),
            "Seniority difference (95% CI)": _effect(item["difference"]),
            "Unadjusted interaction P value": _p(item["difference"]["p"]),
            "BH-adjusted interaction P value": "" if outcome not in adjusted else _p(adjusted[outcome]),
            "Age- and sex-adjusted difference (95% CI)": _effect(sensitivity["difference"]),
            "Age- and sex-adjusted P value": _p(sensitivity["difference"]["p"]),
        })
    _write(rows, output / "multicentre_seniority_effects.csv")
    rows = []
    for condition, item in ext["condition_pdqi_seniority_gaps"].items():
        column = "scw_pdqi_total" if condition == "SCW" else "gca_pdqi_total"
        junior_values = data.loc[data["senior"].eq(0), column]
        senior_values = data.loc[data["senior"].eq(1), column]
        rows.append({
            "Condition": condition,
            "Junior mean (s.d.)": f"{junior_values.mean():.2f} ({junior_values.std(ddof=1):.2f})",
            "Senior mean (s.d.)": f"{senior_values.mean():.2f} ({senior_values.std(ddof=1):.2f})",
            "Senior minus junior (95% CI)": _effect(item["difference"]),
            "Unadjusted P value": _p(item["difference"]["p"]),
        })
    _write(rows, output / "multicentre_pdqi_seniority_gaps.csv")


def multicentre_safety_analyses(results: dict, output: Path) -> None:
    safety = results["external"]["draft_safety"]
    groups = [("Overall", safety)] + list(safety["by_centre"].items())
    blank = {name: "" for name, _ in groups}
    rows = [{"Outcome": CLINICAL_ERROR_SEVERITY_MULTICENTRE_HEADING, **blank}]
    for level, label in enumerate(CLINICAL_ERROR_SEVERITY_LEVELS):
        values = {}
        for name, item in groups:
            severity = item["clinical_error_severity"]["levels"][str(level)]
            values[name] = f"{severity['count']} ({100 * severity['proportion']:.1f}%)"
        rows.append({"Outcome": label, **values})
    rows.append({"Outcome": CHARACTER_LEVEL_FACTUAL_CONFLICT_HEADING, **blank})
    rows.append({"Outcome": "GCA-generated drafts, n", **{name: item["n"] for name, item in groups}})
    rows.append({
        "Outcome": "Factually conflicting characters per GCA-generated draft, mean (s.d.)",
        **{name: f"{item['conflicting_characters']['mean']:.1f} ({item['conflicting_characters']['sd']:.1f})" for name, item in groups},
    })
    rows.append({
        "Outcome": "Factually conflicting characters per GCA-generated draft, median (IQR)",
        **{name: f"{item['conflicting_characters']['median']:.1f} ({item['conflicting_characters']['q1']:.1f} to {item['conflicting_characters']['q3']:.1f})" for name, item in groups},
    })
    rows.append({
        "Outcome": "Drafts containing at least one factually conflicting character, n/N (%) (95% CI)",
        **{name: (
            f"{item['drafts_with_any_detected_conflict']}/{item['n']} "
            f"({100 * item['proportion_with_any_detected_conflict']:.1f}%) "
            f"({100 * item['exact_ci'][0]:.1f}% to {100 * item['exact_ci'][1]:.1f}%)"
        ) for name, item in groups},
    })
    rows.append({
        "Outcome": "Factually conflicting characters per 1,000 final-record characters, mean (s.d.)",
        **{name: f"{item['conflicts_per_1000_final_characters']['mean']:.2f} ({item['conflicts_per_1000_final_characters']['sd']:.2f})" for name, item in groups},
    })
    _write(rows, output / "multicentre_draft_safety.csv")
    q = safety["seniority_bh_q"]
    logistic = safety["seniority_any_conflict"]
    count = safety["seniority_count_rate"]
    rows = [
        {"Outcome": "Drafts containing at least 1 factually conflicting character", "Effect measure": "Centre heterogeneity statistic",
         "Estimate (95% CI)": f"chi-squared={safety['centre_logistic_omnibus']['chi2']:.2f}; df={safety['centre_logistic_omnibus']['df']}",
         "Unadjusted P value": _p(safety["centre_logistic_omnibus"]["p"]), "BH-adjusted P value": "",
         "Model": "HC3-robust omnibus Wald test"},
        {"Outcome": "Drafts containing at least 1 factually conflicting character", "Effect measure": "Senior-minus-junior adjusted risk difference",
         "Estimate (95% CI)": (
             f"{100 * logistic['risk_difference']:.2f} "
             f"({100 * logistic['ci_low']:.2f} to {100 * logistic['ci_high']:.2f}) percentage points"
         ), "Unadjusted P value": _p(logistic["p"]),
         "BH-adjusted P value": _p(q[0]), "Model": "Centre-adjusted binomial logistic regression"},
        {"Outcome": "Factually conflicting characters per 1,000 final-record characters", "Effect measure": "Senior-to-junior incidence-rate ratio",
         "Estimate (95% CI)": f"{count['irr']:.2f} ({count['ci_low']:.2f} to {count['ci_high']:.2f})",
         "Unadjusted P value": _p(count["p"]), "BH-adjusted P value": _p(q[1]),
         "Model": "Centre-adjusted negative-binomial regression"},
    ]
    _write(rows, output / "multicentre_safety_inference.csv")


def sensitivity_tables(results: dict, output: Path) -> None:
    rows = []
    for raw_name, item in results["rct"]["ordinal_logistic_sensitivity"].items():
        label = "Internally consistent" if raw_name == "Internally Consistent" else raw_name
        rows.append({"Outcome": label, "Multiplicity family": item["family"],
                     "Interaction odds ratio": item["interaction_or"], "CI low": item["ci_low"],
                     "CI high": item["ci_high"], "Unadjusted P value": item["p"],
                     "BH-adjusted P value": item["q"], "Converged": item["converged"], "Valid": item["valid"]})
    _write(rows, output / "rct_ordinal_sensitivity.csv")
    rows = []
    for domain, fits in results["external"]["ordinal_gee_sensitivity"].items():
        for estimand, item in fits.items():
            rows.append({"PDQI-9 domain": domain, "Estimand": estimand, "Odds ratio": item["or"],
                         "CI low": item["ci_low"], "CI high": item["ci_high"],
                         "Unadjusted P value": item["p"], "BH-adjusted P value": item["q"],
                         "Converged": item["converged"], "Valid": item["valid"]})
    _write(rows, output / "multicentre_ordinal_gee_sensitivity.csv")


def physician_cluster_sensitivity_tables(results: dict, output: Path) -> None:
    column = "Effect estimate (physician-clustered 95% CI)"
    rct = results["rct"]["physician_cluster_sensitivity"]
    rct_outcomes = [
        ("Total admission workflow time, s", "Total admission workflow time", 1.0),
        ("Interview time, s", "Interview time", 1.0),
        ("Physician follow-up work time, s", "Physician follow-up work time", 1.0),
        ("Text editing workload, characters", "Text editing workload", 1.0),
        ("PDQI-9 total score", "PDQI-9 total score", 1.0),
        ("Accurate preliminary diagnosis, pp", "Accurate preliminary diagnosis", 100.0),
    ]
    rows = []
    for target, key in [
        ("Overall treatment effect", "overall"),
        ("Treatment-by-seniority interaction", "seniority_interaction"),
    ]:
        for label, outcome, scale in rct_outcomes:
            rows.append({
                "Analysis target": target,
                "Outcome": label,
                column: _effect(rct[key][outcome], scale=scale),
            })
    _write(rows, output / "rct_physician_clustered_sensitivity.csv")

    external = results["external"]["physician_cluster_sensitivity"]
    external_outcomes = [
        ("Text editing workload, characters", "Text editing workload"),
        ("PDQI-9 total score", "PDQI-9 total score"),
        ("Final record length, characters", "Final record length"),
    ]
    rows = []
    for target, key in [
        ("Overall paired effect", "overall"),
        ("Senior-minus-junior difference in paired effects", "seniority_difference"),
    ]:
        for label, outcome in external_outcomes:
            rows.append({
                "Analysis target": target,
                "Outcome": label,
                column: _effect(external[key][outcome]),
            })
    _write(rows, output / "multicentre_physician_clustered_sensitivity.csv")


def rct_physician_clustering_outcomes(results: dict, output: Path) -> None:
    rct = results["rct"]
    clustered = rct["physician_cluster_sensitivity"]["patient_reported"]
    patient_outcomes = [
        ("Patient experience score", "patient_experience"),
        ("Communication convenience", "Communication Convenience"),
        ("Perceived physician attention", "Perceived Physician Attention"),
        ("Perceived empathy", "Perceived Empathy"),
        ("Overall satisfaction", "Overall Satisfaction"),
    ]
    rows = []
    for label, key in patient_outcomes:
        conventional = rct["patient_outcomes"][key]
        sensitivity = clustered[label]
        displayed_estimate = f"{conventional['estimate']:.2f}"
        rows.append({
            "Section": "A. Patient-reported outcomes",
            "Outcome": label,
            "Conventional patient-level estimate": displayed_estimate,
            "Physician-clustered estimate": displayed_estimate,
            "Physician-clustered 95% CI": f"{sensitivity['ci_low']:.2f} to {sensitivity['ci_high']:.2f}",
            "Physician-clustered P value": _p(sensitivity["p"]),
            "Physician-level ICC": "",
        })
    for label, result in rct["physician_icc"]["continuous"].items():
        rows.append({
            "Section": "B. Physician-level ICCs",
            "Outcome": label,
            "Conventional patient-level estimate": "",
            "Physician-clustered estimate": "",
            "Physician-clustered 95% CI": "",
            "Physician-clustered P value": "",
            "Physician-level ICC": "<0.0001" if result["icc"] < 0.0001 else f"{result['icc']:.4f}",
        })
    binary = rct["physician_icc"]["binary"]["Accurate preliminary diagnosis"]
    rows.append({
        "Section": "B. Physician-level ICCs",
        "Outcome": "Accurate preliminary diagnosis (observed binary scale)",
        "Conventional patient-level estimate": "",
        "Physician-clustered estimate": "",
        "Physician-clustered 95% CI": "",
        "Physician-clustered P value": "",
        "Physician-level ICC": "<0.0001" if binary["icc"] < 0.0001 else f"{binary['icc']:.4f}",
    })
    _write(rows, output / "rct_physician_clustering_outcomes.csv")


def write_all(results: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rct_baseline_characteristics(output)
    rct_outcomes(results, output)
    rct_ai_safety_and_implementation(results, output)
    multicentre_characteristics(output)
    rct_seniority_analysis(results, output)
    multicentre_paired_analyses(results, output)
    multicentre_seniority_analyses(results, output)
    multicentre_safety_analyses(results, output)
    sensitivity_tables(results, output)
    physician_cluster_sensitivity_tables(results, output)
    rct_physician_clustering_outcomes(results, output)
    manifest = {"generated_tables": sorted(path.name for path in output.glob("*.csv"))}
    (output / "table_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
