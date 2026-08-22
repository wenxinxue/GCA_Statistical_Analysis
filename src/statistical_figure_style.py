from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from . import analysis
from .terminology import IMPLEMENTATION_OUTCOMES, RCT_PDQI_DOMAINS


COLOR_SCW = "#4575B4"
COLOR_GCA = "#D73027"
COLOR_SENIOR = "#1F4E79"
COLOR_JUNIOR = "#C00000"
COLOR_NS = "#B0B0B0"
COLOR_OVERALL = "#8B1A1A"
COLOR_TEXT = "#202020"
COLOR_MUTED = "#666666"
COLOR_GRID = "#E8E8E8"
COLOR_ZERO = "#777777"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    svg: bool = False,
    tight: bool = True,
    dpi: int = 300,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = {"bbox_inches": "tight", "pad_inches": 0.15} if tight else {}
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, facecolor="white", **layout)
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white", **layout)
    if svg:
        fig.savefig(output_dir / f"{stem}.svg", facecolor="white", **layout)
    plt.close(fig)


def p_text(value: float, *, adjusted: bool = False) -> str:
    prefix = "BH-adjusted P" if adjusted else "P"
    return f"{prefix}<0.001" if value < 0.001 else f"{prefix}={value:.3f}"


def q_text(value: float) -> str:
    return "q<0.001" if value < 0.001 else f"q={value:.3f}"


def mean_t_ci(values) -> tuple[float, float, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x)))
    critical = float(stats.t.ppf(0.975, len(x) - 1))
    return mean, mean - critical * sem, mean + critical * sem


def draw_diamond(ax, estimate: float, low: float, high: float, y: float, color: str) -> None:
    height = 0.17
    ax.fill(
        [low, estimate, high, estimate],
        [y, y + height, y, y - height],
        color=color,
        edgecolor=color,
        zorder=3,
    )


def rct_outcome_panels(results: dict, output_dir: Path) -> None:
    gca, scw, _ = analysis.load_rct()
    specs = [
        ("a", "Total admission workflow time", "Total Time", "s", "Primary outcome", None),
        ("b", "Interview time", "Interview Time", "s", "Secondary outcome", results["rct"]["core_secondary_q"][0]),
        ("c", "Physician follow-up work time", "Physician Follow-up Work Time", "s", "Secondary outcome", results["rct"]["core_secondary_q"][1]),
        ("d", "Text editing workload", "Text Editing Workload", "characters", "Secondary outcome", results["rct"]["core_secondary_q"][2]),
        ("e", "PDQI-9 total score", "pdqi_total", "points", "Secondary outcome", results["rct"]["core_secondary_q"][3]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 13.5))
    for seed, (ax, (letter, title, column, unit, outcome_class, adjusted_p)) in enumerate(zip(axes.flat[:5], specs), 10):
        control = scw[column].dropna().to_numpy(float)
        treatment = gca[column].dropna().to_numpy(float)
        item = results["rct"]["core"][title]
        bp = ax.boxplot(
            [control, treatment], positions=[0, 1], widths=0.46,
            patch_artist=True, showfliers=False,
            boxprops=dict(linewidth=0.9), medianprops=dict(color="#222222", linewidth=1.2),
            whiskerprops=dict(color="#555555", linewidth=0.8), capprops=dict(color="#555555", linewidth=0.8),
        )
        for patch, color in zip(bp["boxes"], [COLOR_SCW, COLOR_GCA]):
            patch.set_facecolor(color); patch.set_alpha(0.24); patch.set_edgecolor(color)
        rng = np.random.default_rng(seed)
        ax.scatter(rng.normal(0, 0.055, len(control)), control, s=9, color=COLOR_SCW, alpha=0.32, linewidths=0)
        ax.scatter(rng.normal(1, 0.055, len(treatment)), treatment, s=9, color=COLOR_GCA, alpha=0.32, linewidths=0)
        ax.set_xticks([0, 1], [f"SCW\nn={len(control)}", f"GCA\nn={len(treatment)}"])
        ax.set_xlim(-0.42, 1.42); ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=22)
        ax.text(-0.12, 1.12, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        ax.text(0, 1.015, outcome_class, transform=ax.transAxes, fontsize=8.5,
                color=COLOR_GCA if outcome_class == "Primary outcome" else COLOR_MUTED)
        shown_p = item["p"] if adjusted_p is None else adjusted_p
        ax.text(
            0.98, 0.97,
            f"Mean difference {item['estimate']:+.1f} {unit}\n"
            f"95% CI {item['ci_low']:+.1f} to {item['ci_high']:+.1f}\n"
            f"{p_text(shown_p, adjusted=adjusted_p is not None)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.5),
        )
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)

    ax = axes.flat[5]
    d = results["rct"]["diagnosis"]
    counts = [int(scw["Accuracy of the initial diagnosis"].sum()), int(gca["Accuracy of the initial diagnosis"].sum())]
    totals = [len(scw), len(gca)]
    props = np.asarray(counts) / totals
    cis = [proportion_confint(k, n, method="beta") for k, n in zip(counts, totals)]
    yerr = np.asarray([[props[i] - cis[i][0] for i in range(2)], [cis[i][1] - props[i] for i in range(2)]]) * 100
    bars = ax.bar([0, 1], props * 100, width=0.52, color=[COLOR_SCW, COLOR_GCA], alpha=0.76,
                  yerr=yerr, capsize=4, error_kw=dict(linewidth=0.9))
    for index, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yerr[1, index] + 1.2,
                f"{counts[index]}/{totals[index]}\n({100 * props[index]:.1f}%)", ha="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["SCW", "GCA"]); ax.set_ylabel("Participants with accurate diagnosis (%)"); ax.set_ylim(0, 135)
    ax.set_title("Accurate preliminary diagnosis", loc="left", fontsize=11, fontweight="bold", pad=22)
    ax.text(-0.12, 1.12, "f", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0, 1.015, "Secondary outcome", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED)
    ax.text(0.98, 0.97,
            f"Risk difference {100 * d['risk_difference']:+.1f} pp\n"
            f"95% CI {100 * d['ci_low']:+.1f} to {100 * d['ci_high']:+.1f}\n"
            f"{p_text(results['rct']['core_secondary_q'][4], adjusted=True)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.5))
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.06, hspace=0.48, wspace=0.28)
    save(fig, output_dir, "rct_outcome_panels")


def rct_effect_forest(results: dict, output_dir: Path) -> None:
    order = ["Total admission workflow time", "Interview time", "Physician follow-up work time", "Text editing workload", "PDQI-9 total score"]
    q_values = [None] + results["rct"]["core_secondary_q"][:4]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [2.25, 1.0]})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    for yi, name, shown_q in zip(y, order, q_values):
        item = results["rct"]["hedges_g"][name]
        primary = shown_q is None
        significant = item["p"] < 0.05 if "p" in item else not (item["ci_low"] <= 0 <= item["ci_high"])
        color = COLOR_GCA if primary else (COLOR_SENIOR if significant else COLOR_NS)
        ax.plot([item["ci_low"], item["ci_high"]], [yi, yi], color=color, lw=1.5)
        ax.scatter(item["estimate"], yi, s=85, marker="s", facecolor=color if significant else "white", edgecolor=color, lw=1.1)
        ax.text(-0.03, yi + 0.10, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, fontweight="bold" if primary else "normal")
        ax.text(-0.03, yi - 0.23, "Primary outcome" if primary else "Secondary outcome",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                color=COLOR_GCA if primary else COLOR_MUTED)
        shown_p = results["rct"]["core"][name]["p"] if primary else shown_q
        label = p_text(shown_p, adjusted=not primary)
        ax.text(2.42, yi + 0.11, f"{item['estimate']:+.2f} ({item['ci_low']:+.2f} to {item['ci_high']:+.2f}); {label}",
                ha="right", va="bottom", fontsize=7.5, color=color if significant else COLOR_MUTED)
        ax.axhline(yi - 0.5, color="#EFEFEF", lw=0.6)
    ax.axvline(0, color=COLOR_ZERO, lw=0.8, ls="--"); ax.set_xlim(-3, 2.5); ax.set_ylim(-0.7, 4.3)
    ax.set_yticks([]); ax.set_xlabel("Hedges’ g (95% CI)")
    ax.set_title("Continuous outcomes", loc="left", fontsize=11, fontweight="bold", pad=20)
    ax.text(-0.31, 1.11, "a", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0, 1.02, "GCA minus SCW", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED)
    ax.spines["left"].set_visible(False); ax.grid(axis="x", color=COLOR_GRID, lw=0.6)

    ax = axes[1]; d = results["rct"]["diagnosis"]; est, low, high = [100 * d[x] for x in ["risk_difference", "ci_low", "ci_high"]]
    ax.plot([low, high], [0, 0], color=COLOR_GCA, lw=1.8); ax.scatter(est, 0, s=95, marker="s", color=COLOR_GCA, edgecolor="white")
    ax.axvline(0, color=COLOR_ZERO, lw=0.8, ls="--")
    ax.text(0, 0.71, "Accurate preliminary diagnosis", transform=ax.transAxes, fontsize=9.5, fontweight="bold")
    ax.text(0, 0.64, "Secondary outcome", transform=ax.transAxes, fontsize=7.5, color=COLOR_MUTED)
    ax.text(0, 0.26, f"Risk difference {est:+.1f} pp\n95% CI {low:+.1f} to {high:+.1f}\n{p_text(results['rct']['core_secondary_q'][4], adjusted=True)}",
            transform=ax.transAxes, fontsize=9, linespacing=1.35)
    ax.set_xlim(-4, 26); ax.set_ylim(-0.65, 0.65); ax.set_yticks([]); ax.set_xlabel("Risk difference (percentage points)")
    ax.set_title("Binary outcome", loc="left", fontsize=11, fontweight="bold", pad=20)
    ax.text(-0.12, 1.11, "b", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0, 1.02, "GCA minus SCW", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED)
    ax.spines["left"].set_visible(False); ax.grid(axis="x", color=COLOR_GRID, lw=0.6)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.88, bottom=0.15, wspace=0.32)
    save(fig, output_dir, "rct_effect_panels")


def rct_pdqi_radar(results: dict, output_dir: Path) -> None:
    gca, scw, _ = analysis.load_rct()
    labels = RCT_PDQI_DOMAINS
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
    scw_values = [float(scw[x].mean()) for x in analysis.PDQI_RCT]; scw_values += scw_values[:1]
    gca_values = [float(gca[x].mean()) for x in analysis.PDQI_RCT]; gca_values += gca_values[:1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    ax.set_facecolor("#FAFAFA"); ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5]); ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color="#999999")
    ax.set_rlabel_position(80); ax.grid(color="#E0E0E0", lw=0.4); ax.spines["polar"].set_color("#CCCCCC")
    ax.fill(angles, scw_values, alpha=0.12, color=COLOR_SCW); ax.plot(angles, scw_values, color=COLOR_SCW, lw=2.2, ls="--", label="SCW")
    ax.scatter(angles[:-1], scw_values[:-1], c=COLOR_SCW, s=55, edgecolors="white", lw=1, zorder=10)
    ax.fill(angles, gca_values, alpha=0.15, color=COLOR_GCA); ax.plot(angles, gca_values, color=COLOR_GCA, lw=2.5, label="GCA")
    ax.scatter(angles[:-1], gca_values[:-1], c=COLOR_GCA, s=65, edgecolors="white", lw=1, zorder=11)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    for index, (text, q_value) in enumerate(zip(ax.get_xticklabels(), results["rct"]["pdqi_domain_q"])):
        delta = gca_values[index] - scw_values[index]
        text.set_color(COLOR_GCA if delta > 0 else COLOR_SCW)
        if q_value < 0.05:
            marker = "***" if q_value < 0.001 else ("**" if q_value < 0.01 else "*")
            ax.annotate(marker, xy=(angles[index], max(gca_values[index], scw_values[index]) + 0.25),
                        fontsize=8, fontweight="bold", color=text.get_color(), ha="center")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=True, fontsize=9)
    ax.set_title("PDQI-9 domain scores", fontsize=13, fontweight="bold", pad=28)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.88, bottom=0.08)
    save(fig, output_dir, "rct_pdqi_domain_panel")


def attention_association(results: dict, output_dir: Path) -> None:
    gca, scw, combined = analysis.load_rct()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1))
    ax = axes[0]; rng = np.random.default_rng(2407)
    for group, frame, color in [("SCW", scw, COLOR_SCW), ("GCA", gca, COLOR_GCA)]:
        plot_data = frame[["Patient Attention Time", "patient_experience"]].dropna()
        y = plot_data["patient_experience"].to_numpy(float) + rng.normal(0, 0.10, len(plot_data))
        ax.scatter(plot_data["Patient Attention Time"], y, s=18, color=color, alpha=0.34, linewidths=0, label=group)
        fit = smf.ols("patient_experience ~ Q('Patient Attention Time')", data=plot_data).fit(cov_type="HC3")
        grid = np.linspace(plot_data["Patient Attention Time"].quantile(0.01), plot_data["Patient Attention Time"].quantile(0.99), 120)
        pred = fit.get_prediction(pd.DataFrame({"Patient Attention Time": grid})).summary_frame(alpha=0.05)
        ax.plot(grid, pred["mean"], color=color, lw=1.8); ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color=color, alpha=0.12)
    correlations = results["rct"]["attention_correlations"]
    label = "\n".join(f"{name}: ρ={correlations[key]['rho']:.3f}, {p_text(correlations[key]['p'])}"
                       for name, key in [("Overall cohort", "overall"), ("SCW", "SCW"), ("GCA", "GCA")])
    ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="#D9D9D9", lw=0.6, alpha=0.92, pad=4))
    ax.set_xlabel("Patient Attention Time (s)"); ax.set_ylabel("Patient experience score (4-20)")
    ax.set_xlim(left=0); ax.set_ylim(7.5, 20.7); ax.set_yticks(np.arange(8, 21, 2)); ax.grid(color=COLOR_GRID, lw=0.6)
    ax.set_title("Observed association", loc="left", fontsize=11, fontweight="bold", pad=18)
    ax.text(-0.10, 1.08, "a", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top"); ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    model_data = combined.assign(attention_per_60s=combined["Patient Attention Time"] / 60)
    fit = smf.ols("patient_experience ~ attention_per_60s + treatment + senior + Q('Patient Age') + female", data=model_data).fit(cov_type="HC3")
    grid = np.linspace(model_data["Patient Attention Time"].quantile(0.01), model_data["Patient Attention Time"].quantile(0.99), 160)
    for group, treatment, color in [("SCW", 0, COLOR_SCW), ("GCA", 1, COLOR_GCA)]:
        pred_data = pd.DataFrame({"attention_per_60s": grid / 60, "treatment": treatment,
                                  "senior": model_data["senior"].mean(), "Patient Age": model_data["Patient Age"].mean(),
                                  "female": model_data["female"].mean()})
        pred = fit.get_prediction(pred_data).summary_frame(alpha=0.05)
        ax.plot(grid, pred["mean"], color=color, lw=2.1, label=group); ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color=color, alpha=0.14)
    item = results["rct"]["attention_hc3_regression"]
    ax.text(0.03, 0.97, f"Adjusted change per 60 s: {item['estimate_per_60_s']:+.2f} points\n95% CI {item['ci_low']:+.2f} to {item['ci_high']:+.2f}\n{p_text(item['p'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#D9D9D9", lw=0.6, alpha=0.92, pad=4))
    ax.text(0.03, 0.04, "Adjusted for treatment group, physician seniority,\npatient age, and sex",
            transform=ax.transAxes, fontsize=8, color=COLOR_MUTED)
    ax.set_xlabel("Patient Attention Time (s)"); ax.set_ylabel("Adjusted Patient experience score")
    ax.set_xlim(left=0); ax.set_ylim(7.5, 20.7); ax.set_yticks(np.arange(8, 21, 2)); ax.grid(color=COLOR_GRID, lw=0.6)
    ax.set_title("Adjusted association", loc="left", fontsize=11, fontweight="bold", pad=18)
    ax.text(-0.10, 1.08, "b", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top"); ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.14, wspace=0.28)
    save(fig, output_dir, "rct_attention_panels")


def editing_profile(output_dir: Path) -> None:
    gca, _, _ = analysis.load_rct()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    specs = [("Deleted Characters", "Deleted characters"), ("Factually Conflicting Characters", "Factually conflicting characters"), ("Added Characters", "Added characters")]
    data = [gca[column].dropna() for column, _ in specs]
    bp = ax.boxplot(data, tick_labels=[label for _, label in specs], patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.1))
    for patch, color in zip(bp["boxes"], [COLOR_GCA, "#404040", COLOR_SCW]):
        patch.set_facecolor(color); patch.set_alpha(0.18); patch.set_edgecolor(color)
    rng = np.random.default_rng(7)
    for index, values in enumerate(data, 1):
        ax.scatter(rng.normal(index, 0.05, len(values)), values, s=11, color="#666666", alpha=0.25, edgecolors="white", linewidths=0.2)
    ax.set_ylabel("Characters"); ax.set_title("GCA-assisted editing components", fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelrotation=18); ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    ax = axes[1]
    pairs = gca[["Total Record Word Count Before Editing", "Total Record Word Count After Editing"]].dropna()
    means = pairs.mean()
    ci = pairs.std(ddof=1) / np.sqrt(len(pairs)) * stats.t.ppf(0.975, len(pairs) - 1)
    ax.errorbar([0, 1], means, yerr=ci, color=COLOR_GCA, marker="o", lw=1.8, capsize=4)
    for _, row in pairs.sample(min(len(pairs), 80), random_state=1).iterrows():
        ax.plot([0, 1], row, color="#BBBBBB", lw=0.4, alpha=0.25)
    ax.set_xticks([0, 1], ["GCA-generated draft", "GCA-assisted final record"])
    ax.set_ylabel("Final record length (characters)"); ax.set_title("Final record length before and after clinician editing", fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.20, wspace=0.35)
    save(fig, output_dir, "rct_editing_panels")


def rct_seniority(results: dict, output_dir: Path) -> None:
    names = ["Total admission workflow time", "Interview time", "Physician follow-up work time", "Text editing workload", "PDQI-9 total score", "Accurate preliminary diagnosis"]
    titles = ["Total admission workflow time", "Interview time", "Physician follow-up work time", "Text editing workload", "PDQI-9 total score", "Accurate preliminary diagnosis"]
    units = ["s", "s", "s", "characters", "points", "percentage points"]
    q_values = dict(zip(names[1:], results["rct"]["subgroup_core_q"]))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for panel, (ax, name, title, unit) in enumerate(zip(axes.flat, names, titles, units)):
        item = results["rct"]["subgroup_diagnostic_accuracy"] if name == names[-1] else results["rct"]["subgroup_continuous"][name]
        scale = 100 if name == names[-1] else 1
        for y, subgroup, color, label in [(1, "junior", COLOR_JUNIOR, "Junior physicians"), (0, "senior", COLOR_SENIOR, "Senior physicians")]:
            estimate = scale * item[subgroup]["estimate"]; low = scale * item[subgroup]["ci_low"]; high = scale * item[subgroup]["ci_high"]
            ax.plot([low, high], [y, y], color=color, lw=2); ax.scatter(estimate, y, s=85, marker="s", color=color, edgecolor="white", lw=0.8)
            ax.text(0.01, y + 0.18, label, transform=ax.get_yaxis_transform(), fontsize=8.5, color=color, fontweight="bold")
        inter = item["interaction"]; shown_p = inter["p"] if panel == 0 else q_values[name]
        label = p_text(shown_p) if panel == 0 else q_text(shown_p)
        ax.text(0.01, -0.30, f"Difference in effects: {scale * inter['estimate']:+.1f} ({scale * inter['ci_low']:.1f} to {scale * inter['ci_high']:.1f}); {label}",
                transform=ax.transAxes, fontsize=8.2, color="#333333")
        values = [0, scale * item["junior"]["ci_low"], scale * item["junior"]["ci_high"], scale * item["senior"]["ci_low"], scale * item["senior"]["ci_high"]]
        span = max(values) - min(values); margin = max(0.08 * span, 0.25)
        ax.set_xlim(min(values) - margin, max(values) + margin); ax.set_ylim(-0.55, 1.55); ax.axvline(0, color=COLOR_ZERO, lw=0.9, ls="--")
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel(f"GCA minus SCW ({unit})", fontsize=8.8)
        ax.set_title(f"{chr(97 + panel)}  {title}", loc="left", fontsize=10.5, fontweight="bold", pad=7,
                     color=COLOR_GCA if panel == 0 else COLOR_TEXT)
        ax.text(1, 1.05, "Primary" if panel == 0 else "Secondary", transform=ax.transAxes, ha="right", fontsize=7.5, color=COLOR_MUTED)
    fig.suptitle("Treatment effects by physician seniority", fontsize=13, fontweight="bold", y=0.997)
    fig.text(0.5, 0.958, "Interaction estimand: senior treatment effect minus junior treatment effect. Lines show 95% confidence intervals.", ha="center", fontsize=8.2, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.07, hspace=0.64, wspace=0.27)
    save(fig, output_dir, "rct_seniority_panel")


def multicentre_four_panel(results: dict, output_dir: Path) -> None:
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.4,
        "axes.titlesize": 8.2,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "axes.unicode_minus": False,
    })
    external = results["external"]
    external_data, _ = analysis.load_external()
    centres = ["Centre 1", "Centre 2", "Centre 3", "Centre 4", "Centre 5"]
    centre_n = external_data.groupby("centre", sort=False).size().astype(int).to_dict()
    overall_n = int(len(external_data))
    heterogeneity_q = dict(zip(
        ["Text editing workload", "PDQI-9 total score"],
        external["key_heterogeneity_q"],
    ))
    domain_order = ["Thorough", "Useful", "Synthesized", "Organized", "Comprehensible", "Succinct", "Up-to-date", "Accurate", "Internally consistent"]
    fig = plt.figure(figsize=(7.0866, 6.6929), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=0.122, right=0.985, top=0.958, bottom=0.095,
                             wspace=0.31, hspace=0.47, height_ratios=[1, 1.36])

    def panel_axes(cell):
        grid = cell.subgridspec(1, 2, width_ratios=[2.05, 1.25], wspace=0.08)
        return fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    def style_axis(ax):
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", width=0.55, length=3); ax.grid(axis="x", color=COLOR_GRID, lw=0.45); ax.set_axisbelow(True)

    def forest(ax, text_ax, outcome, title, xlabel, xlim, ticks, letter, left_label, right_label, decimals):
        rows = external["centre_specific"][outcome]
        overall = external["key_outcomes"][outcome]
        labels = centres + ["Overall"]; y = np.arange(6)[::-1]
        ax.axvline(0, color=COLOR_ZERO, ls="--", lw=0.75)
        for yi, label in zip(y, labels):
            item = overall if label == "Overall" else rows[label]
            if label == "Overall":
                draw_diamond(ax, item["estimate"], item["ci_low"], item["ci_high"], yi, COLOR_OVERALL)
            else:
                ax.plot([item["ci_low"], item["ci_high"]], [yi, yi], color=COLOR_GCA, lw=1.25)
                ax.scatter(item["estimate"], yi, s=24, marker="s", color=COLOR_GCA, edgecolor="white", lw=0.45)
                ax.axhline(yi - 0.5, color="#F1F1F1", lw=0.42)
        tick_labels = [f"{c}  (n={centre_n[c]})" for c in centres] + [f"Overall  (n={overall_n})"]
        ax.set_yticks(y, tick_labels); ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.set_xlim(*xlim); ax.set_xticks(ticks); ax.set_ylim(-0.72, 5.75); ax.set_xlabel(xlabel, labelpad=4)
        ax.set_title(title, loc="left", fontweight="bold", pad=12); ax.text(-0.33, 1.14, letter, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")
        style_axis(ax)
        ax.text(0, -0.22, left_label, transform=ax.transAxes, ha="left", va="top", fontsize=5.7,
                color=COLOR_GCA if "GCA" in left_label else COLOR_SCW)
        ax.text(1, -0.22, right_label, transform=ax.transAxes, ha="right", va="top", fontsize=5.7,
                color=COLOR_GCA if "GCA" in right_label else COLOR_SCW)
        adjusted_heterogeneity = heterogeneity_q[outcome]
        heterogeneity_label = p_text(adjusted_heterogeneity, adjusted=True).replace(
            "P", "P for heterogeneity", 1
        )
        ax.text(0, -0.35, heterogeneity_label, transform=ax.transAxes, fontsize=5.7, color=COLOR_MUTED)
        text_ax.set_ylim(ax.get_ylim()); text_ax.axis("off")
        text_ax.text(0, 5.97, "Difference (95% CI)", fontsize=5.8, fontweight="bold", va="bottom")
        fmt = f"{{:+.{decimals}f}} ({{:+.{decimals}f}} to {{:+.{decimals}f}})"
        for yi, label in zip(y, labels):
            item = overall if label == "Overall" else rows[label]
            text_ax.text(0, yi, fmt.format(item["estimate"], item["ci_low"], item["ci_high"]), fontsize=5.75, va="center",
                         fontweight="bold" if label == "Overall" else "normal", color=COLOR_OVERALL if label == "Overall" else COLOR_TEXT)

    ax_a, tx_a = panel_axes(outer[0, 0]); ax_b, tx_b = panel_axes(outer[0, 1])
    ax_c, tx_c = panel_axes(outer[1, 0]); ax_d, tx_d = panel_axes(outer[1, 1])
    forest(ax_a, tx_a, "Text editing workload", "Text editing workload", "Paired difference, characters",
           (-850, 30), [-800, -600, -400, -200, 0], "a", "Favours GCA", "Favours SCW", 1)
    forest(ax_b, tx_b, "PDQI-9 total score", "PDQI-9 total score", "Paired difference, points",
           (-0.05, 4.05), [0, 1, 2, 3, 4], "b", "Favours SCW", "Favours GCA", 2)

    y = np.arange(len(domain_order))[::-1]
    ax_c.axvline(0, color=COLOR_ZERO, ls="--", lw=0.75)
    for yi, domain in zip(y, domain_order):
        item = external["pdqi_domains"][domain]; q_value = external["pdqi_domain_q"][analysis.PDQI_EXTERNAL.index(domain)]
        color = COLOR_GCA if q_value < 0.05 and item["estimate"] > 0 else (COLOR_SCW if q_value < 0.05 else COLOR_NS)
        ax_c.plot([item["ci_low"], item["ci_high"]], [yi, yi], color=color, lw=1.2)
        ax_c.scatter(item["estimate"], yi, s=23, marker="s", facecolor=color if q_value < 0.05 else "white", edgecolor=color, lw=0.85)
        ax_c.axhline(yi - 0.5, color="#F1F1F1", lw=0.42)
    for sep in [5.5, 2.5]: ax_c.axhline(sep, color="#D6D6D6", lw=0.65)
    ax_c.set_yticks(y, domain_order)
    ax_c.set_xlim(-1.25, 1.02); ax_c.set_xticks([-1, -0.5, 0, 0.5, 1]); ax_c.set_ylim(-0.72, 8.75)
    ax_c.set_xlabel("Paired mean difference, points", labelpad=4); ax_c.set_title("Quality-domain trade-offs", loc="left", fontweight="bold", pad=12)
    ax_c.text(-0.33, 1.14, "c", transform=ax_c.transAxes, fontsize=8.2, fontweight="bold", va="top"); style_axis(ax_c)
    ax_c.text(0, -0.20, "Lower with GCA", transform=ax_c.transAxes, fontsize=5.7, color=COLOR_SCW)
    ax_c.text(1, -0.20, "Higher with GCA", transform=ax_c.transAxes, ha="right", fontsize=5.7, color=COLOR_GCA)
    tx_c.set_ylim(ax_c.get_ylim()); tx_c.axis("off"); tx_c.text(0, 8.97, "Difference (95% CI)", fontsize=5.8, fontweight="bold", va="bottom")
    for yi, domain in zip(y, domain_order):
        item = external["pdqi_domains"][domain]; q_value = external["pdqi_domain_q"][analysis.PDQI_EXTERNAL.index(domain)]
        color = COLOR_GCA if q_value < 0.05 and item["estimate"] > 0 else (COLOR_SCW if q_value < 0.05 else COLOR_MUTED)
        tx_c.text(0, yi, f"{item['estimate']:+.2f} ({item['ci_low']:+.2f} to {item['ci_high']:+.2f})", fontsize=5.65, va="center", color=color)
        if domain == "Accurate": tx_c.text(0, yi - 0.30, f"BH-adjusted P={q_value:.3f}", fontsize=5.15, color=COLOR_MUTED)

    safety = external["draft_safety"]; labels = centres + ["Overall"]; y = np.arange(6)[::-1]
    for yi, label in zip(y, labels):
        item = safety if label == "Overall" else safety["by_centre"][label]
        estimate = 100 * item["proportion_with_any_detected_conflict"]; low, high = [100 * x for x in item["exact_ci"]]
        if label == "Overall": draw_diamond(ax_d, estimate, low, high, yi, COLOR_OVERALL)
        else:
            ax_d.plot([low, high], [yi, yi], color=COLOR_GCA, lw=1.25); ax_d.scatter(estimate, yi, s=24, marker="s", color=COLOR_GCA, edgecolor="white", lw=0.45)
            ax_d.axhline(yi - 0.5, color="#F1F1F1", lw=0.42)
    ax_d.set_yticks(y, [f"{c}  (n={centre_n[c]})" for c in centres] + [f"Overall  (n={overall_n})"]); ax_d.get_yticklabels()[-1].set_fontweight("bold")
    ax_d.set_xlim(0, 100); ax_d.set_xticks([0, 20, 40, 60, 80, 100]); ax_d.set_ylim(-0.72, 5.75)
    ax_d.set_xlabel("Drafts with ≥1 factually conflicting character, %", labelpad=4)
    ax_d.set_title("Factually conflicting characters in GCA-generated drafts", loc="left", fontweight="bold", fontsize=7.3, pad=12)
    ax_d.text(-0.33, 1.14, "d", transform=ax_d.transAxes, fontsize=8.2, fontweight="bold", va="top"); style_axis(ax_d)
    tx_d.set_ylim(ax_d.get_ylim()); tx_d.axis("off"); tx_d.text(0, 5.70, "Proportion (95% CI)", fontsize=5.8, fontweight="bold", va="bottom")
    for yi, label in zip(y, labels):
        item = safety if label == "Overall" else safety["by_centre"][label]
        estimate = 100 * item["proportion_with_any_detected_conflict"]; low, high = [100 * x for x in item["exact_ci"]]
        tx_d.text(0, yi, f"{estimate:.1f}% ({low:.1f} to {high:.1f})", fontsize=5.65, va="center",
                  fontweight="bold" if label == "Overall" else "normal", color=COLOR_OVERALL if label == "Overall" else COLOR_TEXT)
    save(fig, output_dir, "multicentre_evaluation_panel", svg=True, tight=False, dpi=600)
    plt.rcParams.update(previous_style)


def exploratory_patient_outcomes(results: dict, output_dir: Path) -> None:
    gca, scw, _ = analysis.load_rct()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), gridspec_kw={"height_ratios": [1.05, 0.82]})
    specs = [("a", "Patient Attention Time", "Patient Attention Time", "s", "Patient Attention Time"),
             ("b", "Final record length", "final_record_length", "characters", "Final record length")]
    for index, (ax, (letter, title, column, unit, key)) in enumerate(zip(axes[0], specs)):
        control = scw[column].dropna().to_numpy(float); treatment = gca[column].dropna().to_numpy(float)
        item = results["rct"]["other_exploratory"][key]
        bp = ax.boxplot([control, treatment], positions=[0, 1], widths=0.46, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#222222", lw=1.2), whiskerprops=dict(color="#555555", lw=0.8), capprops=dict(color="#555555", lw=0.8))
        for patch, color in zip(bp["boxes"], [COLOR_SCW, COLOR_GCA]): patch.set_facecolor(color); patch.set_alpha(0.24); patch.set_edgecolor(color)
        rng = np.random.default_rng(70 + index)
        ax.scatter(rng.normal(0, 0.055, len(control)), control, s=9, color=COLOR_SCW, alpha=0.30, linewidths=0)
        ax.scatter(rng.normal(1, 0.055, len(treatment)), treatment, s=9, color=COLOR_GCA, alpha=0.30, linewidths=0)
        ax.set_xticks([0, 1], [f"SCW\nn={len(control)}", f"GCA\nn={len(treatment)}"]); ax.set_xlim(-0.42, 1.42); ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=22); ax.text(-0.12, 1.12, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        ax.text(0, 1.015, "Selected exploratory outcome", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED)
        ax.text(0.98, 0.97, f"Mean difference {item['estimate']:+.1f} {unit}\n95% CI {item['ci_low']:+.1f} to {item['ci_high']:+.1f}\n{p_text(results['rct']['other_exploratory_q'][index], adjusted=True)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.5))
        ax.grid(axis="y", color=COLOR_GRID, lw=0.6)

    def dot_panel(ax, letter, title, columns, labels, q_values):
        y = np.arange(len(columns))
        for offset, frame, color, group in [(0.12, scw, COLOR_SCW, "SCW"), (-0.12, gca, COLOR_GCA, "GCA")]:
            means, lower, upper = [], [], []
            for column in columns:
                mean, low, high = mean_t_ci(frame[column]); means.append(mean); lower.append(mean - low); upper.append(high - mean)
            ax.errorbar(means, y + offset, xerr=np.vstack([lower, upper]), fmt="o", color=color, markersize=5.5, capsize=2.5, lw=1.1, label=group)
        ax.set_yticks(y, labels); ax.set_xlim(1, 5.15); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("Mean score (95% CI)"); ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=22); ax.text(-0.12, 1.12, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        ax.text(0, 1.015, "Selected exploratory outcomes", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED)
        for yi, q_value in enumerate(q_values): ax.text(5.12, yi, p_text(q_value, adjusted=True), ha="right", va="center", fontsize=7.8, color=COLOR_GCA if q_value < 0.05 else COLOR_MUTED)
        ax.legend(frameon=False, loc="upper left", ncol=2); ax.grid(axis="x", color=COLOR_GRID, lw=0.6)
    dot_panel(axes[1, 0], "c", "Interview quality", ["Interview Fluency", "Interview Completeness"], ["Interview fluency", "Interview completeness"], results["rct"]["interview_quality_q"])
    dot_panel(axes[1, 1], "d", "Patient-reported experience domains", analysis.PATIENT_ITEMS,
              ["Communication convenience", "Perceived physician attention", "Perceived empathy", "Overall satisfaction"], results["rct"]["patient_outcome_q"][1:])
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.08, hspace=0.48, wspace=0.33)
    save(fig, output_dir, "rct_exploratory_patient_outcomes_panel")


def patient_likert(output_dir: Path) -> None:
    gca, scw, _ = analysis.load_rct()
    levels = [1, 2, 3, 4, 5]; colors = {1: "#D73027", 2: "#FC8D59", 3: "#FEE090", 4: "#91BFDB", 5: "#4575B4"}
    labels = ["Communication convenience", "Perceived physician attention", "Perceived empathy", "Overall satisfaction"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    for group_index, (frame, group) in enumerate([(scw, "SCW"), (gca, "GCA")]):
        ax = axes[group_index]; y = np.arange(len(labels)); left = np.zeros(len(labels))
        for score in levels:
            values = np.asarray([(frame[column] == score).mean() * 100 for column in analysis.PATIENT_ITEMS])
            ax.barh(y, values, left=left, color=colors[score], edgecolor="white", lw=0.4, height=0.62, label=f"{score} score")
            for idx, value in enumerate(values):
                if value > 7: ax.text(left[idx] + value / 2, y[idx], f"{value:.0f}%", ha="center", va="center", fontsize=7, color="white" if score in [1, 5] else "#333333", fontweight="bold")
            left += values
        ax.set_xlim(0, 100); ax.set_yticks(y, labels if group_index == 0 else []); ax.set_xlabel("Percentage (%)"); ax.set_title(group, fontsize=11, fontweight="bold", loc="left"); ax.invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels(); fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)
    fig.suptitle("Patient-reported experience rating distributions", fontsize=12, fontweight="bold", y=1.12)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.82, bottom=0.18, wspace=0.08)
    save(fig, output_dir, "rct_patient_rating_distributions_panel")


def implementation_by_seniority(results: dict, output_dir: Path) -> None:
    implementation = results["rct"]["gca_implementation_by_seniority"]
    items = [(title, implementation[title]) for title in IMPLEMENTATION_OUTCOMES]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharey=True)
    for panel, (ax, (title, item)) in enumerate(zip(axes.flat, items)):
        counts = [item["junior_events"], item["senior_events"]]; ns = [item["junior_n"], item["senior_n"]]
        props = np.asarray(counts, float) / ns
        cis = [proportion_confint(k, n, method="wilson") for k, n in zip(counts, ns)]
        yerr = np.asarray([[100 * (props[i] - cis[i][0]) for i in range(2)], [100 * (cis[i][1] - props[i]) for i in range(2)]])
        ax.errorbar([0, 1], props * 100, yerr=yerr, fmt="none", ecolor="#444444", elinewidth=1.2, capsize=3)
        ax.scatter([0, 1], props * 100, s=90, marker="s", color=[COLOR_JUNIOR, COLOR_SENIOR], edgecolor="white", lw=0.8)
        for x, value in enumerate(props * 100): ax.text(x, min(value + 7, 106), f"{value:.1f}%", ha="center", fontsize=8.3, fontweight="bold")
        ax.text(0.5, 0.04, f"Senior - junior: {100 * item['risk_difference']:+.1f} pp\n95% CI {100 * item['ci_low']:.1f} to {100 * item['ci_high']:.1f}; {q_text(item['q'])}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#333333")
        ax.set_xticks([0, 1], ["Junior", "Senior"]); ax.set_xlim(-0.45, 1.45); ax.set_ylim(0, 112); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_title(f"{chr(97 + panel)}  {title}", loc="left", fontsize=8.6, fontweight="bold", pad=6)
        if panel % 3 == 0: ax.set_ylabel("Endorsement (%)", fontsize=8.8)
    fig.suptitle("Clinician-reported implementation outcomes by physician seniority", fontsize=12.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07, hspace=0.40, wspace=0.20)
    save(fig, output_dir, "rct_implementation_by_seniority_panel")
