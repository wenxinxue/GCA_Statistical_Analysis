from __future__ import annotations

import shutil
from pathlib import Path

from . import statistical_figure_style as style
from .compose import compose_rct_outcomes, compose_rct_patient_outcomes, compose_single_panel


def _copy_pair(source_dir: Path, output_dir: Path, source_stem: str, final_stem: str) -> None:
    for suffix in (".png", ".pdf"):
        shutil.copyfile(source_dir / f"{source_stem}{suffix}", output_dir / f"{final_stem}{suffix}")


def write_all(results: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panel_dir = output / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    style.apply_style()
    style.rct_outcome_panels(results, panel_dir)
    style.rct_effect_forest(results, panel_dir)
    style.rct_pdqi_radar(results, panel_dir)
    style.attention_association(results, panel_dir)
    style.editing_profile(panel_dir)
    style.rct_seniority(results, panel_dir)
    style.multicentre_four_panel(results, panel_dir)
    style.exploratory_patient_outcomes(results, panel_dir)
    style.patient_likert(panel_dir)
    style.implementation_by_seniority(results, panel_dir)

    compose_rct_outcomes(panel_dir, output)
    compose_rct_patient_outcomes(panel_dir, output)
    compose_single_panel(panel_dir, output, "rct_seniority_panel", "rct_seniority", (843, 14, 2371, 1876))
    compose_single_panel(panel_dir, output, "multicentre_evaluation_panel", "multicentre_evaluation", (1160, 88, 1868, 1764))
    _copy_pair(panel_dir, output, "rct_exploratory_patient_outcomes_panel", "rct_exploratory_patient_outcomes")
    _copy_pair(panel_dir, output, "rct_patient_rating_distributions_panel", "rct_patient_rating_distributions")
    _copy_pair(panel_dir, output, "rct_implementation_by_seniority_panel", "rct_implementation_by_seniority")
