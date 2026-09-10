from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


EXPECTED_TABLES = [
    "rct_baseline_characteristics.csv",
    "rct_outcomes.csv",
    "rct_ai_safety_and_implementation.csv",
    "multicentre_characteristics.csv",
    "rct_seniority_effects.csv",
    "multicentre_overall_paired_effects.csv",
    "multicentre_centre_specific_effects.csv",
    "multicentre_paired_effect_sensitivity.csv",
    "multicentre_centre_heterogeneity.csv",
    "multicentre_pdqi_domains.csv",
    "multicentre_seniority_effects.csv",
    "multicentre_pdqi_seniority_gaps.csv",
    "multicentre_draft_safety.csv",
    "multicentre_safety_inference.csv",
    "rct_ordinal_sensitivity.csv",
    "multicentre_ordinal_gee_sensitivity.csv",
    "rct_physician_clustered_sensitivity.csv",
    "multicentre_physician_clustered_sensitivity.csv",
    "rct_physician_clustering_outcomes.csv",
]

EXPECTED_FIGURES = [
    "rct_outcomes", "rct_patient_outcomes", "rct_seniority",
    "multicentre_evaluation", "rct_exploratory_patient_outcomes",
    "rct_patient_rating_distributions", "rct_implementation_by_seniority",
]

def _record(rows: list[dict], area: str, check: str, observed, expected, passed: bool, note: str = "") -> None:
    rows.append({
        "Area": area,
        "Check": check,
        "Observed": observed,
        "Expected": expected,
        "Status": "PASS" if passed else "FAIL",
        "Note": note,
    })


def validate_results(
    checks: dict,
    output_root: Path,
    qa_dir: Path,
) -> bool:
    rows: list[dict] = []
    _record(rows, "Analysis", "All embedded regression checks", checks.get("all_passed"), True,
            checks.get("all_passed") is True)

    table_dir = output_root / "tables"
    for name in EXPECTED_TABLES:
        path = table_dir / name
        size = path.stat().st_size if path.exists() else 0
        _record(rows, "Tables", name, size, ">0 bytes", size > 0)

    figure_dir = output_root / "figures"
    for stem in EXPECTED_FIGURES:
        for suffix in (".png", ".pdf"):
            path = figure_dir / f"{stem}{suffix}"
            size = path.stat().st_size if path.exists() else 0
            _record(rows, "Figures", path.name, size, ">0 bytes", size > 0)
        image_path = figure_dir / f"{stem}.png"
        if image_path.exists():
            with Image.open(image_path) as image:
                valid = image.width >= 1200 and image.height >= 700
                _record(rows, "Figures", f"{stem} dimensions", f"{image.width}x{image.height}",
                        ">=1200x700", valid)

    qa_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(qa_dir / "reconciliation_report.csv", index=False, encoding="utf-8-sig")
    passed = all(row["Status"] == "PASS" for row in rows)
    (qa_dir / "reproducibility_status.json").write_text(
        json.dumps({"passed": passed, "checks": len(rows), "failed": sum(row["Status"] == "FAIL" for row in rows)}, indent=2),
        encoding="utf-8",
    )
    return passed
