from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def independent_bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def run(output_dir: Path, qa_dir: Path) -> bool:
    multiplicity = pd.read_csv(output_dir / "results" / "multiplicity_families.csv")
    family_rows = []
    for (study, family), group in multiplicity.groupby(["study", "family"], sort=False):
        expected = independent_bh(group["p"].to_numpy(float))
        observed = group["q"].to_numpy(float)
        difference = float(np.max(np.abs(expected - observed)))
        family_rows.append({
            "study": study,
            "family": family,
            "tests": len(group),
            "max_absolute_q_difference": difference,
            "status": "PASS" if difference <= 1e-12 else "FAIL",
        })

    qa_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(family_rows).to_csv(qa_dir / "bh_independent_recalculation.csv", index=False, encoding="utf-8-sig")
    passed = all(row["status"] == "PASS" for row in family_rows)
    summary = {
        "passed": passed,
        "bh_families": len(family_rows),
        "bh_tests": int(sum(row["tests"] for row in family_rows)),
        "maximum_absolute_bh_difference": max(row["max_absolute_q_difference"] for row in family_rows),
    }
    (qa_dir / "independent_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return passed
