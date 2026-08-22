from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import analysis, audit, figures, table_exports, tables, validate


REQUIRED_INPUTS = [
    "RCT.xlsx", "HM.xlsx", "JJ.xlsx", "JX.xlsx", "WHRM.xlsx", "WHZX.xlsx",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce the GCA statistical analyses, statistical figures and tables.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    qa_dir = output_dir / "qa"
    missing = [name for name in REQUIRED_INPUTS if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source workbooks in {data_dir}: {missing}")

    analysis.SOURCE = data_dir
    analysis.RESULTS = results_dir
    analysis.main()
    results = json.loads((results_dir / "reproducibility_results.json").read_text(encoding="utf-8"))
    checks = json.loads((results_dir / "regression_checks.json").read_text(encoding="utf-8"))
    tables.write_all(results, output_dir / "tables")
    table_exports.write_excel_workbooks(output_dir / "tables", output_dir / "excel_tables")
    figures.write_all(results, output_dir / "figures")

    passed = audit.run(output_dir, qa_dir)
    passed = validate.validate_results(checks, output_dir, qa_dir) and passed
    word_dir = output_dir / "word_tables"
    if passed:
        table_exports.write_physician_cluster_word_tables(output_dir / "tables", word_dir)
        table_exports.write_physician_clustering_word(
            output_dir / "tables", word_dir
        )
    passed = table_exports.validate_exports(
        output_dir / "excel_tables",
        word_dir if word_dir.exists() else None,
        qa_dir,
    ) and passed
    if not passed:
        raise RuntimeError(f"Reconciliation failed. See {qa_dir / 'reconciliation_report.csv'}")
    print(f"Reproduction completed successfully: {output_dir}")


if __name__ == "__main__":
    main()
