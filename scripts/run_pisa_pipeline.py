#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.adapters.pisa import PISAAdapter
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.io import write_table
from sexvary.pisa import estimate_pisa_cells
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


def _default_raw_path(root: Path) -> Path:
    _, candidates = existing_external_dataset_files(
        "pisa_2022",
        start=root,
        patterns=("*.sav", "*.csv", "*.parquet", "*.dta"),
    )
    if not candidates:
        searched = ", ".join(str(path) for path in external_data_dirs("pisa_2022", root))
        raise FileNotFoundError(
            "No PISA raw file found. "
            f"Searched: {searched}. Place a public-use file there or pass --raw-path explicitly."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PISA 2022 ingest -> estimate pipeline.")
    parser.add_argument("--raw-path", help="Path to the PISA raw file. Defaults to the first supported file under the canonical shared-data or legacy path.")
    parser.add_argument("--country-code", action="append", help="Optional country filter, e.g. USA.")
    parser.add_argument("--output-dir", default="results/pisa_2022", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_path = Path(args.raw_path) if args.raw_path else _default_raw_path(root)
    registry = build_registry(root)
    spec = registry.get_dataset("pisa_2022")
    out_dir = ensure_dir(root / args.output_dir)
    country_codes = args.country_code or ["USA"]

    adapter = PISAAdapter(spec, raw_path=raw_path, country_codes=country_codes)
    normalized = adapter.to_long_person_trait()
    estimates = estimate_pisa_cells(normalized.data)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / "pisa_2022_trait_estimates.csv")
    summary = estimates[["country", "country_id", "grade_or_age_band", "trait_id", "log_variance_ratio", "variance_ratio", "se_log_variance_ratio"]]
    summary_path = out_dir / "pisa_2022_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="PISA 2022 trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
