#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.adapters.piaac import PIAACAdapter
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.io import write_table
from sexvary.piaac import estimate_piaac_cells
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


def _default_raw_path(root: Path) -> Path:
    data_dir, candidates = existing_external_dataset_files(
        "piaac_cycle2",
        start=root,
        patterns=("*.sav", "*.csv", "*.parquet", "*.dta"),
    )
    if not candidates:
        searched = ", ".join(str(path) for path in external_data_dirs("piaac_cycle2", root))
        raise FileNotFoundError(
            "No PIAAC raw file found. "
            f"Searched: {searched}. Place a public-use file there or pass --raw-path explicitly."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PIAAC cycle 2 ingest -> estimate pipeline.")
    parser.add_argument("--raw-path", help="Path to the PIAAC raw file. Defaults to the first supported file under the canonical shared-data or legacy path.")
    parser.add_argument("--country-id", action="append", help="Optional country filter, e.g. USA.")
    parser.add_argument("--output-dir", default="results/piaac_cycle2", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_path = Path(args.raw_path) if args.raw_path else _default_raw_path(root)
    registry = build_registry(root)
    spec = registry.get_dataset("piaac_cycle2")
    out_dir = ensure_dir(root / args.output_dir)

    adapter = PIAACAdapter(spec, raw_path=raw_path, country_ids=args.country_id)
    normalized = adapter.to_long_person_trait()
    estimates = estimate_piaac_cells(normalized.data)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / "piaac_cycle2_trait_estimates.csv")
    summary = estimates[["country", "country_id", "grade_or_age_band", "trait_id", "log_variance_ratio", "variance_ratio", "se_log_variance_ratio"]]
    summary_path = out_dir / "piaac_cycle2_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="PIAAC cycle 2 trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
