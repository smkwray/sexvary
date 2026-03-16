#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import re

from sexvary.adapters.icils import ICILSAdapter
from sexvary.archive import extract_matching_zip_members
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.icils import estimate_icils_cells
from sexvary.io import write_table
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


ICILS_PREFERRED_PATTERNS = (
    re.compile(r"^bsgusai3\.sav$", re.IGNORECASE),
)


def _default_raw_path(root: Path) -> Path:
    data_dir, candidates = existing_external_dataset_files(
        "icils_2023",
        start=root,
        patterns=("*.sav", "*.csv", "*.parquet", "*.dta"),
    )
    if data_dir is None:
        data_dir = external_data_dirs("icils_2023", root)[0]
    preferred = [
        path
        for path in candidates
        if any(pattern.match(path.name) for pattern in ICILS_PREFERRED_PATTERNS)
    ]
    if preferred:
        return preferred[0]

    zip_candidates = sorted(data_dir.glob("*.zip"))
    if zip_candidates:
        extracted = extract_matching_zip_members(
            zip_candidates,
            member_patterns=ICILS_PREFERRED_PATTERNS,
            output_dir=data_dir,
        )
        if extracted:
            return extracted[0]

    raise FileNotFoundError(
        "No ICILS raw student file found. "
        f"Searched: {', '.join(str(path) for path in external_data_dirs('icils_2023', root))}. "
        "Place the public-use archive there or pass --raw-path explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ICILS 2023 ingest -> estimate pipeline.")
    parser.add_argument("--raw-path", help="Path to the ICILS raw student file.")
    parser.add_argument("--country-id", action="append", help="Optional country filter, e.g. 840.")
    parser.add_argument("--output-dir", default="results/icils_2023", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_path = Path(args.raw_path) if args.raw_path else _default_raw_path(root)
    registry = build_registry(root)
    spec = registry.get_dataset("icils_2023")
    out_dir = ensure_dir(root / args.output_dir)

    adapter = ICILSAdapter(spec, raw_path=raw_path, country_ids=args.country_id or ["840"])
    normalized = adapter.to_long_person_trait()
    estimates = estimate_icils_cells(normalized.data)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / "icils_2023_trait_estimates.csv")
    summary = estimates[["country", "country_id", "grade_or_age_band", "trait_id", "log_variance_ratio", "variance_ratio", "se_log_variance_ratio"]]
    summary_path = out_dir / "icils_2023_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="ICILS 2023 trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
