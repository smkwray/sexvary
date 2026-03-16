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

from sexvary.adapters.pirls import PIRLSAdapter
from sexvary.archive import extract_matching_zip_members
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.io import write_table
from sexvary.pirls import estimate_pirls_cells
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


PIRLS_PREFERRED_PATTERNS = (
    re.compile(r"^asgusar5\.sav$", re.IGNORECASE),
)


def _default_raw_paths(root: Path) -> list[Path]:
    data_dir, candidates = existing_external_dataset_files(
        "pirls_2021",
        start=root,
        patterns=("*.sav", "*.csv", "*.parquet", "*.dta"),
    )
    if data_dir is None:
        data_dir = external_data_dirs("pirls_2021", root)[0]
    preferred = [
        path
        for pattern in PIRLS_PREFERRED_PATTERNS
        for path in candidates
        if pattern.match(path.name)
    ]
    if preferred:
        return preferred[:1]

    zip_candidates = sorted(data_dir.glob("*.zip"))
    if zip_candidates:
        extracted = extract_matching_zip_members(
            zip_candidates,
            member_patterns=PIRLS_PREFERRED_PATTERNS,
            output_dir=data_dir,
        )
        if extracted:
            return extracted[:1]

    raise FileNotFoundError(
        "No PIRLS raw student file found. "
        f"Searched: {', '.join(str(path) for path in external_data_dirs('pirls_2021', root))}. "
        "Place the public-use archive there or pass --raw-path explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PIRLS 2021 ingest -> estimate pipeline.")
    parser.add_argument("--raw-path", action="append", help="Path to a PIRLS raw student file.")
    parser.add_argument("--country-id", action="append", help="Optional country filter, e.g. 840.")
    parser.add_argument("--output-dir", default="results/pirls_2021", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_paths = [Path(path) for path in args.raw_path] if args.raw_path else _default_raw_paths(root)
    registry = build_registry(root)
    spec = registry.get_dataset("pirls_2021")
    out_dir = ensure_dir(root / args.output_dir)

    adapter = PIRLSAdapter(spec, raw_path=raw_paths, country_ids=args.country_id)
    normalized = adapter.to_long_person_trait()
    estimates = estimate_pirls_cells(normalized.data)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / "pirls_2021_trait_estimates.csv")
    summary = estimates[["country", "country_id", "grade_or_age_band", "trait_id", "log_variance_ratio", "variance_ratio", "se_log_variance_ratio"]]
    summary_path = out_dir / "pirls_2021_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="PIRLS 2021 trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
