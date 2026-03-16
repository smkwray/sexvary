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

from sexvary.adapters.timss import TIMSSAdapter
from sexvary.archive import extract_matching_zip_members
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.io import write_table
from sexvary.reporting import write_markdown_summary
from sexvary.timss import estimate_timss_cells
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


TIMSS_PREFERRED_PATTERNS = (
    re.compile(r"^asgus.*a[mzb][78]\.(sav|csv|parquet|dta)$", re.IGNORECASE),
    re.compile(r"^bsgus.*a[mzb][78]\.(sav|csv|parquet|dta)$", re.IGNORECASE),
)


def _default_raw_paths(root: Path, dataset_id: str) -> list[Path]:
    data_dir, candidates = existing_external_dataset_files(
        dataset_id,
        start=root,
        patterns=("*.sav", "*.csv", "*.parquet", "*.dta"),
    )
    if data_dir is None:
        data_dir = external_data_dirs(dataset_id, root)[0]
    preferred = [
        path
        for path in candidates
        if any(pattern.match(path.name) for pattern in TIMSS_PREFERRED_PATTERNS)
    ]
    if preferred:
        return preferred

    zip_candidates = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file()
        and path.name not in {"FILE_INVENTORY.csv", "SOURCE.md", "TRANSFORM_LOG.md"}
        and path.suffix.lower() == ".zip"
    ]
    if zip_candidates:
        extracted = extract_matching_zip_members(
            zip_candidates,
            member_patterns=TIMSS_PREFERRED_PATTERNS,
            output_dir=data_dir,
        )
        if extracted:
            return extracted

    if not candidates and not zip_candidates:
        raise FileNotFoundError(
            f"No TIMSS raw file found for {dataset_id}. "
            f"Searched: {', '.join(str(path) for path in external_data_dirs(dataset_id, root))}. "
            "Place a public-use student file there or pass --raw-path explicitly."
        )
    return candidates[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TIMSS ingest -> estimate pipeline.")
    parser.add_argument("--dataset-id", default="timss_2019", choices=["timss_2019", "timss_2023"], help="TIMSS dataset id to run.")
    parser.add_argument("--raw-path", action="append", help="Path to a TIMSS raw student file. Repeat to provide multiple grade files.")
    parser.add_argument("--country-id", action="append", help="Optional country filter, e.g. 840 or USA depending on the file.")
    parser.add_argument("--grade", action="append", help="Optional grade filter, e.g. 4 or 8.")
    parser.add_argument("--output-dir", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_paths = [Path(path) for path in args.raw_path] if args.raw_path else _default_raw_paths(root, args.dataset_id)
    registry = build_registry(root)
    spec = registry.get_dataset(args.dataset_id)
    output_dir = args.output_dir or f"results/{args.dataset_id}"
    out_dir = ensure_dir(root / output_dir)

    adapter = TIMSSAdapter(spec, raw_path=raw_paths, country_ids=args.country_id, grades=args.grade)
    normalized = adapter.to_long_person_trait()
    estimates = estimate_timss_cells(normalized.data)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / f"{args.dataset_id}_trait_estimates.csv")
    summary = estimates[["country", "country_id", "grade_or_age_band", "trait_id", "log_variance_ratio", "variance_ratio", "se_log_variance_ratio"]]
    summary_path = out_dir / f"{args.dataset_id}_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title=f"{spec.name} trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
