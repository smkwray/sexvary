#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.adapters.nnyfs import NNYFSAdapter
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.estimation import estimate_dataset_cells, estimation_config_from_analysis
from sexvary.io import write_table
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, external_data_dirs, first_existing_external_data_dir, project_root


def _default_raw_dir(root: Path) -> Path:
    data_dir = first_existing_external_data_dir("nnyfs_2012", root)
    if data_dir is None:
        searched = ", ".join(str(path) for path in external_data_dirs("nnyfs_2012", root))
        raise FileNotFoundError(
            "No NNYFS raw directory found. "
            f"Searched: {searched}. Place Y_DEMO.xpt, Y_BMX.xpt, and Y_MGX.xpt there or pass --raw-dir."
        )
    if not (data_dir / "Y_DEMO.xpt").exists():
        raise FileNotFoundError(
            f"No Y_DEMO.xpt file found under {data_dir}. "
            "Expected NNYFS public-use XPT files such as Y_DEMO.xpt, Y_BMX.xpt, and Y_MGX.xpt."
        )
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NNYFS 2012 ingest -> estimate pipeline.")
    parser.add_argument("--raw-dir", help="Path to a directory containing NNYFS 2012 XPT files.")
    parser.add_argument("--trait", action="append", help="Optional trait filter.")
    parser.add_argument("--output-dir", default="results/nnyfs_2012", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_dir = Path(args.raw_dir) if args.raw_dir else _default_raw_dir(root)
    registry = build_registry(root)
    spec = registry.get_dataset("nnyfs_2012")
    out_dir = ensure_dir(root / args.output_dir)
    config = estimation_config_from_analysis(registry.analysis_config)

    adapter = NNYFSAdapter(spec, raw_path=raw_dir)
    normalized = adapter.to_long_person_trait()
    if args.trait:
        normalized.data = normalized.data[normalized.data["trait_id"].isin(set(args.trait))].copy()
    estimates = estimate_dataset_cells(normalized.data, config=config)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / "nnyfs_2012_trait_estimates.csv")
    summary = estimates[
        [
            "cycle_or_wave",
            "country",
            "age_band",
            "trait_id",
            "log_variance_ratio",
            "variance_ratio",
            "se_log_variance_ratio",
            "inference_method",
            "qa_flags",
            "evidence_status",
        ]
    ]
    summary_path = out_dir / "nnyfs_2012_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="NNYFS 2012 trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
