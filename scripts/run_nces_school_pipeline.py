#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.adapters import NCESSchoolAdapter
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.estimation import estimate_dataset_cells, estimation_config_from_analysis
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


def _default_raw_path(root: Path, dataset_id: str) -> Path:
    _, candidates = existing_external_dataset_files(
        dataset_id,
        start=root,
        patterns=("*.dat", "*.zip", "*.sav", "*.csv", "*.parquet", "*.dta", "*.xlsx"),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No raw file found for {dataset_id}. Searched: {', '.join(str(path) for path in external_data_dirs(dataset_id, root))}. "
            "Place a supported file there or pass --raw-path."
        )
    return candidates[0]


def _mapping_path(root: Path, dataset_id: str) -> Path:
    preferred = root / "config" / "mappings" / f"{dataset_id}.yaml"
    if preferred.exists():
        return preferred
    fallback = root / "config" / "mappings" / f"{dataset_id}.example.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No mapping file found for {dataset_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NCES school ingest -> estimate pipeline.")
    parser.add_argument("--dataset-id", required=True, choices=["ecls_k_2011", "hsls_2009"], help="NCES dataset id.")
    parser.add_argument("--raw-path", help="Optional explicit raw file path.")
    parser.add_argument("--mapping-path", help="Optional explicit mapping yaml path.")
    parser.add_argument("--output-dir", help="Optional output dir relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    dataset_id = args.dataset_id
    spec = registry.get_dataset(dataset_id)
    raw_path = Path(args.raw_path) if args.raw_path else _default_raw_path(root, dataset_id)
    mapping_path = Path(args.mapping_path) if args.mapping_path else _mapping_path(root, dataset_id)
    out_dir = ensure_dir(root / (args.output_dir or f"results/{dataset_id}"))

    adapter = NCESSchoolAdapter(spec, raw_path=raw_path, mapping_path=mapping_path)
    normalized = adapter.to_long_person_trait()
    config = estimation_config_from_analysis(registry.analysis_config)
    estimates = estimate_dataset_cells(normalized.data, config=config)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = out_dir / f"{dataset_id}_trait_estimates.csv"
    summary_path = out_dir / f"{dataset_id}_trait_estimates.md"
    estimates.to_csv(estimates_path, index=False)
    summary = estimates[
        [
            "dataset_id",
            "cycle_or_wave",
            "age_band",
            "trait_id",
            "male_n",
            "female_n",
            "variance_ratio",
            "log_variance_ratio",
            "qa_flags",
        ]
    ]
    write_markdown_summary(summary, summary_path, title=f"{dataset_id} trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
