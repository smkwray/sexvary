#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.adapters import CPPAdapter
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.estimation import estimate_dataset_cells, estimation_config_from_analysis
from sexvary.io import write_table
from sexvary.reporting import write_markdown_summary
from sexvary.utils import ensure_dir, external_data_dirs, first_existing_external_data_dir, project_root


def _default_raw_dir(root: Path, dataset_id: str) -> Path:
    data_dir = first_existing_external_data_dir(dataset_id, root)
    if data_dir is None:
        searched = ", ".join(str(path) for path in external_data_dirs(dataset_id, root))
        raise FileNotFoundError(
            f"No CPP raw directory found for {dataset_id}. "
            f"Searched: {searched}. Place the CPP release files there or pass --raw-dir."
        )
    return data_dir


def _default_mapping_path(root: Path) -> Path:
    preferred = root / "config" / "mappings" / "cpp.yaml"
    if preferred.exists():
        return preferred
    fallback = root / "config" / "mappings" / "cpp.example.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No CPP mapping file found. Expected config/mappings/cpp.yaml or cpp.example.yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPP ingest -> estimate pipeline.")
    parser.add_argument("--dataset-id", required=True, choices=["cpp_core", "cpp_growth"], help="CPP dataset id.")
    parser.add_argument("--raw-dir", help="Path to the CPP release directory.")
    parser.add_argument("--mapping-path", help="Path to the shared CPP mapping yaml.")
    parser.add_argument("--trait", action="append", help="Optional trait filter.")
    parser.add_argument("--output-dir", help="Optional output dir relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    dataset_id = args.dataset_id
    spec = registry.get_dataset(dataset_id)
    raw_dir = Path(args.raw_dir) if args.raw_dir else _default_raw_dir(root, dataset_id)
    mapping_path = Path(args.mapping_path) if args.mapping_path else _default_mapping_path(root)
    out_dir = ensure_dir(root / (args.output_dir or f"results/{dataset_id}"))
    config = estimation_config_from_analysis(registry.analysis_config)

    adapter = CPPAdapter(spec, raw_path=raw_dir, mapping_path=mapping_path, mode=dataset_id)
    normalized = adapter.to_long_person_trait()
    if args.trait:
        normalized.data = normalized.data[normalized.data["trait_id"].isin(set(args.trait))].copy()
    estimates = estimate_dataset_cells(normalized.data, config=config)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    estimates_path = write_table(estimates, out_dir / f"{dataset_id}_trait_estimates.csv")
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
            "inference_method",
            "evidence_status",
            "qa_flags",
        ]
    ]
    summary_path = out_dir / f"{dataset_id}_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title=f"{dataset_id} trait estimates")

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
