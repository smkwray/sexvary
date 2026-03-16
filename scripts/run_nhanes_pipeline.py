#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import numpy as np
import pandas as pd

from sexvary.adapters.nhanes import NHANESAdapter
from sexvary.config import build_registry
from sexvary.evidence import annotate_estimate_evidence
from sexvary.estimation import (
    estimate_dataset_cells,
    estimate_sex_difference_cell,
    estimation_config_from_analysis,
    prepare_analysis_frame,
)
from sexvary.io import write_table
from sexvary.reporting import markdown_table, write_markdown_summary
from sexvary.utils import ensure_dir, existing_external_dataset_files, external_data_dirs, project_root


def _default_raw_dir(root: Path) -> Path:
    data_dir, candidates = existing_external_dataset_files(
        "nhanes_2011_2023",
        start=root,
        patterns=("DEMO_*.xpt",),
    )
    if data_dir is None or not candidates:
        searched = ", ".join(str(path) for path in external_data_dirs("nhanes_2011_2023", root))
        raise FileNotFoundError(
            "No NHANES raw directory found. "
            f"Searched: {searched}. Place NHANES XPT files there or pass --raw-dir explicitly."
        )
    return data_dir


def _estimate_from_prepared_frame(prepared: pd.DataFrame, *, config, registry) -> pd.DataFrame:
    group_cols = ["dataset_id", "cycle_or_wave", "country", "analysis_cell", "trait_id"]
    rows = [estimate_sex_difference_cell(group.copy(), config=config) for _, group in prepared.groupby(group_cols, dropna=False)]
    if not rows:
        return pd.DataFrame()
    estimates = pd.DataFrame(rows).sort_values(
        by=["dataset_id", "trait_id", "cycle_or_wave", "country", "age_band"],
        kind="stable",
    ).reset_index(drop=True)
    return annotate_estimate_evidence(estimates, registry=registry)


def _winsorize_scores(prepared: pd.DataFrame, *, lower_q: float, upper_q: float) -> pd.DataFrame:
    work = prepared.copy()
    group_cols = ["dataset_id", "cycle_or_wave", "country", "analysis_cell", "trait_id"]
    for _, idx in work.groupby(group_cols, dropna=False).groups.items():
        scores = pd.to_numeric(work.loc[idx, "score_raw"], errors="coerce")
        valid = scores.dropna()
        if valid.empty:
            continue
        lo = float(valid.quantile(lower_q))
        hi = float(valid.quantile(upper_q))
        work.loc[idx, "score_raw"] = scores.clip(lower=lo, upper=hi)
    return work


def _build_robustness_variants(*, baseline: pd.DataFrame, registry, config) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    baseline_prepared = prepare_analysis_frame(baseline, config=config)
    variants["baseline_weighted"] = _estimate_from_prepared_frame(baseline_prepared, config=config, registry=registry)

    unweighted = baseline.copy()
    unweighted["weight_main"] = 1.0
    variants["unweighted"] = _estimate_from_prepared_frame(
        prepare_analysis_frame(unweighted, config=config),
        config=config,
        registry=registry,
    )

    lower_q, upper_q = registry.analysis_config.get("analysis_defaults", {}).get("winsorize_for_sensitivity", [0.005, 0.995])
    winsorized_prepared = prepare_analysis_frame(baseline, config=config)
    winsorized_prepared = _winsorize_scores(winsorized_prepared, lower_q=float(lower_q), upper_q=float(upper_q))
    variants["winsorized_weighted"] = _estimate_from_prepared_frame(winsorized_prepared, config=config, registry=registry)
    return variants


def _build_robustness_comparison(variant_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = variant_tables["baseline_weighted"].copy()
    key_cols = ["dataset_id", "cycle_or_wave", "country", "age_band", "trait_id"]
    baseline = baseline[key_cols + ["log_variance_ratio", "evidence_status", "headline_eligible"]].rename(
        columns={
            "log_variance_ratio": "baseline_log_variance_ratio",
            "evidence_status": "baseline_evidence_status",
            "headline_eligible": "baseline_headline_eligible",
        }
    )

    rows: list[pd.DataFrame] = []
    for variant_name, variant_df in variant_tables.items():
        if variant_name == "baseline_weighted":
            continue
        merged = baseline.merge(
            variant_df[key_cols + ["log_variance_ratio", "evidence_status", "headline_eligible"]],
            on=key_cols,
            how="left",
        ).rename(
            columns={
                "log_variance_ratio": "variant_log_variance_ratio",
                "evidence_status": "variant_evidence_status",
                "headline_eligible": "variant_headline_eligible",
            }
        )
        merged.insert(0, "variant", variant_name)
        merged["delta_log_variance_ratio"] = merged["variant_log_variance_ratio"] - merged["baseline_log_variance_ratio"]
        merged["sign_changed"] = (
            np.sign(merged["baseline_log_variance_ratio"]) != np.sign(merged["variant_log_variance_ratio"])
        ) & np.isfinite(merged["baseline_log_variance_ratio"]) & np.isfinite(merged["variant_log_variance_ratio"])
        rows.append(merged)

    comparison = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if comparison.empty:
        return comparison, pd.DataFrame()

    summary = (
        comparison.groupby("variant", sort=True)
        .agg(
            matched_cells=("variant_log_variance_ratio", lambda s: int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())),
            median_abs_delta=("delta_log_variance_ratio", lambda s: float(pd.Series(s).abs().median(skipna=True))),
            sign_change_rate=("sign_changed", lambda s: float(pd.Series(s, dtype=float).mean(skipna=True))),
            headline_cells_retained=("variant_headline_eligible", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
    )
    return comparison, summary


def _write_nhanes_report(
    *,
    out_dir: Path,
    estimates: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> Path:
    report_path = out_dir / "nhanes_2011_2023_report.md"
    by_status = (
        estimates.groupby(["trait_id", "evidence_status"], dropna=False)
        .size()
        .reset_index(name="cells")
        .sort_values(["trait_id", "cells", "evidence_status"], ascending=[True, False, True], kind="stable")
    )
    physical = estimates[estimates["trait_id"].isin(["height_cm", "weight_kg", "bmi", "waist_cm", "grip_strength_kg"])]
    cognition = estimates[estimates["trait_id"] == "adult_cognition_screen"]
    lines = [
        "# NHANES selected-cycles report",
        "",
        "This NHANES pass is stronger than the supporting panel datasets because it already uses design-aware uncertainty, but its bounded cognition screen remains QA-only under the current thresholds.",
        "",
        "## Trait-status summary",
        "",
        markdown_table(by_status),
        "",
        "## Coverage split",
        "",
        f"- Physical-trait rows: `{len(physical)}`",
        f"- Cognition-screen rows: `{len(cognition)}`",
        f"- Headline-eligible rows: `{int(estimates['headline_eligible'].fillna(False).sum())}`",
        "",
        "## Robustness checks",
        "",
        "These compare the weighted baseline against unweighted and winsorized variants.",
        "",
        markdown_table(robustness_summary) if not robustness_summary.empty else "_No robustness summary available._",
        "",
        "## Interpretation guardrails",
        "",
        "- The adult cognition screen is bounded and sparse, so it remains QA-only rather than inferential evidence.",
        "- Grip strength remains the most stable male-greater NHANES trait in the live output.",
        "- Anthropometric traits are informative, but not uniformly male-greater in variability.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NHANES ingest -> estimate pipeline.")
    parser.add_argument("--raw-dir", help="Path to a directory containing NHANES XPT files.")
    parser.add_argument("--cycle", action="append", help="Optional NHANES cycle suffix filter, e.g. G, H, I, J, or L.")
    parser.add_argument("--trait", action="append", help="Optional trait filter.")
    parser.add_argument("--output-dir", default="results/nhanes_2011_2023", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_dir = Path(args.raw_dir) if args.raw_dir else _default_raw_dir(root)
    registry = build_registry(root)
    spec = registry.get_dataset("nhanes_2011_2023")
    out_dir = ensure_dir(root / args.output_dir)
    config = estimation_config_from_analysis(registry.analysis_config)

    adapter = NHANESAdapter(spec, raw_path=raw_dir, cycles=args.cycle, traits=args.trait)
    normalized = adapter.to_long_person_trait()
    estimates = estimate_dataset_cells(normalized.data, config=config)
    estimates = annotate_estimate_evidence(estimates, registry=registry)

    robustness_tables = _build_robustness_variants(baseline=normalized.data, registry=registry, config=config)
    robustness_comparison, robustness_summary = _build_robustness_comparison(robustness_tables)

    estimates_path = write_table(estimates, out_dir / "nhanes_2011_2023_trait_estimates.csv")
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
    summary_path = out_dir / "nhanes_2011_2023_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="NHANES selected-cycles trait estimates")

    robustness_comparison_path = out_dir / "nhanes_2011_2023_robustness_comparison.csv"
    robustness_summary_path = out_dir / "nhanes_2011_2023_robustness_summary.csv"
    robustness_comparison.to_csv(robustness_comparison_path, index=False)
    robustness_summary.to_csv(robustness_summary_path, index=False)
    write_markdown_summary(
        robustness_summary,
        out_dir / "nhanes_2011_2023_robustness_summary.md",
        title="NHANES selected-cycles robustness summary",
    )
    report_path = _write_nhanes_report(out_dir=out_dir, estimates=estimates, robustness_summary=robustness_summary)

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {robustness_comparison_path}")
    print(f"Wrote {robustness_summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
