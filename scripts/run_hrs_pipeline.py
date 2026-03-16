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

from sexvary.adapters.hrs import HRSAdapter
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
from sexvary.utils import ensure_dir, external_data_dirs, first_existing_external_data_dir, project_root


ESTIMATE_OUTPUT_COLUMNS = [
    "dataset_id",
    "cycle_or_wave",
    "country",
    "analysis_cell",
    "age_band",
    "trait_id",
    "male_n",
    "female_n",
    "mean_difference",
    "variance_ratio",
    "log_variance_ratio",
    "se_log_variance_ratio",
    "inference_method",
    "qa_flags",
    "evidence_status",
    "headline_eligible",
    "suppression_reason",
    "comparability_tier",
    "provisional",
    "qa_only",
    "method_limited",
    "trait_family",
    "trait_priority",
    "trait_scale_type",
]


def _default_raw_dir(root: Path) -> Path:
    data_dir = first_existing_external_data_dir("hrs_public", root)
    if data_dir is None:
        searched = ", ".join(str(path) for path in external_data_dirs("hrs_public", root))
        raise FileNotFoundError(
            "No HRS raw directory found. "
            f"Searched: {searched}. Place h18core.zip, h20core.zip, h22core.zip, and trk2022v1.zip there or pass --raw-dir."
        )
    core_archives = sorted(path for path in data_dir.glob("*core.zip") if path.is_file())
    if not core_archives:
        raise FileNotFoundError(
            f"No HRS core archives found under {data_dir}. "
            "Expected files such as h18core.zip, h20core.zip, or h22core.zip."
        )
    if not any(path.name.lower().startswith("trk") for path in data_dir.glob("*.zip")):
        raise FileNotFoundError(
            f"No HRS tracker archive found under {data_dir}. "
            "Expected a tracker file such as trk2022v1.zip."
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
    required_cols = key_cols + ["log_variance_ratio", "evidence_status", "headline_eligible"]
    if baseline.empty or any(col not in baseline.columns for col in required_cols):
        return pd.DataFrame(), pd.DataFrame()
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
        if variant_df.empty or any(col not in variant_df.columns for col in required_cols):
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


def _write_hrs_report(
    *,
    out_dir: Path,
    estimates: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> Path:
    report_path = out_dir / "hrs_public_report.md"
    if estimates.empty:
        evidence_summary = pd.DataFrame({"evidence_status": [], "cells": []})
        inference_summary = pd.DataFrame({"evidence_status": [], "inference_method": [], "cells": []})
    else:
        evidence_summary = (
            estimates.groupby("evidence_status", dropna=False)
            .size()
            .reset_index(name="cells")
            .sort_values(["cells", "evidence_status"], ascending=[False, True], kind="stable")
        )
        inference_summary = (
            estimates.groupby(["evidence_status", "inference_method"], dropna=False)
            .size()
            .reset_index(name="cells")
            .sort_values(["evidence_status", "cells"], ascending=[True, False], kind="stable")
        )
    lines = [
        "# HRS public report",
        "",
        "This HRS pass is method-limited. It uses tracker respondent weights and an approximate household-cluster bootstrap with tracker strata, not official replicate weights.",
        "",
        "## Evidence-status summary",
        "",
        markdown_table(evidence_summary) if not evidence_summary.empty else "_No HRS estimate rows were generated._",
        "",
        "## Inference-method summary",
        "",
        markdown_table(inference_summary) if not inference_summary.empty else "_No inference-method rows were generated._",
        "",
        "## Robustness checks",
        "",
        "These compare the weighted baseline against unweighted and winsorized variants.",
        "",
        markdown_table(robustness_summary) if not robustness_summary.empty else "_No robustness summary available._",
        "",
        "## Interpretation guardrails",
        "",
        "- `total_cognition` cells remain provisional when bounded-scale fragility is high.",
        "- `serial_7s` is bounded and coarse, so many cells stay QA-only under the current variance rules.",
        "- No HRS row is headline-eligible in this pass; use HRS as supporting later-life evidence.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HRS public ingest -> estimate pipeline.")
    parser.add_argument("--raw-dir", help="Path to a directory containing HRS public distribution archives.")
    parser.add_argument("--trait", action="append", help="Optional trait filter.")
    parser.add_argument("--output-dir", default="results/hrs_public", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    raw_dir = Path(args.raw_dir) if args.raw_dir else _default_raw_dir(root)
    registry = build_registry(root)
    spec = registry.get_dataset("hrs_public")
    out_dir = ensure_dir(root / args.output_dir)
    config = estimation_config_from_analysis(registry.analysis_config)

    adapter = HRSAdapter(spec, raw_path=raw_dir)
    normalized = adapter.to_long_person_trait()
    data = normalized.data
    if args.trait:
        data = data[data["trait_id"].isin(set(args.trait))].copy()
    estimates = estimate_dataset_cells(data, config=config)
    if estimates.empty:
        estimates = pd.DataFrame(columns=ESTIMATE_OUTPUT_COLUMNS)
    else:
        estimates = annotate_estimate_evidence(estimates, registry=registry)

    robustness_tables = _build_robustness_variants(baseline=data, registry=registry, config=config)
    robustness_comparison, robustness_summary = _build_robustness_comparison(robustness_tables)

    estimates_path = write_table(estimates, out_dir / "hrs_public_trait_estimates.csv")
    summary_cols = [
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
    summary = estimates[summary_cols].copy() if all(col in estimates.columns for col in summary_cols) else pd.DataFrame(columns=summary_cols)
    summary_path = out_dir / "hrs_public_trait_estimates.md"
    write_markdown_summary(summary, summary_path, title="HRS public trait estimates")

    robustness_comparison_path = out_dir / "hrs_public_robustness_comparison.csv"
    robustness_summary_path = out_dir / "hrs_public_robustness_summary.csv"
    robustness_comparison.to_csv(robustness_comparison_path, index=False)
    robustness_summary.to_csv(robustness_summary_path, index=False)
    write_markdown_summary(robustness_summary, out_dir / "hrs_public_robustness_summary.md", title="HRS public robustness summary")
    report_path = _write_hrs_report(out_dir=out_dir, estimates=estimates, robustness_summary=robustness_summary)

    print(f"Wrote {estimates_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {robustness_comparison_path}")
    print(f"Wrote {robustness_summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
