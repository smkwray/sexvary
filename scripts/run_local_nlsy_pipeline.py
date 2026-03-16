#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from sexvary.adapters import LocalWideTableAdapter
from sexvary.config import build_registry, load_local_paths, resolve_local_dataset_path
from sexvary.evidence import annotate_estimate_evidence
from sexvary.estimation import (
    estimate_dataset_cells,
    estimate_sex_difference_cell,
    estimation_config_from_analysis,
    prepare_analysis_frame,
)
from sexvary.meta import dersimonian_laird_meta, fixed_effect_meta
from sexvary.reporting import forest_plot_from_effects, markdown_table, write_markdown_summary
from sexvary.utils import ensure_dir, project_root


def _default_mapping_path(root: Path, dataset_id: str) -> Path:
    preferred = root / "config" / "mappings" / f"{dataset_id}.yaml"
    if preferred.exists():
        return preferred
    fallback = root / "config" / "mappings" / f"{dataset_id}.example.yaml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No mapping file found for {dataset_id}. Checked {preferred.name} and {fallback.name}.")


def _build_dataset_summary(estimates_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, sub in estimates_df.groupby("dataset_id", sort=True):
        rows.append(
            {
                "dataset_id": dataset_id,
                "cells": int(len(sub)),
                "traits": int(sub["trait_id"].nunique()),
                "age_bands": int(sub["age_band"].nunique()),
                "cells_with_ci": int(sub["se_log_variance_ratio"].notna().sum()),
                "headline_eligible_cells": int(sub.get("headline_eligible", pd.Series(False, index=sub.index)).fillna(False).sum()),
                "provisional_cells": int(sub.get("provisional", pd.Series(False, index=sub.index)).fillna(False).sum()),
                "method_limited_cells": int(sub.get("method_limited", pd.Series(False, index=sub.index)).fillna(False).sum()),
                "flagged_cells": int(sub["qa_flags"].notna().sum()),
                "male_greater_cells": int((sub["log_variance_ratio"] > 0).sum()),
                "female_greater_cells": int((sub["log_variance_ratio"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_qa_summary(estimates_df: pd.DataFrame) -> pd.DataFrame:
    work = estimates_df[["dataset_id", "qa_flags"]].copy()
    work["qa_flags"] = work["qa_flags"].fillna("")
    work["qa_flag"] = work["qa_flags"].str.split(";")
    exploded = work.explode("qa_flag")
    exploded["qa_flag"] = exploded["qa_flag"].replace("", "no_flag")
    summary = (
        exploded.groupby(["dataset_id", "qa_flag"], dropna=False)
        .size()
        .reset_index(name="cells")
        .sort_values(["dataset_id", "cells", "qa_flag"], ascending=[True, False, True], kind="stable")
    )
    return summary.reset_index(drop=True)


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


def _build_robustness_variants(
    *,
    normalized_frames: dict[str, pd.DataFrame],
    registry,
    config,
) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    baseline = pd.concat(normalized_frames.values(), ignore_index=True)
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

    coarse_config = replace(config, default_age_band_width_years=max(config.default_age_band_width_years, 6))
    variants["coarse_age_bands"] = _estimate_from_prepared_frame(
        prepare_analysis_frame(baseline, config=coarse_config),
        config=coarse_config,
        registry=registry,
    )
    if "nlsy79_child_ya" in normalized_frames:
        no_maternal = baseline.loc[
            ~(
                (baseline["dataset_id"] == "nlsy79_child_ya")
                & (baseline.get("weight_source", pd.Series(pd.NA, index=baseline.index)).astype("string") == "mother_sampling_weight_79")
            )
        ].copy()
        variants["child_no_maternal_fallback"] = _estimate_from_prepared_frame(
            prepare_analysis_frame(no_maternal, config=config),
            config=config,
            registry=registry,
        )
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


def _build_top_findings(estimates_df: pd.DataFrame) -> pd.DataFrame:
    eligible = estimates_df[np.isfinite(estimates_df["log_variance_ratio"])].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["direction"] = np.where(eligible["log_variance_ratio"] > 0, "male_greater", "female_greater")
    eligible["abs_log_variance_ratio"] = eligible["log_variance_ratio"].abs()
    top = (
        eligible.sort_values(
            ["dataset_id", "direction", "abs_log_variance_ratio"],
            ascending=[True, True, False],
            kind="stable",
        )
        .groupby(["dataset_id", "direction"], as_index=False, sort=False)
        .head(3)
        .reset_index(drop=True)
    )
    return top[
        [
            "dataset_id",
            "direction",
            "trait_id",
            "age_band",
            "log_variance_ratio",
            "variance_ratio",
            "ci_low_variance_ratio",
            "ci_high_variance_ratio",
            "qa_flags",
        ]
    ]


def _write_dataset_forest_plots(estimates_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    plot_dir = ensure_dir(out_dir / "plots")
    outputs: list[Path] = []
    for dataset_id, sub in estimates_df.groupby("dataset_id", sort=True):
        plot_df = sub[np.isfinite(sub["se_log_variance_ratio"])].copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.assign(label=plot_df["trait_id"] + " | " + plot_df["age_band"].astype(str))
        output = forest_plot_from_effects(
            plot_df,
            label_col="label",
            effect_col="log_variance_ratio",
            se_col="se_log_variance_ratio",
            output_path=plot_dir / f"{dataset_id}_log_variance_ratio_forest.png",
            title=f"{dataset_id}: log variance ratios",
        )
        outputs.append(output)
    return outputs


def _build_meta_estimates(estimates_df: pd.DataFrame) -> pd.DataFrame:
    eligible = estimates_df[
        np.isfinite(estimates_df["log_variance_ratio"]) & np.isfinite(estimates_df["se_log_variance_ratio"])
    ].copy()
    if eligible.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for trait_id, sub in eligible.groupby("trait_id", sort=True):
        variances = np.square(sub["se_log_variance_ratio"].to_numpy(dtype=float))
        effects = sub["log_variance_ratio"].to_numpy(dtype=float)
        for fn in (fixed_effect_meta, dersimonian_laird_meta):
            res = fn(effects, variances)
            rows.append(
                {
                    "trait_id": trait_id,
                    "model": res.model,
                    "k": res.k,
                    "pooled_log_variance_ratio": res.estimate,
                    "pooled_variance_ratio": res.estimate_backtransformed,
                    "pooled_se": res.standard_error,
                    "ci_low_log_variance_ratio": res.ci_low,
                    "ci_high_log_variance_ratio": res.ci_high,
                    "ci_low_variance_ratio": float(np.exp(res.ci_low)),
                    "ci_high_variance_ratio": float(np.exp(res.ci_high)),
                    "tau2": res.tau2,
                    "q": res.q,
                    "i2": res.i2,
                }
            )
    return pd.DataFrame(rows)


def _write_local_report(
    *,
    out_dir: Path,
    ingestion_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    qa_summary: pd.DataFrame,
    top_findings: pd.DataFrame,
    meta_df: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> Path:
    report_path = out_dir / "local_nlsy_report.md"
    lines = [
        "# Local NLSY report",
        "",
        "This report is descriptive. Local NLSY confidence intervals are simple-design approximations based on effective sample size, not design-aware survey inference.",
        "",
        "## Dataset overview",
        "",
        markdown_table(dataset_summary),
        "",
        "## Ingestion summary",
        "",
        markdown_table(ingestion_df),
        "",
        "## QA summary",
        "",
        markdown_table(qa_summary),
        "",
    ]

    if not meta_df.empty:
        lines.extend(
            [
                "## Trait meta-estimates",
                "",
                markdown_table(meta_df),
                "",
            ]
        )

    if not top_findings.empty:
        lines.extend(
            [
                "## Top findings",
                "",
                markdown_table(top_findings),
                "",
            ]
        )

    if not robustness_summary.empty:
        lines.extend(
            [
                "## Robustness checks",
                "",
                "These comparisons benchmark the weighted baseline against unweighted, winsorized, and coarser-age-band variants.",
                "",
                markdown_table(robustness_summary),
                "",
            ]
        )

    child_row = dataset_summary[dataset_summary["dataset_id"] == "nlsy79_child_ya"]
    if not child_row.empty:
        child_cells = int(child_row["cells"].iloc[0])
        child_cells_with_ci = int(child_row["cells_with_ci"].iloc[0])
        child_limitations: list[str] = []
        if child_cells_with_ci == 0:
            child_limitations.append(
                "- `nlsy79_child_ya` remains below the current variance-reporting threshold even after pooled-age fallback, so it is present as a QA-tracked dataset but not yet an inferential result source."
            )
        elif child_cells_with_ci < child_cells:
            child_limitations.append(
                "- `nlsy79_child_ya` now has partial inferential coverage, but many child age-band cells still remain below the current variance-reporting threshold and stay QA-only."
            )
        else:
            child_limitations.append(
                "- `nlsy79_child_ya` now reaches inferential coverage for all configured child traits, but rows using maternal-weight fallback or pooled-age fallback are explicitly labeled provisional and should not drive headline claims."
            )
        lines.extend(
            [
                "## Limitations",
                "",
                *child_limitations,
                "- Adult NLSY79 cells now use 4-year age bands to avoid sparse cells; this is a reporting choice, not a claim that age heterogeneity disappears within each band.",
                "- `nlsy97_main` currently behaves like a single pooled adult cell because the available processed extract lacks a cleaner wave-specific analysis field.",
                "",
            ]
        )

    report_path.write_text("\n".join(lines).replace("\n\n\n", "\n\n") + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local NLSY ingest -> estimate pipeline.")
    parser.add_argument("--dataset-id", action="append", help="Dataset id(s) to run. Defaults to all registered local datasets.")
    parser.add_argument("--output-dir", default="results/local_nlsy", help="Output directory relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    local_paths = load_local_paths(root, missing_ok=False).get("local_datasets", {})
    requested = args.dataset_id or list(local_paths)
    if not requested:
        raise SystemExit("No registered local datasets found in config/local_paths.yaml.")

    out_dir = ensure_dir(root / args.output_dir)
    config = estimation_config_from_analysis(registry.analysis_config)

    estimate_frames: list[pd.DataFrame] = []
    normalized_frames: dict[str, pd.DataFrame] = {}
    ingestion_rows: list[dict[str, object]] = []

    for dataset_id in requested:
        if dataset_id not in local_paths:
            raise SystemExit(f"{dataset_id} is not registered in config/local_paths.yaml.")
        spec = registry.get_dataset(dataset_id)
        raw_path = resolve_local_dataset_path(local_paths[dataset_id], root)
        mapping_path = _default_mapping_path(root, dataset_id)
        adapter = LocalWideTableAdapter(spec, raw_path=raw_path, mapping_path=mapping_path)
        normalized = adapter.to_long_person_trait()
        normalized_frames[dataset_id] = normalized.data.copy()
        estimates = estimate_dataset_cells(normalized.data, config=config)
        estimates = annotate_estimate_evidence(estimates, registry=registry)
        if estimates.empty:
            continue
        estimates.insert(0, "mapping_path", str(mapping_path.relative_to(root)))
        estimates.insert(0, "raw_path", local_paths[dataset_id])
        estimate_frames.append(estimates)
        ingestion_rows.append(
            {
                "dataset_id": dataset_id,
                "raw_path": local_paths[dataset_id],
                "mapping_path": str(mapping_path.relative_to(root)),
                "normalized_rows": int(len(normalized.data)),
                "estimated_cells": int(len(estimates)),
                "traits": int(estimates["trait_id"].nunique()),
            }
        )

    if not estimate_frames:
        raise SystemExit("No estimate rows were produced.")

    estimates_df = pd.concat(estimate_frames, ignore_index=True)
    ingestion_df = pd.DataFrame(ingestion_rows)

    estimates_path = out_dir / "local_nlsy_trait_estimates.csv"
    ingestion_path = out_dir / "local_nlsy_ingestion_summary.csv"
    dataset_summary = _build_dataset_summary(estimates_df)
    qa_summary = _build_qa_summary(estimates_df)
    top_findings = _build_top_findings(estimates_df)
    meta_df = _build_meta_estimates(estimates_df)
    robustness_tables = _build_robustness_variants(normalized_frames=normalized_frames, registry=registry, config=config)
    robustness_comparison, robustness_summary = _build_robustness_comparison(robustness_tables)
    dataset_summary_path = out_dir / "local_nlsy_dataset_summary.csv"
    qa_summary_path = out_dir / "local_nlsy_qa_summary.csv"
    top_findings_path = out_dir / "local_nlsy_top_findings.csv"
    meta_path = out_dir / "local_nlsy_meta_estimates.csv"
    robustness_comparison_path = out_dir / "local_nlsy_robustness_comparison.csv"
    robustness_summary_path = out_dir / "local_nlsy_robustness_summary.csv"
    estimates_df.to_csv(estimates_path, index=False)
    ingestion_df.to_csv(ingestion_path, index=False)
    dataset_summary.to_csv(dataset_summary_path, index=False)
    qa_summary.to_csv(qa_summary_path, index=False)
    top_findings.to_csv(top_findings_path, index=False)
    meta_df.to_csv(meta_path, index=False)
    robustness_comparison.to_csv(robustness_comparison_path, index=False)
    robustness_summary.to_csv(robustness_summary_path, index=False)
    write_markdown_summary(ingestion_df, out_dir / "local_nlsy_ingestion_summary.md", title="Local NLSY ingestion summary")
    write_markdown_summary(dataset_summary, out_dir / "local_nlsy_dataset_summary.md", title="Local NLSY dataset summary")
    write_markdown_summary(qa_summary, out_dir / "local_nlsy_qa_summary.md", title="Local NLSY QA summary")
    if not top_findings.empty:
        write_markdown_summary(top_findings, out_dir / "local_nlsy_top_findings.md", title="Local NLSY top findings")
    if not meta_df.empty:
        write_markdown_summary(meta_df, out_dir / "local_nlsy_meta_estimates.md", title="Local NLSY meta estimates")
    if not robustness_summary.empty:
        write_markdown_summary(
            robustness_summary,
            out_dir / "local_nlsy_robustness_summary.md",
            title="Local NLSY robustness summary",
        )
    plot_paths = _write_dataset_forest_plots(estimates_df, out_dir)
    report_path = _write_local_report(
        out_dir=out_dir,
        ingestion_df=ingestion_df,
        dataset_summary=dataset_summary,
        qa_summary=qa_summary,
        top_findings=top_findings,
        meta_df=meta_df,
        robustness_summary=robustness_summary,
    )

    print(f"Wrote {estimates_path}")
    print(f"Wrote {ingestion_path}")
    print(f"Wrote {dataset_summary_path}")
    print(f"Wrote {qa_summary_path}")
    print(f"Wrote {top_findings_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {robustness_comparison_path}")
    print(f"Wrote {robustness_summary_path}")
    print(f"Wrote {report_path}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
