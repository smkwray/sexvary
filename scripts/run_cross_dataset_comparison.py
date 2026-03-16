#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from sexvary.config import build_registry
from sexvary.cross_dataset import (
    DEFAULT_COMPARISON_SOURCES,
    SUPPORTING_EVIDENCE_DATASET_IDS,
    build_age_profile_summary,
    build_dataset_focus_table,
    build_dataset_focus_split,
    build_dataset_inventory,
    build_priority_summary,
    build_supporting_evidence_summary,
    build_supporting_evidence_top_cells,
    build_top_cells,
    build_trait_family_summary,
    load_comparison_tables,
)
from sexvary.orchestration import discover_pipeline_availability
from sexvary.reporting import forest_plot_from_effects, markdown_table, write_markdown_summary
from sexvary.utils import ensure_dir, project_root


def _write_primary_forest(comparison_df: pd.DataFrame, output_path: Path) -> Path | None:
    plot_df = comparison_df[
        comparison_df["headline_eligible"] & (comparison_df["trait_priority"] == "confirmatory")
    ].copy()
    if plot_df.empty:
        return None
    plot_df["abs_log_variance_ratio"] = plot_df["log_variance_ratio"].abs()
    plot_df = plot_df.sort_values("abs_log_variance_ratio", ascending=False, kind="stable").head(18).copy()
    plot_df["label"] = (
        plot_df["dataset_label"]
        + " | "
        + plot_df["trait_label"]
        + " | "
        + plot_df["cycle_or_wave"]
        + " | "
        + plot_df["age_band"]
    )
    return forest_plot_from_effects(
        plot_df,
        label_col="label",
        effect_col="log_variance_ratio",
        se_col="se_log_variance_ratio",
        output_path=output_path,
        title="Primary confirmatory log variance ratio comparison",
    )


def _write_secondary_forest(comparison_df: pd.DataFrame, output_path: Path) -> Path | None:
    plot_df = comparison_df[
        comparison_df["headline_eligible"] & comparison_df["trait_priority"].isin(["secondary", "exploratory"])
    ].copy()
    if plot_df.empty:
        return None
    plot_df["abs_log_variance_ratio"] = plot_df["log_variance_ratio"].abs()
    plot_df = plot_df.sort_values("abs_log_variance_ratio", ascending=False, kind="stable").head(18).copy()
    plot_df["label"] = (
        plot_df["dataset_label"]
        + " | "
        + plot_df["trait_label"]
        + " | "
        + plot_df["cycle_or_wave"]
        + " | "
        + plot_df["age_band"]
    )
    return forest_plot_from_effects(
        plot_df,
        label_col="label",
        effect_col="log_variance_ratio",
        se_col="se_log_variance_ratio",
        output_path=output_path,
        title="Secondary / exploratory log variance ratio comparison",
    )


def _write_age_profile(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    datasets = list(summary_df["dataset_label"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, max(3.5, 2.8 * len(datasets))), squeeze=False)
    for ax, dataset_label in zip(axes.ravel(), datasets):
        sub = summary_df[summary_df["dataset_label"] == dataset_label].copy()
        x = np.arange(len(sub))
        ax.plot(x, sub["median_log_variance_ratio"], marker="o", linewidth=2)
        ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["age_band"], rotation=45, ha="right")
        ax.set_ylabel("Median log VR")
        ax.set_title(dataset_label)
    axes.ravel()[-1].set_xlabel("Age / grade band")
    fig.suptitle("Age-profile comparison across datasets", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _write_mean_variance_scatter(comparison_df: pd.DataFrame, output_path: Path) -> Path | None:
    plot_df = comparison_df[
        np.isfinite(comparison_df["log_variance_ratio"]) & np.isfinite(comparison_df["mean_difference"])
    ].copy()
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for dataset_label, sub in plot_df.groupby("dataset_label", sort=True):
        ax.scatter(sub["mean_difference"], sub["log_variance_ratio"], alpha=0.75, label=dataset_label)
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
    ax.axvline(0.0, linestyle=":", linewidth=1, color="black")
    ax.set_xlabel("Mean difference (male - female)")
    ax.set_ylabel("Log variance ratio")
    ax.set_title("Mean vs variance differences across analyzed cells")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _write_dataset_family_summary_plot(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    plot_df = summary_df[summary_df["cells_with_ci"] > 0].copy()
    if plot_df.empty:
        return None
    plot_df["label"] = plot_df["dataset_label"] + " | " + plot_df["trait_family"]
    plot_df = plot_df.sort_values("median_log_variance_ratio", kind="stable").tail(18)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.45 * len(plot_df) + 1.5)))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["median_log_variance_ratio"], color=np.where(plot_df["median_log_variance_ratio"] >= 0, "#2f6f4f", "#9b3d2f"))
    ax.axvline(0.0, linestyle="--", linewidth=1, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Median log variance ratio")
    ax.set_title("Dataset-family summary")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _load_robustness_summary(root: Path) -> pd.DataFrame:
    sources = [
        ("local_nlsy", "Local NLSY", root / "results" / "local_nlsy" / "local_nlsy_robustness_summary.csv"),
        ("hrs_public", "HRS public", root / "results" / "hrs_public" / "hrs_public_robustness_summary.csv"),
        ("psid_cds_tas", "PSID CDS / TAS", root / "results" / "psid_cds_tas" / "psid_cds_tas_robustness_summary.csv"),
        ("nhanes_2011_2023", "NHANES selected cycles", root / "results" / "nhanes_2011_2023" / "nhanes_2011_2023_robustness_summary.csv"),
    ]
    frames: list[pd.DataFrame] = []
    for dataset_id, dataset_label, path in sources:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            continue
        if df.empty:
            continue
        df.insert(0, "dataset_id", dataset_id)
        df.insert(1, "dataset_label", dataset_label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_robustness_comparison_plot(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["dataset_label"] + " | " + plot_df["variant"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(plot_df["label"], plot_df["median_abs_delta"], color="#336699")
    axes[0].set_title("Median absolute delta vs baseline")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(plot_df["label"], plot_df["sign_change_rate"], color="#996633")
    axes[1].set_title("Sign change rate vs baseline")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Cross-dataset robustness summary")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _supporting_robustness_summary(robustness_summary: pd.DataFrame) -> pd.DataFrame:
    if robustness_summary.empty:
        return pd.DataFrame()
    return robustness_summary[
        robustness_summary["dataset_id"].isin(SUPPORTING_EVIDENCE_DATASET_IDS)
    ].reset_index(drop=True)


def _build_evidence_audit(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    ecls_target = comparison_df[
        (comparison_df["dataset_id"] == "ecls_k_2011")
        & (comparison_df["trait_id"] == "reading_achievement")
        & (comparison_df["cycle_or_wave"] == "fall_kindergarten_2010")
    ]
    if not ecls_target.empty:
        row = ecls_target.sort_values("log_variance_ratio", kind="stable").iloc[0]
        rows.append(
            {
                "audit_topic": "ECLS-K kindergarten reading reversal",
                "dataset_label": row["dataset_label"],
                "trait_label": row["trait_label"],
                "cycle_or_wave": row["cycle_or_wave"],
                "age_band": row["age_band"],
                "evidence_status": row["evidence_status"],
                "log_variance_ratio": row["log_variance_ratio"],
                "conclusion": "High-priority reversal; keep in main text and audit before any stronger interpretation.",
            }
        )

    child_targets = comparison_df[
        (comparison_df["dataset_id"] == "nlsy79_child_ya") & comparison_df["provisional"] & comparison_df["ci_available"]
    ]
    for _, row in child_targets.iterrows():
        rows.append(
            {
                "audit_topic": "Child/YA fallback-weight inference",
                "dataset_label": row["dataset_label"],
                "trait_label": row["trait_label"],
                "cycle_or_wave": row["cycle_or_wave"],
                "age_band": row["age_band"],
                "evidence_status": row["evidence_status"],
                "log_variance_ratio": row["log_variance_ratio"],
                "conclusion": "Keep in appendix and sensitivity checks; exclude from headline claims.",
            }
        )

    nhanes_targets = comparison_df[
        (comparison_df["dataset_id"] == "nhanes_2011_2023") & (comparison_df["trait_family"] == "later_life_cognition")
    ]
    for _, row in nhanes_targets.head(8).iterrows():
        rows.append(
            {
                "audit_topic": "NHANES bounded cognition rows",
                "dataset_label": row["dataset_label"],
                "trait_label": row["trait_label"],
                "cycle_or_wave": row["cycle_or_wave"],
                "age_band": row["age_band"],
                "evidence_status": row["evidence_status"],
                "log_variance_ratio": row["log_variance_ratio"],
                "conclusion": "Treat as QA-only bounded-screen rows; not part of headline evidence.",
            }
        )

    for dataset_id in ("piaac_cycle2", "pisa_2022", "timss_2019", "timss_2023", "pirls_2021"):
        sub = comparison_df[
            (comparison_df["dataset_id"] == dataset_id)
            & (comparison_df["trait_priority"] == "confirmatory")
            & comparison_df["headline_eligible"]
        ]
        if sub.empty:
            continue
        row = sub.sort_values("log_variance_ratio", ascending=False, kind="stable").iloc[0]
        rows.append(
            {
                "audit_topic": "Strongest positive confirmatory cell",
                "dataset_label": row["dataset_label"],
                "trait_label": row["trait_label"],
                "cycle_or_wave": row["cycle_or_wave"],
                "age_band": row["age_band"],
                "evidence_status": row["evidence_status"],
                "log_variance_ratio": row["log_variance_ratio"],
                "conclusion": "Retain as headline-eligible confirmatory evidence.",
            }
        )

    return pd.DataFrame(rows)


def _write_evidence_audit_report(audit_df: pd.DataFrame, output_path: Path) -> Path:
    lines = [
        "# Evidence audit",
        "",
        "This audit records the current high-priority cells that require explicit interpretation choices.",
        "",
    ]
    if audit_df.empty:
        lines.extend(["No audit rows were generated.", ""])
    else:
        lines.extend([markdown_table(audit_df), ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _build_coverage_gap_summary(root: Path, comparison_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability = discover_pipeline_availability(root)
    present_dataset_ids = set(comparison_df["dataset_id"].dropna().astype(str).unique())
    comparison_output_exists = {
        source_id: (path if path.is_absolute() else root / path).exists()
        for source_id, path in DEFAULT_COMPARISON_SOURCES.items()
    }

    missing_rows: list[dict[str, str]] = []
    reproducibility_rows: list[dict[str, str]] = []
    for item in availability:
        output_exists = comparison_output_exists.get(item.pipeline_id, False)
        if item.status == "missing_input":
            if output_exists or item.pipeline_id in present_dataset_ids:
                reproducibility_rows.append(
                    {
                        "pipeline_id": item.pipeline_id,
                        "dataset_label": item.label,
                        "issue": "result_present_but_input_missing",
                        "details": item.reason or "",
                    }
                )
            else:
                missing_rows.append(
                    {
                        "pipeline_id": item.pipeline_id,
                        "dataset_label": item.label,
                        "issue": "missing_input",
                        "details": item.reason or "",
                    }
                )

    return pd.DataFrame(missing_rows), pd.DataFrame(reproducibility_rows)


def _load_latest_backend_manifest(root: Path) -> dict[str, object] | None:
    manifest_dir = root / "results" / "run_manifests"
    if not manifest_dir.exists():
        return None

    candidates = sorted(
        manifest_dir.glob("backend_run_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    fallback: dict[str, object] | None = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if fallback is None:
            fallback = payload
        if not bool(payload.get("dry_run", False)):
            return payload
    return fallback


def _summarize_backend_manifest(manifest: dict[str, object] | None) -> list[str]:
    if not manifest:
        return []
    selection = manifest.get("pipeline_selection", []) or []
    if not selection:
        selected_pipelines = manifest.get("selected_pipelines", []) or []
        selection = [
            {
                "pipeline_id": row.get("pipeline_id"),
                "selection_status": "selected",
            }
            for row in selected_pipelines
        ]
    runs = manifest.get("pipeline_runs", []) or []
    compare_run = manifest.get("compare_run")

    selection_counts: dict[str, int] = {}
    for row in selection:
        status = str(row.get("selection_status", "unknown"))
        selection_counts[status] = selection_counts.get(status, 0) + 1

    failed_runs = [row for row in runs if row.get("status") == "failed"]
    compare_failed = bool(compare_run and compare_run.get("status") == "failed")
    dry_run = bool(manifest.get("dry_run", False))

    lines = [
        "## Backend state",
        "",
        f"- Manifest mode: `{'dry_run' if dry_run else 'executed'}`.",
        f"- Pipeline selection counts: `selected={selection_counts.get('selected', 0)}`, `missing_input={selection_counts.get('missing_input', 0)}`, `excluded={selection_counts.get('excluded', 0)}`, `not_selected={selection_counts.get('not_selected', 0)}`.",
        f"- Pipeline run failures in the latest manifest: `{len(failed_runs)}`. Comparison rebuild failed: `{'yes' if compare_failed else 'no'}`.",
        "",
    ]
    return lines


def _write_report(
    *,
    comparison_df: pd.DataFrame,
    dataset_inventory: pd.DataFrame,
    trait_family_summary: pd.DataFrame,
    priority_summary: pd.DataFrame,
    confirmatory_top_cells: pd.DataFrame,
    secondary_top_cells: pd.DataFrame,
    headline_confirmatory_cells: pd.DataFrame,
    provisional_inferential_cells: pd.DataFrame,
    method_limited_inferential_cells: pd.DataFrame,
    qa_only_cells: pd.DataFrame,
    child_inferential_cells: pd.DataFrame,
    child_qa_only_cells: pd.DataFrame,
    nhanes_inferential_cells: pd.DataFrame,
    nhanes_qa_cognition_cells: pd.DataFrame,
    supporting_summary: pd.DataFrame,
    supporting_top_cells: pd.DataFrame,
    supporting_robustness: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    missing_coverage: pd.DataFrame,
    reproducibility_gaps: pd.DataFrame,
    output_path: Path,
) -> Path:
    lines = [
        "# Cross-dataset comparison report",
        "",
        "This report is descriptive and combines the currently available local NLSY, PIAAC cycle 2, PISA 2022, TIMSS, PIRLS, ICILS, NHANES, ECLS-K:2011, and HSLS:09 outputs.",
        "",
        "## Dataset inventory",
        "",
        markdown_table(dataset_inventory),
        "",
        "## Trait-family summary",
        "",
        markdown_table(trait_family_summary),
        "",
        "## Priority summary",
        "",
        markdown_table(priority_summary),
        "",
    ]

    inferential_rows = int(comparison_df["ci_available"].sum())
    flagged_rows = int(comparison_df["qa_flags"].notna().sum())
    lines.extend(
        [
            "## Summary",
            "",
            f"- Combined cells: `{len(comparison_df)}`",
            f"- Cells with approximate uncertainty: `{inferential_rows}`",
            f"- Cells carrying QA flags: `{flagged_rows}`",
            "",
        ]
    )

    confirmatory_rows = int((comparison_df["trait_priority"] == "confirmatory").sum())
    secondary_rows = int(comparison_df["trait_priority"].isin(["secondary", "exploratory"]).sum())
    lines.extend(
        [
            f"- Confirmatory cells: `{confirmatory_rows}`",
            f"- Secondary or exploratory cells: `{secondary_rows}`",
            "",
        ]
    )

    if not confirmatory_top_cells.empty:
        lines.extend(
            [
                "## Headline confirmatory cells",
                "",
                markdown_table(headline_confirmatory_cells.head(20) if not headline_confirmatory_cells.empty else confirmatory_top_cells.head(20)),
                "",
                "## Confirmatory cells with the largest absolute log variance ratios",
                "",
                markdown_table(confirmatory_top_cells),
                "",
            ]
        )

    if not secondary_top_cells.empty:
        lines.extend(
            [
                "## Secondary or exploratory cells with the largest absolute log variance ratios",
                "",
                markdown_table(secondary_top_cells),
                "",
            ]
        )

    if not provisional_inferential_cells.empty:
        lines.extend(
            [
                "## Provisional inferential cells",
                "",
                markdown_table(provisional_inferential_cells.head(20)),
                "",
            ]
        )

    if not method_limited_inferential_cells.empty:
        lines.extend(
            [
                "## Method-limited inferential cells",
                "",
                markdown_table(method_limited_inferential_cells.head(20)),
                "",
            ]
        )

    if not qa_only_cells.empty:
        lines.extend(
            [
                "## QA-only cells",
                "",
                markdown_table(qa_only_cells.head(20)),
                "",
            ]
        )

    if not child_inferential_cells.empty or not child_qa_only_cells.empty:
        lines.extend(
            [
                "## NLSY79 Child and Young Adult split",
                "",
                "These local child results are easier to interpret when inferential cells are separated from sparse QA-only cells.",
                "",
            ]
        )
        if not child_inferential_cells.empty:
            lines.extend(
                [
                    "### Inferential child cells",
                    "",
                    markdown_table(child_inferential_cells),
                    "",
                ]
            )
        if not child_qa_only_cells.empty:
            lines.extend(
                [
                    "### QA-only child cells",
                    "",
                    markdown_table(child_qa_only_cells),
                    "",
                ]
            )

    if not nhanes_inferential_cells.empty or not nhanes_qa_cognition_cells.empty:
        lines.extend(
            [
                "## NHANES split",
                "",
                "NHANES is easier to read when the inferential physical-measure cells are separated from the QA-only cognition-screen rows.",
                "",
            ]
        )
        if not nhanes_inferential_cells.empty:
            lines.extend(
                [
                    "### Inferential NHANES cells",
                    "",
                    markdown_table(nhanes_inferential_cells),
                    "",
                ]
            )
        if not nhanes_qa_cognition_cells.empty:
            lines.extend(
                [
                    "### QA-only NHANES cognition cells",
                    "",
                    markdown_table(nhanes_qa_cognition_cells),
                    "",
                ]
            )

    if not supporting_summary.empty:
        lines.extend(
            [
                "## Supporting-evidence appendix",
                "",
                "These datasets broaden scope beyond the headline confirmatory layer. NHANES adds physical-trait evidence, HRS adds later-life cognition, and PSID adds panel-based child/adolescent and young-adult measures.",
                "",
                markdown_table(supporting_summary),
                "",
            ]
        )
        if not supporting_top_cells.empty:
            lines.extend(
                [
                    "### Largest inferential supporting-evidence cells",
                    "",
                    markdown_table(supporting_top_cells),
                    "",
                ]
            )
        if not supporting_robustness.empty:
            lines.extend(
                [
                    "### Supporting-evidence robustness",
                    "",
                    markdown_table(supporting_robustness),
                    "",
                ]
            )

    if not robustness_summary.empty:
        lines.extend(
            [
                "## Robustness summary",
                "",
                "This combines the currently available dataset-specific robustness summaries.",
                "",
                markdown_table(robustness_summary),
                "",
            ]
        )

    if not missing_coverage.empty or not reproducibility_gaps.empty:
        lines.extend(
            [
                "## Coverage gaps",
                "",
                "These are backend coverage gaps, not null findings.",
                "",
            ]
        )
        if not missing_coverage.empty:
            lines.extend(
                [
                    "### Datasets currently absent because inputs are missing",
                    "",
                    markdown_table(missing_coverage),
                    "",
                ]
            )
        if not reproducibility_gaps.empty:
            lines.extend(
                [
                    "### Datasets present in outputs but not currently rerunnable from local raw inputs",
                    "",
                    markdown_table(reproducibility_gaps),
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_final_report(
    *,
    comparison_df: pd.DataFrame,
    dataset_inventory: pd.DataFrame,
    priority_summary: pd.DataFrame,
    confirmatory_top_cells: pd.DataFrame,
    headline_confirmatory_cells: pd.DataFrame,
    provisional_inferential_cells: pd.DataFrame,
    method_limited_inferential_cells: pd.DataFrame,
    qa_only_cells: pd.DataFrame,
    supporting_summary: pd.DataFrame,
    supporting_top_cells: pd.DataFrame,
    supporting_robustness: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    missing_coverage: pd.DataFrame,
    reproducibility_gaps: pd.DataFrame,
    output_path: Path,
) -> Path:
    ecls_rows = comparison_df[comparison_df["dataset_id"] == "ecls_k_2011"]
    ecls_design_aware = int((ecls_rows["inference_method"] == "stratified_cluster_bootstrap_psu").sum()) if "inference_method" in ecls_rows.columns else 0
    hsls_rows = comparison_df[comparison_df["dataset_id"] == "hsls_2009"]
    hsls_brr = int((hsls_rows["inference_method"] == "replicate_weights_brr").sum()) if "inference_method" in hsls_rows.columns else 0
    pisa_rows = comparison_df[comparison_df["dataset_id"] == "pisa_2022"]
    pisa_brr = int((pisa_rows["inference_method"] == "replicate_weights_brr").sum()) if "inference_method" in pisa_rows.columns else 0
    nhanes_rows = comparison_df[comparison_df["dataset_id"] == "nhanes_2011_2023"].copy()
    nhanes_physical = nhanes_rows[
        (nhanes_rows["trait_family"] == "physical") & nhanes_rows["ci_available"]
    ].copy()
    nhanes_cognition_qa = nhanes_rows[
        (nhanes_rows["trait_family"] == "later_life_cognition") & ~nhanes_rows["ci_available"]
    ].copy()
    confirmatory_live = comparison_df[
        (comparison_df["trait_priority"] == "confirmatory") & comparison_df["headline_eligible"]
    ].copy()
    hrs_rows = comparison_df[comparison_df["dataset_id"] == "hrs_public"].copy()
    psid_rows = comparison_df[comparison_df["dataset_id"] == "psid_cds_tas"].copy()
    lines = [
        "# Sex Variability Report",
        "",
        "## Research question",
        "",
        "This report summarizes the current descriptive evidence on male versus female variability across the datasets that are already live in the repo.",
        "",
        "## Scope",
        "",
        "- This report is descriptive, not causal.",
        "- Headline claims are restricted to confirmatory cells that are inferential and not marked provisional or method-limited.",
        "- Secondary, provisional, and QA-only results are retained below but are interpreted separately.",
        "",
    ]

    if not confirmatory_live.empty:
        positive_share = float((confirmatory_live["log_variance_ratio"] > 0).mean())
        strongest_positive = confirmatory_live.loc[confirmatory_live["log_variance_ratio"].idxmax()]
        strongest_negative = confirmatory_live.loc[confirmatory_live["log_variance_ratio"].idxmin()]
        lines.extend(
            [
                "## Headline results",
                "",
                f"- Across the headline-eligible confirmatory cells, male variability is more common than female variability (`{positive_share:.0%}` of cells are positive).",
                f"- The strongest positive confirmatory pattern in the current output is `{strongest_positive['trait_label']}` in `{strongest_positive['dataset_label']}` at `{strongest_positive['age_band']}` during `{strongest_positive['cycle_or_wave']}` (log VR `{strongest_positive['log_variance_ratio']:.3f}`).",
                f"- The strongest negative confirmatory pattern is `{strongest_negative['trait_label']}` in `{strongest_negative['dataset_label']}` at `{strongest_negative['age_band']}` during `{strongest_negative['cycle_or_wave']}` (log VR `{strongest_negative['log_variance_ratio']:.3f}`).",
                "- Public-facing summaries should keep both parts of the pattern visible: adult-skills and later-school datasets mostly lean male-greater in variability, while early-school reading contains the clearest reversal.",
                "",
            ]
        )

    lines.extend(_summarize_backend_manifest(_load_latest_backend_manifest(output_path.parents[2])))

    lines.extend(
        [
        "## Datasets",
        "",
        markdown_table(dataset_inventory[["dataset_label", "rows", "rows_with_ci", "headline_eligible_rows", "provisional_rows", "qa_only_rows"]]),
        "",
        "## Data and methods",
        "",
        "- Point estimates are weighted and cell-based.",
        "- PIAAC uses its plausible-value plus replicate-weight pipeline.",
        f"- PISA 2022 now uses a U.S.-first plausible-value plus BRR replicate-weight pipeline (`{pisa_brr}` cells).",
        f"- ECLS-K now uses stratified PSU bootstrap uncertainty where design metadata support it (`{ecls_design_aware}` cells), with simple-design fallback only where the design bootstrap degenerates.",
        f"- HSLS:09 now uses BRR replicate weights from the 2017 PETS/SR student archive (`{hsls_brr}` cells), so it no longer depends on the masked PSU/strata IDs in that public file.",
        "- NHANES now contributes design-aware physical-trait cells across selected cycles, while the cognition screen remains QA-only under the current cell thresholds.",
        "- PSID CDS / TAS now has its own robustness appendix, but all current PSID inferential rows remain method-limited or provisional.",
        "- Local NLSY uncertainty remains a simple effective-sample-size approximation.",
        "- HRS now uses an approximate household-cluster bootstrap with tracker strata and is still treated as method-limited supporting evidence rather than headline evidence.",
        "",
        "## Confirmatory results",
        "",
        markdown_table(priority_summary[priority_summary["trait_priority"] == "confirmatory"]),
        "",
    ])

    if not headline_confirmatory_cells.empty:
        lines.extend(
            [
                "## Leading headline-eligible confirmatory cells",
                "",
                markdown_table(headline_confirmatory_cells.head(12)),
                "",
            ]
        )

    confirmatory_rows = comparison_df[comparison_df["trait_priority"] == "confirmatory"]
    confirmatory_with_ci = int(confirmatory_rows["headline_eligible"].sum())
    lines.extend(
        [
            "## Secondary and provisional results",
            "",
            f"- Confirmatory coverage spans `{len(confirmatory_rows)}` cells, of which `{confirmatory_with_ci}` are currently headline-eligible.",
            f"- Provisional inferential rows currently total `{len(provisional_inferential_cells)}` and method-limited inferential rows total `{len(method_limited_inferential_cells)}`.",
            "- NLSY outputs remain useful for replication structure, but Child/YA fallback-weight rows are kept out of headline claims by design.",
            "- PSID CDS / TAS remains useful as supporting panel evidence, but it still sits below headline quality because its current inference path is simple-design and some TAS rows rely on alternate public weights.",
            "",
        ]
    )

    if not nhanes_physical.empty or not nhanes_cognition_qa.empty:
        lines.extend(
            [
                "## NHANES findings",
                "",
                f"- NHANES currently contributes `{len(nhanes_physical)}` inferential physical-trait cells and `{len(nhanes_cognition_qa)}` QA-only cognition-screen cells.",
            ]
        )
        if not nhanes_physical.empty:
            positive = nhanes_physical.loc[nhanes_physical["log_variance_ratio"].idxmax()]
            negative = nhanes_physical.loc[nhanes_physical["log_variance_ratio"].idxmin()]
            grip_rows = nhanes_physical[nhanes_physical["trait_label"] == "Grip strength (kg)"]
            bmi_rows = nhanes_physical[nhanes_physical["trait_label"] == "BMI"]
            lines.append(
                f"- The clearest NHANES male-greater variability cell is `{positive['trait_label']}` in `{positive['cycle_or_wave']}` at `{positive['age_band']}` (log VR `{positive['log_variance_ratio']:.3f}`)."
            )
            lines.append(
                f"- The clearest NHANES female-greater variability cell is `{negative['trait_label']}` in `{negative['cycle_or_wave']}` at `{negative['age_band']}` (log VR `{negative['log_variance_ratio']:.3f}`)."
            )
            if not grip_rows.empty and not bmi_rows.empty:
                grip_positive_share = float((grip_rows["log_variance_ratio"] > 0).mean())
                bmi_positive_share = float((bmi_rows["log_variance_ratio"] > 0).mean())
                lines.append(
                    f"- Grip strength is the most consistently male-greater NHANES trait in the live output (`{grip_positive_share:.0%}` positive cells), while BMI is mixed and often reverses direction (`{bmi_positive_share:.0%}` positive cells)."
                )
        if not nhanes_cognition_qa.empty:
            lines.append(
                "- The NHANES cognition screen remains QA-only because the current age-banded cells stay below the variance threshold, especially in older-adult bands."
            )
        lines.append("")

    if not supporting_summary.empty:
        hrs_ci = int(hrs_rows["ci_available"].sum())
        hrs_provisional = int(hrs_rows["provisional"].sum())
        psid_ci = int(psid_rows["ci_available"].sum())
        psid_provisional = int(psid_rows["provisional"].sum())
        psid_method_limited = int(psid_rows["method_limited"].sum())
        lines.extend(
            [
                "## Supporting evidence",
                "",
                "- NHANES broadens the project beyond education and cognition. Its live physical-trait rows are fairly stable under winsorization and only modestly weight-sensitive, while the cognition screen remains QA-only.",
                f"- HRS currently contributes `{hrs_ci}` inferential later-life rows and `{hrs_provisional}` provisional rows. Public-facing summaries should treat HRS as supporting or unavailable evidence, not headline evidence.",
                f"- PSID CDS / TAS currently contributes `{psid_ci}` inferential rows, with `{psid_method_limited}` method-limited rows and `{psid_provisional}` provisional rows. Public-facing summaries should keep PSID in a supporting panel-evidence tier, not a headline tier.",
                "",
                markdown_table(supporting_summary),
                "",
            ]
        )
        if not supporting_top_cells.empty:
            lines.extend(
                [
                    "### Largest supporting-evidence cells",
                    "",
                    markdown_table(supporting_top_cells.head(12)),
                    "",
                ]
            )

    if not robustness_summary.empty:
        lines.extend(
            [
                "## Robustness checks",
                "",
                "The current robustness appendix includes local NLSY plus the main supporting-evidence datasets now live in the backend: HRS, PSID CDS / TAS, and NHANES.",
                "",
                markdown_table(robustness_summary),
                "",
            ]
        )

    if not supporting_robustness.empty:
        lines.extend(
            [
                "### Supporting-evidence robustness detail",
                "",
                markdown_table(supporting_robustness),
                "",
            ]
        )

    if not missing_coverage.empty or not reproducibility_gaps.empty:
        lines.extend(
            [
                "## Coverage gaps",
                "",
            ]
        )
        if not missing_coverage.empty:
            labels = ", ".join(f"`{label}`" for label in missing_coverage["dataset_label"].tolist())
            lines.append(f"- The following supported datasets are currently absent from the live evidence because local raw inputs are missing: {labels}.")
        if not reproducibility_gaps.empty:
            labels = ", ".join(f"`{label}`" for label in reproducibility_gaps["dataset_label"].tolist())
            lines.append(f"- The following datasets still appear in current outputs but are not locally rerunnable until raw inputs are restored: {labels}.")
        lines.append("")

    lines.extend(
        [
            "## Limitations and interpretation guardrails",
            "",
            "- This repo is still descriptive and replication-oriented, not causal.",
            "- HSLS is now included, but only through a math-only repeated-wave student-file path so far.",
            "- The current PISA path defaults to `USA` for tractability; multi-country expansion should be explicit rather than accidental.",
            "- Some ECLS-K fifth-grade cells still fall back to simple-design uncertainty because the design bootstrap provides zero variation under the available PSU/stratum metadata.",
            "- Provisional rows, QA-only rows, and method-limited rows are retained for transparency but excluded from the headline-evidence layer.",
            "- Public-facing pages should use headline-eligible confirmatory rows for the main claim and keep supporting/provisional rows in separate sections.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-dataset comparison tables and figures.")
    parser.add_argument("--output-root", default="results", help="Output root relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    comparison_df = load_comparison_tables(root, registry=registry)

    tables_dir = ensure_dir(root / args.output_root / "tables")
    figures_dir = ensure_dir(root / args.output_root / "figures")
    reports_dir = ensure_dir(root / args.output_root / "reports")

    dataset_inventory = build_dataset_inventory(comparison_df)
    trait_family_summary = build_trait_family_summary(comparison_df)
    priority_summary = build_priority_summary(comparison_df)
    age_profile_summary = build_age_profile_summary(comparison_df)
    confirmatory_top_cells = build_top_cells(comparison_df[comparison_df["ci_available"]], limit=20, priorities=("confirmatory",))
    secondary_top_cells = build_top_cells(comparison_df[comparison_df["ci_available"]], limit=20, priorities=("secondary", "exploratory"))
    headline_confirmatory_cells = build_top_cells(
        comparison_df[comparison_df["headline_eligible"]],
        limit=50,
        priorities=("confirmatory",),
    )
    provisional_inferential_cells = comparison_df[comparison_df["provisional"]].sort_values(
        ["dataset_label", "trait_label", "cycle_or_wave", "age_band"], kind="stable"
    )[
        ["dataset_label", "cycle_or_wave", "age_band", "trait_label", "trait_family", "evidence_status", "log_variance_ratio", "se_log_variance_ratio", "suppression_reason", "qa_flags"]
    ].reset_index(drop=True)
    method_limited_inferential_cells = comparison_df[comparison_df["method_limited"]].sort_values(
        ["dataset_label", "trait_label", "cycle_or_wave", "age_band"], kind="stable"
    )[
        ["dataset_label", "cycle_or_wave", "age_band", "trait_label", "trait_family", "evidence_status", "log_variance_ratio", "se_log_variance_ratio", "suppression_reason", "qa_flags"]
    ].reset_index(drop=True)
    qa_only_cells = comparison_df[comparison_df["qa_only"]].sort_values(
        ["dataset_label", "trait_label", "cycle_or_wave", "age_band"], kind="stable"
    )[
        ["dataset_label", "cycle_or_wave", "age_band", "trait_label", "trait_family", "evidence_status", "suppression_reason", "qa_flags"]
    ].reset_index(drop=True)
    missing_coverage, reproducibility_gaps = _build_coverage_gap_summary(root, comparison_df)
    robustness_summary = _load_robustness_summary(root)
    supporting_summary = build_supporting_evidence_summary(comparison_df)
    supporting_top_cells = build_supporting_evidence_top_cells(comparison_df, limit=30)
    supporting_robustness = _supporting_robustness_summary(robustness_summary)
    audit_df = _build_evidence_audit(comparison_df)
    child_inferential_cells, child_qa_only_cells = build_dataset_focus_split(
        comparison_df, dataset_id="nlsy79_child_ya"
    )
    nhanes_inferential_cells = build_dataset_focus_table(
        comparison_df,
        dataset_id="nhanes_2011_2023",
        ci_available=True,
        trait_families=("physical",),
    )
    nhanes_qa_cognition_cells = build_dataset_focus_table(
        comparison_df,
        dataset_id="nhanes_2011_2023",
        ci_available=False,
        trait_families=("later_life_cognition",),
    )

    analysis_cells_path = tables_dir / "cross_dataset_analysis_cells.csv"
    dataset_inventory_path = tables_dir / "cross_dataset_dataset_inventory.csv"
    trait_family_summary_path = tables_dir / "cross_dataset_trait_family_summary.csv"
    priority_summary_path = tables_dir / "cross_dataset_priority_summary.csv"
    age_profile_summary_path = tables_dir / "cross_dataset_age_profile_summary.csv"
    confirmatory_top_cells_path = tables_dir / "cross_dataset_confirmatory_top_cells.csv"
    secondary_top_cells_path = tables_dir / "cross_dataset_secondary_top_cells.csv"
    headline_confirmatory_cells_path = tables_dir / "cross_dataset_headline_confirmatory_cells.csv"
    provisional_inferential_cells_path = tables_dir / "cross_dataset_provisional_inferential_cells.csv"
    method_limited_inferential_cells_path = tables_dir / "cross_dataset_method_limited_inferential_cells.csv"
    qa_only_cells_path = tables_dir / "cross_dataset_qa_only_cells.csv"
    child_inferential_cells_path = tables_dir / "cross_dataset_nlsy79_child_ya_inferential_cells.csv"
    child_qa_only_cells_path = tables_dir / "cross_dataset_nlsy79_child_ya_qa_only_cells.csv"
    nhanes_inferential_cells_path = tables_dir / "cross_dataset_nhanes_inferential_cells.csv"
    nhanes_qa_cognition_cells_path = tables_dir / "cross_dataset_nhanes_qa_only_cognition_cells.csv"
    supporting_summary_path = tables_dir / "cross_dataset_supporting_evidence_summary.csv"
    supporting_top_cells_path = tables_dir / "cross_dataset_supporting_evidence_top_cells.csv"
    supporting_robustness_path = tables_dir / "cross_dataset_supporting_evidence_robustness.csv"
    robustness_summary_path = tables_dir / "cross_dataset_robustness_comparison.csv"
    audit_path = tables_dir / "cross_dataset_evidence_audit.csv"

    comparison_df.to_csv(analysis_cells_path, index=False)
    dataset_inventory.to_csv(dataset_inventory_path, index=False)
    trait_family_summary.to_csv(trait_family_summary_path, index=False)
    priority_summary.to_csv(priority_summary_path, index=False)
    age_profile_summary.to_csv(age_profile_summary_path, index=False)
    confirmatory_top_cells.to_csv(confirmatory_top_cells_path, index=False)
    secondary_top_cells.to_csv(secondary_top_cells_path, index=False)
    headline_confirmatory_cells.to_csv(headline_confirmatory_cells_path, index=False)
    provisional_inferential_cells.to_csv(provisional_inferential_cells_path, index=False)
    method_limited_inferential_cells.to_csv(method_limited_inferential_cells_path, index=False)
    qa_only_cells.to_csv(qa_only_cells_path, index=False)
    child_inferential_cells.to_csv(child_inferential_cells_path, index=False)
    child_qa_only_cells.to_csv(child_qa_only_cells_path, index=False)
    nhanes_inferential_cells.to_csv(nhanes_inferential_cells_path, index=False)
    nhanes_qa_cognition_cells.to_csv(nhanes_qa_cognition_cells_path, index=False)
    supporting_summary.to_csv(supporting_summary_path, index=False)
    supporting_top_cells.to_csv(supporting_top_cells_path, index=False)
    supporting_robustness.to_csv(supporting_robustness_path, index=False)
    robustness_summary.to_csv(robustness_summary_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    write_markdown_summary(dataset_inventory, reports_dir / "cross_dataset_dataset_inventory.md", title="Cross-dataset dataset inventory")
    write_markdown_summary(trait_family_summary, reports_dir / "cross_dataset_trait_family_summary.md", title="Cross-dataset trait-family summary")
    write_markdown_summary(priority_summary, reports_dir / "cross_dataset_priority_summary.md", title="Cross-dataset priority summary")
    if not confirmatory_top_cells.empty:
        write_markdown_summary(
            confirmatory_top_cells,
            reports_dir / "cross_dataset_confirmatory_top_cells.md",
            title="Cross-dataset confirmatory top cells",
        )
    if not secondary_top_cells.empty:
        write_markdown_summary(
            secondary_top_cells,
            reports_dir / "cross_dataset_secondary_top_cells.md",
            title="Cross-dataset secondary top cells",
        )
    if not headline_confirmatory_cells.empty:
        write_markdown_summary(
            headline_confirmatory_cells,
            reports_dir / "cross_dataset_headline_confirmatory_cells.md",
            title="Cross-dataset headline confirmatory cells",
        )
    if not provisional_inferential_cells.empty:
        write_markdown_summary(
            provisional_inferential_cells,
            reports_dir / "cross_dataset_provisional_inferential_cells.md",
            title="Cross-dataset provisional inferential cells",
        )
    if not method_limited_inferential_cells.empty:
        write_markdown_summary(
            method_limited_inferential_cells,
            reports_dir / "cross_dataset_method_limited_inferential_cells.md",
            title="Cross-dataset method-limited inferential cells",
        )
    if not qa_only_cells.empty:
        write_markdown_summary(
            qa_only_cells,
            reports_dir / "cross_dataset_qa_only_cells.md",
            title="Cross-dataset QA-only cells",
        )
    if not child_inferential_cells.empty:
        write_markdown_summary(
            child_inferential_cells,
            reports_dir / "cross_dataset_nlsy79_child_ya_inferential_cells.md",
            title="NLSY79 Child and Young Adult inferential cells",
        )
    if not child_qa_only_cells.empty:
        write_markdown_summary(
            child_qa_only_cells,
            reports_dir / "cross_dataset_nlsy79_child_ya_qa_only_cells.md",
            title="NLSY79 Child and Young Adult QA-only cells",
        )
    if not nhanes_inferential_cells.empty:
        write_markdown_summary(
            nhanes_inferential_cells,
            reports_dir / "cross_dataset_nhanes_inferential_cells.md",
            title="NHANES inferential cells",
        )
    if not nhanes_qa_cognition_cells.empty:
        write_markdown_summary(
            nhanes_qa_cognition_cells,
            reports_dir / "cross_dataset_nhanes_qa_only_cognition_cells.md",
            title="NHANES QA-only cognition cells",
        )
    if not supporting_summary.empty:
        write_markdown_summary(
            supporting_summary,
            reports_dir / "cross_dataset_supporting_evidence_summary.md",
            title="Supporting-evidence summary",
        )
    if not supporting_top_cells.empty:
        write_markdown_summary(
            supporting_top_cells,
            reports_dir / "cross_dataset_supporting_evidence_top_cells.md",
            title="Supporting-evidence top cells",
        )
    if not supporting_robustness.empty:
        write_markdown_summary(
            supporting_robustness,
            reports_dir / "cross_dataset_supporting_evidence_robustness.md",
            title="Supporting-evidence robustness",
        )

    primary_forest = _write_primary_forest(comparison_df, figures_dir / "forest_log_variance_ratio_primary.png")
    secondary_forest = _write_secondary_forest(comparison_df, figures_dir / "forest_log_variance_ratio_secondary.png")
    age_profile = _write_age_profile(age_profile_summary, figures_dir / "age_profile_log_variance_ratio.png")
    mean_variance = _write_mean_variance_scatter(comparison_df, figures_dir / "mean_vs_variance_scatter.png")
    dataset_family_plot = _write_dataset_family_summary_plot(
        trait_family_summary,
        figures_dir / "dataset_family_summary.png",
    )
    robustness_plot = _write_robustness_comparison_plot(
        robustness_summary,
        figures_dir / "robustness_comparison.png",
    )

    report_path = _write_report(
        comparison_df=comparison_df,
        dataset_inventory=dataset_inventory,
        trait_family_summary=trait_family_summary,
        priority_summary=priority_summary,
        confirmatory_top_cells=confirmatory_top_cells,
        secondary_top_cells=secondary_top_cells,
        headline_confirmatory_cells=headline_confirmatory_cells,
        provisional_inferential_cells=provisional_inferential_cells,
        method_limited_inferential_cells=method_limited_inferential_cells,
        qa_only_cells=qa_only_cells,
        child_inferential_cells=child_inferential_cells,
        child_qa_only_cells=child_qa_only_cells,
        nhanes_inferential_cells=nhanes_inferential_cells,
        nhanes_qa_cognition_cells=nhanes_qa_cognition_cells,
        supporting_summary=supporting_summary,
        supporting_top_cells=supporting_top_cells,
        supporting_robustness=supporting_robustness,
        robustness_summary=robustness_summary,
        missing_coverage=missing_coverage,
        reproducibility_gaps=reproducibility_gaps,
        output_path=reports_dir / "cross_dataset_comparison.md",
    )
    final_report_path = _write_final_report(
        comparison_df=comparison_df,
        dataset_inventory=dataset_inventory,
        priority_summary=priority_summary,
        confirmatory_top_cells=confirmatory_top_cells,
        headline_confirmatory_cells=headline_confirmatory_cells,
        provisional_inferential_cells=provisional_inferential_cells,
        method_limited_inferential_cells=method_limited_inferential_cells,
        qa_only_cells=qa_only_cells,
        supporting_summary=supporting_summary,
        supporting_top_cells=supporting_top_cells,
        supporting_robustness=supporting_robustness,
        robustness_summary=robustness_summary,
        missing_coverage=missing_coverage,
        reproducibility_gaps=reproducibility_gaps,
        output_path=reports_dir / "sexvary_report.md",
    )
    audit_report_path = _write_evidence_audit_report(audit_df, reports_dir / "evidence_audit.md")

    for path in [
        analysis_cells_path,
        dataset_inventory_path,
        trait_family_summary_path,
        priority_summary_path,
        age_profile_summary_path,
        confirmatory_top_cells_path,
        secondary_top_cells_path,
        headline_confirmatory_cells_path,
        provisional_inferential_cells_path,
        method_limited_inferential_cells_path,
        qa_only_cells_path,
        child_inferential_cells_path,
        child_qa_only_cells_path,
        nhanes_inferential_cells_path,
        nhanes_qa_cognition_cells_path,
        supporting_summary_path,
        supporting_top_cells_path,
        supporting_robustness_path,
        robustness_summary_path,
        audit_path,
        primary_forest,
        secondary_forest,
        age_profile,
        mean_variance,
        dataset_family_plot,
        robustness_plot,
        report_path,
        final_report_path,
        audit_report_path,
    ]:
        if path is not None:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
