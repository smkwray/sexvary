from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .reporting import markdown_table


REQUIRED_TABLES = {
    "dataset_inventory": Path("results/tables/cross_dataset_dataset_inventory.csv"),
    "priority_summary": Path("results/tables/cross_dataset_priority_summary.csv"),
    "headline_confirmatory_cells": Path("results/tables/cross_dataset_headline_confirmatory_cells.csv"),
    "supporting_evidence_summary": Path("results/tables/cross_dataset_supporting_evidence_summary.csv"),
    "supporting_evidence_robustness": Path("results/tables/cross_dataset_supporting_evidence_robustness.csv"),
    "robustness_comparison": Path("results/tables/cross_dataset_robustness_comparison.csv"),
    "evidence_audit": Path("results/tables/cross_dataset_evidence_audit.csv"),
    "provisional_inferential_cells": Path("results/tables/cross_dataset_provisional_inferential_cells.csv"),
    "method_limited_inferential_cells": Path("results/tables/cross_dataset_method_limited_inferential_cells.csv"),
    "qa_only_cells": Path("results/tables/cross_dataset_qa_only_cells.csv"),
}

CORE_FIGURES = {
    "figure_1_confirmatory_forest": Path("results/figures/forest_log_variance_ratio_primary.png"),
    "figure_2_dataset_family_summary": Path("results/figures/dataset_family_summary.png"),
    "figure_3_age_profile": Path("results/figures/age_profile_log_variance_ratio.png"),
    "figure_4_robustness": Path("results/figures/robustness_comparison.png"),
}

SOURCE_REPORTS = {
    "main_report": Path("results/reports/sexvary_report.md"),
    "comparison_appendix": Path("results/reports/cross_dataset_comparison.md"),
    "evidence_audit": Path("results/reports/evidence_audit.md"),
}


def _load_latest_backend_manifest(root: Path) -> dict[str, object] | None:
    path = root / "results" / "run_manifests" / "backend_run_latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_str(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_csv(root: Path, relpath: Path) -> pd.DataFrame:
    path = root / relpath
    if not path.exists():
        raise FileNotFoundError(f"Required paper-bundle input is missing: {path}")
    return pd.read_csv(path)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    return markdown_table(df)


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_abstract(
    dataset_inventory: pd.DataFrame,
    headline_cells: pd.DataFrame,
    supporting_summary: pd.DataFrame,
) -> list[str]:
    datasets = int(len(dataset_inventory))
    headline_count = int(len(headline_cells))
    support_count = int(int(supporting_summary["rows_with_ci"].sum()) if not supporting_summary.empty else 0)

    lines = [
        "## Abstract",
        "",
        "This project estimates sex differences in variability across cognitive, achievement, and physical traits using public-use datasets rather than relying on a single instrument or sample.",
        f"The current Phase 1 evidence base spans `{datasets}` live datasets with a shared descriptive backend and a headline-evidence layer restricted to the strongest confirmatory cells.",
        f"The core estimator is the within-cell log variance ratio, paired with dataset-appropriate survey handling where possible and explicit status labels for headline-eligible, provisional, method-limited, and QA-only evidence.",
        f"Across `{headline_count}` headline-eligible confirmatory cells, the prevailing pattern is male-greater variability, but the strongest counterexample is early reading in ECLS-K and the supporting-evidence layer adds `{support_count}` inferential rows from NHANES, HRS, and PSID that remain below headline quality.",
        "The main limitations are public-use design constraints, bounded-scale fragility for some measures, and method-limited inference in several supporting datasets.",
        "",
    ]
    return lines


def _build_manuscript_text(
    *,
    root: Path,
    dataset_inventory: pd.DataFrame,
    priority_summary: pd.DataFrame,
    headline_cells: pd.DataFrame,
    supporting_summary: pd.DataFrame,
    supporting_robustness: pd.DataFrame,
    robustness_comparison: pd.DataFrame,
    evidence_audit: pd.DataFrame,
    backend_manifest: dict[str, object] | None,
) -> str:
    confirmatory_summary = priority_summary[priority_summary["trait_priority"] == "confirmatory"].copy()
    positive_share = None
    strongest_positive = None
    strongest_negative = None
    strongest_counterexample_note = (
        "The clearest counterexample in the current headline layer is early reading in ECLS-K, which should remain visible in any public-facing summary."
    )
    if not headline_cells.empty:
        positives = headline_cells["log_variance_ratio"] > 0
        positive_share = float(positives.mean())
        strongest_positive = headline_cells.loc[headline_cells["log_variance_ratio"].idxmax()]
        strongest_negative = headline_cells.loc[headline_cells["log_variance_ratio"].idxmin()]

    lines: list[str] = [
        "# Trait-Specific Sex Differences in Variability Across Public-Use Datasets",
        "",
    ]
    lines.extend(_build_abstract(dataset_inventory, headline_cells, supporting_summary))
    if backend_manifest:
        selection = backend_manifest.get("pipeline_selection", []) or []
        selected = sum(1 for row in selection if row.get("selection_status") == "selected")
        missing_input = sum(1 for row in selection if row.get("selection_status") == "missing_input")
        failed_runs = [
            row for row in (backend_manifest.get("pipeline_runs", []) or []) if row.get("status") == "failed"
        ]
        compare_run = backend_manifest.get("compare_run") or {}
        compare_failed = compare_run.get("status") == "failed"
        lines.extend(
            [
                "## Build state",
                "",
                f"- Backend manifest mode: `{'dry_run' if backend_manifest.get('dry_run') else 'executed'}`.",
                f"- Selected pipelines: `{selected}`. Missing-input pipelines: `{missing_input}`.",
                f"- Backend pipeline failures: `{len(failed_runs)}`. Comparison rebuild failed: `{'yes' if compare_failed else 'no'}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## 1. Introduction",
            "",
            "This manuscript asks where sex differences in variability appear, rather than assuming a universal pattern. The strategy is descriptive and replication-oriented: estimate comparable within-cell variance contrasts across multiple public datasets, then separate stronger replicated evidence from supporting and provisional findings.",
            "",
            "## 2. Data",
            "",
            "The Phase 1 corpus combines local NLSY anchor datasets with adult-skills, school-age, physical-health, panel, and later-life supporting datasets.",
            "",
            _markdown_table(
                dataset_inventory[
                    [
                        "dataset_label",
                        "rows",
                        "rows_with_ci",
                        "headline_eligible_rows",
                        "provisional_rows",
                        "qa_only_rows",
                    ]
                ]
            ),
            "",
            "## 3. Measures",
            "",
            "Trait families are analyzed within dataset-defined cells using sex, age or grade band, cycle or wave, and the dataset’s native score scale. Confirmatory and secondary priorities are set in the project registry and carried through the outputs.",
            "",
            "## 4. Methods",
            "",
            "The main effect is the log variance ratio. Weighted estimation is used throughout, with plausible values, BRR, JRR, bootstrap, or analytic approximations depending on the dataset. Output rows are labeled as headline-eligible, provisional, method-limited, or QA-only so interpretation does not depend on reading implementation details out of code comments.",
            "",
            "## 5. Confirmatory Results",
            "",
        ]
    )

    if positive_share is not None and strongest_positive is not None and strongest_negative is not None:
        lines.extend(
            [
                f"Across headline-eligible confirmatory cells, male-greater variability appears in `{positive_share:.0%}` of cells.",
                f"The strongest positive confirmatory cell is `{strongest_positive['trait_label']}` in `{strongest_positive['dataset_label']}` at `{strongest_positive['age_band']}` during `{strongest_positive['cycle_or_wave']}`.",
                f"The strongest negative confirmatory cell is `{strongest_negative['trait_label']}` in `{strongest_negative['dataset_label']}` at `{strongest_negative['age_band']}` during `{strongest_negative['cycle_or_wave']}`.",
                strongest_counterexample_note,
                "",
            ]
        )

    lines.extend(
        [
            _markdown_table(confirmatory_summary),
            "",
            "### Table 2. Leading headline-eligible confirmatory cells",
            "",
            _markdown_table(
                headline_cells.head(12)[
                    [
                        "dataset_label",
                        "cycle_or_wave",
                        "age_band",
                        "trait_label",
                        "log_variance_ratio",
                        "se_log_variance_ratio",
                        "qa_flags",
                    ]
                ]
                if not headline_cells.empty
                else pd.DataFrame()
            ),
            "",
            "## 6. Supporting Evidence",
            "",
            "Supporting evidence broadens the project beyond the main confirmatory layer. These datasets are retained because they add domain coverage, not because they meet the same evidentiary standard as the headline cells. They should be presented publicly as supporting or provisional evidence, not folded into the headline claim.",
            "",
            _markdown_table(supporting_summary),
            "",
            "## 7. Sensitivity Checks",
            "",
            "The supporting datasets and the local NLSY anchor all have dataset-specific robustness summaries. Winsorization generally moves the results less than dropping survey weights, and supporting datasets remain below headline quality even when their point estimates are directionally stable.",
            "",
            _markdown_table(robustness_comparison),
            "",
            "### Supporting-dataset robustness detail",
            "",
            _markdown_table(supporting_robustness),
            "",
            "## 8. Discussion",
            "",
            "The broad pattern is not universal but it is not random either. Adult skills and many later-school measures often lean male-greater in variability, while early reading in ECLS-K remains the clearest female-greater counterexample. Supporting datasets broaden the age and domain coverage without collapsing the need to distinguish strong evidence from method-limited evidence.",
            "",
            "## 9. Limitations",
            "",
            "- This project is descriptive, not causal.",
            "- Public-use datasets impose hard limits on survey design handling and harmonization.",
            "- Some scales are bounded or heaped, which constrains tail metrics and inflates fragility.",
            "- Supporting datasets like HRS and PSID remain method-limited or provisional rather than headline-eligible.",
            "- Public-facing summaries should not collapse headline evidence, supporting evidence, and provisional evidence into one pooled claim.",
            "",
            "## 10. Conclusion",
            "",
            "The current public-use evidence supports a restrained conclusion: male-greater variability is common in the strongest confirmatory cells, but it is not universal, and the most important counterexample is early reading in ECLS-K.",
            "",
            "## Figure References",
            "",
            f"- Figure 1: `{_relative_str(CORE_FIGURES['figure_1_confirmatory_forest'], root=root)}`",
            f"- Figure 2: `{_relative_str(CORE_FIGURES['figure_2_dataset_family_summary'], root=root)}`",
            f"- Figure 3: `{_relative_str(CORE_FIGURES['figure_3_age_profile'], root=root)}`",
            f"- Figure 4: `{_relative_str(CORE_FIGURES['figure_4_robustness'], root=root)}`",
            "",
            "## Audit References",
            "",
            _markdown_table(evidence_audit),
            "",
        ]
    )
    return "\n".join(lines)


def _build_appendix_text(
    *,
    provisional: pd.DataFrame,
    method_limited: pd.DataFrame,
    qa_only: pd.DataFrame,
    evidence_audit: pd.DataFrame,
) -> str:
    lines = [
        "# Appendix",
        "",
        "This appendix retains the rows that are not part of the headline-evidence layer but are still important for transparency and interpretation.",
        "",
        "## A1. Evidence audit",
        "",
        _markdown_table(evidence_audit),
        "",
        "## A2. Provisional inferential rows",
        "",
        _markdown_table(provisional.head(40)),
        "",
        "## A3. Method-limited inferential rows",
        "",
        _markdown_table(method_limited.head(40)),
        "",
        "## A4. QA-only rows",
        "",
        _markdown_table(qa_only.head(40)),
        "",
    ]
    return "\n".join(lines)


def _build_public_summary(
    *,
    dataset_inventory: pd.DataFrame,
    headline_cells: pd.DataFrame,
    supporting_summary: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    headline_count = int(len(headline_cells))
    positive_share = float((headline_cells["log_variance_ratio"] > 0).mean()) if headline_count else 0.0
    strongest_positive = headline_cells.loc[headline_cells["log_variance_ratio"].idxmax()] if headline_count else None
    strongest_negative = headline_cells.loc[headline_cells["log_variance_ratio"].idxmin()] if headline_count else None
    support_rows = int(supporting_summary["rows_with_ci"].sum()) if not supporting_summary.empty else 0
    live_datasets = int(len(dataset_inventory))

    headline_lines = [
        "# Public Summary",
        "",
        "## Headline",
        "",
        f"- The strongest confirmatory evidence currently spans `{headline_count}` headline-eligible cells across `{live_datasets}` live datasets.",
        f"- Within that headline layer, male-greater variability appears in about `{positive_share:.0%}` of cells.",
    ]
    if strongest_positive is not None:
        headline_lines.append(
            f"- The clearest headline positive is `{strongest_positive['trait_label']}` in `{strongest_positive['dataset_label']}` at `{strongest_positive['age_band']}` during `{strongest_positive['cycle_or_wave']}`."
        )
    if strongest_negative is not None:
        headline_lines.append(
            f"- The clearest headline reversal is `{strongest_negative['trait_label']}` in `{strongest_negative['dataset_label']}` at `{strongest_negative['age_band']}` during `{strongest_negative['cycle_or_wave']}`."
        )
    headline_lines.extend(
        [
            "",
            "## Public Guardrails",
            "",
            "- This project is descriptive, not causal.",
            "- Headline claims should use only headline-eligible confirmatory evidence.",
            "- Supporting datasets such as NHANES, HRS, and PSID should be labeled as supporting or provisional evidence, not merged into the headline claim.",
            "- Early reading in ECLS-K should remain visible as a real counterexample in any public-facing summary.",
            "",
            "## Supporting Evidence",
            "",
            f"- The supporting-evidence layer currently adds `{support_rows}` inferential rows that broaden age and domain coverage but remain below headline quality.",
            "",
        ]
    )

    payload = {
        "headline_cell_count": headline_count,
        "headline_positive_share": positive_share,
        "headline_positive_cell": (
            {
                "dataset_label": str(strongest_positive["dataset_label"]),
                "trait_label": str(strongest_positive["trait_label"]),
                "cycle_or_wave": str(strongest_positive["cycle_or_wave"]),
                "age_band": str(strongest_positive["age_band"]),
                "log_variance_ratio": float(strongest_positive["log_variance_ratio"]),
            }
            if strongest_positive is not None
            else None
        ),
        "headline_counterexample": (
            {
                "dataset_label": str(strongest_negative["dataset_label"]),
                "trait_label": str(strongest_negative["trait_label"]),
                "cycle_or_wave": str(strongest_negative["cycle_or_wave"]),
                "age_band": str(strongest_negative["age_band"]),
                "log_variance_ratio": float(strongest_negative["log_variance_ratio"]),
            }
            if strongest_negative is not None
            else None
        ),
        "supporting_inferential_rows": support_rows,
        "guardrails": [
            "descriptive_not_causal",
            "headline_only_uses_headline_eligible_confirmatory_cells",
            "supporting_datasets_remain_supporting_or_provisional",
            "ecls_kindergarten_reading_reversal_must_remain_visible",
        ],
    }
    return "\n".join(headline_lines), payload


def build_paper_bundle(root: Path, output_dir: Path) -> dict[str, Path]:
    tables = {name: _read_csv(root, relpath) for name, relpath in REQUIRED_TABLES.items()}
    backend_manifest = _load_latest_backend_manifest(root)

    output_dir.mkdir(parents=True, exist_ok=True)

    manuscript_text = _build_manuscript_text(
        root=root,
        dataset_inventory=tables["dataset_inventory"],
        priority_summary=tables["priority_summary"],
        headline_cells=tables["headline_confirmatory_cells"],
        supporting_summary=tables["supporting_evidence_summary"],
        supporting_robustness=tables["supporting_evidence_robustness"],
        robustness_comparison=tables["robustness_comparison"],
        evidence_audit=tables["evidence_audit"],
        backend_manifest=backend_manifest,
    )
    appendix_text = _build_appendix_text(
        provisional=tables["provisional_inferential_cells"],
        method_limited=tables["method_limited_inferential_cells"],
        qa_only=tables["qa_only_cells"],
        evidence_audit=tables["evidence_audit"],
    )
    public_summary_text, public_payload = _build_public_summary(
        dataset_inventory=tables["dataset_inventory"],
        headline_cells=tables["headline_confirmatory_cells"],
        supporting_summary=tables["supporting_evidence_summary"],
    )

    manuscript_path = output_dir / "manuscript.md"
    appendix_path = output_dir / "appendix.md"
    public_summary_path = output_dir / "public_summary.md"
    site_payload_path = output_dir / "site_content.json"
    manuscript_path.write_text(manuscript_text, encoding="utf-8")
    appendix_path.write_text(appendix_text, encoding="utf-8")
    public_summary_path.write_text(public_summary_text, encoding="utf-8")
    site_payload_path.write_text(json.dumps(public_payload, indent=2), encoding="utf-8")

    figure_manifest = pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "source_path": _relative_str(relpath, root=root),
                "exists": (root / relpath).exists(),
            }
            for figure_id, relpath in CORE_FIGURES.items()
        ]
    )
    figure_manifest_path = output_dir / "figure_manifest.csv"
    figure_manifest.to_csv(figure_manifest_path, index=False)

    table_manifest = pd.DataFrame(
        [
            {
                "table_id": table_id,
                "source_path": _relative_str(relpath, root=root),
                "rows": int(len(tables[table_id])) if table_id in tables else 0,
            }
            for table_id, relpath in REQUIRED_TABLES.items()
        ]
    )
    table_manifest_path = output_dir / "table_manifest.csv"
    table_manifest.to_csv(table_manifest_path, index=False)

    source_manifest = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "source_path": _relative_str(relpath, root=root),
                "exists": (root / relpath).exists(),
            }
            for source_id, relpath in SOURCE_REPORTS.items()
        ]
    )
    source_manifest_path = output_dir / "source_report_manifest.csv"
    source_manifest.to_csv(source_manifest_path, index=False)

    metadata = {
        "output_dir": _relative_str(output_dir, root=root),
        "manuscript_path": _relative_str(manuscript_path, root=root),
        "appendix_path": _relative_str(appendix_path, root=root),
        "figure_manifest_path": _relative_str(figure_manifest_path, root=root),
        "table_manifest_path": _relative_str(table_manifest_path, root=root),
        "source_report_manifest_path": _relative_str(source_manifest_path, root=root),
        "public_summary_path": _relative_str(public_summary_path, root=root),
        "site_content_path": _relative_str(site_payload_path, root=root),
        "backend_manifest_path": _relative_str(root / "results" / "run_manifests" / "backend_run_latest.json", root=root)
        if backend_manifest
        else None,
        "backend_manifest_dry_run": backend_manifest.get("dry_run") if backend_manifest else None,
    }
    metadata_path = output_dir / "bundle_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_lines = [
        "# Paper bundle",
        "",
        "This directory is the canonical manuscript-facing bundle generated from the current comparison outputs.",
        "",
        f"- Manuscript: `{_relative_str(manuscript_path, root=root)}`",
        f"- Appendix: `{_relative_str(appendix_path, root=root)}`",
        f"- Public summary: `{_relative_str(public_summary_path, root=root)}`",
        f"- Figure manifest: `{_relative_str(figure_manifest_path, root=root)}`",
        f"- Table manifest: `{_relative_str(table_manifest_path, root=root)}`",
    ]
    if backend_manifest:
        readme_lines.extend(
            [
                f"- Backend manifest mode: `{'dry_run' if backend_manifest.get('dry_run') else 'executed'}`",
            ]
        )
    readme_lines.extend(
        [
            "",
        ]
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    return {
        "manuscript": manuscript_path,
        "appendix": appendix_path,
        "public_summary": public_summary_path,
        "site_content": site_payload_path,
        "figure_manifest": figure_manifest_path,
        "table_manifest": table_manifest_path,
        "source_report_manifest": source_manifest_path,
        "bundle_manifest": metadata_path,
        "readme": readme_path,
    }
