from pathlib import Path

import pandas as pd

from sexvary.paper_bundle import build_paper_bundle


def test_build_paper_bundle_writes_manuscript_appendix_and_manifests(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    reports_dir = results_dir / "reports"
    tables_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "dataset_label": ["PIAAC cycle 2", "NHANES selected cycles"],
            "rows": [30, 472],
            "rows_with_ci": [30, 132],
            "headline_eligible_rows": [30, 132],
            "provisional_rows": [0, 0],
            "qa_only_rows": [0, 12],
        }
    ).to_csv(tables_dir / "cross_dataset_dataset_inventory.csv", index=False)
    pd.DataFrame(
        {
            "trait_priority": ["confirmatory", "secondary"],
            "dataset_label": ["PIAAC cycle 2", "NHANES selected cycles"],
            "cells": [30, 250],
            "cells_with_ci": [30, 73],
            "headline_eligible_cells": [30, 73],
            "provisional_cells": [0, 0],
            "method_limited_cells": [0, 0],
            "qa_only_cells": [0, 177],
            "median_log_variance_ratio": [0.2, 0.1],
            "share_male_greater": [0.8, 0.7],
        }
    ).to_csv(tables_dir / "cross_dataset_priority_summary.csv", index=False)
    pd.DataFrame(
        {
            "dataset_label": ["PIAAC cycle 2"],
            "cycle_or_wave": ["cycle2"],
            "age_band": ["60-65"],
            "trait_label": ["Numeracy"],
            "log_variance_ratio": [0.4],
            "se_log_variance_ratio": [0.05],
            "qa_flags": [pd.NA],
        }
    ).to_csv(tables_dir / "cross_dataset_headline_confirmatory_cells.csv", index=False)
    pd.DataFrame(
        {
            "dataset_id": ["nhanes_2011_2023"],
            "dataset_label": ["NHANES selected cycles"],
            "trait_priority": ["secondary"],
            "rows": [250],
            "rows_with_ci": [73],
            "headline_eligible_rows": [73],
            "provisional_rows": [0],
            "method_limited_rows": [0],
            "qa_only_rows": [177],
            "trait_families": [1],
            "median_log_variance_ratio": [0.1],
            "share_male_greater": [0.7],
        }
    ).to_csv(tables_dir / "cross_dataset_supporting_evidence_summary.csv", index=False)
    pd.DataFrame(
        {
            "dataset_id": ["nhanes_2011_2023"],
            "dataset_label": ["NHANES selected cycles"],
            "variant": ["unweighted"],
            "matched_cells": [132],
            "median_abs_delta": [0.05],
            "sign_change_rate": [0.02],
            "headline_cells_retained": [132],
        }
    ).to_csv(tables_dir / "cross_dataset_supporting_evidence_robustness.csv", index=False)
    pd.DataFrame(
        {
            "dataset_id": ["nhanes_2011_2023"],
            "dataset_label": ["NHANES selected cycles"],
            "variant": ["unweighted"],
            "matched_cells": [132],
            "median_abs_delta": [0.05],
            "sign_change_rate": [0.02],
            "headline_cells_retained": [132],
        }
    ).to_csv(tables_dir / "cross_dataset_robustness_comparison.csv", index=False)
    pd.DataFrame(
        {
            "audit_topic": ["Evidence audit"],
            "dataset_label": ["PIAAC cycle 2"],
            "trait_label": ["Numeracy"],
            "cycle_or_wave": ["cycle2"],
            "age_band": ["60-65"],
            "evidence_status": ["headline_eligible"],
            "log_variance_ratio": [0.4],
            "conclusion": ["Retain as headline evidence."],
        }
    ).to_csv(tables_dir / "cross_dataset_evidence_audit.csv", index=False)
    pd.DataFrame(
        {
            "dataset_label": ["PSID CDS / TAS"],
            "cycle_or_wave": ["TAS 2019"],
            "age_band": ["all_ages"],
            "trait_label": ["BMI"],
            "trait_family": ["physical"],
            "evidence_status": ["provisional"],
            "log_variance_ratio": [0.2],
            "se_log_variance_ratio": [0.1],
            "suppression_reason": ["nonprimary_weight_fallback"],
            "qa_flags": ["nonprimary_weight_fallback"],
        }
    ).to_csv(tables_dir / "cross_dataset_provisional_inferential_cells.csv", index=False)
    pd.DataFrame(
        {
            "dataset_label": ["HRS public"],
            "cycle_or_wave": ["HRS 2022"],
            "age_band": ["80-83"],
            "trait_label": ["Numeracy"],
            "trait_family": ["adult_skills"],
            "evidence_status": ["method_limited"],
            "log_variance_ratio": [0.3],
            "se_log_variance_ratio": [0.1],
            "suppression_reason": ["approximate_household_cluster_bootstrap"],
            "qa_flags": ["approximate_design_bootstrap"],
        }
    ).to_csv(tables_dir / "cross_dataset_method_limited_inferential_cells.csv", index=False)
    pd.DataFrame(
        {
            "dataset_label": ["NHANES selected cycles"],
            "cycle_or_wave": ["2011-2012"],
            "age_band": ["68-71"],
            "trait_label": ["Adult cognition screen"],
            "trait_family": ["later_life_cognition"],
            "evidence_status": ["qa_only"],
            "suppression_reason": ["low_n_variance"],
            "qa_flags": ["tail_metrics_suppressed"],
        }
    ).to_csv(tables_dir / "cross_dataset_qa_only_cells.csv", index=False)

    for figure_name in [
        "forest_log_variance_ratio_primary.png",
        "dataset_family_summary.png",
        "age_profile_log_variance_ratio.png",
        "robustness_comparison.png",
    ]:
        (figures_dir / figure_name).write_bytes(b"png")

    for report_name in ["sexvary_report.md", "cross_dataset_comparison.md", "evidence_audit.md"]:
        (reports_dir / report_name).write_text("stub", encoding="utf-8")

    outputs = build_paper_bundle(tmp_path, tmp_path / "results" / "paper_bundle")

    manuscript = outputs["manuscript"].read_text(encoding="utf-8")
    appendix = outputs["appendix"].read_text(encoding="utf-8")
    public_summary = outputs["public_summary"].read_text(encoding="utf-8")

    assert "## 5. Confirmatory Results" in manuscript
    assert "## 6. Supporting Evidence" in manuscript
    assert "NHANES selected cycles" in manuscript
    assert "## A2. Provisional inferential rows" in appendix
    assert "## Public Guardrails" in public_summary
    assert outputs["site_content"].exists()
    assert outputs["figure_manifest"].exists()
    assert outputs["table_manifest"].exists()
    assert outputs["bundle_manifest"].exists()
