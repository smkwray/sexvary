import json
from pathlib import Path

import pandas as pd
import pytest

from sexvary.paper_bundle import build_paper_bundle
from sexvary.site_bundle import validate_site_bundle


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
            "source_table": ["piaac_cycle2", "ecls_k_2011", "nhanes_2011_2023", "psid_cds_tas", "hrs_public"],
            "dataset_id": ["piaac_cycle2", "ecls_k_2011", "nhanes_2011_2023", "psid_cds_tas", "hrs_public"],
            "cycle_or_wave": ["cycle2", "fall_kindergarten_2010", "2011-2012", "TAS 2019", "HRS 2022"],
            "country": ["United States"] * 5,
            "age_band": ["60-65", "K", "20-23", "all_ages", "80-83"],
            "trait_id": ["numeracy", "reading_achievement", "grip_strength_kg", "bmi", "numeracy"],
            "log_variance_ratio": [0.4, -0.6, 0.1, 0.2, 0.3],
            "se_log_variance_ratio": [0.05, 0.04, 0.08, 0.1, 0.1],
            "variance_ratio": [1.49, 0.55, 1.11, 1.22, 1.35],
            "ci_low_log_variance_ratio": [0.302, -0.678, -0.057, 0.004, 0.104],
            "ci_high_log_variance_ratio": [0.498, -0.522, 0.257, 0.396, 0.496],
            "mean_difference": [0.0, 0.0, 0.0, 0.0, 0.0],
            "inference_method": [
                "replicate_weights_brr",
                "stratified_cluster_bootstrap_psu",
                "stratified_cluster_bootstrap_psu",
                "analytic_effective_n_simple_design",
                "approximate_household_cluster_bootstrap",
            ],
            "male_n": [500, 800, 250, 180, 200],
            "female_n": [520, 780, 260, 175, 230],
            "qa_flags": [pd.NA, pd.NA, pd.NA, "nonprimary_weight_fallback", "approximate_design_bootstrap"],
            "trait_family": ["adult_skills", "achievement", "physical", "physical", "adult_skills"],
            "trait_priority": ["confirmatory", "confirmatory", "secondary", "secondary", "secondary"],
            "evidence_status": ["headline_eligible", "headline_eligible", "headline_eligible", "provisional", "method_limited"],
            "headline_eligible": [True, True, True, False, False],
            "suppression_reason": [pd.NA, pd.NA, pd.NA, "nonprimary_weight_fallback", "approximate_household_cluster_bootstrap"],
            "comparability_tier": [
                "confirmatory_headline",
                "confirmatory_headline",
                "secondary_headline",
                "provisional",
                "secondary_method_limited",
            ],
            "provisional": [False, False, False, True, False],
            "qa_only": [False, False, False, False, False],
            "method_limited": [False, False, False, False, True],
            "trait_scale_type": ["continuous"] * 5,
            "dataset_label": ["PIAAC cycle 2", "ECLS-K:2011", "NHANES selected cycles", "PSID CDS / TAS", "HRS public"],
            "trait_root": ["numeracy", "reading_achievement", "grip_strength_kg", "bmi", "numeracy"],
            "trait_label": ["Numeracy", "Reading achievement", "Grip strength (kg)", "BMI", "Numeracy"],
            "ci_available": [True, True, True, True, True],
            "effect_available": [True, True, True, True, True],
            "male_greater_variability": [True, False, True, True, True],
            "vr_ci_low": [1.35, 0.51, 0.94, 1.00, 1.11],
            "vr_ci_high": [1.65, 0.59, 1.29, 1.49, 1.64],
            "direction": ["male_greater", "female_greater", "male_greater", "male_greater", "male_greater"],
            "abs_log_vr": [0.4, 0.6, 0.1, 0.2, 0.3],
            "distance_from_equal": [0.49, 0.45, 0.11, 0.22, 0.35],
            "n_total": [1020, 1580, 510, 355, 430],
            "claim_status_display": ["Headline claim", "Headline claim", "Supporting evidence", "Provisional", "Method-limited"],
            "priority_display": ["Confirmatory", "Confirmatory", "Secondary", "Secondary", "Secondary"],
            "display_explanation": [
                "Headline claim. male-greater variability (VR 1.49x, 95% CI 1.35x to 1.65x).",
                "Headline claim. female-greater variability (VR 0.55x, 95% CI 0.51x to 0.59x).",
                "Supporting evidence. male-greater variability (VR 1.11x, 95% CI 0.94x to 1.29x).",
                "Provisional. male-greater variability (VR 1.22x, 95% CI 1.00x to 1.49x); nonprimary weight fallback.",
                "Method-limited. male-greater variability (VR 1.35x, 95% CI 1.11x to 1.64x); approximate household cluster bootstrap.",
            ],
        }
    ).to_csv(tables_dir / "cross_dataset_analysis_cells.csv", index=False)
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
    site_bundle = json.loads(outputs["site_content"].read_text(encoding="utf-8"))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    site_index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    site_results = (tmp_path / "site" / "results.html").read_text(encoding="utf-8")
    site_datasets = (tmp_path / "site" / "datasets.html").read_text(encoding="utf-8")

    assert "## 5. Confirmatory Results" in manuscript
    assert "## 6. Supporting Evidence" in manuscript
    assert "NHANES selected cycles" in manuscript
    assert "## A2. Provisional inferential rows" in appendix
    assert "## Public Guardrails" in public_summary
    assert outputs["site_content"].exists()
    assert outputs["site_bundle_public"].exists()
    assert outputs["figure_manifest"].exists()
    assert outputs["table_manifest"].exists()
    assert outputs["bundle_manifest"].exists()
    assert "Headline confirmatory cells" in site_index
    assert "Searchable Cell Explorer" in site_results
    assert "Per-Dataset Quantiles" in site_datasets
    assert "Selected headline cells" in readme
    validate_site_bundle(site_bundle)


def test_validate_site_bundle_rejects_drifted_counts() -> None:
    bundle = {
        "summary": {
            "live_dataset_count": 2,
            "headline_confirmatory_cell_count": 1,
            "supporting_inferential_cell_count": 1,
            "inferential_cell_count": 1,
        },
        "page_metrics": {
            "home": {"headline_confirmatory_cell_count": 2},
            "results": {"headline_confirmatory_cell_count": 1},
            "readme": {"headline_confirmatory_cell_count": 1},
            "datasets": {"inventory_row_count": 2},
        },
        "tables": {
            "dataset_inventory": [{}, {}],
            "headline_confirmatory_all": [{}],
            "supporting_summary": [{"rows_with_ci": 1}],
            "inferential_cells": [{}],
        },
    }

    with pytest.raises(ValueError, match="page_metrics.home.headline_confirmatory_cell_count"):
        validate_site_bundle(bundle)
