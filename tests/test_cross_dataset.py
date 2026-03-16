from pathlib import Path
import importlib.util
import warnings

import pandas as pd

from sexvary.config import build_registry
from sexvary.cross_dataset import (
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
    normalize_estimate_table,
)


def test_normalize_estimate_table_handles_schema_variants() -> None:
    registry = build_registry()

    local_df = pd.DataFrame(
        {
            "dataset_id": ["nlsy79_main"],
            "cycle_or_wave": ["all"],
            "country": ["all"],
            "age_band": ["16-19"],
            "trait_id": ["asvab_subtests:arithmetic_reasoning"],
            "log_variance_ratio": [0.2],
            "se_log_variance_ratio": [0.05],
            "variance_ratio": [1.22],
            "mean_diff": [0.1],
            "inference_method": ["analytic_effective_n_simple_design"],
            "male_n": [200],
            "female_n": [210],
            "qa_flags": [pd.NA],
        }
    )
    piaac_df = pd.DataFrame(
        {
            "dataset_id": ["piaac_cycle2"],
            "cycle_or_wave": ["cycle2"],
            "country": ["United States"],
            "grade_or_age_band": ["16-19"],
            "trait_id": ["literacy"],
            "log_variance_ratio": [0.1],
            "se_log_variance_ratio": [0.08],
            "variance_ratio": [1.1],
            "mean_difference": [0.03],
            "inference_method": ["brr"],
            "male_n": [1000],
            "female_n": [950],
            "qa_flags": [pd.NA],
        }
    )

    local = normalize_estimate_table(local_df, source_id="local_nlsy", registry=registry)
    piaac = normalize_estimate_table(piaac_df, source_id="piaac_cycle2", registry=registry)
    combined = pd.concat([local, piaac], ignore_index=True)

    assert combined["age_band"].tolist() == ["16-19", "16-19"]
    assert combined["inference_method"].tolist() == ["analytic_effective_n_simple_design", "brr"]
    assert combined["trait_root"].tolist() == ["asvab_subtests", "literacy"]
    assert combined["trait_family"].tolist() == ["cognition", "adult_skills"]
    assert combined["ci_available"].tolist() == [True, True]
    assert combined["evidence_status"].tolist() == ["method_limited", "headline_eligible"]
    assert combined["headline_eligible"].tolist() == [False, True]

    inventory = build_dataset_inventory(combined)
    assert inventory["dataset_id"].tolist() == ["nlsy79_main", "piaac_cycle2"]
    assert "headline_eligible_rows" in inventory.columns
    assert "qa_only_rows" in inventory.columns

    family_summary = build_trait_family_summary(combined)
    assert set(family_summary["trait_family"]) == {"adult_skills", "cognition"}

    priority_summary = build_priority_summary(combined)
    assert set(priority_summary["trait_priority"]) == {"confirmatory", "secondary"}
    assert "headline_eligible_cells" in priority_summary.columns

    top_cells = build_top_cells(combined, limit=2)
    assert len(top_cells) == 2
    assert set(top_cells["dataset_label"]) == {"NLSY79 main", "PIAAC cycle 2"}
    assert set(top_cells["trait_priority"]) == {"confirmatory", "secondary"}
    assert "evidence_status" in top_cells.columns
    assert "suppression_reason" in top_cells.columns

    confirmatory_only = build_top_cells(combined, limit=5, priorities=("confirmatory",))
    assert confirmatory_only["trait_priority"].tolist() == ["confirmatory"]

    age_profile = build_age_profile_summary(combined)
    assert set(age_profile["age_band"]) == {"16-19"}


def test_cross_dataset_summaries_ignore_unavailable_rows_for_direction_shares() -> None:
    registry = build_registry()
    df = pd.DataFrame(
        {
            "dataset_id": ["nlsy79_child_ya", "nlsy79_child_ya", "nlsy79_child_ya"],
            "cycle_or_wave": ["all", "all", "all"],
            "country": ["all", "all", "all"],
            "age_band": ["16-17", "all_ages", "12-13"],
            "trait_id": ["working_memory", "ppvt", "ppvt"],
            "log_variance_ratio": [0.3, 0.1, pd.NA],
            "se_log_variance_ratio": [0.2, 0.2, pd.NA],
            "variance_ratio": [1.35, 1.10, pd.NA],
            "mean_diff": [0.0, 0.0, pd.NA],
            "inference_method": ["analytic_effective_n_simple_design"] * 3,
            "male_n": [100, 100, 30],
            "female_n": [100, 100, 30],
            "qa_flags": [pd.NA, pd.NA, "low_n_variance"],
        }
    )
    normalized = normalize_estimate_table(df, source_id="local_nlsy", registry=registry)

    inventory = build_dataset_inventory(normalized)
    assert inventory.loc[0, "rows"] == 3
    assert inventory.loc[0, "rows_with_ci"] == 2
    assert inventory.loc[0, "share_male_greater"] == 1.0

    family_summary = build_trait_family_summary(normalized)
    assert family_summary["share_male_greater"].max() == 1.0

    priority_summary = build_priority_summary(normalized)
    assert priority_summary.loc[0, "share_male_greater"] == 1.0


def test_build_dataset_focus_split_separates_inferential_and_qa_only_rows() -> None:
    registry = build_registry()
    df = pd.DataFrame(
        {
            "dataset_id": ["nlsy79_child_ya", "nlsy79_child_ya", "nlsy79_main"],
            "cycle_or_wave": ["all", "all", "all"],
            "country": ["all", "all", "all"],
            "age_band": ["16-17", "12-13", "16-19"],
            "trait_id": ["working_memory", "working_memory", "asvab_subtests:word_knowledge"],
            "log_variance_ratio": [0.3, pd.NA, 0.2],
            "se_log_variance_ratio": [0.2, pd.NA, 0.05],
            "variance_ratio": [1.35, pd.NA, 1.22],
            "mean_diff": [0.0, pd.NA, 0.0],
            "inference_method": ["analytic_effective_n_simple_design"] * 3,
            "male_n": [111, 15, 200],
            "female_n": [121, 9, 210],
            "qa_flags": ["flag_a", "flag_b", pd.NA],
        }
    )
    normalized = normalize_estimate_table(df, source_id="local_nlsy", registry=registry)
    inferential, qa_only = build_dataset_focus_split(normalized, dataset_id="nlsy79_child_ya")

    assert len(inferential) == 1
    assert inferential.loc[0, "age_band"] == "16-17"
    assert len(qa_only) == 1
    assert qa_only.loc[0, "age_band"] == "12-13"


def test_build_dataset_focus_table_supports_family_and_ci_filters() -> None:
    registry = build_registry()
    df = pd.DataFrame(
        {
            "dataset_id": ["nhanes_2011_2023", "nhanes_2011_2023", "nhanes_2011_2023"],
            "cycle_or_wave": ["2011-2012", "2011-2012", "2011-2012"],
            "country": ["United States"] * 3,
            "age_band": ["20-23", "64-67", "68-71"],
            "trait_id": ["grip_strength_kg", "adult_cognition_screen", "bmi"],
            "log_variance_ratio": [0.3, pd.NA, 0.1],
            "se_log_variance_ratio": [0.08, pd.NA, 0.05],
            "variance_ratio": [1.35, pd.NA, 1.10],
            "mean_diff": [0.0, pd.NA, 0.0],
            "inference_method": ["stratified_cluster_bootstrap_psu", "unavailable", "stratified_cluster_bootstrap_psu"],
            "male_n": [250, 180, 280],
            "female_n": [260, 170, 290],
            "qa_flags": [pd.NA, "low_n_variance", pd.NA],
        }
    )
    normalized = normalize_estimate_table(df, source_id="nhanes_2011_2023", registry=registry)

    inferential_physical = build_dataset_focus_table(
        normalized,
        dataset_id="nhanes_2011_2023",
        ci_available=True,
        trait_families=("physical",),
    )
    qa_cognition = build_dataset_focus_table(
        normalized,
        dataset_id="nhanes_2011_2023",
        ci_available=False,
        trait_families=("later_life_cognition",),
    )

    assert inferential_physical["trait_label"].tolist() == ["BMI", "Grip strength (kg)"]
    assert qa_cognition["trait_label"].tolist() == ["Adult cognition screen"]


def test_load_comparison_tables_warns_when_expected_source_is_missing(tmp_path: Path) -> None:
    registry = build_registry()
    present = tmp_path / "results" / "piaac_cycle2"
    present.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "dataset_id": ["piaac_cycle2"],
            "cycle_or_wave": ["cycle2"],
            "country": ["United States"],
            "age_band": ["16-19"],
            "trait_id": ["literacy"],
            "log_variance_ratio": [0.1],
            "se_log_variance_ratio": [0.05],
            "variance_ratio": [1.1],
            "mean_diff": [0.0],
            "inference_method": ["replicate_weights_brr"],
            "male_n": [100],
            "female_n": [100],
            "qa_flags": [pd.NA],
        }
    )
    df.to_csv(present / "piaac_cycle2_trait_estimates.csv", index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        combined = load_comparison_tables(
            tmp_path,
            registry=registry,
            sources={
                "piaac_cycle2": Path("results/piaac_cycle2/piaac_cycle2_trait_estimates.csv"),
                "timss_2019": Path("results/timss_2019/timss_2019_trait_estimates.csv"),
            },
        )

    assert len(combined) == 1
    assert any("timss_2019" in str(item.message) for item in caught)


def test_load_robustness_summary_skips_empty_csv(tmp_path: Path) -> None:
    reports_root = tmp_path / "results"
    local_dir = reports_root / "local_nlsy"
    local_dir.mkdir(parents=True)
    (local_dir / "local_nlsy_robustness_summary.csv").write_text("", encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "run_cross_dataset_comparison",
        Path(__file__).resolve().parents[1] / "scripts" / "run_cross_dataset_comparison.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._load_robustness_summary(tmp_path)
    assert summary.empty


def test_supporting_evidence_builders_split_headline_and_method_limited_rows() -> None:
    registry = build_registry()
    df = pd.DataFrame(
        {
            "dataset_id": ["nhanes_2011_2023", "hrs_public", "psid_cds_tas", "piaac_cycle2"],
            "cycle_or_wave": ["2011-2012", "2022", "2021", "cycle2"],
            "country": ["United States"] * 4,
            "age_band": ["20-23", "64-67", "all_ages", "60-65"],
            "trait_id": ["grip_strength_kg", "numeracy", "sat_math", "numeracy"],
            "log_variance_ratio": [0.3, -0.1, 0.2, 0.4],
            "se_log_variance_ratio": [0.08, 0.09, 0.12, 0.05],
            "variance_ratio": [1.35, 0.90, 1.22, 1.49],
            "mean_diff": [0.0, 0.0, 0.0, 0.0],
            "inference_method": [
                "stratified_cluster_bootstrap_psu",
                "approximate_household_cluster_bootstrap",
                "analytic_effective_n_simple_design",
                "replicate_weights_brr",
            ],
            "male_n": [250, 220, 180, 500],
            "female_n": [260, 230, 175, 520],
            "qa_flags": [pd.NA, "approximate_design_bootstrap", pd.NA, pd.NA],
        }
    )
    normalized = normalize_estimate_table(df, source_id="mixed", registry=registry)

    summary = build_supporting_evidence_summary(normalized)
    top_cells = build_supporting_evidence_top_cells(normalized, limit=10)

    assert set(summary["dataset_id"]) == {"nhanes_2011_2023", "hrs_public", "psid_cds_tas"}
    assert "piaac_cycle2" not in set(summary["dataset_id"])
    assert SUPPORTING_EVIDENCE_DATASET_IDS == ("nhanes_2011_2023", "hrs_public", "psid_cds_tas")

    nhanes_row = summary[summary["dataset_id"] == "nhanes_2011_2023"].iloc[0]
    assert nhanes_row["headline_eligible_rows"] == 1
    assert nhanes_row["method_limited_rows"] == 0

    hrs_row = summary[summary["dataset_id"] == "hrs_public"].iloc[0]
    assert hrs_row["method_limited_rows"] == 1
    assert hrs_row["headline_eligible_rows"] == 0

    psid_row = summary[summary["dataset_id"] == "psid_cds_tas"].iloc[0]
    assert psid_row["method_limited_rows"] == 1
    assert psid_row["headline_eligible_rows"] == 0

    assert set(top_cells["dataset_label"]) == {"NHANES selected cycles", "HRS public", "PSID CDS / TAS"}
    assert "headline_eligible" in top_cells.columns
