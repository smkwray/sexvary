import math

import pandas as pd

from sexvary.estimation import (
    EstimationConfig,
    derive_age_band,
    estimation_config_from_analysis,
    estimate_dataset_cells,
    prepare_analysis_frame,
)


def test_derive_age_band_groups_two_year_bins():
    ages = pd.Series([14.2, 15.9, 16.0, None])
    out = derive_age_band(ages, width_years=2)
    assert list(out[:3]) == ["14-15", "14-15", "16-17"]
    assert pd.isna(out.iloc[3])


def test_prepare_analysis_frame_fills_missing_analysis_cells_from_age():
    df = pd.DataFrame(
        {
            "dataset_id": ["demo"] * 2,
            "cycle_or_wave": [pd.NA, pd.NA],
            "country": [pd.NA, pd.NA],
            "grade_or_age_band": [pd.NA, pd.NA],
            "person_id": [1, 2],
            "sex_observed": ["male", "female"],
            "age": [14.1, 15.7],
            "trait_id": ["trait"] * 2,
            "score_raw": [1.0, 2.0],
            "weight_main": [1.0, 1.0],
        }
    )
    prepared = prepare_analysis_frame(df, config=EstimationConfig())
    assert prepared["analysis_cell"].tolist() == ["12-15", "12-15"]
    assert prepared["cycle_or_wave"].tolist() == ["all", "all"]
    assert prepared["country"].tolist() == ["all", "all"]


def test_prepare_analysis_frame_uses_dataset_specific_age_band_width():
    df = pd.DataFrame(
        {
            "dataset_id": ["nlsy79_child_ya", "nlsy79_child_ya", "nlsy79_main", "nlsy79_main"],
            "cycle_or_wave": [pd.NA] * 4,
            "country": [pd.NA] * 4,
            "grade_or_age_band": [pd.NA] * 4,
            "person_id": [1, 2, 3, 4],
            "sex_observed": ["male", "female", "male", "female"],
            "age": [14.1, 15.7, 14.1, 15.7],
            "trait_id": ["trait"] * 4,
            "score_raw": [1.0, 2.0, 1.5, 2.5],
            "weight_main": [1.0] * 4,
        }
    )
    prepared = prepare_analysis_frame(
        df,
        config=EstimationConfig(default_age_band_width_years=4, dataset_age_band_width_years={"nlsy79_child_ya": 2}),
    )
    child = prepared[prepared["dataset_id"] == "nlsy79_child_ya"]["analysis_cell"].tolist()
    adult = prepared[prepared["dataset_id"] == "nlsy79_main"]["analysis_cell"].tolist()
    assert child == ["14-15", "14-15"]
    assert adult == ["12-15", "12-15"]


def test_prepare_analysis_frame_can_fallback_to_all_ages_when_no_variance_cells_exist():
    rows = []
    for i in range(60):
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": pd.NA,
                "country": pd.NA,
                "grade_or_age_band": pd.NA,
                "person_id": i,
                "sex_observed": "male",
                "age": 14.0 + (i % 4),
                "trait_id": "trait",
                "score_raw": float(i),
                "weight_main": 1.0,
            }
        )
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": pd.NA,
                "country": pd.NA,
                "grade_or_age_band": pd.NA,
                "person_id": 1000 + i,
                "sex_observed": "female",
                "age": 14.0 + (i % 4),
                "trait_id": "trait",
                "score_raw": float(i),
                "weight_main": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    prepared = prepare_analysis_frame(
        df,
        config=EstimationConfig(
            default_age_band_width_years=4,
            dataset_age_band_width_years={"nlsy79_child_ya": 2},
            fallback_to_all_ages_if_no_variance_cells=("nlsy79_child_ya",),
            min_n_per_sex_for_variance=200,
        ),
    )
    assert prepared["analysis_cell"].nunique() == 1
    assert prepared["analysis_cell"].iloc[0] == "all_ages"
    assert prepared["used_age_fallback"].all()


def test_prepare_analysis_frame_can_fallback_to_all_ages_per_trait():
    rows = []
    for i in range(110):
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": pd.NA,
                "country": pd.NA,
                "grade_or_age_band": pd.NA,
                "person_id": f"ppvt_m{i}",
                "sex_observed": "male",
                "age": 16.0 if i < 84 else 10.0,
                "trait_id": "ppvt",
                "score_raw": float(i),
                "weight_main": 1.0,
            }
        )
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": pd.NA,
                "country": pd.NA,
                "grade_or_age_band": pd.NA,
                "person_id": f"ppvt_f{i}",
                "sex_observed": "female",
                "age": 16.0 if i < 84 else 10.0,
                "trait_id": "ppvt",
                "score_raw": float(i),
                "weight_main": 1.0,
            }
        )
    for i in range(110):
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": pd.NA,
                "country": pd.NA,
                "grade_or_age_band": pd.NA,
                "person_id": f"piat_m{i}",
                "sex_observed": "male",
                "age": 16.0,
                "trait_id": "piat_math",
                "score_raw": float(i),
                "weight_main": 1.0,
            }
        )
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": pd.NA,
                "country": pd.NA,
                "grade_or_age_band": pd.NA,
                "person_id": f"piat_f{i}",
                "sex_observed": "female",
                "age": 16.0,
                "trait_id": "piat_math",
                "score_raw": float(i),
                "weight_main": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    prepared = prepare_analysis_frame(
        df,
        config=EstimationConfig(
            default_age_band_width_years=4,
            dataset_age_band_width_years={"nlsy79_child_ya": 2},
            dataset_min_n_per_sex_for_variance={"nlsy79_child_ya": 100},
            fallback_to_all_ages_if_no_variance_cells_per_trait=("nlsy79_child_ya",),
        ),
    )
    ppvt_cells = prepared.loc[prepared["trait_id"] == "ppvt", "analysis_cell"].unique().tolist()
    piat_cells = prepared.loc[prepared["trait_id"] == "piat_math", "analysis_cell"].unique().tolist()
    assert ppvt_cells == ["all_ages"]
    assert piat_cells == ["16-17"]


def test_estimation_config_from_analysis_parses_dataset_specific_n_overrides():
    config = estimation_config_from_analysis(
        {
            "analysis_defaults": {
                "min_n_per_sex_for_variance": 200,
                "dataset_min_n_per_sex_for_variance": {"nlsy79_child_ya": 100},
                "boundary_mass_suppress_variance_share": 0.3,
            }
        }
    )
    assert config.min_n_per_sex_for_variance == 200
    assert config.dataset_min_n_per_sex_for_variance == {"nlsy79_child_ya": 100}
    assert config.fallback_to_all_ages_if_no_variance_cells_per_trait == ()
    assert config.boundary_mass_suppress_variance_share == 0.3


def test_estimate_dataset_cells_returns_expected_metrics():
    rows = []
    for i in range(300):
        rows.append(
            {
                "dataset_id": "demo",
                "cycle_or_wave": "all",
                "country": "all",
                "grade_or_age_band": "all",
                "person_id": i,
                "sex_observed": "male",
                "age": 20,
                "trait_id": "trait",
                "score_raw": float(i % 40),
                "weight_main": 1.0,
            }
        )
        rows.append(
            {
                "dataset_id": "demo",
                "cycle_or_wave": "all",
                "country": "all",
                "grade_or_age_band": "all",
                "person_id": 10_000 + i,
                "sex_observed": "female",
                "age": 20,
                "trait_id": "trait",
                "score_raw": float(i % 20),
                "weight_main": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    estimates = estimate_dataset_cells(df, config=EstimationConfig())
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert row["male_n"] == 300
    assert row["female_n"] == 300
    assert not row["used_age_fallback"]
    assert row["variance_ratio"] > 1.0
    assert row["inference_method"] == "analytic_effective_n_simple_design"
    assert row["se_log_variance_ratio"] > 0.0
    assert row["ci_low_log_variance_ratio"] < row["log_variance_ratio"] < row["ci_high_log_variance_ratio"]
    assert row["top90_rate_ratio"] >= 1.0
    assert pd.isna(row["top95_rate_ratio"])
    assert row["qa_flags"] == "low_n_tail_95"


def test_estimate_dataset_cells_respects_dataset_specific_variance_threshold():
    rows = []
    for i in range(110):
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": "all",
                "country": "all",
                "grade_or_age_band": pd.NA,
                "person_id": f"m{i}",
                "sex_observed": "male",
                "age": 14.0 + (i % 4),
                "trait_id": "piat_math",
                "score_raw": float((i % 30) + 10),
                "weight_main": 1.0,
            }
        )
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": "all",
                "country": "all",
                "grade_or_age_band": pd.NA,
                "person_id": f"f{i}",
                "sex_observed": "female",
                "age": 14.0 + (i % 4),
                "trait_id": "piat_math",
                "score_raw": float((i % 20) + 10),
                "weight_main": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    estimates = estimate_dataset_cells(
        df,
        config=EstimationConfig(
            min_n_per_sex_for_variance=200,
            min_n_per_sex_for_95_tail=500,
            dataset_age_band_width_years={"nlsy79_child_ya": 2},
            dataset_min_n_per_sex_for_variance={"nlsy79_child_ya": 100},
            fallback_to_all_ages_if_no_variance_cells=("nlsy79_child_ya",),
        ),
    )
    row = estimates.iloc[0]
    assert row["male_n"] == 110
    assert row["female_n"] == 110
    assert row["inference_method"] == "analytic_effective_n_simple_design"
    assert "low_n_variance" not in str(row["qa_flags"])
    assert "low_n_tail_95" in str(row["qa_flags"])


def test_estimate_dataset_cells_flags_nonprimary_weight_fallback():
    rows = []
    for i in range(110):
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": "all",
                "country": "all",
                "grade_or_age_band": "all_ages",
                "person_id": f"m{i}",
                "sex_observed": "male",
                "age": 16.0,
                "trait_id": "ppvt",
                "score_raw": float((i % 20) + 10),
                "weight_main": 1.0,
                "weight_source": "mother_sampling_weight_79" if i >= 20 else "child_sampling_weight_1998",
                "weight_primary_source": "child_sampling_weight_1998",
            }
        )
        rows.append(
            {
                "dataset_id": "nlsy79_child_ya",
                "cycle_or_wave": "all",
                "country": "all",
                "grade_or_age_band": "all_ages",
                "person_id": f"f{i}",
                "sex_observed": "female",
                "age": 16.0,
                "trait_id": "ppvt",
                "score_raw": float((i % 15) + 10),
                "weight_main": 1.0,
                "weight_source": "mother_sampling_weight_79" if i >= 20 else "child_sampling_weight_1998",
                "weight_primary_source": "child_sampling_weight_1998",
            }
        )
    estimates = estimate_dataset_cells(
        pd.DataFrame(rows),
        config=EstimationConfig(dataset_min_n_per_sex_for_variance={"nlsy79_child_ya": 100}),
    )
    row = estimates.iloc[0]
    assert row["inference_method"] == "analytic_effective_n_simple_design"
    assert row["used_nonprimary_weight_fallback"]
    assert row["weight_sources"] == "child_sampling_weight_1998;mother_sampling_weight_79"
    assert "nonprimary_weight_fallback" in str(row["qa_flags"])
    assert "mixed_weight_sources" in str(row["qa_flags"])


def test_estimate_dataset_cells_prefers_design_aware_inference_when_design_columns_exist():
    rows = []
    strata_psu = [("s1", "p1"), ("s1", "p2"), ("s2", "p3"), ("s2", "p4")]
    for cluster_index, (stratum, psu) in enumerate(strata_psu):
        for i in range(80):
            rows.append(
                {
                    "dataset_id": "ecls_k_2011",
                    "cycle_or_wave": "fall_kindergarten_2010",
                    "country": "United States",
                    "grade_or_age_band": "K",
                    "design_strata": stratum,
                    "design_psu": psu,
                    "person_id": f"m{cluster_index}_{i}",
                    "sex_observed": "male",
                    "age": 5.5,
                    "trait_id": "reading_achievement",
                    "score_raw": float((i % 20) + cluster_index),
                    "weight_main": 1.0,
                }
            )
            rows.append(
                {
                    "dataset_id": "ecls_k_2011",
                    "cycle_or_wave": "fall_kindergarten_2010",
                    "country": "United States",
                    "grade_or_age_band": "K",
                    "design_strata": stratum,
                    "design_psu": psu,
                    "person_id": f"f{cluster_index}_{i}",
                    "sex_observed": "female",
                    "age": 5.5,
                    "trait_id": "reading_achievement",
                    "score_raw": float((i % 15) + (cluster_index * 0.5)),
                    "weight_main": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    estimates = estimate_dataset_cells(
        df,
        config=EstimationConfig(design_bootstrap_replicates=12, design_bootstrap_seed=123),
    )
    row = estimates.iloc[0]
    assert row["inference_method"] == "stratified_cluster_bootstrap_psu"
    assert row["se_log_variance_ratio"] > 0.0
    assert row["ci_low_log_variance_ratio"] < row["log_variance_ratio"] < row["ci_high_log_variance_ratio"]


def test_estimate_dataset_cells_prefers_replicate_weight_inference_when_available():
    rows = []
    for i in range(300):
        male_score = float(i % 40)
        female_score = float(i % 20)
        rows.append(
            {
                "dataset_id": "hsls_2009",
                "cycle_or_wave": "fall_2009_grade_9",
                "country": "United States",
                "grade_or_age_band": "9",
                "person_id": f"m{i}",
                "sex_observed": "male",
                "age": 14.5,
                "trait_id": "math_achievement",
                "score_raw": male_score,
                "weight_main": 1.0,
                "replicate_method": "brr",
                "replicate_weight_001": 1.2 if i < 150 else 0.8,
                "replicate_weight_002": 0.7 if i < 150 else 1.3,
            }
        )
        rows.append(
            {
                "dataset_id": "hsls_2009",
                "cycle_or_wave": "fall_2009_grade_9",
                "country": "United States",
                "grade_or_age_band": "9",
                "person_id": f"f{i}",
                "sex_observed": "female",
                "age": 14.5,
                "trait_id": "math_achievement",
                "score_raw": female_score,
                "weight_main": 1.0,
                "replicate_method": "brr",
                "replicate_weight_001": 0.8 if i < 150 else 1.2,
                "replicate_weight_002": 1.3 if i < 150 else 0.7,
            }
        )
    df = pd.DataFrame(rows)
    estimates = estimate_dataset_cells(df, config=EstimationConfig())
    row = estimates.iloc[0]
    assert row["inference_method"] == "replicate_weights_brr"
    assert row["se_log_variance_ratio"] > 0.0
    assert row["ci_low_log_variance_ratio"] < row["log_variance_ratio"] < row["ci_high_log_variance_ratio"]


def test_estimate_dataset_cells_suppresses_tail_metrics_for_bounded_traits() -> None:
    rows = []
    male_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 30
    female_values = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8] * 30
    for i, score in enumerate(male_values):
        rows.append(
            {
                "dataset_id": "nhanes_2011_2023",
                "cycle_or_wave": "2011-2012",
                "country": "United States",
                "grade_or_age_band": "20-23",
                "person_id": f"m{i}",
                "sex_observed": "male",
                "age": 22.0,
                "trait_id": "adult_cognition_screen",
                "score_raw": float(score),
                "weight_main": 1.0,
            }
        )
    for i, score in enumerate(female_values):
        rows.append(
            {
                "dataset_id": "nhanes_2011_2023",
                "cycle_or_wave": "2011-2012",
                "country": "United States",
                "grade_or_age_band": "20-23",
                "person_id": f"f{i}",
                "sex_observed": "female",
                "age": 22.0,
                "trait_id": "adult_cognition_screen",
                "score_raw": float(score),
                "weight_main": 1.0,
            }
        )
    row = estimate_dataset_cells(pd.DataFrame(rows), config=EstimationConfig()).iloc[0]
    assert row["trait_scale_type"] == "bounded_count"
    assert row["tail_metrics_suppressed"]
    assert row["inference_method"] == "analytic_effective_n_simple_design"
    assert math.isnan(row["top90_rate_ratio"])
    assert math.isnan(row["bottom10_rate_ratio"])
    assert "tail_metrics_suppressed" in str(row["qa_flags"])
    assert "bounded_scale_variance_fragile" in str(row["qa_flags"])


def test_estimate_dataset_cells_suppresses_variance_for_extreme_bounded_piling() -> None:
    rows = []
    male_values = ([0] * 210) + ([1] * 20) + ([2] * 20) + ([3] * 20) + ([4] * 15) + ([5] * 15)
    female_values = ([0] * 205) + ([1] * 25) + ([2] * 20) + ([3] * 20) + ([4] * 15) + ([5] * 15)
    for i, score in enumerate(male_values):
        rows.append(
            {
                "dataset_id": "nhanes_2011_2023",
                "cycle_or_wave": "2011-2012",
                "country": "United States",
                "grade_or_age_band": "68-71",
                "person_id": f"m{i}",
                "sex_observed": "male",
                "age": 69.0,
                "trait_id": "adult_cognition_screen",
                "score_raw": float(score),
                "weight_main": 1.0,
            }
        )
    for i, score in enumerate(female_values):
        rows.append(
            {
                "dataset_id": "nhanes_2011_2023",
                "cycle_or_wave": "2011-2012",
                "country": "United States",
                "grade_or_age_band": "68-71",
                "person_id": f"f{i}",
                "sex_observed": "female",
                "age": 69.0,
                "trait_id": "adult_cognition_screen",
                "score_raw": float(score),
                "weight_main": 1.0,
            }
        )
    row = estimate_dataset_cells(pd.DataFrame(rows), config=EstimationConfig()).iloc[0]
    assert row["trait_scale_type"] == "bounded_count"
    assert row["tail_metrics_suppressed"]
    assert row["inference_method"] == "unavailable"
    assert math.isnan(row["log_variance_ratio"])
    assert math.isnan(row["variance_ratio"])
    assert math.isnan(row["top90_rate_ratio"])
    assert "bounded_scale_variance_suppressed" in str(row["qa_flags"])
