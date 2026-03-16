import pandas as pd

from sexvary.pv_replicate import estimate_pv_replicate_cells, infer_replicate_design


def test_infer_replicate_design_maps_brr_tokens():
    df = pd.DataFrame({"variance_method": ["BRR"], "fay_factor": [0.5], "n_replicates": [80]})
    spec = infer_replicate_design(
        df,
        ["replicate_weight_001", "replicate_weight_002"],
        method_col="variance_method",
        fay_col="fay_factor",
        nrep_col="n_replicates",
        default_method="jackknife",
        default_fay=0.0,
    )
    assert spec.method == "brr"
    assert spec.fay == 0.5
    assert spec.n_replicates == 80


def test_infer_replicate_design_preserves_jrr_token():
    df = pd.DataFrame({"variance_method": ["JRR"], "n_replicates": [150]})
    spec = infer_replicate_design(
        df,
        ["replicate_weight_001", "replicate_weight_002"],
        method_col="variance_method",
        nrep_col="n_replicates",
        default_method="brr",
    )
    assert spec.method == "jrr"
    assert spec.n_replicates == 150


def test_estimate_pv_replicate_cells_runs_on_generic_long_frame():
    rows = []
    for pv_index in [1, 2]:
        for person_id, sex, score, weight, rep1, rep2 in [
            ("m1", "male", 10 + pv_index, 1.0, 0.8, 1.2),
            ("m2", "male", 14 + pv_index, 1.2, 1.0, 1.4),
            ("f1", "female", 8 + pv_index, 0.9, 1.1, 0.7),
            ("f2", "female", 9 + pv_index, 1.1, 1.3, 0.9),
        ]:
            rows.append(
                {
                    "dataset_id": "demo_assessment",
                    "cycle_or_wave": "2022",
                    "country": "United States",
                    "country_id": "USA",
                    "grade_or_age_band": "15-year-olds",
                    "trait_id": "math_achievement",
                    "person_id": person_id,
                    "sex_observed": sex,
                    "score_raw": float(score),
                    "weight_main": weight,
                    "pv_index": pv_index,
                    "variance_method": "BRR",
                    "fay_factor": 0.5,
                    "n_replicates": 2,
                    "replicate_weight_001": rep1,
                    "replicate_weight_002": rep2,
                }
            )
    df = pd.DataFrame(rows)
    estimates = estimate_pv_replicate_cells(
        df,
        group_cols=["dataset_id", "cycle_or_wave", "country", "country_id", "grade_or_age_band", "trait_id"],
        replicate_cols=["replicate_weight_001", "replicate_weight_002"],
        method_col="variance_method",
        fay_col="fay_factor",
        nrep_col="n_replicates",
        default_method="brr",
        default_fay=0.5,
    )
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert row["replicate_method"] == "brr"
    assert row["se_log_variance_ratio"] > 0.0
