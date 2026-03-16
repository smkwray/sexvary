import math

import numpy as np
import pandas as pd

from sexvary.survey import (
    jackknife_zone_replicate_estimates,
    combine_plausible_values,
    combine_plausible_values_and_replicates,
    replicate_variance,
    stratified_cluster_bootstrap_variance,
)


def test_replicate_variance_brr_default_scale():
    point = 10.0
    reps = [9.0, 11.0, 10.0, 12.0]
    res = replicate_variance(point, reps, method="brr")
    expected = ((1.0**2 + 1.0**2 + 0.0**2 + 2.0**2) / 4.0)
    assert math.isclose(res.sampling_variance, expected)


def test_combine_plausible_values_matches_rubin_rules():
    combo = combine_plausible_values([1.0, 2.0], [0.25, 0.25])
    assert math.isclose(combo.estimate, 1.5)
    assert math.isclose(combo.within_variance, 0.25)
    assert math.isclose(combo.between_variance, 0.5)
    assert math.isclose(combo.total_variance, 1.0)


def test_combine_plausible_values_and_replicates_runs():
    combo = combine_plausible_values_and_replicates(
        [1.0, 2.0],
        [
            [0.8, 1.2],
            [1.5, 2.5],
        ],
        method="brr",
    )
    assert combo.n_plausible_values == 2
    assert combo.total_variance > 0.0


def test_combine_plausible_values_and_replicates_skips_nonfinite_point_rows():
    combo = combine_plausible_values_and_replicates(
        [1.0, float("nan"), 2.0],
        [
            [0.8, 1.2],
            [0.5, 0.6],
            [1.5, 2.5],
        ],
        method="brr",
    )
    assert combo.n_plausible_values == 2
    assert combo.total_variance > 0.0


def test_stratified_cluster_bootstrap_variance_runs_deterministically():
    df = pd.DataFrame(
        {
            "score_raw": [1.0, 2.0, 1.5, 2.5, 1.2, 2.1, 1.7, 2.7],
            "sex_observed": ["male", "female"] * 4,
            "weight_main": [1.0] * 8,
            "design_strata": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "design_psu": ["1", "1", "2", "2", "3", "3", "4", "4"],
        }
    )

    def estimator(data: pd.DataFrame, weights: np.ndarray) -> float:
        male = data["sex_observed"] == "male"
        female = data["sex_observed"] == "female"
        male_mean = np.average(data.loc[male, "score_raw"], weights=weights[male.to_numpy()])
        female_mean = np.average(data.loc[female, "score_raw"], weights=weights[female.to_numpy()])
        return float(male_mean - female_mean)

    first = stratified_cluster_bootstrap_variance(
        df,
        estimator=estimator,
        weight_col="weight_main",
        strata_col="design_strata",
        cluster_col="design_psu",
        n_boot=12,
        random_state=123,
    )
    second = stratified_cluster_bootstrap_variance(
        df,
        estimator=estimator,
        weight_col="weight_main",
        strata_col="design_strata",
        cluster_col="design_psu",
        n_boot=12,
        random_state=123,
    )
    assert math.isclose(first.point_estimate, second.point_estimate)
    assert math.isclose(first.sampling_variance, second.sampling_variance)
    assert first.standard_error > 0.0


def test_jackknife_zone_replicate_estimates_runs():
    df = pd.DataFrame(
        {
            "score_raw": [10.0, 11.0, 9.0, 10.0, 14.0, 13.0, 8.0, 7.0],
            "sex_observed": ["male", "male", "female", "female"] * 2,
            "weight_main": [1.0] * 8,
            "jk_zone": [1, 1, 1, 1, 2, 2, 2, 2],
            "jk_rep": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )

    def estimator(data: pd.DataFrame, weights: np.ndarray) -> float:
        male = data["sex_observed"] == "male"
        female = data["sex_observed"] == "female"
        male_mean = np.average(data.loc[male, "score_raw"], weights=weights[male.to_numpy()])
        female_mean = np.average(data.loc[female, "score_raw"], weights=weights[female.to_numpy()])
        return float(male_mean - female_mean)

    reps = jackknife_zone_replicate_estimates(
        df,
        estimator=estimator,
        weight_col="weight_main",
        zone_col="jk_zone",
        rep_col="jk_rep",
    )
    assert len(reps) == 4
    assert np.isfinite(reps).all()
