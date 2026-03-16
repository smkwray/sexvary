import math

import numpy as np

from sexvary.metrics import (
    log_variance_ratio_from_groups,
    tail_rate_ratio_from_groups,
    weighted_mean,
    weighted_quantile,
    weighted_var,
)


def test_weighted_mean_and_variance_match_unweighted_for_equal_weights():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])
    assert weighted_mean(x, w) == 2.0
    assert math.isclose(weighted_var(x, w), 1.0)


def test_weighted_quantile_returns_reasonable_linear_interpolation():
    x = np.array([0.0, 10.0])
    w = np.array([1.0, 3.0])
    q50 = weighted_quantile(x, 0.5, w)
    assert math.isclose(q50, 10.0 / 3.0)


def test_log_variance_ratio_detects_larger_numerator_variance():
    female = np.array([0.0, 1.0, 2.0, 3.0])
    male = np.array([-1.0, 1.0, 3.0, 6.0])
    vr, md = log_variance_ratio_from_groups(male, female)
    assert vr.variance_ratio > 1.0
    assert vr.log_variance_ratio > 0.0
    assert md.mean_difference > 0.0


def test_tail_rate_ratio_returns_valid_upper_tail_result():
    female = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
    male = np.array([-2.0, -1.0, 0.0, 1.5, 2.0, 2.5, 3.0])
    res = tail_rate_ratio_from_groups(male, female, quantile=0.8, tail="upper")
    assert res.tail == "upper"
    assert res.rate_numerator >= res.rate_denominator
    assert res.rate_ratio >= 1.0
