from sexvary.meta import dersimonian_laird_meta, fixed_effect_meta


def test_fixed_effect_meta_runs():
    res = fixed_effect_meta([0.1, 0.2, 0.3], [0.04, 0.04, 0.04])
    assert res.k == 3
    assert 0.1 < res.estimate < 0.3
    assert res.tau2 == 0.0


def test_random_effects_meta_runs():
    res = dersimonian_laird_meta([0.1, 0.5, 0.3], [0.04, 0.04, 0.04])
    assert res.k == 3
    assert res.standard_error > 0.0
    assert res.estimate_backtransformed > 0.0
