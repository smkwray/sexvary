from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class VarianceRatioResult:
    numerator_label: str
    denominator_label: str
    n_numerator: int
    n_denominator: int
    weight_sum_numerator: float
    weight_sum_denominator: float
    mean_numerator: float
    mean_denominator: float
    var_numerator: float
    var_denominator: float
    variance_ratio: float
    log_variance_ratio: float


@dataclass(frozen=True)
class MeanDifferenceResult:
    numerator_label: str
    denominator_label: str
    mean_difference: float
    standardized_mean_difference: float


@dataclass(frozen=True)
class TailRatioResult:
    numerator_label: str
    denominator_label: str
    tail: str
    quantile: float
    threshold: float
    rate_numerator: float
    rate_denominator: float
    rate_ratio: float
    baseline_share_numerator: float
    tail_share_numerator: float
    representation_ratio: float


def _to_1d_float_array(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def _clean_values_and_weights(
    values: Iterable[float] | np.ndarray,
    weights: Iterable[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = _to_1d_float_array(values, name="values")
    if weights is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = _to_1d_float_array(weights, name="weights")
        if w.shape != x.shape:
            raise ValueError("values and weights must have the same shape.")
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        raise ValueError("No valid observations remain after dropping missing or non-positive weights.")
    return x, w


def effective_sample_size(weights: Iterable[float] | np.ndarray) -> float:
    w = _to_1d_float_array(weights, name="weights")
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(np.square(w.sum()) / np.square(w).sum())


def weighted_mean(values: Iterable[float] | np.ndarray, weights: Iterable[float] | np.ndarray | None = None) -> float:
    x, w = _clean_values_and_weights(values, weights)
    return float(np.average(x, weights=w))


def weighted_var(
    values: Iterable[float] | np.ndarray,
    weights: Iterable[float] | np.ndarray | None = None,
    *,
    ddof: int = 1,
) -> float:
    x, w = _clean_values_and_weights(values, weights)
    mu = np.average(x, weights=w)
    centered = x - mu
    numerator = float(np.sum(w * centered * centered))
    if ddof == 0:
        denominator = float(np.sum(w))
    elif ddof == 1:
        sum_w = float(np.sum(w))
        sum_w2 = float(np.sum(np.square(w)))
        denominator = sum_w - (sum_w2 / sum_w)
    else:
        raise ValueError("weighted_var supports ddof 0 or 1 only.")
    if denominator <= 0:
        raise ValueError("Non-positive denominator encountered while computing weighted variance.")
    return numerator / denominator


def weighted_std(
    values: Iterable[float] | np.ndarray,
    weights: Iterable[float] | np.ndarray | None = None,
    *,
    ddof: int = 1,
) -> float:
    return float(np.sqrt(weighted_var(values, weights, ddof=ddof)))


def weighted_quantile(
    values: Iterable[float] | np.ndarray,
    quantiles: float | Sequence[float],
    weights: Iterable[float] | np.ndarray | None = None,
) -> float | np.ndarray:
    """Compute weighted quantiles.

    The implementation uses linear interpolation on the cumulative weight distribution.
    This is suitable for pooled-threshold construction in this project, but dataset-specific
    adapters may substitute a survey package's official quantile implementation if needed.
    """
    x, w = _clean_values_and_weights(values, weights)
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Quantiles must lie in [0, 1].")

    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    total = cum_w[-1]
    probs = cum_w / total

    interp_x = np.concatenate(([x_sorted[0]], x_sorted))
    interp_p = np.concatenate(([0.0], probs))
    out = np.interp(q, interp_p, interp_x)
    if np.isscalar(quantiles):
        return float(out[0])
    return out


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.inf if numerator > 0 else np.nan
    return numerator / denominator


def log_variance_ratio(var_numerator: float, var_denominator: float) -> float:
    if var_numerator <= 0 or var_denominator <= 0:
        raise ValueError("Variances must be strictly positive to compute a log variance ratio.")
    return float(np.log(var_numerator / var_denominator))


def standardized_mean_difference(
    mean_numerator: float,
    mean_denominator: float,
    var_numerator: float,
    var_denominator: float,
) -> float:
    pooled_sd = np.sqrt((var_numerator + var_denominator) / 2.0)
    if pooled_sd <= 0:
        return np.nan
    return float((mean_numerator - mean_denominator) / pooled_sd)


def log_variance_ratio_from_groups(
    numerator_values: Iterable[float] | np.ndarray,
    denominator_values: Iterable[float] | np.ndarray,
    *,
    numerator_weights: Iterable[float] | np.ndarray | None = None,
    denominator_weights: Iterable[float] | np.ndarray | None = None,
    numerator_label: str = "male",
    denominator_label: str = "female",
) -> tuple[VarianceRatioResult, MeanDifferenceResult]:
    x_num, w_num = _clean_values_and_weights(numerator_values, numerator_weights)
    x_den, w_den = _clean_values_and_weights(denominator_values, denominator_weights)

    mean_num = weighted_mean(x_num, w_num)
    mean_den = weighted_mean(x_den, w_den)
    var_num = weighted_var(x_num, w_num)
    var_den = weighted_var(x_den, w_den)
    vr = _safe_ratio(var_num, var_den)
    log_vr = log_variance_ratio(var_num, var_den)

    vr_result = VarianceRatioResult(
        numerator_label=numerator_label,
        denominator_label=denominator_label,
        n_numerator=int(x_num.size),
        n_denominator=int(x_den.size),
        weight_sum_numerator=float(w_num.sum()),
        weight_sum_denominator=float(w_den.sum()),
        mean_numerator=mean_num,
        mean_denominator=mean_den,
        var_numerator=var_num,
        var_denominator=var_den,
        variance_ratio=vr,
        log_variance_ratio=log_vr,
    )
    md_result = MeanDifferenceResult(
        numerator_label=numerator_label,
        denominator_label=denominator_label,
        mean_difference=float(mean_num - mean_den),
        standardized_mean_difference=standardized_mean_difference(mean_num, mean_den, var_num, var_den),
    )
    return vr_result, md_result


def tail_rate_ratio_from_groups(
    numerator_values: Iterable[float] | np.ndarray,
    denominator_values: Iterable[float] | np.ndarray,
    *,
    numerator_weights: Iterable[float] | np.ndarray | None = None,
    denominator_weights: Iterable[float] | np.ndarray | None = None,
    quantile: float = 0.95,
    tail: str = "upper",
    numerator_label: str = "male",
    denominator_label: str = "female",
) -> TailRatioResult:
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be either 'upper' or 'lower'.")
    x_num, w_num = _clean_values_and_weights(numerator_values, numerator_weights)
    x_den, w_den = _clean_values_and_weights(denominator_values, denominator_weights)

    pooled_values = np.concatenate([x_num, x_den])
    pooled_weights = np.concatenate([w_num, w_den])
    threshold = weighted_quantile(pooled_values, quantile, pooled_weights)

    if tail == "upper":
        num_mask = x_num >= threshold
        den_mask = x_den >= threshold
    else:
        num_mask = x_num <= threshold
        den_mask = x_den <= threshold

    rate_num = float(w_num[num_mask].sum() / w_num.sum())
    rate_den = float(w_den[den_mask].sum() / w_den.sum())
    rr = _safe_ratio(rate_num, rate_den)

    baseline_num = float(w_num.sum() / (w_num.sum() + w_den.sum()))
    total_tail_weight = float(w_num[num_mask].sum() + w_den[den_mask].sum())
    tail_share_num = np.nan
    representation_ratio = np.nan
    if total_tail_weight > 0:
        tail_share_num = float(w_num[num_mask].sum() / total_tail_weight)
        if baseline_num > 0:
            representation_ratio = float(tail_share_num / baseline_num)

    return TailRatioResult(
        numerator_label=numerator_label,
        denominator_label=denominator_label,
        tail=tail,
        quantile=quantile,
        threshold=float(threshold),
        rate_numerator=rate_num,
        rate_denominator=rate_den,
        rate_ratio=rr,
        baseline_share_numerator=baseline_num,
        tail_share_numerator=tail_share_num,
        representation_ratio=representation_ratio,
    )
