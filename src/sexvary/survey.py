from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplicateVarianceResult:
    point_estimate: float
    sampling_variance: float
    standard_error: float
    method: str
    n_replicates: int
    scale: float


@dataclass(frozen=True)
class PlausibleValueCombination:
    estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    standard_error: float
    n_plausible_values: int


@dataclass(frozen=True)
class StratifiedClusterBootstrapResult:
    point_estimate: float
    sampling_variance: float
    standard_error: float
    method: str
    n_replicates: int
    n_strata: int
    n_clusters: int


def replicate_variance(
    point_estimate: float,
    replicate_estimates: Iterable[float],
    *,
    method: str = "brr",
    fay: float = 0.0,
    scale: float | None = None,
) -> ReplicateVarianceResult:
    reps = np.asarray(list(replicate_estimates), dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        raise ValueError("replicate_estimates must contain at least one finite value.")

    method = method.lower()
    if scale is None:
        if method == "brr":
            scale = 1.0 / (reps.size * (1.0 - fay) ** 2)
        elif method in {"jackknife", "jrr"}:
            scale = (reps.size - 1.0) / reps.size
        else:
            scale = 1.0 / reps.size

    diffs = reps - float(point_estimate)
    var = float(scale * np.sum(np.square(diffs)))
    return ReplicateVarianceResult(
        point_estimate=float(point_estimate),
        sampling_variance=var,
        standard_error=float(np.sqrt(max(var, 0.0))),
        method=method,
        n_replicates=int(reps.size),
        scale=float(scale),
    )


def apply_estimator_to_replicates(
    data: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame, str], float],
    weight_col: str,
    replicate_weight_cols: list[str],
) -> tuple[float, np.ndarray]:
    point = float(estimator(data, weight_col))
    reps = np.asarray([float(estimator(data, col)) for col in replicate_weight_cols], dtype=float)
    return point, reps


def combine_plausible_values(
    point_estimates: Iterable[float],
    sampling_variances: Iterable[float],
) -> PlausibleValueCombination:
    est = np.asarray(list(point_estimates), dtype=float)
    var = np.asarray(list(sampling_variances), dtype=float)
    mask = np.isfinite(est) & np.isfinite(var)
    est = est[mask]
    var = var[mask]
    if est.size == 0:
        raise ValueError("No valid plausible value estimates supplied.")

    m = est.size
    estimate = float(np.mean(est))
    within = float(np.mean(var))
    between = 0.0 if m == 1 else float(np.var(est, ddof=1))
    total = float(within + (1.0 + 1.0 / m) * between)
    return PlausibleValueCombination(
        estimate=estimate,
        within_variance=within,
        between_variance=between,
        total_variance=total,
        standard_error=float(np.sqrt(max(total, 0.0))),
        n_plausible_values=int(m),
    )


def combine_plausible_values_and_replicates(
    point_estimates: Iterable[float],
    replicate_estimates: Iterable[Iterable[float]],
    *,
    method: str = "brr",
    fay: float = 0.0,
    scale: float | None = None,
) -> PlausibleValueCombination:
    point_arr = np.asarray(list(point_estimates), dtype=float)
    rep_rows = [np.asarray(list(reps), dtype=float) for reps in replicate_estimates]
    if point_arr.size != len(rep_rows):
        raise ValueError("Number of point estimates must match replicate_estimates.")
    if not rep_rows:
        raise ValueError("replicate_estimates must contain at least one row.")

    sampling_vars = []
    valid_points = []
    for point, reps in zip(point_arr, rep_rows):
        if not np.isfinite(point):
            continue
        if reps.ndim != 1:
            raise ValueError("Each replicate-estimate row must be one-dimensional.")
        rv = replicate_variance(point, reps, method=method, fay=fay, scale=scale)
        sampling_vars.append(rv.sampling_variance)
        valid_points.append(float(point))
    return combine_plausible_values(valid_points, sampling_vars)


def stratified_cluster_bootstrap_variance(
    data: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame, np.ndarray], float],
    weight_col: str,
    strata_col: str,
    cluster_col: str,
    n_boot: int = 40,
    random_state: int = 0,
) -> StratifiedClusterBootstrapResult:
    if n_boot <= 1:
        raise ValueError("n_boot must be greater than 1.")

    work = data.reset_index(drop=True).copy()
    weights = pd.to_numeric(work[weight_col], errors="coerce").to_numpy(dtype=float)
    strata = work[strata_col].astype("string")
    clusters = work[cluster_col].astype("string")
    valid = np.isfinite(weights) & (weights > 0) & strata.notna().to_numpy() & clusters.notna().to_numpy()
    if valid.sum() == 0:
        raise ValueError("No valid rows available for stratified cluster bootstrap variance.")

    work = work.loc[valid].reset_index(drop=True)
    weights = weights[valid]
    strata = strata.loc[valid].reset_index(drop=True)
    clusters = clusters.loc[valid].reset_index(drop=True)

    strata_groups: dict[str, list[np.ndarray]] = {}
    for stratum_value in strata.drop_duplicates().tolist():
        stratum_mask = strata == stratum_value
        stratum_clusters = clusters.loc[stratum_mask]
        cluster_arrays: list[np.ndarray] = []
        for cluster_value in stratum_clusters.drop_duplicates().tolist():
            idx = np.flatnonzero(stratum_mask.to_numpy() & (clusters.to_numpy() == cluster_value))
            if idx.size:
                cluster_arrays.append(idx)
        if cluster_arrays:
            strata_groups[str(stratum_value)] = cluster_arrays

    n_clusters = int(sum(len(cluster_arrays) for cluster_arrays in strata_groups.values()))
    if n_clusters < 2:
        raise ValueError("At least two clusters are required for stratified cluster bootstrap variance.")

    point_estimate = float(estimator(work, weights))
    rng = np.random.default_rng(random_state)
    replicate_estimates: list[float] = []

    for _ in range(n_boot):
        replicate_multiplier = np.zeros(len(work), dtype=float)
        for cluster_arrays in strata_groups.values():
            n_psu = len(cluster_arrays)
            sampled = rng.integers(0, n_psu, size=n_psu)
            counts = np.bincount(sampled, minlength=n_psu)
            for pos, count in enumerate(counts):
                if count > 0:
                    replicate_multiplier[cluster_arrays[pos]] = float(count)
        replicate_weights = weights * replicate_multiplier
        try:
            estimate = float(estimator(work, replicate_weights))
        except ValueError:
            continue
        if np.isfinite(estimate):
            replicate_estimates.append(estimate)

    reps = np.asarray(replicate_estimates, dtype=float)
    if reps.size < 2:
        raise ValueError("Too few valid bootstrap replicates were available for design-aware variance.")
    sampling_variance = float(np.var(reps, ddof=1))
    return StratifiedClusterBootstrapResult(
        point_estimate=point_estimate,
        sampling_variance=sampling_variance,
        standard_error=float(np.sqrt(max(sampling_variance, 0.0))),
        method="stratified_cluster_bootstrap",
        n_replicates=int(reps.size),
        n_strata=int(len(strata_groups)),
        n_clusters=n_clusters,
    )


def jackknife_zone_replicate_estimates(
    data: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame, np.ndarray], float],
    weight_col: str,
    zone_col: str,
    rep_col: str,
) -> np.ndarray:
    work = data.reset_index(drop=True).copy()
    weights = pd.to_numeric(work[weight_col], errors="coerce").to_numpy(dtype=float)
    zones = work[zone_col]
    reps = work[rep_col]
    valid = np.isfinite(weights) & (weights > 0) & zones.notna().to_numpy() & reps.notna().to_numpy()
    if valid.sum() == 0:
        raise ValueError("No valid rows available for jackknife zone replicate estimation.")

    work = work.loc[valid].reset_index(drop=True)
    weights = weights[valid]
    zones = zones.loc[valid].reset_index(drop=True)
    reps = reps.loc[valid].reset_index(drop=True)

    replicate_estimates: list[float] = []
    for zone_value in zones.drop_duplicates().tolist():
        zone_mask = (zones == zone_value).to_numpy()
        zone_rep_values = reps.loc[zone_mask].drop_duplicates().tolist()
        if len(zone_rep_values) < 2:
            continue
        for rep_value in zone_rep_values[:2]:
            replicate_weights = weights.copy()
            keep_mask = zone_mask & (reps.to_numpy() == rep_value)
            drop_mask = zone_mask & (reps.to_numpy() != rep_value)
            replicate_weights[keep_mask] *= 2.0
            replicate_weights[drop_mask] = 0.0
            try:
                estimate = float(estimator(work, replicate_weights))
            except ValueError:
                continue
            if np.isfinite(estimate):
                replicate_estimates.append(estimate)

    reps_arr = np.asarray(replicate_estimates, dtype=float)
    if reps_arr.size < 2:
        raise ValueError("Too few valid jackknife replicate estimates were available.")
    return reps_arr
