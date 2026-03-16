from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import log_variance_ratio_from_groups, tail_rate_ratio_from_groups
from .survey import combine_plausible_values_and_replicates, jackknife_zone_replicate_estimates


@dataclass(frozen=True)
class TIMSSReplicateSpec:
    method: str
    scale: float
    n_replicates: int


def infer_timss_replicate_spec(df: pd.DataFrame) -> TIMSSReplicateSpec:
    valid = df[["jk_zone", "jk_rep"]].dropna() if {"jk_zone", "jk_rep"}.issubset(df.columns) else pd.DataFrame()
    if valid.empty:
        return TIMSSReplicateSpec(method="jrr", scale=0.5, n_replicates=0)
    n_replicates = int(valid.drop_duplicates().shape[0])
    return TIMSSReplicateSpec(method="jrr", scale=0.5, n_replicates=n_replicates)


def _estimate_log_vr_for_weights(group: pd.DataFrame, weights: np.ndarray) -> float:
    sex = group["sex_observed"].astype("string").str.lower().to_numpy()
    scores = pd.to_numeric(group["score_raw"], errors="coerce").to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    male = valid & (sex == "male")
    female = valid & (sex == "female")
    if male.sum() < 2 or female.sum() < 2:
        raise ValueError("Need at least two positive-weight observations per sex for log variance ratio estimation.")
    vr, _ = log_variance_ratio_from_groups(
        scores[male],
        scores[female],
        numerator_weights=weights[male],
        denominator_weights=weights[female],
    )
    return float(vr.log_variance_ratio)


def _estimate_mean_diff_for_weights(group: pd.DataFrame, weights: np.ndarray) -> float:
    sex = group["sex_observed"].astype("string").str.lower().to_numpy()
    scores = pd.to_numeric(group["score_raw"], errors="coerce").to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    male = valid & (sex == "male")
    female = valid & (sex == "female")
    if male.sum() < 2 or female.sum() < 2:
        raise ValueError("Need at least two positive-weight observations per sex for mean-difference estimation.")
    _, md = log_variance_ratio_from_groups(
        scores[male],
        scores[female],
        numerator_weights=weights[male],
        denominator_weights=weights[female],
    )
    return float(md.mean_difference)


def _estimate_tail_ratio_for_weights(group: pd.DataFrame, weights: np.ndarray, *, quantile: float, tail: str) -> float:
    sex = group["sex_observed"].astype("string").str.lower().to_numpy()
    scores = pd.to_numeric(group["score_raw"], errors="coerce").to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    male = valid & (sex == "male")
    female = valid & (sex == "female")
    if male.sum() < 2 or female.sum() < 2:
        raise ValueError("Need at least two positive-weight observations per sex for tail-ratio estimation.")
    res = tail_rate_ratio_from_groups(
        scores[male],
        scores[female],
        numerator_weights=weights[male],
        denominator_weights=weights[female],
        quantile=quantile,
        tail=tail,
    )
    return float(res.rate_ratio)


def estimate_timss_cells(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["dataset_id", "cycle_or_wave", "country", "country_id", "grade_or_age_band", "trait_id"]
    rows: list[dict[str, Any]] = []

    for keys, cell in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        spec = infer_timss_replicate_spec(cell)
        male = cell[cell["sex_observed"] == "male"]
        female = cell[cell["sex_observed"] == "female"]
        row: dict[str, Any] = {
            **base,
            "male_n": int(male["person_id"].nunique()),
            "female_n": int(female["person_id"].nunique()),
            "inference_method": "unavailable",
            "replicate_method": spec.method,
            "fay_factor": 0.0,
            "n_replicates": spec.n_replicates,
        }
        if row["male_n"] == 0 or row["female_n"] == 0:
            row.update(
                {
                    "log_variance_ratio": np.nan,
                    "variance_ratio": np.nan,
                    "se_log_variance_ratio": np.nan,
                    "ci_low_log_variance_ratio": np.nan,
                    "ci_high_log_variance_ratio": np.nan,
                    "mean_difference": np.nan,
                    "se_mean_difference": np.nan,
                    "top90_rate_ratio": np.nan,
                    "top95_rate_ratio": np.nan,
                    "bottom10_rate_ratio": np.nan,
                    "bottom05_rate_ratio": np.nan,
                    "qa_flags": "missing_sex_group",
                }
            )
            rows.append(row)
            continue
        if row["male_n"] < 2 or row["female_n"] < 2:
            row.update(
                {
                    "log_variance_ratio": np.nan,
                    "variance_ratio": np.nan,
                    "se_log_variance_ratio": np.nan,
                    "ci_low_log_variance_ratio": np.nan,
                    "ci_high_log_variance_ratio": np.nan,
                    "mean_difference": np.nan,
                    "se_mean_difference": np.nan,
                    "top90_rate_ratio": np.nan,
                    "top95_rate_ratio": np.nan,
                    "bottom10_rate_ratio": np.nan,
                    "bottom05_rate_ratio": np.nan,
                    "qa_flags": "low_n_variance",
                }
            )
            rows.append(row)
            continue

        pv_points_logvr: list[float] = []
        pv_reps_logvr: list[np.ndarray] = []
        pv_points_md: list[float] = []
        pv_reps_md: list[np.ndarray] = []
        tail90: list[float] = []
        tail95: list[float] = []
        bottom10: list[float] = []
        bottom05: list[float] = []

        for _, pv_group in cell.groupby("pv_index", sort=True):
            weights = pd.to_numeric(pv_group["weight_main"], errors="coerce").to_numpy(dtype=float)
            pv_points_logvr.append(_estimate_log_vr_for_weights(pv_group, weights))
            pv_reps_logvr.append(
                jackknife_zone_replicate_estimates(
                    pv_group,
                    estimator=_estimate_log_vr_for_weights,
                    weight_col="weight_main",
                    zone_col="jk_zone",
                    rep_col="jk_rep",
                )
            )
            pv_points_md.append(_estimate_mean_diff_for_weights(pv_group, weights))
            pv_reps_md.append(
                jackknife_zone_replicate_estimates(
                    pv_group,
                    estimator=_estimate_mean_diff_for_weights,
                    weight_col="weight_main",
                    zone_col="jk_zone",
                    rep_col="jk_rep",
                )
            )
            tail90.append(_estimate_tail_ratio_for_weights(pv_group, weights, quantile=0.90, tail="upper"))
            tail95.append(_estimate_tail_ratio_for_weights(pv_group, weights, quantile=0.95, tail="upper"))
            bottom10.append(_estimate_tail_ratio_for_weights(pv_group, weights, quantile=0.10, tail="lower"))
            bottom05.append(_estimate_tail_ratio_for_weights(pv_group, weights, quantile=0.05, tail="lower"))

        logvr_combo = combine_plausible_values_and_replicates(
            pv_points_logvr,
            pv_reps_logvr,
            method="jrr",
            scale=spec.scale,
        )
        md_combo = combine_plausible_values_and_replicates(
            pv_points_md,
            pv_reps_md,
            method="jrr",
            scale=spec.scale,
        )
        row.update(
            {
                "inference_method": "replicate_weights_jrr",
                "log_variance_ratio": logvr_combo.estimate,
                "variance_ratio": float(np.exp(logvr_combo.estimate)),
                "se_log_variance_ratio": logvr_combo.standard_error,
                "ci_low_log_variance_ratio": float(logvr_combo.estimate - 1.96 * logvr_combo.standard_error),
                "ci_high_log_variance_ratio": float(logvr_combo.estimate + 1.96 * logvr_combo.standard_error),
                "mean_difference": md_combo.estimate,
                "se_mean_difference": md_combo.standard_error,
                "top90_rate_ratio": float(np.nanmean(tail90)),
                "top95_rate_ratio": float(np.nanmean(tail95)),
                "bottom10_rate_ratio": float(np.nanmean(bottom10)),
                "bottom05_rate_ratio": float(np.nanmean(bottom05)),
                "qa_flags": np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by=["country", "grade_or_age_band", "trait_id"], kind="stable").reset_index(drop=True)
