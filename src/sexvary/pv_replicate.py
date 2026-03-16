from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import log_variance_ratio_from_groups, tail_rate_ratio_from_groups
from .survey import combine_plausible_values_and_replicates


@dataclass(frozen=True)
class ReplicateDesignSpec:
    method: str
    fay: float
    n_replicates: int


def infer_replicate_design(
    df: pd.DataFrame,
    replicate_cols: list[str],
    *,
    method_col: str | None = None,
    fay_col: str | None = None,
    nrep_col: str | None = None,
    default_method: str = "brr",
    default_fay: float = 0.0,
) -> ReplicateDesignSpec:
    method_raw = df[method_col].dropna().astype(str).str.upper().unique().tolist() if method_col and method_col in df.columns else []
    fay_raw = df[fay_col].dropna().astype(float).tolist() if fay_col and fay_col in df.columns else []
    n_reps_raw = df[nrep_col].dropna().astype(int).tolist() if nrep_col and nrep_col in df.columns else []

    method = default_method.lower()
    if method_raw:
        token = method_raw[0]
        try:
            numeric_token = str(int(float(token)))
        except ValueError:
            numeric_token = token
        if numeric_token in {"3", "4", "BRR", "FAY"} or token in {"BRR", "FAY"}:
            method = "brr"
        elif numeric_token in {"JRR"} or token in {"JRR"}:
            method = "jrr"
        elif numeric_token in {"1", "2", "JK1", "JK2", "JACKKNIFE"} or token in {"JK1", "JK2", "JACKKNIFE"}:
            method = "jackknife"
    fay = float(fay_raw[0]) if fay_raw else float(default_fay)
    n_replicates = int(n_reps_raw[0]) if n_reps_raw else len(replicate_cols)
    return ReplicateDesignSpec(method=method, fay=fay, n_replicates=n_replicates)


def _estimate_log_vr_for_weight(group: pd.DataFrame, weight_col: str) -> float:
    male = group[group["sex_observed"] == "male"]
    female = group[group["sex_observed"] == "female"]
    vr, _ = log_variance_ratio_from_groups(
        male["score_raw"],
        female["score_raw"],
        numerator_weights=male[weight_col],
        denominator_weights=female[weight_col],
    )
    return vr.log_variance_ratio


def _estimate_full_stats_for_weight(group: pd.DataFrame, weight_col: str) -> dict:
    """Return log_vr, mean_diff, and sex-specific means/variances for one weight column."""
    male = group[group["sex_observed"] == "male"]
    female = group[group["sex_observed"] == "female"]
    vr, md = log_variance_ratio_from_groups(
        male["score_raw"],
        female["score_raw"],
        numerator_weights=male[weight_col],
        denominator_weights=female[weight_col],
    )
    return {
        "log_vr": vr.log_variance_ratio,
        "mean_diff": md.mean_difference,
        "male_mean": vr.mean_numerator,
        "female_mean": vr.mean_denominator,
        "male_var": vr.var_numerator,
        "female_var": vr.var_denominator,
    }


def _estimate_mean_diff_for_weight(group: pd.DataFrame, weight_col: str) -> float:
    male = group[group["sex_observed"] == "male"]
    female = group[group["sex_observed"] == "female"]
    _, md = log_variance_ratio_from_groups(
        male["score_raw"],
        female["score_raw"],
        numerator_weights=male[weight_col],
        denominator_weights=female[weight_col],
    )
    return md.mean_difference


def _estimate_tail_ratio_for_weight(group: pd.DataFrame, weight_col: str, *, quantile: float, tail: str) -> float:
    male = group[group["sex_observed"] == "male"]
    female = group[group["sex_observed"] == "female"]
    res = tail_rate_ratio_from_groups(
        male["score_raw"],
        female["score_raw"],
        numerator_weights=male[weight_col],
        denominator_weights=female[weight_col],
        quantile=quantile,
        tail=tail,
    )
    return res.rate_ratio


def estimate_pv_replicate_cells(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    replicate_cols: list[str],
    method_col: str | None = None,
    fay_col: str | None = None,
    nrep_col: str | None = None,
    default_method: str = "brr",
    default_fay: float = 0.0,
    default_scale: float | None = None,
) -> pd.DataFrame:
    if not replicate_cols:
        raise ValueError("PV+replicate estimation requires replicate weight columns.")
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for keys, cell in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        spec = infer_replicate_design(
            cell,
            replicate_cols,
            method_col=method_col,
            fay_col=fay_col,
            nrep_col=nrep_col,
            default_method=default_method,
            default_fay=default_fay,
        )
        male = cell[cell["sex_observed"] == "male"]
        female = cell[cell["sex_observed"] == "female"]
        row: dict[str, Any] = {
            **base,
            "male_n": int(male["person_id"].nunique()),
            "female_n": int(female["person_id"].nunique()),
            "inference_method": "unavailable",
            "replicate_method": spec.method,
            "fay_factor": spec.fay,
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
                    "male_weighted_mean": np.nan,
                    "female_weighted_mean": np.nan,
                    "male_weighted_variance": np.nan,
                    "female_weighted_variance": np.nan,
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
                    "male_weighted_mean": np.nan,
                    "female_weighted_mean": np.nan,
                    "male_weighted_variance": np.nan,
                    "female_weighted_variance": np.nan,
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
        pv_reps_logvr: list[list[float]] = []
        pv_points_md: list[float] = []
        pv_reps_md: list[list[float]] = []
        pv_male_means: list[float] = []
        pv_female_means: list[float] = []
        pv_male_vars: list[float] = []
        pv_female_vars: list[float] = []
        tail90: list[float] = []
        tail95: list[float] = []
        bottom10: list[float] = []
        bottom05: list[float] = []

        for _, pv_group in cell.groupby("pv_index", sort=True):
            main_stats = _estimate_full_stats_for_weight(pv_group, "weight_main")
            pv_points_logvr.append(main_stats["log_vr"])
            pv_points_md.append(main_stats["mean_diff"])
            pv_male_means.append(main_stats["male_mean"])
            pv_female_means.append(main_stats["female_mean"])
            pv_male_vars.append(main_stats["male_var"])
            pv_female_vars.append(main_stats["female_var"])
            pv_reps_logvr.append([_estimate_log_vr_for_weight(pv_group, rep_col) for rep_col in replicate_cols])
            pv_reps_md.append([_estimate_mean_diff_for_weight(pv_group, rep_col) for rep_col in replicate_cols])
            tail90.append(_estimate_tail_ratio_for_weight(pv_group, "weight_main", quantile=0.90, tail="upper"))
            tail95.append(_estimate_tail_ratio_for_weight(pv_group, "weight_main", quantile=0.95, tail="upper"))
            bottom10.append(_estimate_tail_ratio_for_weight(pv_group, "weight_main", quantile=0.10, tail="lower"))
            bottom05.append(_estimate_tail_ratio_for_weight(pv_group, "weight_main", quantile=0.05, tail="lower"))

        logvr_combo = combine_plausible_values_and_replicates(
            pv_points_logvr,
            pv_reps_logvr,
            method=spec.method,
            fay=spec.fay,
            scale=default_scale,
        )
        md_combo = combine_plausible_values_and_replicates(
            pv_points_md,
            pv_reps_md,
            method=spec.method,
            fay=spec.fay,
            scale=default_scale,
        )
        row.update(
            {
                "inference_method": f"replicate_weights_{spec.method}",
                "log_variance_ratio": logvr_combo.estimate,
                "variance_ratio": float(np.exp(logvr_combo.estimate)),
                "se_log_variance_ratio": logvr_combo.standard_error,
                "ci_low_log_variance_ratio": float(logvr_combo.estimate - 1.96 * logvr_combo.standard_error),
                "ci_high_log_variance_ratio": float(logvr_combo.estimate + 1.96 * logvr_combo.standard_error),
                "mean_difference": md_combo.estimate,
                "se_mean_difference": md_combo.standard_error,
                "male_weighted_mean": float(np.nanmean(pv_male_means)),
                "female_weighted_mean": float(np.nanmean(pv_female_means)),
                "male_weighted_variance": float(np.nanmean(pv_male_vars)),
                "female_weighted_variance": float(np.nanmean(pv_female_vars)),
                "top90_rate_ratio": float(np.nanmean(tail90)),
                "top95_rate_ratio": float(np.nanmean(tail95)),
                "bottom10_rate_ratio": float(np.nanmean(bottom10)),
                "bottom05_rate_ratio": float(np.nanmean(bottom05)),
                "qa_flags": np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
