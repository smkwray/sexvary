from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import TraitSpec, build_registry
from .metrics import (
    effective_sample_size,
    log_variance_ratio_from_groups,
    tail_rate_ratio_from_groups,
)
from .survey import replicate_variance, stratified_cluster_bootstrap_variance


@dataclass(frozen=True)
class EstimationConfig:
    min_n_per_sex_for_variance: int = 200
    min_n_per_sex_for_95_tail: int = 500
    min_n_per_sex_for_99_tail: int = 2000
    min_unique_values: int = 10
    default_age_band_width_years: int = 4
    dataset_age_band_width_years: dict[str, int] | None = None
    dataset_min_n_per_sex_for_variance: dict[str, int] | None = None
    dataset_min_n_per_sex_for_95_tail: dict[str, int] | None = None
    fallback_to_all_ages_if_no_variance_cells: tuple[str, ...] = ()
    fallback_to_all_ages_if_no_variance_cells_per_trait: tuple[str, ...] = ()
    pooled_thresholds_within_cell: bool = True
    design_bootstrap_replicates: int = 40
    design_bootstrap_seed: int = 20260314
    suppress_tail_metrics_for_scale_types: tuple[str, ...] = ("bounded_count", "ordinal_or_bounded_composite")
    fragile_variance_scale_types: tuple[str, ...] = ("bounded_count", "ordinal_or_bounded_composite")
    boundary_mass_fragile_share: float = 0.10
    boundary_mass_suppress_variance_share: float = 0.25
    bounded_scale_min_unique_values_for_variance: int = 8


def estimation_config_from_analysis(analysis_config: dict[str, Any]) -> EstimationConfig:
    defaults = analysis_config.get("analysis_defaults", {})
    return EstimationConfig(
        min_n_per_sex_for_variance=int(defaults.get("min_n_per_sex_for_variance", 200)),
        min_n_per_sex_for_95_tail=int(defaults.get("min_n_per_sex_for_95_tail", 500)),
        min_n_per_sex_for_99_tail=int(defaults.get("min_n_per_sex_for_99_tail", 2000)),
        min_unique_values=int(defaults.get("min_unique_values", 10)),
        default_age_band_width_years=int(defaults.get("default_age_band_width_years", defaults.get("age_band_width_years", 4))),
        dataset_age_band_width_years={
            str(key): int(value) for key, value in (defaults.get("dataset_age_band_width_years", {}) or {}).items()
        },
        dataset_min_n_per_sex_for_variance={
            str(key): int(value) for key, value in (defaults.get("dataset_min_n_per_sex_for_variance", {}) or {}).items()
        },
        dataset_min_n_per_sex_for_95_tail={
            str(key): int(value) for key, value in (defaults.get("dataset_min_n_per_sex_for_95_tail", {}) or {}).items()
        },
        fallback_to_all_ages_if_no_variance_cells=tuple(defaults.get("fallback_to_all_ages_if_no_variance_cells", []) or ()),
        fallback_to_all_ages_if_no_variance_cells_per_trait=tuple(
            defaults.get("fallback_to_all_ages_if_no_variance_cells_per_trait", []) or ()
        ),
        design_bootstrap_replicates=int(defaults.get("design_bootstrap_replicates", 40)),
        design_bootstrap_seed=int(defaults.get("design_bootstrap_seed", 20260314)),
        suppress_tail_metrics_for_scale_types=tuple(
            defaults.get(
                "suppress_tail_metrics_for_scale_types",
                ["bounded_count", "ordinal_or_bounded_composite"],
            )
            or ()
        ),
        fragile_variance_scale_types=tuple(
            defaults.get(
                "fragile_variance_scale_types",
                ["bounded_count", "ordinal_or_bounded_composite"],
            )
            or ()
        ),
        boundary_mass_fragile_share=float(defaults.get("boundary_mass_fragile_share", 0.10)),
        boundary_mass_suppress_variance_share=float(defaults.get("boundary_mass_suppress_variance_share", 0.25)),
        bounded_scale_min_unique_values_for_variance=int(
            defaults.get("bounded_scale_min_unique_values_for_variance", 8)
        ),
    )


def derive_age_band(series: pd.Series, *, width_years: int = 2) -> pd.Series:
    ages = pd.to_numeric(series, errors="coerce")
    if width_years <= 0:
        raise ValueError("width_years must be positive.")
    starts = np.floor(ages / width_years) * width_years
    ends = starts + width_years - 1
    labels = pd.Series(pd.NA, index=series.index, dtype="object")
    mask = ages.notna()
    labels.loc[mask] = starts.loc[mask].astype(int).astype(str) + "-" + ends.loc[mask].astype(int).astype(str)
    return labels


def _age_band_width_for_dataset(dataset_id: str, config: EstimationConfig) -> int:
    if config.dataset_age_band_width_years and dataset_id in config.dataset_age_band_width_years:
        return config.dataset_age_band_width_years[dataset_id]
    return config.default_age_band_width_years


def _min_n_for_variance(dataset_id: str, config: EstimationConfig) -> int:
    if config.dataset_min_n_per_sex_for_variance and dataset_id in config.dataset_min_n_per_sex_for_variance:
        return config.dataset_min_n_per_sex_for_variance[dataset_id]
    return config.min_n_per_sex_for_variance


def _min_n_for_95_tail(dataset_id: str, config: EstimationConfig) -> int:
    if config.dataset_min_n_per_sex_for_95_tail and dataset_id in config.dataset_min_n_per_sex_for_95_tail:
        return config.dataset_min_n_per_sex_for_95_tail[dataset_id]
    return config.min_n_per_sex_for_95_tail


def prepare_analysis_frame(df: pd.DataFrame, *, config: EstimationConfig) -> pd.DataFrame:
    work = df.copy()
    work["sex_observed"] = work["sex_observed"].astype("string").str.lower()
    work = work[work["sex_observed"].isin(["male", "female"])].copy()
    work["score_raw"] = pd.to_numeric(work["score_raw"], errors="coerce")
    work["weight_main"] = pd.to_numeric(work["weight_main"], errors="coerce")
    work = work[np.isfinite(work["score_raw"]) & np.isfinite(work["weight_main"]) & (work["weight_main"] > 0)].copy()

    for col in ["cycle_or_wave", "country", "grade_or_age_band"]:
        if col not in work.columns:
            work[col] = pd.NA

    derived_age_band = pd.Series(pd.NA, index=work.index, dtype="object")
    for dataset_id, idx in work.groupby("dataset_id").groups.items():
        width = _age_band_width_for_dataset(str(dataset_id), config)
        derived_age_band.loc[idx] = derive_age_band(work.loc[idx, "age"], width_years=width)
    work["analysis_cell"] = work["grade_or_age_band"]
    missing_band = work["analysis_cell"].isna() | (work["analysis_cell"].astype("string").str.strip() == "")
    work.loc[missing_band, "analysis_cell"] = derived_age_band.loc[missing_band]
    work["analysis_cell"] = work["analysis_cell"].fillna("all")

    work["cycle_or_wave"] = work["cycle_or_wave"].fillna("all")
    work["country"] = work["country"].fillna("all")
    work["used_age_fallback"] = False

    for dataset_id in config.fallback_to_all_ages_if_no_variance_cells:
        mask = work["dataset_id"] == dataset_id
        if not mask.any():
            continue
        min_n_for_variance = _min_n_for_variance(str(dataset_id), config)
        grouped = work.loc[mask].groupby(["analysis_cell", "trait_id", "sex_observed"]).size().unstack(fill_value=0)
        if grouped.empty:
            continue
        male = grouped.get("male", pd.Series(0, index=grouped.index))
        female = grouped.get("female", pd.Series(0, index=grouped.index))
        has_variance_cells = bool(
            ((male >= min_n_for_variance) & (female >= min_n_for_variance)).any()
        )
        if has_variance_cells:
            continue
        work.loc[mask, "analysis_cell"] = "all_ages"
        work.loc[mask, "used_age_fallback"] = True

    for dataset_id in config.fallback_to_all_ages_if_no_variance_cells_per_trait:
        mask = work["dataset_id"] == dataset_id
        if not mask.any():
            continue
        min_n_for_variance = _min_n_for_variance(str(dataset_id), config)
        dataset_work = work.loc[mask]
        for trait_id, trait_idx in dataset_work.groupby("trait_id").groups.items():
            trait_slice = work.loc[trait_idx]
            grouped = trait_slice.groupby(["analysis_cell", "sex_observed"]).size().unstack(fill_value=0)
            if grouped.empty:
                continue
            male = grouped.get("male", pd.Series(0, index=grouped.index))
            female = grouped.get("female", pd.Series(0, index=grouped.index))
            has_variance_cells = bool(((male >= min_n_for_variance) & (female >= min_n_for_variance)).any())
            if has_variance_cells:
                continue
            work.loc[trait_idx, "analysis_cell"] = "all_ages"
            work.loc[trait_idx, "used_age_fallback"] = True
    return work


def _compose_qa_flags(
    *,
    male_n: int,
    female_n: int,
    pooled_unique_values: int,
    pooled_min_share: float,
    pooled_max_share: float,
    min_n_per_sex_for_variance: int,
    min_n_per_sex_for_95_tail: int,
    config: EstimationConfig,
    extra_flags: list[str] | None = None,
) -> str | float:
    flags: list[str] = []
    if male_n < min_n_per_sex_for_variance or female_n < min_n_per_sex_for_variance:
        flags.append("low_n_variance")
    if male_n < min_n_per_sex_for_95_tail or female_n < min_n_per_sex_for_95_tail:
        flags.append("low_n_tail_95")
    if pooled_unique_values < config.min_unique_values:
        flags.append("low_unique_values")
    if pooled_min_share >= 0.1:
        flags.append("mass_at_min")
    if pooled_max_share >= 0.1:
        flags.append("mass_at_max")
    if extra_flags:
        flags.extend(extra_flags)
    if not flags:
        return np.nan
    return ";".join(flags)


def _root_trait_id(trait_id: str) -> str:
    return str(trait_id).split(":", 1)[0]


@lru_cache(maxsize=1)
def _trait_spec_map() -> dict[str, TraitSpec]:
    registry = build_registry()
    return registry.traits


def _lookup_trait_spec(trait_id: str) -> TraitSpec | None:
    return _trait_spec_map().get(_root_trait_id(trait_id))


def _simple_design_log_variance_ratio_inference(
    log_variance_ratio: float,
    *,
    male_effective_n: float,
    female_effective_n: float,
    alpha: float = 0.05,
) -> dict[str, float | str]:
    if not np.isfinite(log_variance_ratio):
        return {
            "inference_method": "unavailable",
            "se_log_variance_ratio": np.nan,
            "ci_low_log_variance_ratio": np.nan,
            "ci_high_log_variance_ratio": np.nan,
            "ci_low_variance_ratio": np.nan,
            "ci_high_variance_ratio": np.nan,
        }
    if male_effective_n <= 1 or female_effective_n <= 1:
        return {
            "inference_method": "unavailable",
            "se_log_variance_ratio": np.nan,
            "ci_low_log_variance_ratio": np.nan,
            "ci_high_log_variance_ratio": np.nan,
            "ci_low_variance_ratio": np.nan,
            "ci_high_variance_ratio": np.nan,
        }

    sampling_variance = (2.0 / (male_effective_n - 1.0)) + (2.0 / (female_effective_n - 1.0))
    se = float(np.sqrt(max(sampling_variance, 0.0)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(log_variance_ratio - z * se)
    ci_high = float(log_variance_ratio + z * se)
    return {
        "inference_method": "analytic_effective_n_simple_design",
        "se_log_variance_ratio": se,
        "ci_low_log_variance_ratio": ci_low,
        "ci_high_log_variance_ratio": ci_high,
        "ci_low_variance_ratio": float(np.exp(ci_low)),
        "ci_high_variance_ratio": float(np.exp(ci_high)),
    }


def _cell_seed(cell_df: pd.DataFrame, *, base_seed: int) -> int:
    key = "|".join(
        [
            str(cell_df["dataset_id"].iloc[0]),
            str(cell_df["cycle_or_wave"].iloc[0]),
            str(cell_df["country"].iloc[0]),
            str(cell_df["analysis_cell"].iloc[0]),
            str(cell_df["trait_id"].iloc[0]),
            str(base_seed),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _weighted_log_variance_ratio_estimator(cell_df: pd.DataFrame, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    sex = cell_df["sex_observed"].astype("string").str.lower().to_numpy()
    scores = pd.to_numeric(cell_df["score_raw"], errors="coerce").to_numpy(dtype=float)
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
        numerator_label="male",
        denominator_label="female",
    )
    return float(vr.log_variance_ratio)


def _replicate_weight_columns(cell_df: pd.DataFrame) -> list[str]:
    return [col for col in cell_df.columns if str(col).startswith("replicate_weight_")]


def _replicate_based_log_variance_ratio_inference(
    cell_df: pd.DataFrame,
    log_variance_ratio: float,
    *,
    alpha: float = 0.05,
) -> dict[str, float | str]:
    unavailable = {
        "inference_method": "unavailable",
        "se_log_variance_ratio": np.nan,
        "ci_low_log_variance_ratio": np.nan,
        "ci_high_log_variance_ratio": np.nan,
        "ci_low_variance_ratio": np.nan,
        "ci_high_variance_ratio": np.nan,
    }
    if not np.isfinite(log_variance_ratio):
        return unavailable

    replicate_cols = _replicate_weight_columns(cell_df)
    if len(replicate_cols) < 2:
        return unavailable

    replicate_estimates: list[float] = []
    for col in replicate_cols:
        weights = pd.to_numeric(cell_df[col], errors="coerce").to_numpy(dtype=float)
        try:
            estimate = _weighted_log_variance_ratio_estimator(cell_df, weights)
        except ValueError:
            continue
        if np.isfinite(estimate):
            replicate_estimates.append(float(estimate))

    if len(replicate_estimates) < 2:
        return unavailable

    method = str(cell_df.get("replicate_method", pd.Series(["brr"])).iloc[0]).strip().lower() or "brr"
    if method in {"nan", "<na>", "none"}:
        method = "brr"
    fay = float(pd.to_numeric(cell_df.get("replicate_fay", pd.Series([0.0])).iloc[0], errors="coerce"))
    if not np.isfinite(fay):
        fay = 0.0

    try:
        result = replicate_variance(log_variance_ratio, replicate_estimates, method=method, fay=fay)
    except ValueError:
        return unavailable

    if not np.isfinite(result.standard_error) or result.standard_error <= 0:
        return unavailable

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(log_variance_ratio - z * result.standard_error)
    ci_high = float(log_variance_ratio + z * result.standard_error)
    return {
        "inference_method": f"replicate_weights_{result.method}",
        "se_log_variance_ratio": result.standard_error,
        "ci_low_log_variance_ratio": ci_low,
        "ci_high_log_variance_ratio": ci_high,
        "ci_low_variance_ratio": float(np.exp(ci_low)),
        "ci_high_variance_ratio": float(np.exp(ci_high)),
    }


def _design_based_log_variance_ratio_inference(
    cell_df: pd.DataFrame,
    log_variance_ratio: float,
    *,
    config: EstimationConfig,
    alpha: float = 0.05,
) -> dict[str, float | str]:
    unavailable = {
        "inference_method": "unavailable",
        "se_log_variance_ratio": np.nan,
        "ci_low_log_variance_ratio": np.nan,
        "ci_high_log_variance_ratio": np.nan,
        "ci_low_variance_ratio": np.nan,
        "ci_high_variance_ratio": np.nan,
    }
    if not np.isfinite(log_variance_ratio):
        return unavailable
    required = {"design_strata", "design_psu"}
    if not required.issubset(cell_df.columns):
        return unavailable
    design_ready = cell_df[list(required)].notna().all(axis=1)
    if design_ready.sum() < 4:
        return unavailable

    try:
        result = stratified_cluster_bootstrap_variance(
            cell_df.loc[design_ready].copy(),
            estimator=_weighted_log_variance_ratio_estimator,
            weight_col="weight_main",
            strata_col="design_strata",
            cluster_col="design_psu",
            n_boot=config.design_bootstrap_replicates,
            random_state=_cell_seed(cell_df, base_seed=config.design_bootstrap_seed),
        )
    except ValueError:
        return unavailable

    if not np.isfinite(result.standard_error) or result.standard_error <= 0:
        return unavailable

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(log_variance_ratio - z * result.standard_error)
    ci_high = float(log_variance_ratio + z * result.standard_error)
    method_label = (
        str(cell_df.get("design_inference_label", pd.Series([pd.NA])).iloc[0]).strip().lower()
        if "design_inference_label" in cell_df.columns
        else ""
    )
    if method_label in {"", "nan", "<na>", "none"}:
        method_label = f"{result.method}_psu"
    return {
        "inference_method": method_label,
        "se_log_variance_ratio": result.standard_error,
        "ci_low_log_variance_ratio": ci_low,
        "ci_high_log_variance_ratio": ci_high,
        "ci_low_variance_ratio": float(np.exp(ci_low)),
        "ci_high_variance_ratio": float(np.exp(ci_high)),
    }


def estimate_sex_difference_cell(cell_df: pd.DataFrame, *, config: EstimationConfig) -> dict[str, Any]:
    male = cell_df[cell_df["sex_observed"] == "male"].copy()
    female = cell_df[cell_df["sex_observed"] == "female"].copy()
    dataset_id = str(cell_df["dataset_id"].iloc[0])
    trait_id = str(cell_df["trait_id"].iloc[0])
    trait_spec = _lookup_trait_spec(trait_id)
    trait_scale_type = trait_spec.scale_type if trait_spec else "unknown"
    recommended_metrics = set(trait_spec.recommended_metrics) if trait_spec else {
        "log_variance_ratio",
        "variance_ratio",
        "tail_rate_ratios",
    }
    min_n_per_sex_for_variance = _min_n_for_variance(dataset_id, config)
    min_n_per_sex_for_95_tail = _min_n_for_95_tail(dataset_id, config)

    male_n = int(len(male))
    female_n = int(len(female))
    pooled_scores = cell_df["score_raw"].to_numpy(dtype=float)
    pooled_unique_values = int(pd.Series(pooled_scores).nunique(dropna=True))
    pooled_min = float(np.nanmin(pooled_scores))
    pooled_max = float(np.nanmax(pooled_scores))
    pooled_min_share = float(np.mean(pooled_scores == pooled_min))
    pooled_max_share = float(np.mean(pooled_scores == pooled_max))

    weight_sources = sorted({str(value) for value in cell_df.get("weight_source", pd.Series(dtype="object")).dropna().tolist()})
    primary_weight_sources = sorted(
        {str(value) for value in cell_df.get("weight_primary_source", pd.Series(dtype="object")).dropna().tolist()}
    )
    used_nonprimary_weight_fallback = bool(weight_sources and set(weight_sources) - set(primary_weight_sources))
    suppress_tail_metrics = (
        "tail_rate_ratios" not in recommended_metrics
        or trait_scale_type in set(config.suppress_tail_metrics_for_scale_types)
    )
    bounded_variance_fragile = (
        trait_scale_type in set(config.fragile_variance_scale_types)
        and (
            pooled_min_share >= config.boundary_mass_fragile_share
            or pooled_max_share >= config.boundary_mass_fragile_share
            or pooled_unique_values < config.bounded_scale_min_unique_values_for_variance
        )
    )
    suppress_variance_metrics = (
        trait_scale_type in set(config.fragile_variance_scale_types)
        and (
            pooled_min_share >= config.boundary_mass_suppress_variance_share
            or pooled_max_share >= config.boundary_mass_suppress_variance_share
        )
        and pooled_unique_values < max(config.min_unique_values, config.bounded_scale_min_unique_values_for_variance)
    )

    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "cycle_or_wave": cell_df["cycle_or_wave"].iloc[0],
        "country": cell_df["country"].iloc[0],
        "age_band": cell_df["analysis_cell"].iloc[0],
        "trait_id": trait_id,
        "trait_scale_type": trait_scale_type,
        "used_age_fallback": bool(cell_df["used_age_fallback"].iloc[0]) if "used_age_fallback" in cell_df.columns else False,
        "male_n": male_n,
        "female_n": female_n,
        "male_weight_sum": float(male["weight_main"].sum()),
        "female_weight_sum": float(female["weight_main"].sum()),
        "male_effective_n": float(effective_sample_size(male["weight_main"])),
        "female_effective_n": float(effective_sample_size(female["weight_main"])),
        "weight_sources": ";".join(weight_sources) if weight_sources else np.nan,
        "weight_primary_sources": ";".join(primary_weight_sources) if primary_weight_sources else np.nan,
        "used_nonprimary_weight_fallback": used_nonprimary_weight_fallback,
        "pooled_unique_values": pooled_unique_values,
        "pooled_min_score": pooled_min,
        "pooled_max_score": pooled_max,
        "pooled_share_at_min": pooled_min_share,
        "pooled_share_at_max": pooled_max_share,
        "inference_method": "unavailable",
        "se_log_variance_ratio": np.nan,
        "ci_low_log_variance_ratio": np.nan,
        "ci_high_log_variance_ratio": np.nan,
        "ci_low_variance_ratio": np.nan,
        "ci_high_variance_ratio": np.nan,
        "tail_metrics_suppressed": suppress_tail_metrics,
    }

    if suppress_variance_metrics or male_n < min_n_per_sex_for_variance or female_n < min_n_per_sex_for_variance:
        row.update(
            {
                "male_weighted_mean": np.nan,
                "female_weighted_mean": np.nan,
                "mean_diff": np.nan,
                "smd": np.nan,
                "male_weighted_variance": np.nan,
                "female_weighted_variance": np.nan,
                "variance_ratio": np.nan,
                "log_variance_ratio": np.nan,
                "top90_rate_ratio": np.nan,
                "top95_rate_ratio": np.nan,
                "bottom10_rate_ratio": np.nan,
                "bottom05_rate_ratio": np.nan,
                "top90_representation_ratio": np.nan,
                "top95_representation_ratio": np.nan,
                "bottom10_representation_ratio": np.nan,
                "bottom05_representation_ratio": np.nan,
            }
        )
    else:
        vr, md = log_variance_ratio_from_groups(
            male["score_raw"],
            female["score_raw"],
            numerator_weights=male["weight_main"],
            denominator_weights=female["weight_main"],
            numerator_label="male",
            denominator_label="female",
        )
        row.update(
            {
                "male_weighted_mean": vr.mean_numerator,
                "female_weighted_mean": vr.mean_denominator,
                "mean_diff": md.mean_difference,
                "smd": md.standardized_mean_difference,
                "male_weighted_variance": vr.var_numerator,
                "female_weighted_variance": vr.var_denominator,
                "variance_ratio": vr.variance_ratio,
                "log_variance_ratio": vr.log_variance_ratio,
            }
        )
        row.update(
            _replicate_based_log_variance_ratio_inference(cell_df, vr.log_variance_ratio)
        )
        if row["inference_method"] == "unavailable":
            row.update(
                _design_based_log_variance_ratio_inference(cell_df, vr.log_variance_ratio, config=config)
            )
        if row["inference_method"] == "unavailable":
            row.update(
                _simple_design_log_variance_ratio_inference(
                    vr.log_variance_ratio,
                    male_effective_n=row["male_effective_n"],
                    female_effective_n=row["female_effective_n"],
                )
            )

        if suppress_tail_metrics:
            row.update(
                {
                    "top90_rate_ratio": np.nan,
                    "bottom10_rate_ratio": np.nan,
                    "top90_representation_ratio": np.nan,
                    "bottom10_representation_ratio": np.nan,
                    "top95_rate_ratio": np.nan,
                    "bottom05_rate_ratio": np.nan,
                    "top95_representation_ratio": np.nan,
                    "bottom05_representation_ratio": np.nan,
                }
            )
        else:
            top90 = tail_rate_ratio_from_groups(
                male["score_raw"],
                female["score_raw"],
                numerator_weights=male["weight_main"],
                denominator_weights=female["weight_main"],
                quantile=0.90,
                tail="upper",
            )
            bottom10 = tail_rate_ratio_from_groups(
                male["score_raw"],
                female["score_raw"],
                numerator_weights=male["weight_main"],
                denominator_weights=female["weight_main"],
                quantile=0.10,
                tail="lower",
            )
            row.update(
                {
                    "top90_rate_ratio": top90.rate_ratio,
                    "bottom10_rate_ratio": bottom10.rate_ratio,
                    "top90_representation_ratio": top90.representation_ratio,
                    "bottom10_representation_ratio": bottom10.representation_ratio,
                }
            )

            if male_n >= min_n_per_sex_for_95_tail and female_n >= min_n_per_sex_for_95_tail:
                top95 = tail_rate_ratio_from_groups(
                    male["score_raw"],
                    female["score_raw"],
                    numerator_weights=male["weight_main"],
                    denominator_weights=female["weight_main"],
                    quantile=0.95,
                    tail="upper",
                )
                bottom05 = tail_rate_ratio_from_groups(
                    male["score_raw"],
                    female["score_raw"],
                    numerator_weights=male["weight_main"],
                    denominator_weights=female["weight_main"],
                    quantile=0.05,
                    tail="lower",
                )
                row.update(
                    {
                        "top95_rate_ratio": top95.rate_ratio,
                        "bottom05_rate_ratio": bottom05.rate_ratio,
                        "top95_representation_ratio": top95.representation_ratio,
                        "bottom05_representation_ratio": bottom05.representation_ratio,
                    }
                )
            else:
                row.update(
                    {
                        "top95_rate_ratio": np.nan,
                        "bottom05_rate_ratio": np.nan,
                        "top95_representation_ratio": np.nan,
                        "bottom05_representation_ratio": np.nan,
                    }
                )

    extra_flags: list[str] = []
    if row["used_age_fallback"]:
        extra_flags.append("pooled_age_fallback")
    if used_nonprimary_weight_fallback:
        extra_flags.append("nonprimary_weight_fallback")
    if len(weight_sources) > 1:
        extra_flags.append("mixed_weight_sources")
    if suppress_tail_metrics:
        extra_flags.append("tail_metrics_suppressed")
    if (
        "design_inference_label" in cell_df.columns
        and str(cell_df["design_inference_label"].iloc[0]).strip().lower() == "approximate_household_cluster_bootstrap"
    ):
        extra_flags.append("approximate_design_bootstrap")
    if bounded_variance_fragile:
        extra_flags.append("bounded_scale_variance_fragile")
    if suppress_variance_metrics:
        extra_flags.append("bounded_scale_variance_suppressed")

    row["qa_flags"] = _compose_qa_flags(
        male_n=male_n,
        female_n=female_n,
        pooled_unique_values=pooled_unique_values,
        pooled_min_share=pooled_min_share,
        pooled_max_share=pooled_max_share,
        min_n_per_sex_for_variance=min_n_per_sex_for_variance,
        min_n_per_sex_for_95_tail=min_n_per_sex_for_95_tail,
        config=config,
        extra_flags=extra_flags or None,
    )
    return row


def estimate_dataset_cells(df: pd.DataFrame, *, config: EstimationConfig) -> pd.DataFrame:
    work = prepare_analysis_frame(df, config=config)
    group_cols = ["dataset_id", "cycle_or_wave", "country", "analysis_cell", "trait_id"]
    rows = [estimate_sex_difference_cell(group.copy(), config=config) for _, group in work.groupby(group_cols, dropna=False)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["dataset_id", "trait_id", "cycle_or_wave", "country", "age_band"],
        kind="stable",
    ).reset_index(drop=True)
