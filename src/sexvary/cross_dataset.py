from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from .config import Registry
from .evidence import annotate_estimate_evidence


DEFAULT_COMPARISON_SOURCES = {
    "local_nlsy": Path("results/local_nlsy/local_nlsy_trait_estimates.csv"),
    "piaac_cycle2": Path("results/piaac_cycle2/piaac_cycle2_trait_estimates.csv"),
    "pisa_2022": Path("results/pisa_2022/pisa_2022_trait_estimates.csv"),
    "timss_2019": Path("results/timss_2019/timss_2019_trait_estimates.csv"),
    "timss_2023": Path("results/timss_2023/timss_2023_trait_estimates.csv"),
    "pirls_2021": Path("results/pirls_2021/pirls_2021_trait_estimates.csv"),
    "icils_2023": Path("results/icils_2023/icils_2023_trait_estimates.csv"),
    "nhanes_2011_2023": Path("results/nhanes_2011_2023/nhanes_2011_2023_trait_estimates.csv"),
    "nnyfs_2012": Path("results/nnyfs_2012/nnyfs_2012_trait_estimates.csv"),
    "cpp_core": Path("results/cpp_core/cpp_core_trait_estimates.csv"),
    "cpp_growth": Path("results/cpp_growth/cpp_growth_trait_estimates.csv"),
    "psid_cds_tas": Path("results/psid_cds_tas/psid_cds_tas_trait_estimates.csv"),
    "hrs_public": Path("results/hrs_public/hrs_public_trait_estimates.csv"),
    "ecls_k_2011": Path("results/ecls_k_2011/ecls_k_2011_trait_estimates.csv"),
    "hsls_2009": Path("results/hsls_2009/hsls_2009_trait_estimates.csv"),
}

SUPPORTING_EVIDENCE_DATASET_IDS = (
    "nhanes_2011_2023",
    "hrs_public",
    "psid_cds_tas",
)

NEAR_EQUAL_VR_TOLERANCE = 0.005
DISPLAY_COLUMNS = [
    "dataset_id",
    "dataset_label",
    "cycle_or_wave",
    "country",
    "age_band",
    "trait_id",
    "trait_label",
    "trait_family",
    "trait_priority",
    "priority_display",
    "evidence_status",
    "claim_status_display",
    "comparability_tier",
    "direction",
    "log_variance_ratio",
    "abs_log_vr",
    "variance_ratio",
    "distance_from_equal",
    "se_log_variance_ratio",
    "ci_low_log_variance_ratio",
    "ci_high_log_variance_ratio",
    "vr_ci_low",
    "vr_ci_high",
    "male_n",
    "female_n",
    "n_total",
    "mean_difference",
    "inference_method",
    "headline_eligible",
    "provisional",
    "method_limited",
    "qa_only",
    "suppression_reason",
    "qa_flags",
    "display_explanation",
]

VR_HISTOGRAM_BINS = (
    (-np.inf, 0.75, "<0.75x"),
    (0.75, 0.9, "0.75x-0.90x"),
    (0.9, 0.98, "0.90x-0.98x"),
    (0.98, 1.02, "0.98x-1.02x"),
    (1.02, 1.1, "1.02x-1.10x"),
    (1.1, 1.25, "1.10x-1.25x"),
    (1.25, np.inf, ">1.25x"),
)


@dataclass(frozen=True)
class ComparisonSource:
    source_id: str
    path: Path


def _root_trait_id(trait_id: str) -> str:
    return str(trait_id).split(":", 1)[0]


def _dataset_label(dataset_id: str, registry: Registry) -> str:
    try:
        return registry.get_dataset(dataset_id).name
    except KeyError:
        return dataset_id


def _trait_metadata(trait_id: str, registry: Registry) -> tuple[str, str, str, str]:
    root = _root_trait_id(trait_id)
    try:
        spec = registry.get_trait(root)
        return root, spec.label, spec.family, spec.priority
    except KeyError:
        return root, root.replace("_", " ").title(), "unknown", "unknown"


def _priority_display(value: object) -> str:
    mapping = {
        "confirmatory": "Confirmatory",
        "secondary": "Secondary",
        "exploratory": "Exploratory",
        "unknown": "Unknown",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def _claim_status_display(*, evidence_status: object, trait_priority: object) -> str:
    status = str(evidence_status)
    priority = str(trait_priority)
    if status == "qa_only":
        return "QA only"
    if status == "provisional":
        return "Provisional"
    if status == "method_limited":
        return "Method-limited"
    if priority == "confirmatory":
        return "Headline claim"
    return "Supporting evidence"


def _direction_from_values(
    *,
    log_variance_ratio: float | None,
    variance_ratio: float | None,
    effect_available: bool,
) -> str:
    if not effect_available:
        return "unavailable"
    vr = variance_ratio
    if vr is not None and np.isfinite(vr) and abs(vr - 1.0) <= NEAR_EQUAL_VR_TOLERANCE:
        return "near_equal"
    if log_variance_ratio is not None and np.isfinite(log_variance_ratio):
        return "male_greater" if log_variance_ratio > 0 else "female_greater"
    return "unavailable"


def _direction_display(value: object) -> str:
    mapping = {
        "male_greater": "Male-greater variability",
        "female_greater": "Female-greater variability",
        "near_equal": "Near equal variance",
        "unavailable": "Unavailable",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def _build_display_explanation(row: pd.Series) -> str:
    claim = str(row["claim_status_display"])
    direction = _direction_display(row["direction"])
    if not bool(row["effect_available"]):
        detail = "no usable variance-ratio estimate"
    elif bool(row["ci_available"]):
        detail = (
            f"{direction.lower()} (VR {row['variance_ratio']:.2f}x, "
            f"95% CI {row['vr_ci_low']:.2f}x to {row['vr_ci_high']:.2f}x)"
        )
    else:
        detail = f"{direction.lower()} (VR {row['variance_ratio']:.2f}x)"
    note = ""
    if not pd.isna(row["suppression_reason"]):
        note = f"; {str(row['suppression_reason']).replace('_', ' ')}"
    return f"{claim}. {detail}{note}."


def _display_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in DISPLAY_COLUMNS if column in df.columns]


def _rank_cells(
    df: pd.DataFrame,
    *,
    limit: int,
    sort_by: str,
    ascending: bool,
    priorities: tuple[str, ...] | None = None,
    require_ci: bool = True,
    direction: str | None = None,
) -> pd.DataFrame:
    eligible = df.copy()
    if require_ci:
        eligible = eligible[eligible["ci_available"]].copy()
    else:
        eligible = eligible[eligible["effect_available"]].copy()
    if priorities is not None:
        eligible = eligible[eligible["trait_priority"].isin(priorities)].copy()
    if direction is not None:
        eligible = eligible[eligible["direction"] == direction].copy()
    eligible = eligible[np.isfinite(pd.to_numeric(eligible.get(sort_by), errors="coerce"))].copy()
    if eligible.empty:
        return pd.DataFrame(columns=_display_columns(df))
    ranked = eligible.sort_values(sort_by, ascending=ascending, kind="stable").head(limit).copy()
    return ranked[_display_columns(ranked)].reset_index(drop=True)


def ensure_display_ready_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["log_variance_ratio"] = pd.to_numeric(work.get("log_variance_ratio"), errors="coerce")
    work["se_log_variance_ratio"] = pd.to_numeric(work.get("se_log_variance_ratio"), errors="coerce")
    work["variance_ratio"] = pd.to_numeric(work.get("variance_ratio"), errors="coerce")
    work["variance_ratio"] = work["variance_ratio"].where(np.isfinite(work["variance_ratio"]), np.exp(work["log_variance_ratio"]))
    work["ci_low_log_variance_ratio"] = pd.to_numeric(
        work.get("ci_low_log_variance_ratio", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    work["ci_high_log_variance_ratio"] = pd.to_numeric(
        work.get("ci_high_log_variance_ratio", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    fallback_half_width = 1.96 * work["se_log_variance_ratio"]
    work["ci_low_log_variance_ratio"] = work["ci_low_log_variance_ratio"].where(
        np.isfinite(work["ci_low_log_variance_ratio"]),
        work["log_variance_ratio"] - fallback_half_width,
    )
    work["ci_high_log_variance_ratio"] = work["ci_high_log_variance_ratio"].where(
        np.isfinite(work["ci_high_log_variance_ratio"]),
        work["log_variance_ratio"] + fallback_half_width,
    )
    work["headline_eligible"] = work.get("headline_eligible", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work["provisional"] = work.get("provisional", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work["method_limited"] = work.get("method_limited", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work["qa_only"] = work.get("qa_only", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work["ci_available"] = work.get(
        "ci_available",
        work["headline_eligible"] | work["provisional"] | work["method_limited"],
    )
    work["ci_available"] = work["ci_available"].fillna(False).astype(bool)
    work["effect_available"] = work.get("effect_available", np.isfinite(work["log_variance_ratio"]))
    work["effect_available"] = work["effect_available"].fillna(False).astype(bool)
    work["male_n"] = pd.to_numeric(work.get("male_n"), errors="coerce")
    work["female_n"] = pd.to_numeric(work.get("female_n"), errors="coerce")
    work["trait_priority"] = work.get("trait_priority", pd.Series("unknown", index=work.index)).astype("string")
    work["evidence_status"] = work.get("evidence_status", pd.Series(pd.NA, index=work.index)).astype("string")
    work["suppression_reason"] = work.get("suppression_reason", pd.Series(pd.NA, index=work.index))
    work["vr_ci_low"] = np.exp(work["ci_low_log_variance_ratio"])
    work["vr_ci_high"] = np.exp(work["ci_high_log_variance_ratio"])
    work["direction"] = [
        _direction_from_values(
            log_variance_ratio=float(log_vr_value) if np.isfinite(log_vr_value) else None,
            variance_ratio=float(vr_value) if np.isfinite(vr_value) else None,
            effect_available=bool(effect_available),
        )
        for log_vr_value, vr_value, effect_available in zip(
            work["log_variance_ratio"],
            work["variance_ratio"],
            work["effect_available"],
            strict=False,
        )
    ]
    work["abs_log_vr"] = work["log_variance_ratio"].abs()
    work["distance_from_equal"] = (work["variance_ratio"] - 1.0).abs()
    work["n_total"] = work["male_n"] + work["female_n"]
    work.loc[~(np.isfinite(work["male_n"]) & np.isfinite(work["female_n"])), "n_total"] = np.nan
    work["claim_status_display"] = [
        _claim_status_display(evidence_status=status, trait_priority=priority)
        for status, priority in zip(work["evidence_status"], work["trait_priority"], strict=False)
    ]
    work["priority_display"] = work["trait_priority"].map(_priority_display)
    work["display_explanation"] = work.apply(_build_display_explanation, axis=1)
    work["male_greater_variability"] = work["log_variance_ratio"] > 0
    return work


def normalize_estimate_table(df: pd.DataFrame, *, source_id: str, registry: Registry) -> pd.DataFrame:
    work = annotate_estimate_evidence(df, registry=registry)
    age_col = "age_band" if "age_band" in work.columns else "grade_or_age_band"
    mean_col = "mean_diff" if "mean_diff" in work.columns else "mean_difference" if "mean_difference" in work.columns else None
    log_vr = pd.to_numeric(work.get("log_variance_ratio"), errors="coerce")
    se_log_vr = pd.to_numeric(work.get("se_log_variance_ratio"), errors="coerce")
    variance_ratio = pd.to_numeric(work.get("variance_ratio"), errors="coerce")
    variance_ratio = variance_ratio.where(np.isfinite(variance_ratio), np.exp(log_vr))
    ci_low_log = pd.to_numeric(
        work.get("ci_low_log_variance_ratio", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    ci_high_log = pd.to_numeric(
        work.get("ci_high_log_variance_ratio", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    fallback_half_width = 1.96 * se_log_vr
    ci_low_log = ci_low_log.where(np.isfinite(ci_low_log), log_vr - fallback_half_width)
    ci_high_log = ci_high_log.where(np.isfinite(ci_high_log), log_vr + fallback_half_width)

    normalized = pd.DataFrame(
        {
            "source_table": source_id,
            "dataset_id": work["dataset_id"].astype("string"),
            "cycle_or_wave": work.get("cycle_or_wave", pd.Series("all", index=work.index)).astype("string"),
            "country": work.get("country", pd.Series("all", index=work.index)).astype("string"),
            "age_band": work.get(age_col, pd.Series(pd.NA, index=work.index)).astype("string"),
            "trait_id": work["trait_id"].astype("string"),
            "log_variance_ratio": log_vr,
            "se_log_variance_ratio": se_log_vr,
            "variance_ratio": variance_ratio,
            "ci_low_log_variance_ratio": ci_low_log,
            "ci_high_log_variance_ratio": ci_high_log,
            "mean_difference": pd.to_numeric(work.get(mean_col) if mean_col else np.nan, errors="coerce"),
            "inference_method": work.get("inference_method", pd.Series(pd.NA, index=work.index)).astype("string"),
            "male_n": pd.to_numeric(work.get("male_n"), errors="coerce"),
            "female_n": pd.to_numeric(work.get("female_n"), errors="coerce"),
            "qa_flags": work.get("qa_flags", pd.Series(pd.NA, index=work.index)),
            "trait_family": work.get("trait_family", pd.Series(pd.NA, index=work.index)).astype("string"),
            "trait_priority": work.get("trait_priority", pd.Series(pd.NA, index=work.index)).astype("string"),
            "evidence_status": work.get("evidence_status", pd.Series(pd.NA, index=work.index)).astype("string"),
            "headline_eligible": work.get("headline_eligible", pd.Series(False, index=work.index)).fillna(False).astype(bool),
            "suppression_reason": work.get("suppression_reason", pd.Series(pd.NA, index=work.index)),
            "comparability_tier": work.get("comparability_tier", pd.Series(pd.NA, index=work.index)).astype("string"),
            "provisional": work.get("provisional", pd.Series(False, index=work.index)).fillna(False).astype(bool),
            "qa_only": work.get("qa_only", pd.Series(False, index=work.index)).fillna(False).astype(bool),
            "method_limited": work.get("method_limited", pd.Series(False, index=work.index)).fillna(False).astype(bool),
            "trait_scale_type": work.get("trait_scale_type", pd.Series(pd.NA, index=work.index)).astype("string"),
        }
    )

    normalized["dataset_label"] = normalized["dataset_id"].map(lambda value: _dataset_label(str(value), registry))
    trait_meta = normalized["trait_id"].map(lambda value: _trait_metadata(str(value), registry))
    normalized["trait_root"] = trait_meta.map(lambda item: item[0])
    normalized["trait_label"] = trait_meta.map(lambda item: item[1])
    normalized["trait_family"] = normalized["trait_family"].fillna(trait_meta.map(lambda item: item[2]))
    normalized["trait_priority"] = normalized["trait_priority"].fillna(trait_meta.map(lambda item: item[3]))
    return ensure_display_ready_columns(normalized)


def load_comparison_tables(
    root: Path,
    *,
    registry: Registry,
    sources: dict[str, Path] | None = None,
) -> pd.DataFrame:
    source_map = sources or DEFAULT_COMPARISON_SOURCES
    frames: list[pd.DataFrame] = []
    missing_sources: list[str] = []
    for source_id, relative_path in source_map.items():
        path = relative_path if relative_path.is_absolute() else root / relative_path
        if not path.exists():
            missing_sources.append(f"{source_id} -> {path}")
            continue
        df = pd.read_csv(path)
        frames.append(normalize_estimate_table(df, source_id=source_id, registry=registry))
    if missing_sources:
        warnings.warn(
            "Missing comparison input(s): " + "; ".join(missing_sources),
            stacklevel=2,
        )
    if not frames:
        raise FileNotFoundError("No comparison-ready estimate tables were found.")
    return pd.concat(frames, ignore_index=True)


def _share_male_greater(sub: pd.DataFrame) -> float:
    eligible = sub[sub["effect_available"]].copy()
    if eligible.empty:
        return float("nan")
    return float(eligible["male_greater_variability"].mean())


def build_dataset_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, sub in df.groupby("dataset_id", sort=True):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": sub["dataset_label"].iloc[0],
                "source_table": ",".join(sorted(sub["source_table"].dropna().astype(str).unique())),
                "rows": int(len(sub)),
                "rows_with_ci": int(sub["ci_available"].sum()),
                "headline_eligible_rows": int(sub["headline_eligible"].sum()),
                "provisional_rows": int(sub["provisional"].sum()),
                "method_limited_rows": int(sub["method_limited"].sum()),
                "qa_only_rows": int(sub["qa_only"].sum()),
                "trait_roots": int(sub["trait_root"].nunique()),
                "trait_families": int(sub["trait_family"].nunique()),
                "age_bands": int(sub["age_band"].nunique()),
                "countries": int(sub["country"].nunique()),
                "median_log_variance_ratio": float(sub["log_variance_ratio"].median(skipna=True)),
                "share_male_greater": _share_male_greater(sub),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_label", kind="stable").reset_index(drop=True)


def build_trait_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_label, trait_family), sub in df.groupby(["dataset_label", "trait_family"], sort=True):
        rows.append(
            {
                "dataset_label": dataset_label,
                "trait_family": trait_family,
                "cells": int(len(sub)),
                "cells_with_ci": int(sub["ci_available"].sum()),
                "median_log_variance_ratio": float(sub["log_variance_ratio"].median(skipna=True)),
                "share_male_greater": _share_male_greater(sub),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["dataset_label", "trait_family"], kind="stable").reset_index(drop=True)


def build_priority_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (trait_priority, dataset_label), sub in df.groupby(["trait_priority", "dataset_label"], sort=True):
        rows.append(
            {
                "trait_priority": trait_priority,
                "dataset_label": dataset_label,
                "cells": int(len(sub)),
                "cells_with_ci": int(sub["ci_available"].sum()),
                "headline_eligible_cells": int(sub["headline_eligible"].sum()),
                "provisional_cells": int(sub["provisional"].sum()),
                "method_limited_cells": int(sub["method_limited"].sum()),
                "qa_only_cells": int(sub["qa_only"].sum()),
                "median_log_variance_ratio": float(sub["log_variance_ratio"].median(skipna=True)),
                "share_male_greater": _share_male_greater(sub),
            }
        )
    summary = pd.DataFrame(rows)
    priority_order = {"confirmatory": 0, "secondary": 1, "exploratory": 2, "unknown": 3}
    summary["priority_order"] = summary["trait_priority"].map(priority_order).fillna(99)
    summary = summary.sort_values(["priority_order", "dataset_label"], kind="stable").drop(columns="priority_order")
    return summary.reset_index(drop=True)


def build_top_cells(df: pd.DataFrame, *, limit: int = 20, priorities: tuple[str, ...] | None = None) -> pd.DataFrame:
    return _rank_cells(
        df,
        limit=limit,
        sort_by="abs_log_vr",
        ascending=False,
        priorities=priorities,
        require_ci=True,
    )


def build_dataset_focus_table(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    ci_available: bool | None = None,
    trait_families: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    focus = df[df["dataset_id"] == dataset_id].copy()
    if focus.empty:
        return pd.DataFrame()

    if ci_available is not None:
        focus = focus[focus["ci_available"] == ci_available].copy()
    if trait_families is not None:
        focus = focus[focus["trait_family"].isin(trait_families)].copy()
    if focus.empty:
        return pd.DataFrame()

    display_cols = [
        "dataset_label",
        "cycle_or_wave",
        "age_band",
        "trait_label",
        "trait_family",
        "evidence_status",
        "male_n",
        "female_n",
        "log_variance_ratio",
        "variance_ratio",
        "se_log_variance_ratio",
        "suppression_reason",
        "qa_flags",
    ]
    return focus.sort_values(
        ["trait_family", "trait_label", "cycle_or_wave", "age_band"],
        kind="stable",
    )[display_cols].reset_index(drop=True)


def build_dataset_focus_split(df: pd.DataFrame, *, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        build_dataset_focus_table(df, dataset_id=dataset_id, ci_available=True),
        build_dataset_focus_table(df, dataset_id=dataset_id, ci_available=False),
    )


def build_supporting_evidence_summary(
    df: pd.DataFrame,
    *,
    dataset_ids: tuple[str, ...] = SUPPORTING_EVIDENCE_DATASET_IDS,
) -> pd.DataFrame:
    support = df[df["dataset_id"].isin(dataset_ids)].copy()
    if support.empty:
        return pd.DataFrame()

    rows = []
    for (dataset_id, dataset_label, trait_priority), sub in support.groupby(
        ["dataset_id", "dataset_label", "trait_priority"], sort=True
    ):
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_label": dataset_label,
                "trait_priority": trait_priority,
                "rows": int(len(sub)),
                "rows_with_ci": int(sub["ci_available"].sum()),
                "headline_eligible_rows": int(sub["headline_eligible"].sum()),
                "provisional_rows": int(sub["provisional"].sum()),
                "method_limited_rows": int(sub["method_limited"].sum()),
                "qa_only_rows": int(sub["qa_only"].sum()),
                "trait_families": int(sub["trait_family"].nunique()),
                "median_log_variance_ratio": float(sub["log_variance_ratio"].median(skipna=True)),
                "share_male_greater": _share_male_greater(sub),
            }
        )
    summary = pd.DataFrame(rows)
    priority_order = {"confirmatory": 0, "secondary": 1, "exploratory": 2, "unknown": 3}
    summary["priority_order"] = summary["trait_priority"].map(priority_order).fillna(99)
    summary = summary.sort_values(["dataset_label", "priority_order"], kind="stable").drop(columns="priority_order")
    return summary.reset_index(drop=True)


def build_supporting_evidence_top_cells(
    df: pd.DataFrame,
    *,
    dataset_ids: tuple[str, ...] = SUPPORTING_EVIDENCE_DATASET_IDS,
    limit: int = 30,
) -> pd.DataFrame:
    support = df[df["dataset_id"].isin(dataset_ids)].copy()
    return _rank_cells(
        support,
        limit=limit,
        sort_by="abs_log_vr",
        ascending=False,
        require_ci=True,
    )


def build_strongest_male_greater_cells(
    df: pd.DataFrame,
    *,
    limit: int = 20,
    priorities: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return _rank_cells(
        df,
        limit=limit,
        sort_by="log_variance_ratio",
        ascending=False,
        priorities=priorities,
        require_ci=True,
        direction="male_greater",
    )


def build_strongest_female_greater_cells(
    df: pd.DataFrame,
    *,
    limit: int = 20,
    priorities: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return _rank_cells(
        df,
        limit=limit,
        sort_by="log_variance_ratio",
        ascending=True,
        priorities=priorities,
        require_ci=True,
        direction="female_greater",
    )


def build_closest_to_equal_cells(
    df: pd.DataFrame,
    *,
    limit: int = 20,
    priorities: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return _rank_cells(
        df,
        limit=limit,
        sort_by="distance_from_equal",
        ascending=True,
        priorities=priorities,
        require_ci=True,
    )


def build_largest_n_cells(
    df: pd.DataFrame,
    *,
    limit: int = 20,
    priorities: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return _rank_cells(
        df,
        limit=limit,
        sort_by="n_total",
        ascending=False,
        priorities=priorities,
        require_ci=False,
    )


def build_widest_ci_cells(
    df: pd.DataFrame,
    *,
    limit: int = 20,
    priorities: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work["vr_ci_width"] = work["vr_ci_high"] - work["vr_ci_low"]
    ranked = _rank_cells(
        work,
        limit=limit,
        sort_by="vr_ci_width",
        ascending=False,
        priorities=priorities,
        require_ci=True,
    )
    return ranked


def _age_band_order_value(value: str | float | int | None) -> float:
    if value is None or pd.isna(value):
        return np.inf
    text = str(value).strip()
    lower = text.lower()
    if lower in {"k", "kindergarten"}:
        return -1.0
    if lower in {"all", "all_ages"}:
        return np.inf
    match = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return np.inf


def build_age_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["effect_available"]].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["age_order"] = eligible["age_band"].map(_age_band_order_value)
    eligible = eligible[np.isfinite(eligible["age_order"])].copy()
    summary = (
        eligible.groupby(["dataset_id", "dataset_label", "age_band", "age_order"], sort=True)
        .agg(
            median_log_variance_ratio=("log_variance_ratio", "median"),
            mean_log_variance_ratio=("log_variance_ratio", "mean"),
            cells=("trait_id", "size"),
        )
        .reset_index()
        .sort_values(["dataset_label", "age_order", "age_band"], kind="stable")
    )
    return summary.reset_index(drop=True)


def build_dataset_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["effect_available"]].copy()
    if eligible.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    labels = ["p10", "p25", "p50", "p75", "p90"]
    for dataset_id, sub in eligible.groupby("dataset_id", sort=True):
        vr_quantiles = sub["variance_ratio"].quantile(quantiles)
        row: dict[str, object] = {
            "dataset_id": dataset_id,
            "dataset_label": sub["dataset_label"].iloc[0],
            "cells": int(len(sub)),
            "cells_with_ci": int(sub["ci_available"].sum()),
            "share_male_greater": _share_male_greater(sub),
            "male_greater_cells": int((sub["direction"] == "male_greater").sum()),
            "female_greater_cells": int((sub["direction"] == "female_greater").sum()),
            "near_equal_cells": int((sub["direction"] == "near_equal").sum()),
        }
        for label, value in zip(labels, vr_quantiles.tolist(), strict=False):
            row[f"variance_ratio_{label}"] = float(value)
        for lower, upper, bucket_label in VR_HISTOGRAM_BINS:
            if np.isinf(lower):
                mask = sub["variance_ratio"] < upper
            elif np.isinf(upper):
                mask = sub["variance_ratio"] >= lower
            else:
                mask = (sub["variance_ratio"] >= lower) & (sub["variance_ratio"] < upper)
            row[f"hist_{bucket_label}"] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset_label", kind="stable").reset_index(drop=True)
