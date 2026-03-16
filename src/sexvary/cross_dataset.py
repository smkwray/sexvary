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


def normalize_estimate_table(df: pd.DataFrame, *, source_id: str, registry: Registry) -> pd.DataFrame:
    work = annotate_estimate_evidence(df, registry=registry)
    age_col = "age_band" if "age_band" in work.columns else "grade_or_age_band"
    mean_col = "mean_diff" if "mean_diff" in work.columns else "mean_difference" if "mean_difference" in work.columns else None

    normalized = pd.DataFrame(
        {
            "source_table": source_id,
            "dataset_id": work["dataset_id"].astype("string"),
            "cycle_or_wave": work.get("cycle_or_wave", pd.Series("all", index=work.index)).astype("string"),
            "country": work.get("country", pd.Series("all", index=work.index)).astype("string"),
            "age_band": work.get(age_col, pd.Series(pd.NA, index=work.index)).astype("string"),
            "trait_id": work["trait_id"].astype("string"),
            "log_variance_ratio": pd.to_numeric(work.get("log_variance_ratio"), errors="coerce"),
            "se_log_variance_ratio": pd.to_numeric(work.get("se_log_variance_ratio"), errors="coerce"),
            "variance_ratio": pd.to_numeric(work.get("variance_ratio"), errors="coerce"),
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
    normalized["ci_available"] = normalized["headline_eligible"] | normalized["provisional"] | normalized["method_limited"]
    normalized["effect_available"] = np.isfinite(normalized["log_variance_ratio"])
    normalized["male_greater_variability"] = normalized["log_variance_ratio"] > 0
    return normalized


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
    eligible = df[df["ci_available"]].copy()
    if priorities is not None:
        eligible = eligible[eligible["trait_priority"].isin(priorities)].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["abs_log_variance_ratio"] = eligible["log_variance_ratio"].abs()
    top = eligible.sort_values("abs_log_variance_ratio", ascending=False, kind="stable").head(limit)
    return top[
        [
            "dataset_label",
            "trait_priority",
            "evidence_status",
            "cycle_or_wave",
            "age_band",
            "trait_label",
            "trait_family",
            "log_variance_ratio",
            "variance_ratio",
            "se_log_variance_ratio",
            "suppression_reason",
            "qa_flags",
        ]
    ].reset_index(drop=True)


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
    support = df[df["dataset_id"].isin(dataset_ids) & df["ci_available"]].copy()
    if support.empty:
        return pd.DataFrame()
    support["abs_log_variance_ratio"] = support["log_variance_ratio"].abs()
    top = support.sort_values("abs_log_variance_ratio", ascending=False, kind="stable").head(limit)
    return top[
        [
            "dataset_label",
            "trait_priority",
            "evidence_status",
            "cycle_or_wave",
            "age_band",
            "trait_label",
            "trait_family",
            "headline_eligible",
            "log_variance_ratio",
            "variance_ratio",
            "se_log_variance_ratio",
            "suppression_reason",
            "qa_flags",
        ]
    ].reset_index(drop=True)


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
