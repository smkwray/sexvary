from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import Registry, build_registry


PROVISIONAL_FLAG_TOKENS = {
    "nonprimary_weight_fallback",
    "mixed_weight_sources",
    "pooled_age_fallback",
    "bounded_scale_variance_fragile",
}

SUPPRESSION_FLAG_TOKENS = {
    "missing_sex_group",
    "low_n_variance",
    "bounded_scale_variance_suppressed",
}

METHOD_LIMITED_INFERENCE_METHODS = {
    "analytic_effective_n_simple_design",
    "approximate_household_cluster_bootstrap",
}


def _root_trait_id(trait_id: str) -> str:
    return str(trait_id).split(":", 1)[0]


def _split_flags(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [flag.strip() for flag in str(value).split(";") if flag and str(flag).strip()]


def _attach_trait_metadata(df: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    work = df.copy()
    roots = work["trait_id"].astype("string").map(_root_trait_id)
    work["trait_root"] = work.get("trait_root", roots)
    work["trait_priority"] = work.get(
        "trait_priority",
        roots.map(lambda value: registry.traits.get(str(value)).priority if str(value) in registry.traits else "unknown"),
    )
    work["trait_family"] = work.get(
        "trait_family",
        roots.map(lambda value: registry.traits.get(str(value)).family if str(value) in registry.traits else "unknown"),
    )
    work["trait_scale_type"] = work.get(
        "trait_scale_type",
        roots.map(lambda value: registry.traits.get(str(value)).scale_type if str(value) in registry.traits else "unknown"),
    )
    return work


def _derive_suppression_reason(
    *,
    ci_available: bool,
    qa_flags: Iterable[str],
    inference_method: str,
    method_limited: bool,
    provisional: bool,
) -> str | float:
    flags = list(qa_flags)
    if not ci_available:
        reasons = [flag for flag in flags if flag in SUPPRESSION_FLAG_TOKENS]
        if reasons:
            return ";".join(reasons)
        if inference_method == "unavailable":
            return "no_inference"
        if flags:
            return ";".join(flags)
        return "no_ci"
    if provisional:
        reasons = [flag for flag in flags if flag in PROVISIONAL_FLAG_TOKENS]
        if reasons:
            return ";".join(reasons)
    if method_limited:
        return inference_method or "method_limited"
    if flags:
        soft_reasons = [flag for flag in flags if flag not in {"low_n_tail_95"}]
        if soft_reasons:
            return ";".join(soft_reasons)
    return np.nan


def annotate_estimate_evidence(
    df: pd.DataFrame,
    *,
    registry: Registry | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    registry = registry or build_registry()
    work = _attach_trait_metadata(df, registry)
    work["log_variance_ratio"] = pd.to_numeric(work.get("log_variance_ratio"), errors="coerce")
    work["se_log_variance_ratio"] = pd.to_numeric(work.get("se_log_variance_ratio"), errors="coerce")
    work["inference_method"] = work.get("inference_method", pd.Series(pd.NA, index=work.index)).astype("string")
    work["qa_flags"] = work.get("qa_flags", pd.Series(pd.NA, index=work.index))

    ci_available = np.isfinite(work["log_variance_ratio"]) & np.isfinite(work["se_log_variance_ratio"])
    work["ci_available"] = ci_available
    qa_flag_lists = work["qa_flags"].map(_split_flags)
    provisional = ci_available & qa_flag_lists.map(lambda flags: any(flag in PROVISIONAL_FLAG_TOKENS for flag in flags))
    method_limited = ci_available & work["inference_method"].isin(METHOD_LIMITED_INFERENCE_METHODS)
    headline_eligible = ci_available & ~provisional & ~method_limited

    work["headline_eligible"] = headline_eligible
    work["provisional"] = provisional
    work["qa_only"] = ~ci_available
    work["method_limited"] = method_limited

    def _status_for_row(index: int) -> str:
        if bool(work["qa_only"].iloc[index]):
            return "qa_only"
        if bool(work["provisional"].iloc[index]):
            return "provisional"
        if bool(work["method_limited"].iloc[index]):
            return "method_limited"
        return "headline_eligible"

    work["evidence_status"] = [_status_for_row(i) for i in range(len(work))]

    def _tier_for_row(index: int) -> str:
        if bool(work["qa_only"].iloc[index]):
            return "qa_only"
        priority = str(work["trait_priority"].iloc[index])
        if bool(work["provisional"].iloc[index]):
            return "provisional"
        if bool(work["method_limited"].iloc[index]):
            return "confirmatory_method_limited" if priority == "confirmatory" else "secondary_method_limited"
        return "confirmatory_headline" if priority == "confirmatory" else "secondary_headline"

    work["comparability_tier"] = [_tier_for_row(i) for i in range(len(work))]
    work["suppression_reason"] = [
        _derive_suppression_reason(
            ci_available=bool(ci_available.iloc[i]),
            qa_flags=qa_flag_lists.iloc[i],
            inference_method=str(work["inference_method"].iloc[i]) if not pd.isna(work["inference_method"].iloc[i]) else "",
            method_limited=bool(method_limited.iloc[i]),
            provisional=bool(provisional.iloc[i]),
        )
        for i in range(len(work))
    ]
    return work
