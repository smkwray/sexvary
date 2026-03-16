from __future__ import annotations

import pandas as pd

from .pv_replicate import estimate_pv_replicate_cells


def detect_icils_replicate_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("SRWGT")]


def estimate_icils_cells(df: pd.DataFrame) -> pd.DataFrame:
    replicate_cols = detect_icils_replicate_cols(df)
    estimates = estimate_pv_replicate_cells(
        df,
        group_cols=["dataset_id", "cycle_or_wave", "country", "country_id", "grade_or_age_band", "trait_id"],
        replicate_cols=replicate_cols,
        method_col="variance_method" if "variance_method" in df.columns else None,
        nrep_col="n_replicates" if "n_replicates" in df.columns else None,
        default_method="jrr",
        default_scale=1.0,
    )
    if estimates.empty:
        return estimates
    return estimates.sort_values(by=["country", "trait_id"], kind="stable").reset_index(drop=True)
