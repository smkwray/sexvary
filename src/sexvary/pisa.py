from __future__ import annotations

import pandas as pd

from .pv_replicate import ReplicateDesignSpec, estimate_pv_replicate_cells, infer_replicate_design


PISAReplicateSpec = ReplicateDesignSpec


def infer_pisa_replicate_spec(df: pd.DataFrame, replicate_cols: list[str]) -> PISAReplicateSpec:
    return infer_replicate_design(
        df,
        replicate_cols,
        method_col="variance_method" if "variance_method" in df.columns else None,
        fay_col="fay_factor" if "fay_factor" in df.columns else None,
        nrep_col="n_replicates" if "n_replicates" in df.columns else None,
        default_method="brr",
        default_fay=0.5,
    )


def detect_pisa_replicate_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("W_FSTURWT")]


def estimate_pisa_cells(df: pd.DataFrame) -> pd.DataFrame:
    replicate_cols = detect_pisa_replicate_cols(df)
    estimates = estimate_pv_replicate_cells(
        df,
        group_cols=["dataset_id", "cycle_or_wave", "country", "country_id", "grade_or_age_band", "trait_id"],
        replicate_cols=replicate_cols,
        method_col="variance_method" if "variance_method" in df.columns else None,
        fay_col="fay_factor" if "fay_factor" in df.columns else None,
        nrep_col="n_replicates" if "n_replicates" in df.columns else None,
        default_method="brr",
        default_fay=0.5,
    )
    if estimates.empty:
        return estimates
    return estimates.sort_values(by=["country", "trait_id"], kind="stable").reset_index(drop=True)
