from __future__ import annotations

import pandas as pd

from .pv_replicate import ReplicateDesignSpec, estimate_pv_replicate_cells, infer_replicate_design


PIAACReplicateSpec = ReplicateDesignSpec


def infer_piaac_replicate_spec(df: pd.DataFrame, replicate_cols: list[str]) -> PIAACReplicateSpec:
    return infer_replicate_design(
        df,
        replicate_cols,
        method_col="variance_method" if "variance_method" in df.columns else "VEMETHOD" if "VEMETHOD" in df.columns else None,
        fay_col="fay_factor" if "fay_factor" in df.columns else "VEFAYFAC" if "VEFAYFAC" in df.columns else None,
        nrep_col="n_replicates" if "n_replicates" in df.columns else "VENREPS" if "VENREPS" in df.columns else None,
        default_method="brr",
        default_fay=0.0,
    )


def detect_piaac_replicate_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("SPFWT") and str(col) != "SPFWT0"]


def estimate_piaac_cells(df: pd.DataFrame) -> pd.DataFrame:
    replicate_cols = detect_piaac_replicate_cols(df)
    estimates = estimate_pv_replicate_cells(
        df,
        group_cols=["dataset_id", "cycle_or_wave", "country", "country_id", "grade_or_age_band", "trait_id"],
        replicate_cols=replicate_cols,
        method_col="variance_method" if "variance_method" in df.columns else None,
        fay_col="fay_factor" if "fay_factor" in df.columns else None,
        nrep_col="n_replicates" if "n_replicates" in df.columns else None,
        default_method="brr",
        default_fay=0.0,
    )
    return estimates.sort_values(by=["country", "grade_or_age_band", "trait_id"], kind="stable").reset_index(drop=True)
