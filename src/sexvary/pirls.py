from __future__ import annotations

import pandas as pd

from .timss import estimate_timss_cells


def estimate_pirls_cells(df: pd.DataFrame) -> pd.DataFrame:
    estimates = estimate_timss_cells(df)
    if estimates.empty:
        return estimates
    return estimates.sort_values(by=["country", "grade_or_age_band", "trait_id"], kind="stable").reset_index(drop=True)
