from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightCheck:
    n_rows: int
    n_missing: int
    n_nonpositive: int
    min_weight: float | None
    max_weight: float | None


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def summarize_weights(df: pd.DataFrame, weight_col: str) -> WeightCheck:
    require_columns(df, [weight_col])
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    finite = weights[np.isfinite(weights)]
    return WeightCheck(
        n_rows=int(len(df)),
        n_missing=int(weights.isna().sum()),
        n_nonpositive=int((finite <= 0).sum()),
        min_weight=float(finite.min()) if len(finite) else None,
        max_weight=float(finite.max()) if len(finite) else None,
    )


def missingness_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "n_rows": len(df),
                "n_missing": int(series.isna().sum()),
                "pct_missing": float(series.isna().mean() * 100),
                "n_unique_nonmissing": int(series.dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)
