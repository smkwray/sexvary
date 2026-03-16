from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MetaAnalysisResult:
    model: str
    k: int
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    tau2: float
    q: float
    i2: float
    weights: list[float]

    @property
    def estimate_backtransformed(self) -> float:
        return float(np.exp(self.estimate))


def _clean_effects(effect_sizes: Iterable[float], variances: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(effect_sizes), dtype=float)
    v = np.asarray(list(variances), dtype=float)
    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y = y[mask]
    v = v[mask]
    if y.size == 0:
        raise ValueError("No valid effect sizes with positive variances.")
    return y, v


def _heterogeneity(y: np.ndarray, w: np.ndarray, pooled: float) -> tuple[float, float]:
    q = float(np.sum(w * np.square(y - pooled)))
    df = y.size - 1
    i2 = 0.0 if df <= 0 or q <= 0 else max(0.0, (q - df) / q) * 100.0
    return q, i2


def fixed_effect_meta(effect_sizes: Iterable[float], variances: Iterable[float], *, alpha: float = 0.05) -> MetaAnalysisResult:
    y, v = _clean_effects(effect_sizes, variances)
    w = 1.0 / v
    pooled = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    q, i2 = _heterogeneity(y, w, pooled)
    return MetaAnalysisResult(
        model="fixed_effect",
        k=int(y.size),
        estimate=pooled,
        standard_error=se,
        ci_low=float(pooled - z * se),
        ci_high=float(pooled + z * se),
        tau2=0.0,
        q=q,
        i2=i2,
        weights=w.tolist(),
    )


def dersimonian_laird_meta(effect_sizes: Iterable[float], variances: Iterable[float], *, alpha: float = 0.05) -> MetaAnalysisResult:
    y, v = _clean_effects(effect_sizes, variances)
    w_fe = 1.0 / v
    pooled_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    q, i2 = _heterogeneity(y, w_fe, pooled_fe)
    df = y.size - 1
    c = float(np.sum(w_fe) - (np.sum(np.square(w_fe)) / np.sum(w_fe)))
    tau2 = 0.0
    if df > 0 and c > 0:
        tau2 = max(0.0, (q - df) / c)

    w_re = 1.0 / (v + tau2)
    pooled = float(np.sum(w_re * y) / np.sum(w_re))
    se = float(np.sqrt(1.0 / np.sum(w_re)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return MetaAnalysisResult(
        model="random_effects_dl",
        k=int(y.size),
        estimate=pooled,
        standard_error=se,
        ci_low=float(pooled - z * se),
        ci_high=float(pooled + z * se),
        tau2=tau2,
        q=q,
        i2=i2,
        weights=w_re.tolist(),
    )
