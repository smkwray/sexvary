from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame


NHANES_CYCLE_LABELS = {
    "G": "2011-2012",
    "H": "2013-2014",
    "I": "2015-2016",
    "J": "2017-2018",
    "L": "2021-2023",
}


def _normalize_nhanes_sex(value: Any) -> str | None:
    if pd.isna(value):
        return None
    sval = str(value).strip().lower()
    if sval in {"1", "1.0", "male", "m"}:
        return "male"
    if sval in {"2", "2.0", "female", "f"}:
        return "female"
    return None


def _max_grip_strength(df: pd.DataFrame) -> pd.Series:
    trial_cols = [
        col
        for col in ["MGXH1T1", "MGXH2T1", "MGXH1T2", "MGXH2T2", "MGXH1T3", "MGXH2T3"]
        if col in df.columns
    ]
    if not trial_cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[trial_cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True)


@dataclass(frozen=True)
class NHANESCycleBundle:
    suffix: str
    cycle_label: str
    demo_path: Path
    bmx_path: Path | None = None
    cfq_path: Path | None = None
    mgx_path: Path | None = None


class NHANESAdapter(BaseAdapter):
    """Adapter for selected NHANES cycles using demographics, body measures, cognition, and grip files."""

    TRAIT_COLUMNS = {
        "height_cm": "BMXHT",
        "weight_kg": "BMXWT",
        "bmi": "BMXBMI",
        "waist_cm": "BMXWAIST",
        "adult_cognition_screen": "CFDDS",
        "grip_strength_kg": "__derived_grip_strength_kg__",
    }

    def __init__(
        self,
        dataset_spec,
        raw_path: str | Path,
        *,
        cycles: list[str] | None = None,
        traits: list[str] | None = None,
    ):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.requested_cycles = {item.upper() for item in cycles} if cycles else None
        self.requested_traits = set(traits) if traits else None

    def _component_path(self, component: str, suffix: str) -> Path | None:
        candidates = sorted(self.raw_path.glob(f"{component}_{suffix}.xpt"))
        return candidates[0] if candidates else None

    def discover_cycle_bundles(self) -> list[NHANESCycleBundle]:
        bundles: list[NHANESCycleBundle] = []
        for suffix, cycle_label in NHANES_CYCLE_LABELS.items():
            if self.requested_cycles and suffix not in self.requested_cycles:
                continue
            demo_path = self._component_path("DEMO", suffix)
            if demo_path is None:
                continue
            bundles.append(
                NHANESCycleBundle(
                    suffix=suffix,
                    cycle_label=cycle_label,
                    demo_path=demo_path,
                    bmx_path=self._component_path("BMX", suffix),
                    cfq_path=self._component_path("CFQ", suffix),
                    mgx_path=self._component_path("MGX", suffix),
                )
            )
        return bundles

    def _load_cycle_frame(self, bundle: NHANESCycleBundle) -> pd.DataFrame:
        demo = read_table(bundle.demo_path).copy()
        demo["SEQN"] = pd.to_numeric(demo["SEQN"], errors="coerce")
        merged = demo

        for extra_path in [bundle.bmx_path, bundle.cfq_path, bundle.mgx_path]:
            if extra_path is None:
                continue
            extra = read_table(extra_path).copy()
            extra["SEQN"] = pd.to_numeric(extra["SEQN"], errors="coerce")
            merged = merged.merge(extra, on="SEQN", how="left", suffixes=("", "_dup"))
            dupes = [col for col in merged.columns if col.endswith("_dup")]
            if dupes:
                merged = merged.drop(columns=dupes)

        if bundle.mgx_path is not None:
            merged["__derived_grip_strength_kg__"] = _max_grip_strength(merged)
        return merged

    def load_raw(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for bundle in self.discover_cycle_bundles():
            cycle_df = self._load_cycle_frame(bundle)
            cycle_df["__cycle_suffix__"] = bundle.suffix
            cycle_df["__cycle_label__"] = bundle.cycle_label
            frames.append(cycle_df)
        if not frames:
            raise FileNotFoundError(
                "No NHANES demographics files were found. Expected DEMO_<suffix>.xpt files in the raw directory."
            )
        return pd.concat(frames, ignore_index=True, sort=False)

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        required = ["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "__cycle_label__"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"NHANES raw files are missing required columns: {missing}")

        trait_map = self.TRAIT_COLUMNS
        if self.requested_traits:
            trait_map = {trait: col for trait, col in trait_map.items() if trait in self.requested_traits}

        long_frames: list[pd.DataFrame] = []
        for trait_id, source_col in trait_map.items():
            if source_col not in df.columns:
                continue
            scores = pd.to_numeric(df[source_col], errors="coerce")
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": df["__cycle_label__"].astype("string"),
                    "country": pd.Series("United States", index=df.index, dtype="string"),
                    "country_id": pd.Series("840", index=df.index, dtype="string"),
                    "grade_or_age_band": pd.Series(pd.NA, index=df.index, dtype="object"),
                    "person_id": pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64").astype("string"),
                    "sex_observed": df["RIAGENDR"].map(_normalize_nhanes_sex),
                    "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
                    "trait_id": trait_id,
                    "score_raw": scores,
                    "weight_main": pd.to_numeric(df["WTMEC2YR"], errors="coerce"),
                    "weight_source": pd.Series("WTMEC2YR", index=df.index, dtype="string"),
                    "weight_primary_source": pd.Series("WTMEC2YR", index=df.index, dtype="string"),
                    "design_strata": pd.to_numeric(df["SDMVSTRA"], errors="coerce").astype("Int64").astype("string"),
                    "design_psu": pd.to_numeric(df["SDMVPSU"], errors="coerce").astype("Int64").astype("string"),
                    "source_variable": pd.Series(source_col, index=df.index, dtype="string"),
                }
            )
            long_frames.append(out)

        if not long_frames:
            raise ValueError("No NHANES trait columns were found for the requested configuration.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "weight_main"])
        provenance = {
            "raw_path": str(self.raw_path),
            "cycle_bundles": [
                {
                    "suffix": bundle.suffix,
                    "cycle_label": bundle.cycle_label,
                    "demo_path": str(bundle.demo_path),
                    "bmx_path": str(bundle.bmx_path) if bundle.bmx_path else None,
                    "cfq_path": str(bundle.cfq_path) if bundle.cfq_path else None,
                    "mgx_path": str(bundle.mgx_path) if bundle.mgx_path else None,
                }
                for bundle in self.discover_cycle_bundles()
            ],
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)
