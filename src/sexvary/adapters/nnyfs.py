from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame
from .nhanes import _max_grip_strength, _normalize_nhanes_sex


class NNYFSAdapter(BaseAdapter):
    """Adapter for the 2012 National Youth Fitness Survey public-use files."""

    TRAIT_COLUMNS = {
        "height_cm": "BMXHT",
        "weight_kg": "BMXWT",
        "bmi": "BMXBMI",
        "waist_cm": "BMXWAIST",
        "grip_strength_kg": "__derived_grip_strength_kg__",
    }

    def _component_path(self, component: str) -> Path | None:
        candidates = sorted(self.raw_path.glob(f"Y_{component}.xpt"))
        return candidates[0] if candidates else None

    def _load_bundle(self) -> pd.DataFrame:
        demo_path = self._component_path("DEMO")
        if demo_path is None:
            raise FileNotFoundError(
                "No NNYFS demographics file found. Expected Y_DEMO.xpt in the raw directory."
            )
        demo = read_table(demo_path).copy()
        demo["SEQN"] = pd.to_numeric(demo["SEQN"], errors="coerce")
        merged = demo
        for component in ("BMX", "MGX"):
            extra_path = self._component_path(component)
            if extra_path is None:
                continue
            extra = read_table(extra_path).copy()
            extra["SEQN"] = pd.to_numeric(extra["SEQN"], errors="coerce")
            merged = merged.merge(extra, on="SEQN", how="left", suffixes=("", "_dup"))
            dupes = [col for col in merged.columns if col.endswith("_dup")]
            if dupes:
                merged = merged.drop(columns=dupes)
        if self._component_path("MGX") is not None:
            merged["__derived_grip_strength_kg__"] = _max_grip_strength(merged)
        return merged

    def load_raw(self) -> pd.DataFrame:
        return self._load_bundle()

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        required = ["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC", "SDMVSTRA", "SDMVPSU"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"NNYFS raw files are missing required columns: {missing}")

        long_frames: list[pd.DataFrame] = []
        for trait_id, source_col in self.TRAIT_COLUMNS.items():
            if source_col not in df.columns:
                continue
            scores = pd.to_numeric(df[source_col], errors="coerce")
            long_frames.append(
                pd.DataFrame(
                    {
                        "source_id": self.dataset_spec.id,
                        "dataset_id": self.dataset_spec.id,
                        "cycle_or_wave": pd.Series("2012", index=df.index, dtype="string"),
                        "country": pd.Series("United States", index=df.index, dtype="string"),
                        "country_id": pd.Series("840", index=df.index, dtype="string"),
                        "grade_or_age_band": pd.Series(pd.NA, index=df.index, dtype="object"),
                        "person_id": pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64").astype("string"),
                        "sex_observed": df["RIAGENDR"].map(_normalize_nhanes_sex),
                        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
                        "trait_id": trait_id,
                        "score_raw": scores,
                        "weight_main": pd.to_numeric(df["WTMEC"], errors="coerce"),
                        "weight_source": pd.Series("WTMEC", index=df.index, dtype="string"),
                        "weight_primary_source": pd.Series("WTMEC", index=df.index, dtype="string"),
                        "design_strata": pd.to_numeric(df["SDMVSTRA"], errors="coerce").astype("Int64").astype("string"),
                        "design_psu": pd.to_numeric(df["SDMVPSU"], errors="coerce").astype("Int64").astype("string"),
                        "source_variable": pd.Series(source_col, index=df.index, dtype="string"),
                    }
                )
            )

        if not long_frames:
            raise ValueError("No NNYFS trait columns were found in the available files.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "weight_main"])
        provenance = {
            "raw_path": str(self.raw_path),
            "demo_path": str(self._component_path("DEMO")),
            "bmx_path": str(self._component_path("BMX")) if self._component_path("BMX") else None,
            "mgx_path": str(self._component_path("MGX")) if self._component_path("MGX") else None,
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)
