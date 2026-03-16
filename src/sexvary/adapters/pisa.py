from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import pyreadstat

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter
from .base import NormalizedTraitFrame


def _normalize_pisa_sex(value: Any) -> str | None:
    if pd.isna(value):
        return None
    sval = str(value).strip().lower()
    if sval in {"2", "2.0", "m", "male"}:
        return "male"
    if sval in {"1", "1.0", "f", "female"}:
        return "female"
    return None


@dataclass(frozen=True)
class PISAColumnSpec:
    person_id: str = "CNTSTUID"
    country: str = "CNT"
    sex: str = "ST004D01T"
    age: str = "AGE"
    weight_main: str = "W_FSTUWT"
    school_id: str = "CNTSCHID"


class PISAAdapter(BaseAdapter):
    """Adapter for PISA 2022 student files with plausible values and BRR replicate weights."""

    TRAIT_PATTERNS = {
        "math_achievement": r"^PV(\d+)MATH$",
        "reading_achievement": r"^PV(\d+)READ$",
        "science_achievement": r"^PV(\d+)SCIE$",
        "creative_thinking": r"^PV(\d+)CREAT$",
    }

    def __init__(
        self,
        dataset_spec,
        raw_path: str | Path,
        *,
        country_codes: list[str] | None = None,
        traits: list[str] | None = None,
        columns: PISAColumnSpec | None = None,
    ):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.country_codes = {str(item) for item in country_codes} if country_codes else None
        self.requested_traits = set(traits) if traits else None
        self.columns = columns or PISAColumnSpec()

    def _detect_pv_columns(self, columns: list[str]) -> dict[str, list[tuple[int, str]]]:
        out: dict[str, list[tuple[int, str]]] = {}
        for trait_id, pattern in self.TRAIT_PATTERNS.items():
            if self.requested_traits and trait_id not in self.requested_traits:
                continue
            regex = re.compile(pattern)
            matches: list[tuple[int, str]] = []
            for col in columns:
                match = regex.match(str(col))
                if match:
                    matches.append((int(match.group(1)), str(col)))
            if matches:
                out[trait_id] = sorted(matches)
        return out

    @staticmethod
    def detect_replicate_weight_cols(columns: list[str] | pd.Index) -> list[str]:
        pairs: list[tuple[int, str]] = []
        for col in columns:
            match = re.match(r"^W_FSTURWT(\d+)$", str(col))
            if match:
                pairs.append((int(match.group(1)), str(col)))
        return [col for _, col in sorted(pairs)]

    def load_raw(self) -> pd.DataFrame:
        if self.raw_path.suffix.lower() != ".sav":
            return read_table(self.raw_path)

        _, meta = pyreadstat.read_sav(self.raw_path, metadataonly=True)
        columns = list(meta.column_names)
        pv_columns = self._detect_pv_columns(columns)
        replicate_cols = self.detect_replicate_weight_cols(columns)
        required = [self.columns.person_id, self.columns.country, self.columns.sex, self.columns.weight_main]
        usecols = sorted({*required, self.columns.age, self.columns.school_id, *replicate_cols, *(col for pairs in pv_columns.values() for _, col in pairs)})
        df, _ = pyreadstat.read_sav(self.raw_path, usecols=usecols, apply_value_formats=False)
        return df

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        cols = self.columns
        required = [cols.person_id, cols.country, cols.sex, cols.weight_main]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"PISA raw file is missing required columns: {missing}")

        if self.country_codes is not None:
            df = df[df[cols.country].astype(str).isin(self.country_codes)].copy()

        pv_columns = self._detect_pv_columns(list(df.columns))
        if not pv_columns:
            raise ValueError("No plausible-value columns were found for the requested PISA traits.")

        replicate_cols = self.detect_replicate_weight_cols(df.columns)
        if not replicate_cols:
            raise ValueError("No PISA replicate-weight columns were found (expected W_FSTR1+).")

        long_frames: list[pd.DataFrame] = []
        age_series = pd.to_numeric(df[cols.age], errors="coerce") if cols.age in df.columns else pd.Series(pd.NA, index=df.index)
        school_series = df[cols.school_id].astype("string") if cols.school_id in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

        for trait_id, indexed_cols in pv_columns.items():
            for pv_index, pv_col in indexed_cols:
                out_columns = {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": "2022",
                    "country": df[cols.country].astype("string"),
                    "country_id": df[cols.country].astype("string"),
                    "grade_or_age_band": pd.Series("15-year-olds", index=df.index, dtype="object"),
                    "design_psu": school_series,
                    "person_id": df[cols.person_id].astype("string"),
                    "sex_observed": df[cols.sex].map(_normalize_pisa_sex),
                    "age": age_series,
                    "trait_id": trait_id,
                    "score_raw": pd.to_numeric(df[pv_col], errors="coerce"),
                    "weight_main": pd.to_numeric(df[cols.weight_main], errors="coerce"),
                    "pv_domain": trait_id,
                    "pv_index": pv_index,
                    "variance_method": "BRR",
                    "fay_factor": 0.5,
                    "n_replicates": len(replicate_cols),
                }
                for rep_col in replicate_cols:
                    out_columns[rep_col] = pd.to_numeric(df[rep_col], errors="coerce")
                long_frames.append(pd.DataFrame(out_columns))

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "weight_main"])
        provenance = {
            "raw_path": str(self.raw_path),
            "replicate_weight_cols": replicate_cols,
            "pv_traits": sorted(pv_columns),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)
