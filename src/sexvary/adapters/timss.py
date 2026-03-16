from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame


TIMSS_COUNTRY_LABELS = {
    "840": "United States",
}


def _normalize_timss_country_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if numeric.is_integer():
        return str(int(numeric))
    text = str(value).strip()
    return text or None


def _normalize_timss_country_label(country_id: Any, country_label: Any) -> str | None:
    cid = _normalize_timss_country_id(country_id)
    if not pd.isna(country_label):
        label = str(country_label).strip()
        if label and label not in {str(country_id).strip(), cid, f'"{cid}"'}:
            return label
    if cid is None:
        return None
    return TIMSS_COUNTRY_LABELS.get(cid, cid)


def _normalize_timss_sex(value: Any) -> str | None:
    if pd.isna(value):
        return None
    sval = str(value).strip().lower()
    if sval in {"2", "2.0", "boy", "male", "m"}:
        return "male"
    if sval in {"1", "1.0", "girl", "female", "f"}:
        return "female"
    return None


@dataclass(frozen=True)
class TIMSSColumnSpec:
    person_id: str = "IDSTUD"
    country_id: str = "IDCNTRY"
    country_label: str = "IDCNTRYL"
    sex: str = "ITSEX"
    weight_main: str = "TOTWGT"
    jk_zone: str = "JKZONE"
    jk_rep: str = "JKREP"


class TIMSSAdapter(BaseAdapter):
    """Adapter for IEA-style student achievement files with plausible values and JRR design fields."""

    TRAIT_PATTERNS = {
        # TIMSS 2019 uses "A" student-file prefixes for grade 4 and "B" for grade 8.
        ("4", "math_achievement"): r"^ASMMAT(\d{2})$",
        ("4", "science_achievement"): r"^ASSSCI(\d{2})$",
        ("8", "math_achievement"): r"^BSMMAT(\d{2})$",
        ("8", "science_achievement"): r"^BSSSCI(\d{2})$",
    }

    def __init__(
        self,
        dataset_spec,
        raw_path: str | Path | Iterable[str | Path],
        *,
        country_ids: list[str] | None = None,
        grades: list[str] | None = None,
        traits: list[str] | None = None,
        columns: TIMSSColumnSpec | None = None,
        trait_patterns: dict[tuple[str, str], str] | None = None,
        cycle_label_template: str | None = None,
    ):
        self.dataset_spec = dataset_spec
        if isinstance(raw_path, (str, Path)):
            self.raw_path = Path(raw_path)
        else:
            self.raw_path = [Path(path) for path in raw_path]
        self.country_ids = {str(item) for item in country_ids} if country_ids else None
        self.requested_grades = {str(item) for item in grades} if grades else None
        self.requested_traits = set(traits) if traits else None
        self.columns = columns or TIMSSColumnSpec()
        self.trait_patterns = trait_patterns or self.TRAIT_PATTERNS
        self.cycle_label_template = cycle_label_template or f"{self.dataset_spec.id.replace('_', '')}_grade{{grade}}"

    @property
    def raw_paths(self) -> list[Path]:
        if isinstance(self.raw_path, (str, Path)):
            return [Path(self.raw_path)]
        return [Path(path) for path in self.raw_path]

    def load_raw(self) -> pd.DataFrame:
        frames = [read_table(path) for path in self.raw_paths]
        if not frames:
            raise FileNotFoundError("No TIMSS raw files were provided.")
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True, sort=False)

    def _detect_pv_columns(self, df: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, str]]]:
        out: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for (grade, trait_id), pattern in self.trait_patterns.items():
            if self.requested_grades and grade not in self.requested_grades:
                continue
            if self.requested_traits and trait_id not in self.requested_traits:
                continue
            regex = re.compile(pattern)
            matches: list[tuple[int, str]] = []
            for col in df.columns:
                match = regex.match(str(col))
                if match:
                    matches.append((int(match.group(1)), str(col)))
            if matches:
                out[(grade, trait_id)] = sorted(matches)
        return out

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        cols = self.columns
        required = [cols.sex, cols.weight_main, cols.jk_zone, cols.jk_rep]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"TIMSS raw file is missing required columns: {missing}")

        if self.country_ids is not None and cols.country_id in df.columns:
            normalized_country_ids = df[cols.country_id].map(_normalize_timss_country_id)
            df = df[normalized_country_ids.isin(self.country_ids)].copy()

        pv_columns = self._detect_pv_columns(df)
        if not pv_columns:
            raise ValueError("No TIMSS plausible-value columns were found for the requested grades/traits.")

        person_id = (
            df[cols.person_id].astype("string")
            if cols.person_id in df.columns
            else pd.Series(df.index.astype(str), index=df.index, dtype="string")
        )
        country_id = (
            df[cols.country_id].map(_normalize_timss_country_id).astype("string")
            if cols.country_id in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string")
        )
        country = pd.Series(
            [
                _normalize_timss_country_label(
                    raw_country_id,
                    df[cols.country_label].iloc[idx] if cols.country_label in df.columns else raw_country_id,
                )
                for idx, raw_country_id in enumerate(df[cols.country_id] if cols.country_id in df.columns else [pd.NA] * len(df))
            ],
            index=df.index,
            dtype="string",
        )

        long_frames: list[pd.DataFrame] = []
        for (grade, trait_id), indexed_cols in pv_columns.items():
            cycle_label = self.cycle_label_template.format(grade=grade)
            grade_label = grade
            for pv_index, pv_col in indexed_cols:
                out = pd.DataFrame(
                    {
                        "source_id": self.dataset_spec.id,
                        "dataset_id": self.dataset_spec.id,
                        "cycle_or_wave": cycle_label,
                        "country": country,
                        "country_id": country_id,
                        "grade_or_age_band": grade_label,
                        "person_id": person_id,
                        "sex_observed": df[cols.sex].map(_normalize_timss_sex),
                        "age": pd.NA,
                        "trait_id": trait_id,
                        "score_raw": pd.to_numeric(df[pv_col], errors="coerce"),
                        "weight_main": pd.to_numeric(df[cols.weight_main], errors="coerce"),
                        "pv_index": pv_index,
                        "variance_method": "JRR",
                        "jk_zone": df[cols.jk_zone],
                        "jk_rep": df[cols.jk_rep],
                    }
                )
                long_frames.append(out)

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "weight_main"])
        provenance = {
            "raw_path": [str(path) for path in self.raw_paths],
            "pv_groups": sorted([f"{grade}:{trait}" for grade, trait in pv_columns]),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)
