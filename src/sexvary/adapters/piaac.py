from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame


PIAAC_COUNTRY_LABELS = {
    "36": "Australia",
    "40": "Austria",
    "56": "Belgium",
    "124": "Canada",
    "152": "Chile",
    "191": "Croatia",
    "196": "Cyprus",
    "203": "Czech Republic",
    "208": "Denmark",
    "233": "Estonia",
    "246": "Finland",
    "250": "France",
    "276": "Germany",
    "300": "Greece",
    "348": "Hungary",
    "372": "Ireland",
    "376": "Israel",
    "380": "Italy",
    "392": "Japan",
    "410": "Korea",
    "440": "Lithuania",
    "484": "Mexico",
    "528": "Netherlands",
    "554": "New Zealand",
    "578": "Norway",
    "616": "Poland",
    "620": "Portugal",
    "702": "Singapore",
    "703": "Slovakia",
    "705": "Slovenia",
    "724": "Spain",
    "752": "Sweden",
    "756": "Switzerland",
    "792": "Turkey",
    "826": "United Kingdom",
    "840": "United States",
}

PIAAC_AGE_BAND_LABELS = {
    "1": "16-19",
    "2": "20-24",
    "3": "25-29",
    "4": "30-34",
    "5": "35-39",
    "6": "40-44",
    "7": "45-49",
    "8": "50-54",
    "9": "55-59",
    "10": "60-65",
}


def _normalize_piaac_sex(value: Any) -> str | None:
    if pd.isna(value):
        return None
    sval = str(value).strip().lower()
    if sval in {"1", "m", "male"}:
        return "male"
    if sval in {"2", "f", "female"}:
        return "female"
    return None


def _normalize_piaac_country_label(country_id: Any, country_label: Any) -> str | None:
    cid = None if pd.isna(country_id) else str(country_id).strip()
    if not pd.isna(country_label):
        label = str(country_label).strip()
        if label and label not in {cid, f'"{cid}"'}:
            return label
    if cid is None:
        return None
    return PIAAC_COUNTRY_LABELS.get(cid, cid)


def _normalize_piaac_age_band(value: Any) -> str | None:
    if pd.isna(value):
        return None
    sval = str(value).strip()
    return PIAAC_AGE_BAND_LABELS.get(sval, sval)


@dataclass(frozen=True)
class PIAACColumnSpec:
    person_id: str = "SEQID"
    country_id: str = "CNTRYID"
    country_label: str = "CNTRYID_E"
    sex: str = "GENDER_R"
    age: str = "AGE_R"
    age_band: str = "AGEG5LFS"
    weight_main: str = "SPFWT0"
    variance_method: str = "VEMETHOD"
    fay_factor: str = "VEFAYFAC"
    n_replicates: str = "VENREPS"


class PIAACAdapter(BaseAdapter):
    """Adapter for PIAAC public-use microdata with plausible values and replicate weights."""

    TRAIT_PATTERNS = {
        "literacy": r"^PVLIT(\d+)$",
        "numeracy": r"^PVNUM(\d+)$",
        "adaptive_problem_solving": r"^PVAPS(\d+)$",
    }

    def __init__(
        self,
        dataset_spec,
        raw_path: str | Path,
        *,
        country_ids: list[str] | None = None,
        traits: list[str] | None = None,
        columns: PIAACColumnSpec | None = None,
    ):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.country_ids = {str(item) for item in country_ids} if country_ids else None
        self.requested_traits = set(traits) if traits else None
        self.columns = columns or PIAACColumnSpec()

    def load_raw(self) -> pd.DataFrame:
        return read_table(self.raw_path)

    def _detect_pv_columns(self, df: pd.DataFrame) -> dict[str, list[tuple[int, str]]]:
        out: dict[str, list[tuple[int, str]]] = {}
        for trait_id, pattern in self.TRAIT_PATTERNS.items():
            if self.requested_traits and trait_id not in self.requested_traits:
                continue
            matches: list[tuple[int, str]] = []
            regex = re.compile(pattern)
            for col in df.columns:
                match = regex.match(col)
                if match:
                    matches.append((int(match.group(1)), col))
            if matches:
                out[trait_id] = sorted(matches)
        return out

    @staticmethod
    def detect_replicate_weight_cols(df: pd.DataFrame) -> list[str]:
        pairs: list[tuple[int, str]] = []
        for col in df.columns:
            match = re.match(r"^SPFWT(\d+)$", str(col))
            if match and match.group(1) != "0":
                pairs.append((int(match.group(1)), col))
        return [col for _, col in sorted(pairs)]

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        cols = self.columns
        required = [cols.person_id, cols.country_id, cols.sex, cols.age, cols.weight_main]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"PIAAC raw file is missing required columns: {missing}")

        if self.country_ids is not None:
            df = df[df[cols.country_id].astype(str).isin(self.country_ids)].copy()

        pv_columns = self._detect_pv_columns(df)
        if not pv_columns:
            raise ValueError("No plausible-value columns were found for the requested PIAAC traits.")

        replicate_cols = self.detect_replicate_weight_cols(df)
        if not replicate_cols:
            raise ValueError("No PIAAC replicate-weight columns were found (expected SPFWT1+).")

        long_frames: list[pd.DataFrame] = []
        for trait_id, indexed_cols in pv_columns.items():
            for pv_index, pv_col in indexed_cols:
                country_labels = [
                    _normalize_piaac_country_label(country_id, country_label)
                    for country_id, country_label in zip(
                        df[cols.country_id],
                        df[cols.country_label] if cols.country_label in df.columns else df[cols.country_id],
                    )
                ]
                out = pd.DataFrame(
                    {
                        "source_id": self.dataset_spec.id,
                        "dataset_id": self.dataset_spec.id,
                        "cycle_or_wave": "cycle2",
                        "country": country_labels,
                        "country_id": df[cols.country_id].astype(str),
                        "grade_or_age_band": (
                            df[cols.age_band].map(_normalize_piaac_age_band) if cols.age_band in df.columns else pd.NA
                        ),
                        "person_id": df[cols.person_id],
                        "sex_observed": df[cols.sex].map(_normalize_piaac_sex),
                        "age": pd.to_numeric(df[cols.age], errors="coerce"),
                        "trait_id": trait_id,
                        "score_raw": pd.to_numeric(df[pv_col], errors="coerce"),
                        "weight_main": pd.to_numeric(df[cols.weight_main], errors="coerce"),
                        "pv_domain": trait_id,
                        "pv_index": pv_index,
                        "variance_method": df[cols.variance_method] if cols.variance_method in df.columns else pd.NA,
                        "fay_factor": pd.to_numeric(df[cols.fay_factor], errors="coerce") if cols.fay_factor in df.columns else pd.NA,
                        "n_replicates": pd.to_numeric(df[cols.n_replicates], errors="coerce") if cols.n_replicates in df.columns else len(replicate_cols),
                    }
                )
                for rep_col in replicate_cols:
                    out[rep_col] = pd.to_numeric(df[rep_col], errors="coerce")
                long_frames.append(out)

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "weight_main"])
        provenance = {
            "raw_path": str(self.raw_path),
            "replicate_weight_cols": replicate_cols,
            "pv_traits": sorted(pv_columns),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)
