from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame
from .nlsy_local import _normalize_sex


HRS_TRACKER_SPEC = {
    "zip_name": "trk2022v1.zip",
    "stata_member": "trk2022tr_r.dta",
}

HRS_WAVE_SPECS = (
    {
        "wave_id": "hrs_2018",
        "label": "HRS 2018",
        "core_zip": "h18core.zip",
        "stata_zip": "h18sta.zip",
        "cognition_member": "h18d_r.dta",
        "hhid": "hhid",
        "pn": "pn",
        "immediate": ("QD174", "QD174W"),
        "delayed": ("QD184", "QD184W"),
        "total_cognition": "QD170",
        "serial_7s": ("QD142", "QD143", "QD144", "QD145", "QD146"),
        "numeracy": None,
        "age_col": "QAGE",
        "weight_col": "QWGTR",
    },
    {
        "wave_id": "hrs_2020",
        "label": "HRS 2020",
        "core_zip": "h20core.zip",
        "stata_zip": "h20sta.zip",
        "cognition_member": "H20D_R.dta",
        "hhid": "HHID",
        "pn": "PN",
        "immediate": ("RD174", "RD174W"),
        "delayed": ("RD184", "RD184W"),
        "total_cognition": "RD170",
        "serial_7s": ("RD142", "RD143", "RD144", "RD145", "RD146"),
        "numeracy": "RNSSCORE",
        "age_col": "RAGE",
        "weight_col": "RWGTR",
    },
    {
        "wave_id": "hrs_2022",
        "label": "HRS 2022",
        "core_zip": "h22core.zip",
        "stata_zip": "H22sta.zip",
        "cognition_member": "H22D_R.dta",
        "hhid": "HHID",
        "pn": "PN",
        "immediate": ("SD174", "SD174W"),
        "delayed": ("SD184", "SD184W"),
        "total_cognition": "SD170",
        "serial_7s": ("SD142", "SD143", "SD144", "SD145", "SD146"),
        "numeracy": "SNSSCORE",
        "age_col": "SAGE",
        "weight_col": "SWGTR",
    },
)

SERIAL_7S_EXPECTED = np.array([93.0, 86.0, 79.0, 72.0, 65.0], dtype=float)


def _coerce_identifier(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def _coerce_score(
    series: pd.Series,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    missing_values: set[float] | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if missing_values:
        numeric = numeric.mask(numeric.isin(missing_values))
    if min_value is not None:
        numeric = numeric.mask(numeric < min_value)
    if max_value is not None:
        numeric = numeric.mask(numeric > max_value)
    return numeric


def _serial_7s_score(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    arr = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    arr[arr >= 998] = np.nan
    arr[arr < 0] = np.nan
    matches = np.isfinite(arr)
    scores = (arr == SERIAL_7S_EXPECTED).sum(axis=1).astype(float)
    scores[~matches.any(axis=1)] = np.nan
    return pd.Series(scores, index=frame.index, dtype="float64")


class HRSAdapter(BaseAdapter):
    """Adapter for HRS public cognition waves backed by the tracker file."""

    def __init__(self, dataset_spec, raw_path: str | Path):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.raw_dir = self.raw_path if self.raw_path.is_dir() else self.raw_path.parent
        self.cache_dir = self.raw_dir / ".hrs_extracted"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _extract_member(self, zip_path: Path, member: str, *, cache_prefix: str) -> Path:
        target = self.cache_dir / cache_prefix / member
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(member, path=target.parent)
        return target

    def _load_tracker(self) -> pd.DataFrame:
        tracker_path = self._extract_member(
            self.raw_dir / HRS_TRACKER_SPEC["zip_name"],
            HRS_TRACKER_SPEC["stata_member"],
            cache_prefix="tracker",
        )
        columns = ["HHID", "PN", "SEX", "QAGE", "RAGE", "SAGE", "QWGTR", "RWGTR", "SWGTR", "STRATUM"]
        tracker = pd.read_stata(tracker_path, columns=columns, convert_categoricals=False)
        tracker = tracker.rename(columns=str.upper)
        tracker["HHID"] = _coerce_identifier(tracker["HHID"])
        tracker["PN"] = _coerce_identifier(tracker["PN"])
        tracker["STRATUM"] = _coerce_identifier(tracker["STRATUM"])
        return tracker

    def _wave_member_path(self, spec: dict[str, object]) -> Path:
        inner_zip = self._extract_member(
            self.raw_dir / str(spec["core_zip"]),
            str(spec["stata_zip"]),
            cache_prefix=str(spec["wave_id"]),
        )
        return self._extract_member(
            inner_zip,
            str(spec["cognition_member"]),
            cache_prefix=f"{spec['wave_id']}_stata",
        )

    def _load_wave(self, spec: dict[str, object], tracker: pd.DataFrame) -> pd.DataFrame:
        path = self._wave_member_path(spec)
        columns = [
            str(spec["hhid"]),
            str(spec["pn"]),
            *[str(value) for value in spec["immediate"]],
            *[str(value) for value in spec["delayed"]],
            str(spec["total_cognition"]),
            *[str(value) for value in spec["serial_7s"]],
        ]
        numeracy_col = spec.get("numeracy")
        if numeracy_col:
            columns.append(str(numeracy_col))
        wave = pd.read_stata(path, columns=columns, convert_categoricals=False)
        wave["HHID"] = _coerce_identifier(wave[str(spec["hhid"])])
        wave["PN"] = _coerce_identifier(wave[str(spec["pn"])])
        wave = wave.merge(
            tracker[["HHID", "PN", "SEX", "STRATUM", str(spec["age_col"]), str(spec["weight_col"])]],
            on=["HHID", "PN"],
            how="left",
            validate="m:1",
        )

        immediate = _coerce_score(wave[str(spec["immediate"][0])], min_value=0, max_value=20).combine_first(
            _coerce_score(wave[str(spec["immediate"][1])], min_value=0, max_value=20)
        )
        delayed = _coerce_score(wave[str(spec["delayed"][0])], min_value=0, max_value=20).combine_first(
            _coerce_score(wave[str(spec["delayed"][1])], min_value=0, max_value=20)
        )
        total_cognition = _coerce_score(wave[str(spec["total_cognition"])], min_value=0, max_value=35)
        serial_7s = _serial_7s_score(wave, tuple(str(value) for value in spec["serial_7s"]))
        numeracy = (
            _coerce_score(wave[str(numeracy_col)], min_value=0, max_value=900, missing_values={995, 996, 998, 999})
            if numeracy_col
            else pd.Series(np.nan, index=wave.index, dtype="float64")
        )
        age = _coerce_score(wave[str(spec["age_col"])], min_value=18, max_value=120)
        weights = _coerce_score(wave[str(spec["weight_col"])], min_value=0).mask(lambda s: s <= 0)
        sex = wave["SEX"].map(_normalize_sex)
        person_id = wave["HHID"].fillna(pd.NA).astype("string") + "_" + wave["PN"].fillna(pd.NA).astype("string")

        trait_map = {
            "immediate_recall": immediate,
            "delayed_recall": delayed,
            "serial_7s": serial_7s,
            "total_cognition": total_cognition,
        }
        if numeracy_col:
            trait_map["numeracy"] = numeracy

        frames = []
        for trait_id, scores in trait_map.items():
            frames.append(
                pd.DataFrame(
                    {
                        "source_id": self.dataset_spec.id,
                        "dataset_id": self.dataset_spec.id,
                        "cycle_or_wave": str(spec["label"]),
                        "country": "United States",
                        "grade_or_age_band": pd.NA,
                        "person_id": person_id,
                        "sex_observed": sex,
                        "age": age,
                        "trait_id": trait_id,
                        "score_raw": scores,
                        "weight_main": weights,
                        "design_strata": wave["STRATUM"],
                        "design_psu": wave["HHID"],
                        "design_inference_label": "approximate_household_cluster_bootstrap",
                        "weight_source": str(spec["weight_col"]),
                        "weight_primary_source": str(spec["weight_col"]),
                    }
                )
            )
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=["person_id", "sex_observed", "score_raw"]).reset_index(drop=True)
        return combined

    def load_raw(self) -> pd.DataFrame:
        return self._load_tracker()

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        tracker = self._load_tracker()
        frames = []
        included_waves: list[str] = []
        for spec in HRS_WAVE_SPECS:
            if not (self.raw_dir / str(spec["core_zip"])).exists():
                continue
            frames.append(self._load_wave(spec, tracker))
            included_waves.append(str(spec["wave_id"]))
        if not frames:
            raise FileNotFoundError(
                f"No HRS core distribution zips found under {self.raw_dir}. "
                "Expected files such as h18core.zip, h20core.zip, and h22core.zip."
            )
        data = pd.concat(frames, ignore_index=True)
        provenance = {
            "raw_dir": str(self.raw_dir),
            "generated_utc": utc_timestamp(),
            "included_waves": included_waves,
            "weight_source": "tracker_wave_respondent_weight",
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=data, provenance=provenance)
