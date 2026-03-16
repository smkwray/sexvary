from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
import yaml

from ..io import read_fixed_width_from_sas_input, read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame
from .nlsy_local import LocalWideTableAdapter, _normalize_sex


NUMERIC_MISSING_SENTINELS = {float(code) for code in range(-9, 0)}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(numeric.isin(NUMERIC_MISSING_SENTINELS))


def _coerce_string(series: pd.Series) -> pd.Series:
    stringy = series.astype("string").str.strip()
    missing_codes = {str(int(code)) for code in NUMERIC_MISSING_SENTINELS}
    return stringy.mask(stringy.isin(missing_codes) | stringy.eq(""))


@dataclass(frozen=True)
class ECLSKRepeatedWaveMapping:
    dataset_id: str
    source_format: str
    person_id: str
    sex: str
    country: str = "United States"
    setup_sas: str = "ECLSK2011_K5PUF.sas"
    waves: dict[str, dict[str, Any]] = field(default_factory=dict)
    traits: dict[str, dict[str, str]] = field(default_factory=dict)
    value_maps: dict[str, Any] | None = None


@dataclass(frozen=True)
class HSLSRepeatedWaveMapping:
    dataset_id: str
    source_format: str
    person_id: str
    sex: str
    country: str = "United States"
    archive_member: str | None = None
    design_strata: str | None = None
    design_psu: str | None = None
    waves: dict[str, dict[str, Any]] = field(default_factory=dict)
    traits: dict[str, dict[str, str]] = field(default_factory=dict)
    value_maps: dict[str, Any] | None = None


class ECLSK2011Adapter(BaseAdapter):
    def __init__(self, dataset_spec, raw_path: str | Path, mapping_path: str | Path):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.mapping_path = Path(mapping_path)
        self.mapping = self.load_mapping(self.mapping_path)

    @staticmethod
    def load_mapping(path: str | Path) -> ECLSKRepeatedWaveMapping:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return ECLSKRepeatedWaveMapping(**raw)

    def _resolve_data_path(self) -> Path:
        path = self.raw_path
        if path.is_dir():
            dat_files = sorted(path.glob("*.dat"))
            if dat_files:
                return dat_files[0]
            zip_files = sorted(path.glob("*.zip"))
            if zip_files:
                return self._extract_dat_from_zip(zip_files[0])
            raise FileNotFoundError(f"No .dat or .zip ECLS file found in {path}.")
        if path.suffix.lower() == ".dat":
            return path
        if path.suffix.lower() == ".zip":
            return self._extract_dat_from_zip(path)
        raise ValueError(f"Unsupported ECLS raw path: {path}")

    def _extract_dat_from_zip(self, path: Path) -> Path:
        try:
            with zipfile.ZipFile(path) as zf:
                members = [name for name in zf.namelist() if name.lower().endswith(".dat")]
                if not members:
                    raise FileNotFoundError(f"No .dat member found in {path}.")
                member = members[0]
                target = path.parent / Path(member).name
                if not target.exists():
                    zf.extract(member, path.parent)
                    extracted = path.parent / member
                    if extracted != target and extracted.exists():
                        extracted.replace(target)
                return target
        except zipfile.BadZipFile as exc:
            raise ValueError(f"ECLS archive is not a valid zip yet: {path}") from exc

    def _resolve_sas_path(self) -> Path:
        base = self._resolve_data_path().parent
        sas_path = base / self.mapping.setup_sas
        if not sas_path.exists():
            raise FileNotFoundError(
                f"ECLS setup file not found: {sas_path}. Download {self.mapping.setup_sas} into {base}."
            )
        return sas_path

    def _required_columns(self) -> list[str]:
        required = {self.mapping.person_id, self.mapping.sex}
        for wave_spec in self.mapping.waves.values():
            for key in ("weight", "age", "grade_column", "strata", "psu"):
                value = wave_spec.get(key)
                if value:
                    required.add(str(value))
        for wave_columns in self.mapping.traits.values():
            required.update(str(col) for col in wave_columns.values())
        return sorted(required)

    def load_raw(self) -> pd.DataFrame:
        return read_fixed_width_from_sas_input(
            self._resolve_data_path(),
            sas_path=self._resolve_sas_path(),
            usecols=self._required_columns(),
        )

    def _grade_series_for_wave(self, df: pd.DataFrame, wave_spec: dict[str, Any]) -> pd.Series:
        grade_column = wave_spec.get("grade_column")
        if grade_column and grade_column in df.columns:
            grade_map = {str(key): value for key, value in (wave_spec.get("grade_value_map") or {}).items()}
            raw = df[grade_column].astype("string")
            if grade_map:
                mapped = raw.map(lambda value: grade_map.get(str(value)) if pd.notna(value) else pd.NA)
                return mapped.fillna(wave_spec.get("grade_band", pd.NA))
            return raw
        return pd.Series(wave_spec.get("grade_band", pd.NA), index=df.index, dtype="object")

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        sex_map = (self.mapping.value_maps or {}).get("sex")

        long_frames: list[pd.DataFrame] = []
        for wave_id, wave_spec in self.mapping.waves.items():
            age = _coerce_numeric(df[wave_spec["age"]]) if wave_spec.get("age") in df.columns else pd.Series(pd.NA, index=df.index)
            weight = (
                _coerce_numeric(df[wave_spec["weight"]])
                if wave_spec.get("weight") in df.columns
                else pd.Series(1.0, index=df.index)
            )
            design_strata = _coerce_string(df[wave_spec["strata"]]) if wave_spec.get("strata") in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
            design_psu = _coerce_string(df[wave_spec["psu"]]) if wave_spec.get("psu") in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
            grade_band = self._grade_series_for_wave(df, wave_spec)

            for trait_id, trait_columns in self.mapping.traits.items():
                column = trait_columns.get(str(wave_id))
                if not column or column not in df.columns:
                    continue
                out = pd.DataFrame(
                    {
                        "source_id": self.dataset_spec.id,
                        "dataset_id": self.dataset_spec.id,
                        "cycle_or_wave": wave_spec["label"],
                        "country": self.mapping.country,
                        "grade_or_age_band": grade_band,
                        "design_strata": design_strata,
                        "design_psu": design_psu,
                        "person_id": df[self.mapping.person_id].astype("string").str.strip(),
                        "sex_observed": df[self.mapping.sex].map(lambda value: _normalize_sex(value, sex_map)),
                        "age": age,
                        "trait_id": trait_id,
                        "score_raw": _coerce_numeric(df[column]),
                        "weight_main": weight,
                    }
                )
                long_frames.append(out)

        if not long_frames:
            raise ValueError("No repeated-wave ECLS trait columns were found in the raw dataset.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "person_id"])
        provenance = {
            "raw_path": str(self.raw_path),
            "resolved_data_path": str(self._resolve_data_path()),
            "mapping_path": str(self.mapping_path),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)


class HSLS2009Adapter(BaseAdapter):
    def __init__(self, dataset_spec, raw_path: str | Path, mapping_path: str | Path):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.mapping_path = Path(mapping_path)
        self.mapping = self.load_mapping(self.mapping_path)

    @staticmethod
    def load_mapping(path: str | Path) -> HSLSRepeatedWaveMapping:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return HSLSRepeatedWaveMapping(**raw)

    def _resolve_csv_path(self) -> Path:
        path = self.raw_path
        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            if csv_files:
                return csv_files[0]
            zip_files = sorted(path.glob("*.zip"))
            if zip_files:
                return self._extract_csv_from_zip(zip_files[0])
            raise FileNotFoundError(f"No .csv or .zip HSLS file found in {path}.")
        if path.suffix.lower() == ".csv":
            return path
        if path.suffix.lower() == ".zip":
            return self._extract_csv_from_zip(path)
        raise ValueError(f"Unsupported HSLS raw path: {path}")

    def _extract_csv_from_zip(self, path: Path) -> Path:
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError(f"No .csv member found in {path}.")
            preferred = self.mapping.archive_member
            member = preferred if preferred in members else next((name for name in members if "student" in name.lower()), members[0])
            target = path.parent / Path(member).name
            if not target.exists():
                zf.extract(member, path.parent)
                extracted = path.parent / member
                if extracted != target and extracted.exists():
                    extracted.replace(target)
            return target

    def _required_columns(self) -> list[str]:
        required = {self.mapping.person_id, self.mapping.sex}
        if self.mapping.design_strata:
            required.add(self.mapping.design_strata)
        if self.mapping.design_psu:
            required.add(self.mapping.design_psu)
        for wave_spec in self.mapping.waves.values():
            for key in ("weight", "age", "grade_column"):
                value = wave_spec.get(key)
                if value:
                    required.add(str(value))
            required.update(self._raw_replicate_weight_columns(wave_spec))
        for wave_columns in self.mapping.traits.values():
            required.update(str(col) for col in wave_columns.values())
        return sorted(required)

    @staticmethod
    def _raw_replicate_weight_columns(wave_spec: dict[str, Any]) -> list[str]:
        prefix = wave_spec.get("replicate_weight_prefix")
        if not prefix:
            return []
        count = int(wave_spec.get("replicate_weight_count", 0) or 0)
        digits = int(wave_spec.get("replicate_weight_digits", 3) or 3)
        return [f"{prefix}{idx:0{digits}d}" for idx in range(1, count + 1)]

    def load_raw(self) -> pd.DataFrame:
        return read_table(self._resolve_csv_path(), usecols=self._required_columns())

    def _grade_series_for_wave(self, df: pd.DataFrame, wave_spec: dict[str, Any]) -> pd.Series:
        grade_column = wave_spec.get("grade_column")
        if grade_column and grade_column in df.columns:
            grade_map = {str(key): value for key, value in (wave_spec.get("grade_value_map") or {}).items()}
            raw = df[grade_column].astype("string")
            if grade_map:
                mapped = raw.map(lambda value: grade_map.get(str(value)) if pd.notna(value) else pd.NA)
                return mapped.fillna(wave_spec.get("grade_band", pd.NA))
            return raw
        return pd.Series(wave_spec.get("grade_band", pd.NA), index=df.index, dtype="object")

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        sex_map = (self.mapping.value_maps or {}).get("sex")
        design_strata = (
            _coerce_string(df[self.mapping.design_strata])
            if self.mapping.design_strata and self.mapping.design_strata in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string")
        )
        design_psu = (
            _coerce_string(df[self.mapping.design_psu])
            if self.mapping.design_psu and self.mapping.design_psu in df.columns
            else pd.Series(pd.NA, index=df.index, dtype="string")
        )

        long_frames: list[pd.DataFrame] = []
        for wave_id, wave_spec in self.mapping.waves.items():
            age = _coerce_numeric(df[wave_spec["age"]]) if wave_spec.get("age") in df.columns else pd.Series(pd.NA, index=df.index)
            weight = _coerce_numeric(df[wave_spec["weight"]]) if wave_spec.get("weight") in df.columns else pd.Series(1.0, index=df.index)
            grade_band = self._grade_series_for_wave(df, wave_spec)
            replicate_weight_cols = self._raw_replicate_weight_columns(wave_spec)
            replicate_weight_map = {
                f"replicate_weight_{idx:03d}": _coerce_numeric(df[col])
                for idx, col in enumerate(replicate_weight_cols, start=1)
                if col in df.columns
            }

            for trait_id, trait_columns in self.mapping.traits.items():
                column = trait_columns.get(str(wave_id))
                if not column or column not in df.columns:
                    continue
                out_columns = {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": wave_spec["label"],
                    "country": self.mapping.country,
                    "grade_or_age_band": grade_band,
                    "design_strata": design_strata,
                    "design_psu": design_psu,
                    "person_id": _coerce_string(df[self.mapping.person_id]),
                    "sex_observed": df[self.mapping.sex].map(lambda value: _normalize_sex(value, sex_map)),
                    "age": age,
                    "trait_id": trait_id,
                    "score_raw": _coerce_numeric(df[column]),
                    "weight_main": weight,
                    "replicate_method": wave_spec.get("replicate_method", pd.NA),
                    "replicate_fay": float(wave_spec.get("replicate_fay", 0.0) or 0.0),
                }
                out_columns.update(replicate_weight_map)
                out = pd.DataFrame(out_columns)
                long_frames.append(out)

        if not long_frames:
            raise ValueError("No repeated-wave HSLS trait columns were found in the raw dataset.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed", "person_id"])
        provenance = {
            "raw_path": str(self.raw_path),
            "resolved_data_path": str(self._resolve_csv_path()),
            "mapping_path": str(self.mapping_path),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)


class NCESSchoolAdapter(BaseAdapter):
    """Dispatch NCES school datasets to the right ingest path."""

    def __init__(self, dataset_spec, raw_path: str | Path, mapping_path: str | Path):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.mapping_path = Path(mapping_path)
        mapping_raw = yaml.safe_load(self.mapping_path.read_text(encoding="utf-8")) or {}
        source_format = str(mapping_raw.get("source_format", ""))
        if dataset_spec.id == "ecls_k_2011" and source_format == "ecls_k2011_fixed_width_sas":
            self.impl = ECLSK2011Adapter(dataset_spec=dataset_spec, raw_path=raw_path, mapping_path=mapping_path)
        elif dataset_spec.id == "hsls_2009" and source_format == "hsls_repeated_wave_csv":
            self.impl = HSLS2009Adapter(dataset_spec=dataset_spec, raw_path=raw_path, mapping_path=mapping_path)
        else:
            self.impl = LocalWideTableAdapter(dataset_spec=dataset_spec, raw_path=raw_path, mapping_path=mapping_path)

    def load_raw(self) -> pd.DataFrame:
        return self.impl.load_raw()

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        return self.impl.to_long_person_trait()
