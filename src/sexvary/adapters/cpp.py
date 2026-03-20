from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame


STRING_ID_DTYPES = {
    "case_id": "string",
    "mother_id": "string",
    "id_1": "string",
    "id_2": "string",
}


def _normalize_cpp_sex(value: Any) -> str | None:
    if pd.isna(value):
        return None
    sval = str(value).strip().lower()
    if sval in {"1", "1.0", "male", "m"}:
        return "male"
    if sval in {"2", "2.0", "female", "f"}:
        return "female"
    return None


def _as_string_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _first_existing_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_grade_or_age_band(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


@dataclass(frozen=True)
class CPPModeConfig:
    files: dict[str, str]
    id_columns: dict[str, list[str]]
    weight_columns: list[str]
    waves: dict[str, dict[str, Any]] | None = None
    measure_columns: dict[str, list[str]] | None = None
    birthweight: dict[str, Any] | None = None


class CPPAdapter(BaseAdapter):
    """Mapping-driven adapter for the Collaborative Perinatal Project public release."""

    def __init__(
        self,
        dataset_spec,
        raw_path: str | Path,
        *,
        mapping_path: str | Path,
        mode: str = "cpp_core",
    ):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.mapping_path = Path(mapping_path)
        self.mode = mode
        self.mapping = self._load_mapping(self.mapping_path, mode)

    @staticmethod
    def _load_mapping(path: str | Path, mode: str) -> CPPModeConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        if mode not in payload:
            raise KeyError(f"Mode `{mode}` not found in mapping file: {path}")
        raw = payload[mode]
        if not isinstance(raw, dict):
            raise TypeError(f"Mapping section `{mode}` must be a dictionary.")
        return CPPModeConfig(**raw)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        return read_table(path, dtype=STRING_ID_DTYPES, low_memory=False)

    def _resolve_file(self, logical_name: str) -> Path:
        filename = self.mapping.files.get(logical_name)
        if not filename:
            raise KeyError(f"Logical file `{logical_name}` missing from mapping for mode {self.mode}.")
        path = self.raw_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected CPP file not found: {path}")
        return path

    def _ensure_case_id_column(self, df: pd.DataFrame, *, logical_name: str) -> pd.DataFrame:
        case_candidates = self.mapping.id_columns.get("case_id", ["case_id"])
        case_col = _first_existing_column(df, case_candidates)
        if case_col is None:
            raise KeyError(f"Could not resolve case_id in CPP file `{logical_name}`.")
        out = df.rename(columns={case_col: "case_id"}) if case_col != "case_id" else df.copy()
        out["case_id"] = _as_string_id(out["case_id"])
        if "mother_id" in out.columns:
            out["mother_id"] = _as_string_id(out["mother_id"])
        return out

    def _load_core_frame(self) -> pd.DataFrame:
        frames: list[tuple[str, pd.DataFrame]] = []
        for logical_name, filename in self.mapping.files.items():
            path = self.raw_path / filename
            if not path.exists():
                continue
            frame = self._ensure_case_id_column(self._read_csv(path), logical_name=logical_name)
            frames.append((logical_name, frame))

        if not frames:
            raise FileNotFoundError(f"No mapped CPP core files were found in {self.raw_path}")

        merged = frames[0][1].copy()
        for logical_name, frame in frames[1:]:
            right = frame.copy()
            keep_cols = ["case_id"] + [col for col in right.columns if col != "case_id" and col not in merged.columns]
            merged = merged.merge(right[keep_cols], on="case_id", how="left")
        return merged

    def _load_growth_frame(self) -> pd.DataFrame:
        growth = self._ensure_case_id_column(self._read_csv(self._resolve_file("growth")), logical_name="growth")
        if {"measure", "value"}.issubset(growth.columns):
            index_cols = ["case_id"]
            if "age_months" in growth.columns:
                index_cols.append("age_months")
            pivoted = growth.pivot_table(index=index_cols, columns="measure", values="value", aggfunc="first").reset_index()
            pivoted.columns.name = None
            if "weight_g" in pivoted.columns and "weight_kg" not in pivoted.columns:
                pivoted["weight_kg"] = _coerce_numeric(pivoted["weight_g"]) / 1000.0
            growth = pivoted

        for logical_name in ("clean", "weights", "birthweight"):
            filename = self.mapping.files.get(logical_name)
            if not filename:
                continue
            path = self.raw_path / filename
            if not path.exists():
                continue
            extra = self._ensure_case_id_column(self._read_csv(path), logical_name=logical_name)
            keep_cols = ["case_id"] + [col for col in extra.columns if col != "case_id" and col not in growth.columns]
            growth = growth.merge(extra[keep_cols], on="case_id", how="left")
        return growth

    def load_raw(self) -> pd.DataFrame:
        if self.mode == "cpp_core":
            return self._load_core_frame()
        if self.mode == "cpp_growth":
            return self._load_growth_frame()
        raise ValueError(f"Unsupported CPP mode: {self.mode}")

    def _resolve_id_columns(self, df: pd.DataFrame) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for logical_name, candidates in self.mapping.id_columns.items():
            out[logical_name] = _first_existing_column(df, candidates)
        return out

    def _weight_series(self, df: pd.DataFrame) -> tuple[pd.Series, str, str | None]:
        return self._weight_series_from_candidates(df, self.mapping.weight_columns)

    def _weight_series_from_candidates(self, df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, str, str | None]:
        weight_col = _first_existing_column(df, candidates)
        primary = candidates[0] if candidates else None
        if weight_col is None:
            return pd.Series(1.0, index=df.index, dtype=float), "unit_weight_fallback", primary
        return _coerce_numeric(df[weight_col]), weight_col, primary

    def _core_to_long(self, df: pd.DataFrame) -> NormalizedTraitFrame:
        ids = self._resolve_id_columns(df)
        case_col = ids.get("case_id")
        sex_col = ids.get("sex")
        mother_col = ids.get("mother_id")
        site_col = ids.get("site")

        if case_col is None or sex_col is None:
            raise KeyError("CPP core mapping must resolve both case_id and sex columns.")

        sex_series = df[sex_col].map(_normalize_cpp_sex)

        long_frames: list[pd.DataFrame] = []
        resolved_trait_columns: dict[str, str] = {}
        resolved_weight_columns: dict[str, str] = {}
        resolved_primary_weight_columns: dict[str, str | None] = {}

        for wave_id, wave_cfg in (self.mapping.waves or {}).items():
            presence_any = [col for col in wave_cfg.get("presence_any", []) if col in df.columns]
            wave_mask = df[presence_any].notna().any(axis=1) if presence_any else pd.Series(True, index=df.index)
            age_value = wave_cfg.get("age")
            age_band = wave_cfg.get("grade_or_age_band", wave_id)
            weight_candidates = list(wave_cfg.get("weight_columns", self.mapping.weight_columns))
            weight_series, weight_source, primary_weight_source = self._weight_series_from_candidates(df, weight_candidates)
            resolved_weight_columns[wave_id] = weight_source
            resolved_primary_weight_columns[wave_id] = primary_weight_source

            for trait_id, candidates in wave_cfg.get("traits", {}).items():
                source_col = _first_existing_column(df, candidates)
                if source_col is None:
                    continue
                resolved_trait_columns[f"{wave_id}:{trait_id}"] = source_col
                out = pd.DataFrame(
                    {
                        "source_id": self.dataset_spec.id,
                        "dataset_id": self.dataset_spec.id,
                        "cycle_or_wave": pd.Series(wave_id, index=df.index, dtype="string"),
                        "country": pd.Series("United States", index=df.index, dtype="string"),
                        "country_id": pd.Series("840", index=df.index, dtype="string"),
                        "grade_or_age_band": pd.Series(_safe_grade_or_age_band(age_band), index=df.index, dtype="string"),
                        "person_id": _as_string_id(df[case_col]),
                        "sex_observed": sex_series.astype("string"),
                        "age": pd.Series(age_value, index=df.index, dtype=float),
                        "trait_id": pd.Series(trait_id, index=df.index, dtype="string"),
                        "score_raw": _coerce_numeric(df[source_col]),
                        "weight_main": weight_series,
                        "weight_source": pd.Series(weight_source, index=df.index, dtype="string"),
                        "weight_primary_source": pd.Series(primary_weight_source, index=df.index, dtype="string"),
                        "source_variable": pd.Series(source_col, index=df.index, dtype="string"),
                    }
                )
                if mother_col is not None:
                    out["family_id"] = _as_string_id(df[mother_col])
                if site_col is not None:
                    out["site"] = df[site_col].astype("string")
                out = out.loc[wave_mask].copy()
                long_frames.append(out)

        if not long_frames:
            raise ValueError("CPP core mapping did not resolve any trait columns. Update the mapping YAML.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed"])
        long_df = long_df[long_df["sex_observed"].isin(["male", "female"])].reset_index(drop=True)

        provenance = {
            "raw_path": str(self.raw_path),
            "mapping_path": str(self.mapping_path),
            "mode": self.mode,
            "loaded_files": self.mapping.files,
            "weight_source": resolved_weight_columns,
            "weight_primary_source": resolved_primary_weight_columns,
            "resolved_trait_columns": resolved_trait_columns,
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)

    def _growth_to_long(self, df: pd.DataFrame) -> NormalizedTraitFrame:
        ids = self._resolve_id_columns(df)
        case_col = ids.get("case_id")
        sex_col = ids.get("sex")
        mother_col = ids.get("mother_id")
        site_col = ids.get("site")

        if case_col is None or sex_col is None:
            raise KeyError("CPP growth mapping must resolve both case_id and sex columns.")

        measure_columns = self.mapping.measure_columns or {}
        wave_col = _first_existing_column(df, measure_columns.get("wave", []))
        age_col = _first_existing_column(df, measure_columns.get("age", []))
        height_col = _first_existing_column(df, measure_columns.get("height_cm", []))
        measure_weight_col = _first_existing_column(df, measure_columns.get("weight_kg", []))

        if wave_col is not None:
            wave_series = df[wave_col].astype("string")
        elif age_col is not None:
            age_wave = _coerce_numeric(df[age_col]).round().astype("Int64").astype("string")
            wave_series = age_wave + pd.Series("m", index=df.index, dtype="string")
        else:
            wave_series = pd.Series("growth_visit", index=df.index, dtype="string")
        age_series = _coerce_numeric(df[age_col]) if age_col is not None else pd.Series(pd.NA, index=df.index, dtype="object")
        weight_series, weight_source, primary_weight_source = self._weight_series(df)
        sex_series = df[sex_col].map(_normalize_cpp_sex)

        long_frames: list[pd.DataFrame] = []
        resolved_trait_columns: dict[str, str] = {}

        def _build_measure_rows(trait_id: str, source_col: str | None) -> None:
            if source_col is None:
                return
            resolved_trait_columns[trait_id] = source_col
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": wave_series,
                    "country": pd.Series("United States", index=df.index, dtype="string"),
                    "country_id": pd.Series("840", index=df.index, dtype="string"),
                    "grade_or_age_band": wave_series,
                    "person_id": _as_string_id(df[case_col]),
                    "sex_observed": sex_series.astype("string"),
                    "age": age_series,
                    "trait_id": pd.Series(trait_id, index=df.index, dtype="string"),
                    "score_raw": _coerce_numeric(df[source_col]),
                    "weight_main": weight_series,
                    "weight_source": pd.Series(weight_source, index=df.index, dtype="string"),
                    "weight_primary_source": pd.Series(primary_weight_source, index=df.index, dtype="string"),
                    "source_variable": pd.Series(source_col, index=df.index, dtype="string"),
                }
            )
            if mother_col is not None:
                out["family_id"] = _as_string_id(df[mother_col])
            if site_col is not None:
                out["site"] = df[site_col].astype("string")
            long_frames.append(out)

        _build_measure_rows("height_cm", height_col)
        _build_measure_rows("weight_kg", measure_weight_col)

        birth_cfg = self.mapping.birthweight or {}
        birth_col = _first_existing_column(df, birth_cfg.get("value_columns", []))
        if birth_col is not None:
            unique_df = df.drop_duplicates(subset=[case_col]).copy()
            unique_weights, birth_weight_source, birth_primary_source = self._weight_series(unique_df)
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": pd.Series(birth_cfg.get("wave", "birth"), index=unique_df.index, dtype="string"),
                    "country": pd.Series("United States", index=unique_df.index, dtype="string"),
                    "country_id": pd.Series("840", index=unique_df.index, dtype="string"),
                    "grade_or_age_band": pd.Series("birth", index=unique_df.index, dtype="string"),
                    "person_id": _as_string_id(unique_df[case_col]),
                    "sex_observed": unique_df[sex_col].map(_normalize_cpp_sex).astype("string"),
                    "age": pd.Series(float(birth_cfg.get("age", 0.0)), index=unique_df.index, dtype=float),
                    "trait_id": pd.Series("birthweight_z", index=unique_df.index, dtype="string"),
                    "score_raw": _coerce_numeric(unique_df[birth_col]),
                    "weight_main": unique_weights,
                    "weight_source": pd.Series(birth_weight_source, index=unique_df.index, dtype="string"),
                    "weight_primary_source": pd.Series(birth_primary_source, index=unique_df.index, dtype="string"),
                    "source_variable": pd.Series(birth_col, index=unique_df.index, dtype="string"),
                }
            )
            if mother_col is not None:
                out["family_id"] = _as_string_id(unique_df[mother_col])
            if site_col is not None:
                out["site"] = unique_df[site_col].astype("string")
            resolved_trait_columns["birthweight_z"] = birth_col
            long_frames.append(out)

        if not long_frames:
            raise ValueError("CPP growth mapping did not resolve any growth or birthweight trait columns.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed"])
        long_df = long_df[long_df["sex_observed"].isin(["male", "female"])].reset_index(drop=True)

        provenance = {
            "raw_path": str(self.raw_path),
            "mapping_path": str(self.mapping_path),
            "mode": self.mode,
            "loaded_files": self.mapping.files,
            "weight_source": weight_source,
            "weight_primary_source": primary_weight_source,
            "resolved_trait_columns": resolved_trait_columns,
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        df = self.load_raw().copy()
        if self.mode == "cpp_core":
            return self._core_to_long(df)
        if self.mode == "cpp_growth":
            return self._growth_to_long(df)
        raise ValueError(f"Unsupported CPP mode: {self.mode}")
