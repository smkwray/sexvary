from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..io import read_table
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame


def _normalize_sex(value: Any, value_map: dict | None = None) -> str | None:
    if pd.isna(value):
        return None
    if value_map:
        male_values = {str(v) for v in value_map.get("male", [])}
        female_values = {str(v) for v in value_map.get("female", [])}
        sval = str(value)
        if sval in male_values:
            return "male"
        if sval in female_values:
            return "female"
    sval = str(value).strip().lower()
    if sval.endswith(".0"):
        sval = sval[:-2]
    if sval in {"m", "male", "1"}:
        return "male"
    if sval in {"f", "female", "2"}:
        return "female"
    return sval


@dataclass(frozen=True)
class LocalWideMapping:
    dataset_id: str
    table_shape: str
    columns: dict[str, str]
    traits: dict[str, Any]
    value_maps: dict[str, Any] | None = None
    weight_fallback_columns: list[str] | None = None


class LocalWideTableAdapter(BaseAdapter):
    """Generic adapter for user-supplied wide local tables.

    This is intentionally simple but already useful for NLSY-style extracts when the user
    can provide a small mapping YAML. Nested trait groups (e.g., ASVAB subtests) are
    flattened into `trait_id:subtrait` identifiers.
    """

    def __init__(self, dataset_spec, raw_path: str | Path, mapping_path: str | Path):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.mapping_path = Path(mapping_path)
        self.mapping = self.load_mapping(self.mapping_path)

    @staticmethod
    def load_mapping(path: str | Path) -> LocalWideMapping:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return LocalWideMapping(**raw)

    def load_raw(self) -> pd.DataFrame:
        return read_table(self.raw_path)

    def _iter_trait_columns(self):
        for trait_id, spec in self.mapping.traits.items():
            if isinstance(spec, dict):
                for subtrait, column in spec.items():
                    yield f"{trait_id}:{subtrait}", column
            else:
                yield trait_id, spec

    def _mapped_series(self, df: pd.DataFrame, key: str, default: Any = pd.NA):
        cols = self.mapping.columns
        if key in cols and cols[key] in df.columns:
            return df[cols[key]]
        return default

    def _mapped_weight_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        primary_weight = str(self.mapping.columns.get("weight", "unit_weight"))
        fallback_columns = [col for col in (self.mapping.weight_fallback_columns or []) if col in df.columns]
        if fallback_columns:
            weights = df[fallback_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            # Use the first strictly positive weight in the configured fallback order.
            fallback = weights.mask(weights <= 0.0).bfill(axis=1).iloc[:, 0].fillna(0.0)
            source = pd.Series(pd.NA, index=df.index, dtype="object")
            for column in fallback_columns:
                use_column = source.isna() & (weights[column] > 0.0)
                source.loc[use_column] = column
            source = source.fillna(primary_weight)
            return pd.DataFrame(
                {
                    "weight_main": fallback,
                    "weight_source": source,
                    "weight_primary_source": primary_weight,
                }
            )
        return pd.DataFrame(
            {
                "weight_main": pd.to_numeric(self._mapped_series(df, "weight", 1.0), errors="coerce"),
                "weight_source": primary_weight,
                "weight_primary_source": primary_weight,
            }
        )

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        if self.mapping.table_shape != "wide":
            raise NotImplementedError("Only wide local tables are supported in this seed adapter.")

        df = self.load_raw().copy()
        cols = self.mapping.columns
        required = [cols["person_id"], cols["sex"]]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Raw file is missing mapped required columns: {missing}")
        weight_frame = self._mapped_weight_frame(df)

        long_frames = []
        for trait_id, col in self._iter_trait_columns():
            if col not in df.columns:
                continue
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": self._mapped_series(df, "cycle_or_wave"),
                    "country": self._mapped_series(df, "country"),
                    "grade_or_age_band": self._mapped_series(df, "grade_or_age_band"),
                    "design_strata": self._mapped_series(df, "design_strata"),
                    "design_psu": self._mapped_series(df, "design_psu"),
                    "person_id": df[cols["person_id"]],
                    "sex_observed": df[cols["sex"]].map(lambda x: _normalize_sex(x, (self.mapping.value_maps or {}).get("sex"))),
                    "age": self._mapped_series(df, "age"),
                    "trait_id": trait_id,
                    "score_raw": pd.to_numeric(df[col], errors="coerce"),
                    "weight_main": weight_frame["weight_main"],
                    "weight_source": weight_frame["weight_source"],
                    "weight_primary_source": weight_frame["weight_primary_source"],
                }
            )
            long_frames.append(out)

        if not long_frames:
            raise ValueError("No mapped trait columns were found in the raw dataset.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["score_raw", "sex_observed"])
        provenance = {
            "raw_path": str(self.raw_path),
            "mapping_path": str(self.mapping_path),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df, provenance=provenance)
