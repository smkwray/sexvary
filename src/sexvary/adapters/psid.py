from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import pandas as pd

from ..io import parse_sas_input_columns, read_fixed_width_from_sas_input
from ..utils import utc_timestamp
from .base import BaseAdapter, NormalizedTraitFrame
from .nlsy_local import _normalize_sex


CDS_WAVE_SPECS = (
    {
        "wave_id": "cds_2014",
        "label": "CDS 2014",
        "zip_name": "CDS2014.zip",
        "demog": {
            "txt": "2014/DEMOG14.txt",
            "sas": "2014/DEMOG14.sas",
            "family": "X14YRID",
            "seq": "X14CYPSN",
            "weight": "X14CHWGT",
            "roster_prefix": "X14R",
            "roster_count": 20,
        },
        "assess": {
            "txt": "2014/ASSESS14.txt",
            "sas": "2014/ASSESS14.sas",
            "family": "A14YRID",
            "seq": "A14CYPSN",
            "age": "A14AGEX",
            "age_months": "A14IWAGE",
            "traits": {
                "woodcock_johnson:letter_word": "A14LWSS",
                "woodcock_johnson:passage_comprehension": "A14PCSS",
                "woodcock_johnson:broad_reading": "A14BRSS",
                "woodcock_johnson:applied_problems": "A14APSS",
                "woodcock_johnson:math_reasoning": "A14MRSS",
            },
        },
        "anthro": {
            "txt": "2014/PCGCHILD14.txt",
            "sas": "2014/PCGCHILD14.sas",
            "family": "P14YRID",
            "seq": "P14CYPSN",
            "age": "P14CHAGE",
            "grade": "P14CHGRADE",
            "traits": {
                "height_cm": "P14HW1CM",
                "weight_kg": "P14HW2KG",
                "bmi": "P14BMI",
            },
        },
    },
    {
        "wave_id": "cds_2019",
        "label": "CDS 2019",
        "zip_name": "CDS2019.zip",
        "demog": {
            "txt": "2019/DEMOG2019.txt",
            "sas": "2019/DEMOG2019.sas",
            "family": "X19YRID",
            "seq": "X19CYPSN",
            "weight": "X19CHWGT",
            "roster_prefix": "X19R",
            "roster_count": 20,
        },
        "assess": {
            "txt": "2019/ASSESS2019.txt",
            "sas": "2019/ASSESS2019.sas",
            "family": "A19YRID",
            "seq": "A19CYPSN",
            "age": "A19AGEX",
            "age_months": "A19IWAGE",
            "traits": {
                "woodcock_johnson:letter_word": "A19LWSS",
                "woodcock_johnson:passage_comprehension": "A19PCSS",
                "woodcock_johnson:broad_reading": "A19BRSS",
                "woodcock_johnson:applied_problems": "A19APSS",
                "woodcock_johnson:math_reasoning": "A19MRSS",
            },
        },
        "anthro": {
            "txt": "2019/PCGCHILD2019.txt",
            "sas": "2019/PCGCHILD2019.sas",
            "family": "P19YRID",
            "seq": "P19CYPSN",
            "age": "P19CHAGE",
            "grade": "P19CHGRADE",
            "traits": {
                "height_cm": "P19HW1CM",
                "weight_kg": "P19HW2KG",
                "bmi": "P19BMI",
            },
        },
    },
    {
        "wave_id": "cds_2021",
        "label": "CDS 2021",
        "zip_name": "CDS2021.zip",
        "demog": {
            "txt": "2021/DEMOG2021.txt",
            "sas": "2021/DEMOG2021.sas",
            "family": "X21YRID",
            "seq": "X21CYPSN",
            "weight": "X21CHWGT",
            "roster_prefix": "X21R",
            "roster_count": 20,
        },
        "anthro": {
            "txt": "2021/PCGCHILD2021.txt",
            "sas": "2021/PCGCHILD2021.sas",
            "family": "P21YRID",
            "seq": "P21CYPSN",
            "age": "P21CHAGE",
            "grade": "P21CHGRADE",
            "traits": {
                "height_cm": "P21HW1CM",
                "weight_kg": "P21HW2KG",
                "bmi": "P21BMI",
            },
        },
    },
)

TAS_WAVE_SPECS = (
    {
        "wave_id": "tas_2017",
        "label": "TAS 2017",
        "zip_name": "TA2017.zip",
        "txt": "TA2017.txt",
        "sas": "TA2017.sas",
        "family": "TA170003",
        "seq": "TA170004",
        "sex": "TA170138",
        "weight": "TA171987",
        "weight_fallbacks": ["TA171989", "TA171988"],
        "traits": {
            "sat_critical_reading": "TA170787",
            "sat_math": "TA170788",
            "act_composite": "TA170789",
            "weight_kg": "TA171787",
            "height_cm": ("TA171789", "TA171790"),
        },
    },
    {
        "wave_id": "tas_2019",
        "label": "TAS 2019",
        "zip_name": "TA2019.zip",
        "txt": "TA2019.txt",
        "sas": "TA2019.sas",
        "family": "TA190003",
        "seq": "TA190004",
        "sex": "TA190180",
        "weight": "TA192199",
        "weight_fallbacks": ["TA192202", "TA192201", "TA192200"],
        "traits": {
            "sat_critical_reading": "TA190924",
            "sat_math": "TA190925",
            "act_composite": "TA190926",
            "weight_kg": "TA191949",
            "height_cm": ("TA191951", "TA191952"),
        },
    },
    {
        "wave_id": "tas_2021",
        "label": "TAS 2021",
        "zip_name": "TA2021.zip",
        "txt": "TA2021.txt",
        "sas": "TA2021.sas",
        "family": "TA210003",
        "seq": "TA210004",
        "sex": "TA210175",
        "weight": "TA212394",
        "weight_fallbacks": ["TA212395"],
        "traits": {
            "sat_critical_reading": "TA210961",
            "sat_math": "TA210962",
            "act_composite": "TA210963",
            "weight_kg": "TA212063",
            "height_cm": ("TA212065", "TA212066"),
        },
    },
    {
        "wave_id": "tas_2023",
        "label": "TAS 2023",
        "zip_name": "TA2023.zip",
        "txt": "TA2023.txt",
        "sas": "TA2023.sas",
        "family": "TA230003",
        "seq": "TA230004",
        "sex": "TA230171",
        "weight": "TA232404",
        "weight_fallbacks": ["TA232405"],
        "traits": {
            "sat_critical_reading": "TA230997",
            "sat_math": "TA230998",
            "act_composite": "TA230999",
            "weight_kg": "TA232116",
            "height_cm": ("TA232118", "TA232119"),
        },
    },
)


def _coerce_numeric(series: pd.Series, *, min_value: float | None = None, max_value: float | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    if min_value is not None:
        numeric = numeric.mask(numeric < min_value)
    if max_value is not None:
        numeric = numeric.mask(numeric > max_value)
    return numeric


def _coerce_numeric_with_missing(
    series: pd.Series,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    missing_values: set[float] | None = None,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if missing_values:
        numeric = numeric.mask(numeric.isin(missing_values))
    return _coerce_numeric(numeric, min_value=min_value, max_value=max_value)


def _pounds_to_kg(series: pd.Series) -> pd.Series:
    pounds = _coerce_numeric_with_missing(series, min_value=40, max_value=700, missing_values={0, 998, 999})
    return pounds * 0.45359237


def _feet_inches_to_cm(feet: pd.Series, inches: pd.Series) -> pd.Series:
    feet_numeric = _coerce_numeric_with_missing(feet, min_value=3, max_value=8, missing_values={0, 98, 99})
    inches_numeric = _coerce_numeric_with_missing(inches, min_value=0, max_value=11, missing_values={98, 99})
    total_inches = (feet_numeric * 12.0) + inches_numeric
    total_inches = total_inches.mask(feet_numeric.isna())
    return total_inches * 2.54


class PSIDAdapter(BaseAdapter):
    """Adapter for PSID CDS/TAS public files.

    First live pass focuses on the self-contained CDS public waves that expose
    clear child weights plus harmonizable Woodcock-Johnson and anthropometry
    measures directly in the public release.
    """

    def __init__(self, dataset_spec, raw_path: str | Path):
        super().__init__(dataset_spec=dataset_spec, raw_path=raw_path)
        self.raw_dir = self.raw_path if self.raw_path.is_dir() else self.raw_path.parent
        self.cache_dir = self.raw_dir / ".psid_extracted"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_raw(self) -> pd.DataFrame:
        rows = []
        for spec in CDS_WAVE_SPECS:
            zip_path = self.raw_dir / spec["zip_name"]
            rows.append(
                {
                    "wave_id": spec["wave_id"],
                    "label": spec["label"],
                    "zip_name": spec["zip_name"],
                    "available": zip_path.exists(),
                }
            )
        return pd.DataFrame(rows)

    def _extract_member(self, zip_name: str, member: str) -> Path:
        zip_path = self.raw_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Required PSID archive not found: {zip_path}")
        target = self.cache_dir / f"{Path(zip_name).stem}__{Path(member).name}"
        if target.exists():
            return target
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return target

    @staticmethod
    def _roster_columns(prefix: str, count: int) -> list[str]:
        out: list[str] = []
        for idx in range(1, count + 1):
            tag = f"{idx:02d}"
            out.extend([f"{prefix}{tag}SEX", f"{prefix}{tag}AGE"])
        return out

    @staticmethod
    def _person_id(family: pd.Series, seq: pd.Series) -> pd.Series:
        fam = pd.to_numeric(family, errors="coerce")
        num = pd.to_numeric(seq, errors="coerce")
        return pd.Series(
            [
                f"{int(f):05d}_{int(s):02d}" if pd.notna(f) and pd.notna(s) else pd.NA
                for f, s in zip(fam, num, strict=False)
            ],
            index=family.index,
            dtype="string",
        )

    @staticmethod
    def _resolve_roster_series(df: pd.DataFrame, *, seq_col: str, prefix: str, suffix: str) -> pd.Series:
        def _lookup(row: pd.Series):
            seq = pd.to_numeric(row.get(seq_col), errors="coerce")
            if pd.isna(seq):
                return pd.NA
            col = f"{prefix}{int(seq):02d}{suffix}"
            return row.get(col, pd.NA)

        return df.apply(_lookup, axis=1)

    def _load_demog(self, spec: dict) -> pd.DataFrame:
        demog_spec = spec["demog"]
        sas_path = self._extract_member(spec["zip_name"], demog_spec["sas"])
        txt_path = self._extract_member(spec["zip_name"], demog_spec["txt"])
        available = {col.name for col in parse_sas_input_columns(sas_path)}
        usecols = [
            demog_spec["family"],
            demog_spec["seq"],
            demog_spec["weight"],
            *[
                col
                for col in self._roster_columns(demog_spec["roster_prefix"], demog_spec["roster_count"])
                if col in available
            ],
        ]
        demog = read_fixed_width_from_sas_input(
            txt_path,
            sas_path=sas_path,
            usecols=usecols,
        ).copy()
        demog["person_id"] = self._person_id(demog[demog_spec["family"]], demog[demog_spec["seq"]])
        demog["sex_observed"] = self._resolve_roster_series(
            demog,
            seq_col=demog_spec["seq"],
            prefix=demog_spec["roster_prefix"],
            suffix="SEX",
        ).map(_normalize_sex)
        demog["age_demog"] = _coerce_numeric(
            self._resolve_roster_series(
                demog,
                seq_col=demog_spec["seq"],
                prefix=demog_spec["roster_prefix"],
                suffix="AGE",
            ),
            min_value=0,
            max_value=30,
        )
        demog["weight_main"] = _coerce_numeric(demog[demog_spec["weight"]], min_value=0)
        return demog[
            [
                demog_spec["family"],
                demog_spec["seq"],
                "person_id",
                "sex_observed",
                "age_demog",
                "weight_main",
            ]
        ].copy()

    def _load_assessment_frame(self, spec: dict, demog: pd.DataFrame) -> pd.DataFrame:
        assess = spec.get("assess")
        if not assess:
            return pd.DataFrame()
        usecols = [assess["family"], assess["seq"], assess["age"], assess["age_months"], *assess["traits"].values()]
        frame = read_fixed_width_from_sas_input(
            self._extract_member(spec["zip_name"], assess["txt"]),
            sas_path=self._extract_member(spec["zip_name"], assess["sas"]),
            usecols=usecols,
        ).copy()
        merged = frame.merge(
            demog,
            left_on=[assess["family"], assess["seq"]],
            right_on=[spec["demog"]["family"], spec["demog"]["seq"]],
            how="left",
        )
        age_years = _coerce_numeric(merged[assess["age"]], min_value=0, max_value=30)
        if age_years.isna().all():
            age_months = _coerce_numeric(merged[assess["age_months"]], min_value=0, max_value=360)
            age_years = age_months / 12.0

        long_frames: list[pd.DataFrame] = []
        for trait_id, column in assess["traits"].items():
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": spec["label"],
                    "country": "United States",
                    "grade_or_age_band": pd.Series(pd.NA, index=merged.index, dtype="string"),
                    "person_id": merged["person_id"],
                    "sex_observed": merged["sex_observed"],
                    "age": age_years,
                    "trait_id": trait_id,
                    "score_raw": _coerce_numeric(merged[column], min_value=1, max_value=300),
                    "weight_main": merged["weight_main"],
                }
            )
            long_frames.append(out)
        return pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

    def _load_anthro_frame(self, spec: dict, demog: pd.DataFrame) -> pd.DataFrame:
        anthro = spec.get("anthro")
        if not anthro:
            return pd.DataFrame()
        usecols = [anthro["family"], anthro["seq"], anthro["age"], anthro["grade"], *anthro["traits"].values()]
        frame = read_fixed_width_from_sas_input(
            self._extract_member(spec["zip_name"], anthro["txt"]),
            sas_path=self._extract_member(spec["zip_name"], anthro["sas"]),
            usecols=usecols,
        ).copy()
        merged = frame.merge(
            demog,
            left_on=[anthro["family"], anthro["seq"]],
            right_on=[spec["demog"]["family"], spec["demog"]["seq"]],
            how="left",
        )
        age_years = _coerce_numeric(merged[anthro["age"]], min_value=0, max_value=30).fillna(merged["age_demog"])

        trait_bounds = {
            "height_cm": (40, 250),
            "weight_kg": (2, 300),
            "bmi": (5, 80),
        }
        long_frames: list[pd.DataFrame] = []
        for trait_id, column in anthro["traits"].items():
            min_value, max_value = trait_bounds[trait_id]
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": spec["label"],
                    "country": "United States",
                    "grade_or_age_band": pd.Series(pd.NA, index=merged.index, dtype="string"),
                    "person_id": merged["person_id"],
                    "sex_observed": merged["sex_observed"],
                    "age": age_years,
                    "trait_id": trait_id,
                    "score_raw": _coerce_numeric(merged[column], min_value=min_value, max_value=max_value),
                    "weight_main": merged["weight_main"],
                }
            )
            long_frames.append(out)
        return pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

    def _load_tas_frame(self, spec: dict) -> pd.DataFrame:
        sas_path = self._extract_member(spec["zip_name"], spec["sas"])
        txt_path = self._extract_member(spec["zip_name"], spec["txt"])
        weight_columns = [spec["weight"], *spec.get("weight_fallbacks", [])]
        usecols: list[str] = [spec["family"], spec["seq"], spec["sex"], *weight_columns]
        for value in spec["traits"].values():
            if isinstance(value, tuple):
                usecols.extend(value)
            else:
                usecols.append(value)
        available = {column.name for column in parse_sas_input_columns(sas_path)}
        available_usecols = sorted({column for column in usecols if column in available})
        frame = read_fixed_width_from_sas_input(txt_path, sas_path=sas_path, usecols=available_usecols).copy()
        frame["person_id"] = self._person_id(frame[spec["family"]], frame[spec["seq"]])
        frame["sex_observed"] = frame[spec["sex"]].map(_normalize_sex)
        present_weight_columns = [column for column in weight_columns if column in frame.columns]
        weight_series = {
            column: _coerce_numeric_with_missing(frame[column], min_value=0, missing_values={0})
            for column in present_weight_columns
        }
        frame["weight_main"] = weight_series[spec["weight"]]
        frame["weight_source"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["weight_primary_source"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        for column in present_weight_columns:
            candidate = weight_series[column]
            use_mask = frame["weight_main"].isna() & candidate.notna()
            if column == spec["weight"]:
                use_mask = candidate.notna()
            frame.loc[use_mask, "weight_main"] = candidate[use_mask]
            frame.loc[use_mask, "weight_source"] = column
            frame.loc[use_mask, "weight_primary_source"] = spec["weight"] if column == spec["weight"] else pd.NA

        transformed: dict[str, pd.Series] = {
            "sat_critical_reading": _coerce_numeric_with_missing(
                frame[spec["traits"]["sat_critical_reading"]],
                min_value=200,
                max_value=800,
                missing_values={0, 998, 999},
            ),
            "sat_math": _coerce_numeric_with_missing(
                frame[spec["traits"]["sat_math"]],
                min_value=200,
                max_value=800,
                missing_values={0, 998, 999},
            ),
            "act_composite": _coerce_numeric_with_missing(
                frame[spec["traits"]["act_composite"]],
                min_value=1,
                max_value=36,
                missing_values={0, 98, 99},
            ),
            "weight_kg": _pounds_to_kg(frame[spec["traits"]["weight_kg"]]),
        }
        height_feet, height_inches = spec["traits"]["height_cm"]
        transformed["height_cm"] = _feet_inches_to_cm(frame[height_feet], frame[height_inches])

        weight_m = transformed["weight_kg"]
        height_m = transformed["height_cm"] / 100.0
        bmi = weight_m / (height_m**2)
        transformed["bmi"] = bmi.mask((bmi < 5) | (bmi > 80))

        long_frames: list[pd.DataFrame] = []
        for trait_id, series in transformed.items():
            out = pd.DataFrame(
                {
                    "source_id": self.dataset_spec.id,
                    "dataset_id": self.dataset_spec.id,
                    "cycle_or_wave": spec["label"],
                    "country": "United States",
                    "grade_or_age_band": pd.Series("all_ages", index=frame.index, dtype="string"),
                    "person_id": frame["person_id"],
                    "sex_observed": frame["sex_observed"],
                    "age": pd.Series(pd.NA, index=frame.index, dtype="Float64"),
                    "trait_id": trait_id,
                    "score_raw": series,
                    "weight_main": frame["weight_main"],
                    "weight_source": frame["weight_source"],
                    "weight_primary_source": frame["weight_primary_source"],
                }
            )
            long_frames.append(out)
        return pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        available_specs = [spec for spec in CDS_WAVE_SPECS if (self.raw_dir / spec["zip_name"]).exists()]
        available_tas_specs = [spec for spec in TAS_WAVE_SPECS if (self.raw_dir / spec["zip_name"]).exists()]
        if not available_specs and not available_tas_specs:
            raise FileNotFoundError(
                f"No supported PSID CDS/TAS archives found in {self.raw_dir}. "
                "Expected at least one of: "
                + ", ".join(spec["zip_name"] for spec in [*CDS_WAVE_SPECS, *TAS_WAVE_SPECS])
            )

        long_frames: list[pd.DataFrame] = []
        for spec in available_specs:
            demog = self._load_demog(spec)
            assessment = self._load_assessment_frame(spec, demog)
            if not assessment.empty:
                long_frames.append(assessment)
            anthro = self._load_anthro_frame(spec, demog)
            if not anthro.empty:
                long_frames.append(anthro)
        for spec in available_tas_specs:
            tas = self._load_tas_frame(spec)
            if not tas.empty:
                long_frames.append(tas)

        if not long_frames:
            raise ValueError("Supported PSID archives were present, but no harmonizable CDS/TAS traits were extracted.")

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df = long_df.dropna(subset=["person_id", "sex_observed", "score_raw"])
        long_df = long_df[long_df["weight_main"].fillna(0) > 0].copy()
        provenance = {
            "raw_path": str(self.raw_path),
            "raw_dir": str(self.raw_dir),
            "included_waves": [spec["wave_id"] for spec in [*available_specs, *available_tas_specs]],
            "skipped_archives": sorted(
                path.name
                for path in self.raw_dir.glob("*.zip")
                if path.name not in {spec["zip_name"] for spec in [*available_specs, *available_tas_specs]}
            ),
            "created_utc": utc_timestamp(),
        }
        return NormalizedTraitFrame(dataset_id=self.dataset_spec.id, data=long_df.reset_index(drop=True), provenance=provenance)
