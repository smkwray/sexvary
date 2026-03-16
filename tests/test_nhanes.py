from pathlib import Path

import pandas as pd
import pyreadstat

from sexvary.adapters.nhanes import NHANESAdapter, _normalize_nhanes_sex
from sexvary.config import build_registry
from sexvary.estimation import EstimationConfig, estimate_dataset_cells


def _write_xpt(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_xport(df, path)
    return path


def _demo_fixture(seqn_start: int, *, cycle_code: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SEQN": seqn_start + 1, "RIAGENDR": 1, "RIDAGEYR": 62, "WTMEC2YR": 1.0, "SDMVPSU": 1, "SDMVSTRA": 10},
            {"SEQN": seqn_start + 2, "RIAGENDR": 2, "RIDAGEYR": 61, "WTMEC2YR": 1.0, "SDMVPSU": 1, "SDMVSTRA": 10},
            {"SEQN": seqn_start + 3, "RIAGENDR": 1, "RIDAGEYR": 64, "WTMEC2YR": 1.1, "SDMVPSU": 2, "SDMVSTRA": 10},
            {"SEQN": seqn_start + 4, "RIAGENDR": 2, "RIDAGEYR": 65, "WTMEC2YR": 1.1, "SDMVPSU": 2, "SDMVSTRA": 10},
            {"SEQN": seqn_start + 5, "RIAGENDR": 1, "RIDAGEYR": 66, "WTMEC2YR": 1.0, "SDMVPSU": 1, "SDMVSTRA": 11},
            {"SEQN": seqn_start + 6, "RIAGENDR": 2, "RIDAGEYR": 63, "WTMEC2YR": 1.0, "SDMVPSU": 1, "SDMVSTRA": 11},
            {"SEQN": seqn_start + 7, "RIAGENDR": 1, "RIDAGEYR": 68, "WTMEC2YR": 1.1, "SDMVPSU": 2, "SDMVSTRA": 11},
            {"SEQN": seqn_start + 8, "RIAGENDR": 2, "RIDAGEYR": 69, "WTMEC2YR": 1.1, "SDMVPSU": 2, "SDMVSTRA": 11},
        ]
    )


def _bmx_fixture(seqn_start: int, *, male_shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SEQN": seqn_start + 1, "BMXHT": 178 + male_shift, "BMXWT": 82 + male_shift, "BMXBMI": 25.9, "BMXWAIST": 96 + male_shift},
            {"SEQN": seqn_start + 2, "BMXHT": 166, "BMXWT": 64, "BMXBMI": 23.1, "BMXWAIST": 78},
            {"SEQN": seqn_start + 3, "BMXHT": 181 + male_shift, "BMXWT": 88 + male_shift, "BMXBMI": 26.8, "BMXWAIST": 100 + male_shift},
            {"SEQN": seqn_start + 4, "BMXHT": 169, "BMXWT": 67, "BMXBMI": 23.5, "BMXWAIST": 80},
            {"SEQN": seqn_start + 5, "BMXHT": 183 + male_shift, "BMXWT": 90 + male_shift, "BMXBMI": 27.0, "BMXWAIST": 102 + male_shift},
            {"SEQN": seqn_start + 6, "BMXHT": 167, "BMXWT": 65, "BMXBMI": 23.3, "BMXWAIST": 79},
            {"SEQN": seqn_start + 7, "BMXHT": 179 + male_shift, "BMXWT": 84 + male_shift, "BMXBMI": 26.0, "BMXWAIST": 97 + male_shift},
            {"SEQN": seqn_start + 8, "BMXHT": 170, "BMXWT": 68, "BMXBMI": 23.6, "BMXWAIST": 81},
        ]
    )


def _cfq_fixture(seqn_start: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SEQN": seqn_start + 1, "CFDDS": 60},
            {"SEQN": seqn_start + 2, "CFDDS": 54},
            {"SEQN": seqn_start + 3, "CFDDS": 63},
            {"SEQN": seqn_start + 4, "CFDDS": 56},
            {"SEQN": seqn_start + 5, "CFDDS": 62},
            {"SEQN": seqn_start + 6, "CFDDS": 55},
            {"SEQN": seqn_start + 7, "CFDDS": 64},
            {"SEQN": seqn_start + 8, "CFDDS": 57},
        ]
    )


def _mgx_fixture(seqn_start: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SEQN": seqn_start + 1, "MGXH1T1": 48.0, "MGXH2T1": 46.0, "MGXH1T2": 49.0, "MGXH2T2": 45.0, "MGXH1T3": 47.0, "MGXH2T3": 44.0},
            {"SEQN": seqn_start + 2, "MGXH1T1": 31.0, "MGXH2T1": 29.0, "MGXH1T2": 30.0, "MGXH2T2": 28.0, "MGXH1T3": 32.0, "MGXH2T3": 27.0},
            {"SEQN": seqn_start + 3, "MGXH1T1": 50.0, "MGXH2T1": 48.0, "MGXH1T2": 51.0, "MGXH2T2": 47.0, "MGXH1T3": 49.0, "MGXH2T3": 46.0},
            {"SEQN": seqn_start + 4, "MGXH1T1": 33.0, "MGXH2T1": 31.0, "MGXH1T2": 34.0, "MGXH2T2": 30.0, "MGXH1T3": 35.0, "MGXH2T3": 29.0},
            {"SEQN": seqn_start + 5, "MGXH1T1": 49.0, "MGXH2T1": 47.0, "MGXH1T2": 50.0, "MGXH2T2": 46.0, "MGXH1T3": 48.0, "MGXH2T3": 45.0},
            {"SEQN": seqn_start + 6, "MGXH1T1": 32.0, "MGXH2T1": 30.0, "MGXH1T2": 31.0, "MGXH2T2": 29.0, "MGXH1T3": 33.0, "MGXH2T3": 28.0},
            {"SEQN": seqn_start + 7, "MGXH1T1": 47.0, "MGXH2T1": 45.0, "MGXH1T2": 48.0, "MGXH2T2": 44.0, "MGXH1T3": 46.0, "MGXH2T3": 43.0},
            {"SEQN": seqn_start + 8, "MGXH1T1": 34.0, "MGXH2T1": 32.0, "MGXH1T2": 35.0, "MGXH2T2": 31.0, "MGXH1T3": 36.0, "MGXH2T3": 30.0},
        ]
    )


def _build_nhanes_fixture_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "nhanes"
    _write_xpt(_demo_fixture(1000, cycle_code="G"), raw_dir / "DEMO_G.xpt")
    _write_xpt(_bmx_fixture(1000), raw_dir / "BMX_G.xpt")
    _write_xpt(_cfq_fixture(1000), raw_dir / "CFQ_G.xpt")
    _write_xpt(_mgx_fixture(1000), raw_dir / "MGX_G.xpt")
    _write_xpt(_demo_fixture(2000, cycle_code="L"), raw_dir / "DEMO_L.xpt")
    _write_xpt(_bmx_fixture(2000, male_shift=2.0), raw_dir / "BMX_L.xpt")
    return raw_dir


def test_nhanes_adapter_longifies_multicycle_traits(tmp_path: Path):
    raw_dir = _build_nhanes_fixture_dir(tmp_path)
    registry = build_registry()
    adapter = NHANESAdapter(registry.get_dataset("nhanes_2011_2023"), raw_path=raw_dir)
    normalized = adapter.to_long_person_trait().data
    assert {"height_cm", "weight_kg", "bmi", "waist_cm", "adult_cognition_screen", "grip_strength_kg"} <= set(
        normalized["trait_id"].unique()
    )
    assert set(normalized["cycle_or_wave"].unique()) == {"2011-2012", "2021-2023"}
    assert normalized["country"].dropna().unique().tolist() == ["United States"]


def test_nhanes_estimation_uses_design_bootstrap_when_design_columns_exist(tmp_path: Path):
    raw_dir = _build_nhanes_fixture_dir(tmp_path)
    registry = build_registry()
    adapter = NHANESAdapter(
        registry.get_dataset("nhanes_2011_2023"),
        raw_path=raw_dir,
        cycles=["G"],
        traits=["grip_strength_kg"],
    )
    normalized = adapter.to_long_person_trait()
    estimates = estimate_dataset_cells(
        normalized.data,
        config=EstimationConfig(
            min_n_per_sex_for_variance=2,
            min_n_per_sex_for_95_tail=2,
            min_unique_values=2,
            design_bootstrap_replicates=8,
            default_age_band_width_years=10,
        ),
    )
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert row["inference_method"] == "stratified_cluster_bootstrap_psu"
    assert row["se_log_variance_ratio"] > 0.0


def test_nhanes_sex_normalizer_uses_cdc_coding():
    assert _normalize_nhanes_sex(1) == "male"
    assert _normalize_nhanes_sex(2) == "female"
