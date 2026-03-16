from pathlib import Path

import pandas as pd
import pyreadstat

from sexvary.adapters.nnyfs import NNYFSAdapter
from sexvary.config import build_registry
from sexvary.estimation import EstimationConfig, estimate_dataset_cells


def _write_xpt(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_xport(df, path)
    return path


def _build_nnyfs_fixture_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "nnyfs"
    demo = pd.DataFrame(
        [
            {"SEQN": 1, "RIAGENDR": 1, "RIDAGEYR": 14, "WTMEC": 1.0, "SDMVPSU": 1, "SDMVSTRA": 10},
            {"SEQN": 2, "RIAGENDR": 2, "RIDAGEYR": 14, "WTMEC": 1.0, "SDMVPSU": 1, "SDMVSTRA": 10},
            {"SEQN": 3, "RIAGENDR": 1, "RIDAGEYR": 15, "WTMEC": 1.1, "SDMVPSU": 2, "SDMVSTRA": 10},
            {"SEQN": 4, "RIAGENDR": 2, "RIDAGEYR": 15, "WTMEC": 1.1, "SDMVPSU": 2, "SDMVSTRA": 10},
            {"SEQN": 5, "RIAGENDR": 1, "RIDAGEYR": 12, "WTMEC": 1.0, "SDMVPSU": 1, "SDMVSTRA": 11},
            {"SEQN": 6, "RIAGENDR": 2, "RIDAGEYR": 12, "WTMEC": 1.0, "SDMVPSU": 1, "SDMVSTRA": 11},
            {"SEQN": 7, "RIAGENDR": 1, "RIDAGEYR": 13, "WTMEC": 1.1, "SDMVPSU": 2, "SDMVSTRA": 11},
            {"SEQN": 8, "RIAGENDR": 2, "RIDAGEYR": 13, "WTMEC": 1.1, "SDMVPSU": 2, "SDMVSTRA": 11},
        ]
    )
    bmx = pd.DataFrame(
        [
            {"SEQN": 1, "BMXHT": 165, "BMXWT": 58, "BMXBMI": 21.3, "BMXWAIST": 74},
            {"SEQN": 2, "BMXHT": 160, "BMXWT": 50, "BMXBMI": 19.5, "BMXWAIST": 69},
            {"SEQN": 3, "BMXHT": 170, "BMXWT": 63, "BMXBMI": 21.8, "BMXWAIST": 76},
            {"SEQN": 4, "BMXHT": 161, "BMXWT": 51, "BMXBMI": 19.7, "BMXWAIST": 70},
            {"SEQN": 5, "BMXHT": 152, "BMXWT": 44, "BMXBMI": 19.0, "BMXWAIST": 66},
            {"SEQN": 6, "BMXHT": 149, "BMXWT": 41, "BMXBMI": 18.5, "BMXWAIST": 64},
            {"SEQN": 7, "BMXHT": 158, "BMXWT": 49, "BMXBMI": 19.6, "BMXWAIST": 68},
            {"SEQN": 8, "BMXHT": 154, "BMXWT": 45, "BMXBMI": 19.0, "BMXWAIST": 66},
        ]
    )
    mgx = pd.DataFrame(
        [
            {"SEQN": 1, "MGXH1T1": 34.0, "MGXH2T1": 32.0, "MGXH1T2": 35.0, "MGXH2T2": 31.0, "MGXH1T3": 33.0, "MGXH2T3": 30.0},
            {"SEQN": 2, "MGXH1T1": 23.0, "MGXH2T1": 22.0, "MGXH1T2": 24.0, "MGXH2T2": 21.0, "MGXH1T3": 25.0, "MGXH2T3": 20.0},
            {"SEQN": 3, "MGXH1T1": 36.0, "MGXH2T1": 34.0, "MGXH1T2": 37.0, "MGXH2T2": 33.0, "MGXH1T3": 35.0, "MGXH2T3": 32.0},
            {"SEQN": 4, "MGXH1T1": 24.0, "MGXH2T1": 23.0, "MGXH1T2": 25.0, "MGXH2T2": 22.0, "MGXH1T3": 26.0, "MGXH2T3": 21.0},
            {"SEQN": 5, "MGXH1T1": 28.0, "MGXH2T1": 26.0, "MGXH1T2": 29.0, "MGXH2T2": 25.0, "MGXH1T3": 27.0, "MGXH2T3": 24.0},
            {"SEQN": 6, "MGXH1T1": 20.0, "MGXH2T1": 18.0, "MGXH1T2": 21.0, "MGXH2T2": 17.0, "MGXH1T3": 22.0, "MGXH2T3": 16.0},
            {"SEQN": 7, "MGXH1T1": 30.0, "MGXH2T1": 28.0, "MGXH1T2": 31.0, "MGXH2T2": 27.0, "MGXH1T3": 29.0, "MGXH2T3": 26.0},
            {"SEQN": 8, "MGXH1T1": 21.0, "MGXH2T1": 19.0, "MGXH1T2": 22.0, "MGXH2T2": 18.0, "MGXH1T3": 23.0, "MGXH2T3": 17.0},
        ]
    )
    _write_xpt(demo, raw_dir / "Y_DEMO.xpt")
    _write_xpt(bmx, raw_dir / "Y_BMX.xpt")
    _write_xpt(mgx, raw_dir / "Y_MGX.xpt")
    return raw_dir


def test_nnyfs_adapter_longifies_expected_traits(tmp_path: Path):
    raw_dir = _build_nnyfs_fixture_dir(tmp_path)
    registry = build_registry()
    adapter = NNYFSAdapter(registry.get_dataset("nnyfs_2012"), raw_path=raw_dir)
    normalized = adapter.to_long_person_trait().data

    assert {"height_cm", "weight_kg", "bmi", "waist_cm", "grip_strength_kg"} <= set(normalized["trait_id"].unique())
    assert normalized["cycle_or_wave"].dropna().unique().tolist() == ["2012"]
    assert normalized["country"].dropna().unique().tolist() == ["United States"]


def test_nnyfs_estimation_produces_inferential_grip_cell(tmp_path: Path):
    raw_dir = _build_nnyfs_fixture_dir(tmp_path)
    registry = build_registry()
    adapter = NNYFSAdapter(registry.get_dataset("nnyfs_2012"), raw_path=raw_dir)
    normalized = adapter.to_long_person_trait()
    data = normalized.data[normalized.data["trait_id"] == "grip_strength_kg"].copy()

    estimates = estimate_dataset_cells(
        data,
        config=EstimationConfig(
            min_n_per_sex_for_variance=2,
            min_n_per_sex_for_95_tail=2,
            min_unique_values=2,
            design_bootstrap_replicates=8,
            default_age_band_width_years=20,
        ),
    )

    assert len(estimates) >= 1
    assert set(estimates["inference_method"].dropna().unique()) <= {
        "stratified_cluster_bootstrap_psu",
        "analytic_effective_n_simple_design",
    }
    assert "unavailable" not in set(estimates["inference_method"].dropna().unique())
