from pathlib import Path

import pandas as pd

from sexvary.adapters.timss import (
    TIMSSAdapter,
    _normalize_timss_country_id,
    _normalize_timss_country_label,
    _normalize_timss_sex,
)
from sexvary.config import build_registry
from sexvary.io import write_table
from sexvary.timss import estimate_timss_cells, infer_timss_replicate_spec


def _timss_fixture() -> pd.DataFrame:
    rows = []
    for grade_prefix, grade_label in [("AS", "4"), ("BS", "8")]:
        for zone in [1, 2]:
            for rep in [1, 2]:
                for sex in [2, 1]:
                    idx = len(rows) + 1
                    math_shift = (zone * 5) + (rep * (4 if sex == 2 else 1)) + (6 if sex == 2 else 0)
                    sci_shift = (zone * 3) + (rep * (3 if sex == 2 else 1)) + (4 if sex == 2 else 0)
                    rows.append(
                        {
                            "IDSTUD": f"{grade_prefix}{idx}",
                            "IDCNTRY": "840",
                            "IDCNTRYL": "United States",
                            "ITSEX": sex,
                            "TOTWGT": 1.0 + (0.1 * rep),
                            "JKZONE": zone,
                            "JKREP": rep,
                            f"{grade_prefix}MMAT01": 500 + idx + math_shift,
                            f"{grade_prefix}MMAT02": 501 + idx + math_shift,
                            f"{grade_prefix}SSCI01": 490 + idx + sci_shift,
                            f"{grade_prefix}SSCI02": 491 + idx + sci_shift,
                        }
                    )
    return pd.DataFrame(rows)


def test_timss_adapter_longifies_grade_specific_pvs(tmp_path: Path):
    df = _timss_fixture()
    raw_path = write_table(df, tmp_path / "timss_fixture.csv")
    registry = build_registry()
    adapter = TIMSSAdapter(registry.get_dataset("timss_2019"), raw_path=raw_path, country_ids=["840"])
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["trait_id"].unique()) == {"math_achievement", "science_achievement"}
    assert set(normalized["grade_or_age_band"].unique()) == {"4", "8"}
    assert set(normalized["cycle_or_wave"].unique()) == {"timss2019_grade4", "timss2019_grade8"}
    assert sorted(normalized["pv_index"].unique().tolist()) == [1, 2]
    assert normalized["country"].dropna().unique().tolist() == ["United States"]
    assert normalized["country_id"].dropna().unique().tolist() == ["840"]


def test_timss_adapter_supports_timss_2023_cycle_labels(tmp_path: Path):
    raw_path = write_table(_timss_fixture(), tmp_path / "timss_fixture.csv")
    registry = build_registry()
    adapter = TIMSSAdapter(registry.get_dataset("timss_2023"), raw_path=raw_path, country_ids=["840"])
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["cycle_or_wave"].unique()) == {"timss2023_grade4", "timss2023_grade8"}


def test_timss_adapter_accepts_multiple_raw_files(tmp_path: Path):
    grade4 = _timss_fixture().loc[lambda df: df["IDSTUD"].str.startswith("AS")].copy()
    grade8 = _timss_fixture().loc[lambda df: df["IDSTUD"].str.startswith("BS")].copy()
    grade4_path = write_table(grade4, tmp_path / "timss_grade4.csv")
    grade8_path = write_table(grade8, tmp_path / "timss_grade8.csv")
    registry = build_registry()
    adapter = TIMSSAdapter(
        registry.get_dataset("timss_2019"),
        raw_path=[grade4_path, grade8_path],
        country_ids=["840"],
    )
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["grade_or_age_band"].unique()) == {"4", "8"}


def test_infer_timss_replicate_spec_counts_zone_replicates():
    df = pd.DataFrame({"jk_zone": [1, 1, 2, 2], "jk_rep": [1, 2, 1, 2]})
    spec = infer_timss_replicate_spec(df)
    assert spec.method == "jrr"
    assert spec.scale == 0.5
    assert spec.n_replicates == 4


def test_estimate_timss_cells_combines_pvs_and_jrr(tmp_path: Path):
    df = _timss_fixture()
    raw_path = write_table(df, tmp_path / "timss_fixture.csv")
    registry = build_registry()
    adapter = TIMSSAdapter(
        registry.get_dataset("timss_2019"),
        raw_path=raw_path,
        country_ids=["840"],
        grades=["4"],
        traits=["math_achievement"],
    )
    normalized = adapter.to_long_person_trait()
    estimates = estimate_timss_cells(normalized.data)
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert row["inference_method"] == "replicate_weights_jrr"
    assert row["replicate_method"] == "jrr"
    assert row["se_log_variance_ratio"] > 0.0


def test_timss_sex_normalizer_uses_iea_coding():
    assert _normalize_timss_sex(1) == "female"
    assert _normalize_timss_sex(2) == "male"
    assert _normalize_timss_sex("Girl") == "female"
    assert _normalize_timss_sex("Boy") == "male"


def test_timss_country_normalizers_handle_numeric_codes():
    assert _normalize_timss_country_id("840.0") == "840"
    assert _normalize_timss_country_id(840.0) == "840"
    assert _normalize_timss_country_label("840.0", "840.0") == "United States"
    assert _normalize_timss_country_label("840.0", None) == "United States"
