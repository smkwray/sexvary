from pathlib import Path

import pandas as pd

from sexvary.adapters.piaac import (
    PIAACAdapter,
    _normalize_piaac_age_band,
    _normalize_piaac_country_label,
)
from sexvary.config import build_registry
from sexvary.io import write_table
from sexvary.piaac import estimate_piaac_cells, infer_piaac_replicate_spec


def _piaac_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SEQID": 1, "CNTRYID": "USA", "CNTRYID_E": "United States", "GENDER_R": 1, "AGE_R": 30, "AGEG5LFS": "25-44", "SPFWT0": 1.0, "SPFWT1": 0.8, "SPFWT2": 1.2, "PVLIT1": 300, "PVLIT2": 310, "PVNUM1": 290, "PVNUM2": 295, "PVAPS1": 280, "PVAPS2": 282, "VEMETHOD": 3, "VEFAYFAC": 0.5, "VENREPS": 2},
            {"SEQID": 2, "CNTRYID": "USA", "CNTRYID_E": "United States", "GENDER_R": 1, "AGE_R": 42, "AGEG5LFS": "25-44", "SPFWT0": 1.5, "SPFWT1": 1.4, "SPFWT2": 1.6, "PVLIT1": 320, "PVLIT2": 325, "PVNUM1": 305, "PVNUM2": 307, "PVAPS1": 299, "PVAPS2": 301, "VEMETHOD": 3, "VEFAYFAC": 0.5, "VENREPS": 2},
            {"SEQID": 3, "CNTRYID": "USA", "CNTRYID_E": "United States", "GENDER_R": 2, "AGE_R": 28, "AGEG5LFS": "25-44", "SPFWT0": 1.1, "SPFWT1": 1.0, "SPFWT2": 1.2, "PVLIT1": 280, "PVLIT2": 285, "PVNUM1": 275, "PVNUM2": 278, "PVAPS1": 270, "PVAPS2": 272, "VEMETHOD": 3, "VEFAYFAC": 0.5, "VENREPS": 2},
            {"SEQID": 4, "CNTRYID": "USA", "CNTRYID_E": "United States", "GENDER_R": 2, "AGE_R": 45, "AGEG5LFS": "25-44", "SPFWT0": 1.3, "SPFWT1": 1.2, "SPFWT2": 1.4, "PVLIT1": 295, "PVLIT2": 300, "PVNUM1": 285, "PVNUM2": 288, "PVAPS1": 276, "PVAPS2": 279, "VEMETHOD": 3, "VEFAYFAC": 0.5, "VENREPS": 2},
        ]
    )


def test_piaac_adapter_longifies_pv_domains(tmp_path: Path):
    df = _piaac_fixture()
    raw_path = write_table(df, tmp_path / "piaac_fixture.csv")
    registry = build_registry()
    adapter = PIAACAdapter(registry.get_dataset("piaac_cycle2"), raw_path=raw_path, country_ids=["USA"])
    normalized = adapter.to_long_person_trait()
    out = normalized.data
    assert set(out["trait_id"].unique()) == {"literacy", "numeracy", "adaptive_problem_solving"}
    assert sorted(out["pv_index"].unique().tolist()) == [1, 2]
    assert {"SPFWT1", "SPFWT2"}.issubset(out.columns)
    assert out["country_id"].nunique() == 1


def test_infer_piaac_replicate_spec_supports_fay_brr():
    df = _piaac_fixture()
    spec = infer_piaac_replicate_spec(df, ["SPFWT1", "SPFWT2"])
    assert spec.method == "brr"
    assert spec.fay == 0.5
    assert spec.n_replicates == 2


def test_estimate_piaac_cells_combines_pvs_and_replicates(tmp_path: Path):
    df = _piaac_fixture()
    raw_path = write_table(df, tmp_path / "piaac_fixture.csv")
    registry = build_registry()
    adapter = PIAACAdapter(registry.get_dataset("piaac_cycle2"), raw_path=raw_path, country_ids=["USA"], traits=["literacy"])
    normalized = adapter.to_long_person_trait()
    estimates = estimate_piaac_cells(normalized.data)
    assert len(estimates) == 1
    assert estimates["log_variance_ratio"].notna().all()
    assert estimates["se_log_variance_ratio"].notna().all()
    assert (estimates["replicate_method"] == "brr").all()


def test_piaac_adapter_reads_semicolon_delimited_csv(tmp_path: Path):
    df = _piaac_fixture()
    raw_path = tmp_path / "piaac_fixture_semicolon.csv"
    df.to_csv(raw_path, index=False, sep=";")
    registry = build_registry()
    adapter = PIAACAdapter(registry.get_dataset("piaac_cycle2"), raw_path=raw_path, country_ids=["USA"], traits=["literacy"])
    normalized = adapter.to_long_person_trait()
    assert not normalized.data.empty
    assert set(normalized.data["country_id"]) == {"USA"}


def test_piaac_label_normalizers_cover_common_export_codes():
    assert _normalize_piaac_country_label("840", "840") == "United States"
    assert _normalize_piaac_country_label("840", "United States") == "United States"
    assert _normalize_piaac_age_band("1") == "16-19"
    assert _normalize_piaac_age_band("10") == "60-65"
