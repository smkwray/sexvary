from pathlib import Path

import pandas as pd

from sexvary.adapters.pisa import PISAAdapter, _normalize_pisa_sex
from sexvary.config import build_registry
from sexvary.io import write_table
from sexvary.pisa import estimate_pisa_cells, infer_pisa_replicate_spec


def _pisa_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"CNT": "USA", "CNTSTUID": "1", "CNTSCHID": "SCH1", "ST004D01T": 2, "AGE": 15.2, "W_FSTUWT": 1.0, "W_FSTURWT1": 0.8, "W_FSTURWT2": 1.2, "PV1MATH": 510, "PV2MATH": 515, "PV1READ": 500, "PV2READ": 505, "PV1SCIE": 490, "PV2SCIE": 493, "PV1CREAT": 480, "PV2CREAT": 482},
            {"CNT": "USA", "CNTSTUID": "2", "CNTSCHID": "SCH1", "ST004D01T": 2, "AGE": 15.5, "W_FSTUWT": 1.3, "W_FSTURWT1": 1.1, "W_FSTURWT2": 1.5, "PV1MATH": 530, "PV2MATH": 535, "PV1READ": 520, "PV2READ": 522, "PV1SCIE": 515, "PV2SCIE": 518, "PV1CREAT": 500, "PV2CREAT": 503},
            {"CNT": "USA", "CNTSTUID": "3", "CNTSCHID": "SCH2", "ST004D01T": 1, "AGE": 15.1, "W_FSTUWT": 0.9, "W_FSTURWT1": 1.0, "W_FSTURWT2": 0.8, "PV1MATH": 495, "PV2MATH": 500, "PV1READ": 505, "PV2READ": 509, "PV1SCIE": 485, "PV2SCIE": 488, "PV1CREAT": 470, "PV2CREAT": 472},
            {"CNT": "USA", "CNTSTUID": "4", "CNTSCHID": "SCH2", "ST004D01T": 1, "AGE": 15.4, "W_FSTUWT": 1.1, "W_FSTURWT1": 1.3, "W_FSTURWT2": 0.9, "PV1MATH": 505, "PV2MATH": 508, "PV1READ": 515, "PV2READ": 518, "PV1SCIE": 492, "PV2SCIE": 495, "PV1CREAT": 476, "PV2CREAT": 479},
        ]
    )


def test_pisa_adapter_longifies_pv_domains(tmp_path: Path):
    df = _pisa_fixture()
    raw_path = write_table(df, tmp_path / "pisa_fixture.csv")
    registry = build_registry()
    adapter = PISAAdapter(registry.get_dataset("pisa_2022"), raw_path=raw_path, country_codes=["USA"])
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["trait_id"].unique()) == {
        "math_achievement",
        "reading_achievement",
        "science_achievement",
        "creative_thinking",
    }
    assert sorted(normalized["pv_index"].unique().tolist()) == [1, 2]
    assert {"W_FSTURWT1", "W_FSTURWT2"}.issubset(normalized.columns)
    assert (normalized["grade_or_age_band"] == "15-year-olds").all()


def test_infer_pisa_replicate_spec_defaults_to_fay_brr():
    df = _pisa_fixture()
    spec = infer_pisa_replicate_spec(df, ["W_FSTURWT1", "W_FSTURWT2"])
    assert spec.method == "brr"
    assert spec.fay == 0.5
    assert spec.n_replicates == 2


def test_estimate_pisa_cells_combines_pvs_and_replicates(tmp_path: Path):
    df = _pisa_fixture()
    raw_path = write_table(df, tmp_path / "pisa_fixture.csv")
    registry = build_registry()
    adapter = PISAAdapter(registry.get_dataset("pisa_2022"), raw_path=raw_path, country_codes=["USA"], traits=["math_achievement"])
    normalized = adapter.to_long_person_trait()
    estimates = estimate_pisa_cells(normalized.data)
    assert len(estimates) == 1
    assert estimates["log_variance_ratio"].notna().all()
    assert estimates["se_log_variance_ratio"].notna().all()
    assert (estimates["replicate_method"] == "brr").all()


def test_pisa_sex_normalizer_uses_official_coding():
    assert _normalize_pisa_sex(1) == "female"
    assert _normalize_pisa_sex(2) == "male"
    assert _normalize_pisa_sex(1.0) == "female"
    assert _normalize_pisa_sex(2.0) == "male"
