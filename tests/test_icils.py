from pathlib import Path

import pandas as pd

from sexvary.adapters.icils import ICILSAdapter, _normalize_icils_sex
from sexvary.config import build_registry
from sexvary.icils import estimate_icils_cells
from sexvary.io import write_table


def _icils_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"IDCNTRY": "840", "IDCNTRYL": "United States", "IDSTUD": "1", "S_SEX": 2, "S_AGE": 14.1, "TOTWGTS": 1.0, "SRWGT01": 0.9, "SRWGT02": 1.1, "PV1CIL": 520, "PV2CIL": 525, "PV1CT": 505, "PV2CT": 509},
            {"IDCNTRY": "840", "IDCNTRYL": "United States", "IDSTUD": "2", "S_SEX": 2, "S_AGE": 14.4, "TOTWGTS": 1.2, "SRWGT01": 1.0, "SRWGT02": 1.3, "PV1CIL": 540, "PV2CIL": 543, "PV1CT": 520, "PV2CT": 523},
            {"IDCNTRY": "840", "IDCNTRYL": "United States", "IDSTUD": "3", "S_SEX": 1, "S_AGE": 14.2, "TOTWGTS": 1.1, "SRWGT01": 1.0, "SRWGT02": 0.8, "PV1CIL": 500, "PV2CIL": 503, "PV1CT": 490, "PV2CT": 493},
            {"IDCNTRY": "840", "IDCNTRYL": "United States", "IDSTUD": "4", "S_SEX": 1, "S_AGE": 14.5, "TOTWGTS": 1.0, "SRWGT01": 1.2, "SRWGT02": 0.9, "PV1CIL": 508, "PV2CIL": 511, "PV1CT": 498, "PV2CT": 500},
        ]
    )


def test_icils_adapter_longifies_domains_and_replicates(tmp_path: Path):
    raw_path = write_table(_icils_fixture(), tmp_path / "icils_fixture.csv")
    registry = build_registry()
    adapter = ICILSAdapter(registry.get_dataset("icils_2023"), raw_path=raw_path, country_ids=["840"])
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["trait_id"].unique()) == {"computer_information_literacy", "computational_thinking"}
    assert sorted(normalized["pv_index"].unique().tolist()) == [1, 2]
    assert {"SRWGT01", "SRWGT02"}.issubset(normalized.columns)
    assert (normalized["grade_or_age_band"] == "grade_8").all()


def test_estimate_icils_cells_uses_jrr_replicates(tmp_path: Path):
    raw_path = write_table(_icils_fixture(), tmp_path / "icils_fixture.csv")
    registry = build_registry()
    adapter = ICILSAdapter(
        registry.get_dataset("icils_2023"),
        raw_path=raw_path,
        country_ids=["840"],
        traits=["computer_information_literacy"],
    )
    normalized = adapter.to_long_person_trait()
    estimates = estimate_icils_cells(normalized.data)
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert row["inference_method"] == "replicate_weights_jrr"
    assert row["se_log_variance_ratio"] > 0.0


def test_icils_sex_normalizer_uses_iea_coding():
    assert _normalize_icils_sex(1) == "female"
    assert _normalize_icils_sex(2) == "male"
    assert _normalize_icils_sex(0) == "male"
