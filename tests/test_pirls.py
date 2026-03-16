from pathlib import Path

import pandas as pd

from sexvary.adapters.pirls import PIRLSAdapter
from sexvary.config import build_registry
from sexvary.io import write_table
from sexvary.pirls import estimate_pirls_cells


def _pirls_fixture() -> pd.DataFrame:
    rows = []
    for zone in [1, 2]:
        for rep in [1, 2]:
            for sex in [2, 1]:
                idx = len(rows) + 1
                overall_shift = (zone * 4) + (rep * (3 if sex == 2 else 1)) + (5 if sex == 2 else 0)
                informational_shift = (zone * 3) + (rep * (2 if sex == 2 else 1)) + (4 if sex == 2 else 0)
                literary_shift = (zone * 2) + (rep * (2 if sex == 2 else 1)) + (3 if sex == 2 else 0)
                rows.append(
                    {
                        "IDSTUD": f"P{idx}",
                        "IDCNTRY": "840",
                        "IDCNTRYL": "United States",
                        "ITSEX": sex,
                        "TOTWGT": 1.0 + (0.1 * rep),
                        "JKZONE": zone,
                        "JKREP": rep,
                        "ASRREA01": 500 + idx + overall_shift,
                        "ASRREA02": 502 + idx + overall_shift,
                        "ASRINF01": 495 + idx + informational_shift,
                        "ASRINF02": 497 + idx + informational_shift,
                        "ASRLIT01": 490 + idx + literary_shift,
                        "ASRLIT02": 492 + idx + literary_shift,
                    }
                )
    return pd.DataFrame(rows)


def test_pirls_adapter_longifies_reading_domains(tmp_path: Path):
    raw_path = write_table(_pirls_fixture(), tmp_path / "pirls_fixture.csv")
    registry = build_registry()
    adapter = PIRLSAdapter(registry.get_dataset("pirls_2021"), raw_path=raw_path, country_ids=["840"])
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["trait_id"].unique()) == {"reading_achievement", "reading_informational", "reading_literary"}
    assert set(normalized["grade_or_age_band"].unique()) == {"4"}
    assert set(normalized["cycle_or_wave"].unique()) == {"pirls2021_grade4"}


def test_estimate_pirls_cells_uses_jrr(tmp_path: Path):
    raw_path = write_table(_pirls_fixture(), tmp_path / "pirls_fixture.csv")
    registry = build_registry()
    adapter = PIRLSAdapter(
        registry.get_dataset("pirls_2021"),
        raw_path=raw_path,
        country_ids=["840"],
        traits=["reading_achievement"],
    )
    normalized = adapter.to_long_person_trait()
    estimates = estimate_pirls_cells(normalized.data)
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert row["inference_method"] == "replicate_weights_jrr"
    assert row["se_log_variance_ratio"] > 0.0
