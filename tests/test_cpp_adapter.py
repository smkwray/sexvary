from __future__ import annotations

from pathlib import Path

import pandas as pd

from sexvary.adapters.cpp import CPPAdapter
from sexvary.config import build_registry


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_mapping(path: Path) -> None:
    path.write_text(
        """
cpp_core:
  files:
    clean: cpp_clean_v1.csv
    cognitive: cpp_cognitive_scores.csv
    g_factors: cpp_g_factors.csv
    weights: cpp_weights.csv
  id_columns:
    case_id: [case_id]
    mother_id: [mother_id]
    sex: [sex]
    site: [site]
  weight_columns:
    - weight_combined
    - wt_recommended
  waves:
    age_4:
      age: 4.0
      grade_or_age_band: "4"
      weight_columns:
        - weight_combined
        - wt_recommended
      presence_any: [g_age4_default]
      traits:
        general_intelligence_g: [g_age4_default]
    age_7:
      age: 7.0
      grade_or_age_band: "7"
      weight_columns:
        - age7_weight
        - weight_combined
      presence_any: [g_age7_default, wrat_reading]
      traits:
        general_intelligence_g: [g_age7_default]
        reading_achievement: [wrat_reading]
        working_memory: [wisc_digit_span]

cpp_growth:
  files:
    growth: cpp_growth_trajectories.csv
    birthweight: cpp_birthweight_zscores.csv
    weights: cpp_weights.csv
    clean: cpp_clean_v1.csv
  id_columns:
    case_id: [case_id]
    mother_id: [mother_id]
    sex: [sex]
    site: [site]
  weight_columns:
    - weight_combined
    - wt_recommended
  measure_columns:
    age: [age_months]
    height_cm: [height_cm]
    weight_kg: [weight_kg]
  birthweight:
    wave: birth
    age: 0.0
    value_columns: [birthweight_z]
""",
        encoding="utf-8",
    )


def test_cpp_core_adapter_emits_expected_rows(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "cpp_clean_v1.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002", "0003"],
                "mother_id": ["0100", "0101", "0102"],
                "sex": [1, 2, 3],
                "site": ["A", "A", "B"],
            }
        ),
    )
    write_csv(
        tmp_path / "cpp_g_factors.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002", "0003"],
                "g_age4_default": [100.0, 101.0, 102.0],
                "g_age7_default": [103.0, 104.0, 105.0],
            }
        ),
    )
    write_csv(
        tmp_path / "cpp_cognitive_scores.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002", "0003"],
                "wrat_reading": [50.0, 55.0, 60.0],
                "wisc_digit_span": [9.0, 10.0, 11.0],
            }
        ),
    )
    write_csv(
        tmp_path / "cpp_weights.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002", "0003"],
                "wt_recommended": [1.2, 0.8, 1.1],
                "age7_weight": [1.7, 1.4, 1.2],
            }
        ),
    )

    mapping_path = tmp_path / "cpp.yaml"
    write_mapping(mapping_path)

    adapter = CPPAdapter(
        dataset_spec=build_registry().get_dataset("cpp_core"),
        raw_path=tmp_path,
        mapping_path=mapping_path,
        mode="cpp_core",
    )
    normalized = adapter.to_long_person_trait()
    df = normalized.data

    assert not df.empty
    assert set(df["cycle_or_wave"]) == {"age_4", "age_7"}
    assert set(df["trait_id"]) >= {"general_intelligence_g", "reading_achievement", "working_memory"}
    assert set(df["sex_observed"]) == {"male", "female"}
    assert "0001" in set(df["person_id"])
    assert "0003" not in set(df["person_id"])
    age4 = df[df["cycle_or_wave"] == "age_4"]
    age7 = df[df["cycle_or_wave"] == "age_7"]
    assert set(age4["weight_source"].dropna()) == {"wt_recommended"}
    assert set(age7["weight_source"].dropna()) == {"age7_weight"}
    assert set(age4["weight_primary_source"].dropna()) == {"weight_combined"}
    assert set(age7["weight_primary_source"].dropna()) == {"age7_weight"}


def test_cpp_growth_adapter_emits_height_weight_and_birthweight(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "cpp_clean_v1.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002"],
                "mother_id": ["0100", "0101"],
                "sex": [1, 2],
                "site": ["A", "B"],
            }
        ),
    )
    write_csv(
        tmp_path / "cpp_weights.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002"],
                "weight_combined": [1.2, 0.8],
            }
        ),
    )
    write_csv(
        tmp_path / "cpp_growth_trajectories.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0001", "0002", "0002"],
                "age_months": [4, 12, 4, 12],
                "measure": ["height_cm", "weight_g", "height_cm", "weight_g"],
                "value": [60.0, 8500.0, 59.0, 8100.0],
                "unit": ["cm", "grams", "cm", "grams"],
            }
        ),
    )
    write_csv(
        tmp_path / "cpp_birthweight_zscores.csv",
        pd.DataFrame(
            {
                "case_id": ["0001", "0002"],
                "birthweight_z": [0.3, -0.4],
            }
        ),
    )

    mapping_path = tmp_path / "cpp.yaml"
    write_mapping(mapping_path)

    adapter = CPPAdapter(
        dataset_spec=build_registry().get_dataset("cpp_growth"),
        raw_path=tmp_path,
        mapping_path=mapping_path,
        mode="cpp_growth",
    )
    normalized = adapter.to_long_person_trait()
    df = normalized.data

    assert not df.empty
    assert set(df["trait_id"]) >= {"height_cm", "weight_kg", "birthweight_z"}
    assert set(df["sex_observed"]) == {"male", "female"}
    assert "birth" in set(df["cycle_or_wave"])
    assert {"4m", "12m"} <= set(df["cycle_or_wave"])
    assert set(df["weight_source"].dropna()) == {"weight_combined"}
