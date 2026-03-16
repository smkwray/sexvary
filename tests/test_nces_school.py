from pathlib import Path
import zipfile

import pandas as pd

from sexvary.adapters import NCESSchoolAdapter
from sexvary.config import build_registry
from sexvary.io import parse_sas_input_columns
from sexvary.io import write_table


def test_nces_school_adapter_preserves_wave_grade_and_country(tmp_path: Path):
    df = pd.DataFrame(
        {
            "STUDENT_ID": [1, 2, 3, 4],
            "SEX": [1, 2, 1, 2],
            "AGE": [10, 10, 11, 11],
            "WAVE": ["fall_k", "fall_k", "spring_1", "spring_1"],
            "GRADE_BAND": ["K", "K", "1", "1"],
            "COUNTRY": ["USA", "USA", "USA", "USA"],
            "STUDENT_WEIGHT": [1.0, 1.1, 0.9, 1.2],
            "READING_SCORE": [50, 48, 60, 58],
            "MATH_SCORE": [49, 47, 61, 57],
        }
    )
    raw_path = write_table(df, tmp_path / "ecls_fixture.csv")
    mapping_path = tmp_path / "ecls_mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "dataset_id: ecls_k_2011",
                "table_shape: wide",
                "columns:",
                "  person_id: STUDENT_ID",
                "  sex: SEX",
                "  age: AGE",
                "  cycle_or_wave: WAVE",
                "  grade_or_age_band: GRADE_BAND",
                "  country: COUNTRY",
                "  weight: STUDENT_WEIGHT",
                "traits:",
                "  reading_achievement: READING_SCORE",
                "  math_achievement: MATH_SCORE",
                "value_maps:",
                "  sex:",
                "    male: [1]",
                "    female: [2]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    registry = build_registry()
    adapter = NCESSchoolAdapter(registry.get_dataset("ecls_k_2011"), raw_path=raw_path, mapping_path=mapping_path)
    normalized = adapter.to_long_person_trait().data
    assert set(normalized["trait_id"].unique()) == {"reading_achievement", "math_achievement"}
    assert normalized["country"].dropna().unique().tolist() == ["USA"]
    assert set(normalized["grade_or_age_band"].dropna().unique()) == {"K", "1"}
    assert set(normalized["cycle_or_wave"].dropna().unique()) == {"fall_k", "spring_1"}


def _fixed_width_line(width: int, values: dict[int, str]) -> str:
    chars = [" "] * width
    for start, value in values.items():
        offset = start - 1
        chars[offset : offset + len(value)] = list(value)
    return "".join(chars)


def test_parse_sas_input_columns_supports_fixed_width_layout(tmp_path: Path):
    sas_path = tmp_path / "toy.sas"
    sas_path.write_text(
        "\n".join(
            [
                "filename in1 'toy.dat';",
                "data toy;",
                "  infile in1 lrecl=24;",
                "  input",
                "    @1 STUDENT_ID $8.",
                "    @9 SEX 2.",
                "    @11 SCORE 8.4",
                "    @19 AGE 5.2",
                "  ;",
                "run;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    specs = parse_sas_input_columns(sas_path)
    assert [spec.name for spec in specs] == ["STUDENT_ID", "SEX", "SCORE", "AGE"]
    assert specs[0].start == 0
    assert specs[0].end == 8
    assert specs[0].is_string is True
    assert specs[2].start == 10
    assert specs[2].end == 18


def test_ecls_k_adapter_reads_repeated_wave_fixed_width_data(tmp_path: Path):
    sas_path = tmp_path / "ECLSK2011_K5PUF.sas"
    sas_path.write_text(
        "\n".join(
            [
                "filename in1 'childK5p.dat';",
                "data toy;",
                "  infile in1 lrecl=64;",
                "  input",
                "    @1 CHILDID $8.",
                "    @9 X_CHSEX_R 2.",
                "    @11 W1C0 6.2",
                "    @17 W2P0 6.2",
                "    @23 X1KAGE_R 5.2",
                "    @28 X2KAGE_R 5.2",
                "    @33 X1RTHETK5 8.4",
                "    @41 X2RTHETK5 8.4",
                "    @49 X1MTHETK5 8.4",
                "    @57 X2STHETK5 8.4",
                "  ;",
                "run;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    raw_path = tmp_path / "childK5p.dat"
    raw_path.write_text(
        "\n".join(
            [
                _fixed_width_line(
                    64,
                    {
                        1: "STUD0001",
                        9: "01",
                        11: "001.50",
                        17: "001.70",
                        23: "05.50",
                        28: "06.10",
                        33: "012.3456",
                        41: "013.4567",
                        49: "011.2233",
                        57: "009.8765",
                    },
                ),
                _fixed_width_line(
                    64,
                    {
                        1: "STUD0002",
                        9: "02",
                        11: "002.50",
                        17: "002.90",
                        23: "05.70",
                        28: "06.30",
                        33: "010.1000",
                        41: "011.2000",
                        49: "010.5000",
                        57: "008.8000",
                    },
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "ecls_mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "dataset_id: ecls_k_2011",
                "source_format: ecls_k2011_fixed_width_sas",
                "person_id: CHILDID",
                "sex: X_CHSEX_R",
                "country: United States",
                "setup_sas: ECLSK2011_K5PUF.sas",
                "waves:",
                '  "1":',
                "    label: fall_kindergarten_2010",
                "    grade_band: K",
                "    weight: W1C0",
                "    age: X1KAGE_R",
                '  "2":',
                "    label: spring_kindergarten_2011",
                "    grade_band: K",
                "    weight: W2P0",
                "    age: X2KAGE_R",
                "traits:",
                "  reading_achievement:",
                '    "1": X1RTHETK5',
                '    "2": X2RTHETK5',
                "  math_achievement:",
                '    "1": X1MTHETK5',
                "  science_achievement:",
                '    "2": X2STHETK5',
                "value_maps:",
                "  sex:",
                '    male: [1, "1"]',
                '    female: [2, "2"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_registry()
    adapter = NCESSchoolAdapter(registry.get_dataset("ecls_k_2011"), raw_path=raw_path, mapping_path=mapping_path)
    normalized = adapter.to_long_person_trait().data.sort_values(["person_id", "cycle_or_wave", "trait_id"]).reset_index(drop=True)

    assert len(normalized) == 8
    assert normalized["country"].dropna().unique().tolist() == ["United States"]
    assert set(normalized["grade_or_age_band"].dropna().unique()) == {"K"}
    assert set(normalized["cycle_or_wave"].unique()) == {"fall_kindergarten_2010", "spring_kindergarten_2011"}
    assert set(normalized["sex_observed"].unique()) == {"male", "female"}

    first = normalized.iloc[0]
    assert first["person_id"] == "STUD0001"
    assert first["cycle_or_wave"] == "fall_kindergarten_2010"
    assert first["trait_id"] == "math_achievement"
    assert float(first["score_raw"]) == 11.2233
    assert float(first["weight_main"]) == 1.5


def test_hsls_adapter_reads_repeated_wave_csv_from_zip(tmp_path: Path):
    csv_path = tmp_path / "hsls_student.csv"
    pd.DataFrame(
        {
            "STU_ID": ["1001", "1002", "1003", "1004"],
            "X1SEX": [1, 2, 1, 2],
            "STRAT_ID": ["10", "10", "11", "11"],
            "PSU": ["101", "102", "201", "202"],
            "W2W1STU": [100.0, 120.0, 95.0, 110.0],
            "W2W1STU001": [110.0, 115.0, 90.0, 105.0],
            "W2W1STU002": [95.0, 125.0, 100.0, 120.0],
            "X1TXMSCR": [50.0, 48.0, 55.0, 49.0],
            "X2TXMSCR": [60.0, 57.0, 63.0, 58.0],
        }
    ).to_csv(csv_path, index=False)
    zip_path = tmp_path / "HSLS_fixture.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="hsls_student.csv")

    mapping_path = tmp_path / "hsls_mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "dataset_id: hsls_2009",
                "source_format: hsls_repeated_wave_csv",
                "person_id: STU_ID",
                "sex: X1SEX",
                "country: United States",
                "archive_member: hsls_student.csv",
                "design_strata: STRAT_ID",
                "design_psu: PSU",
                "waves:",
                '  "1":',
                "    label: fall_2009_grade_9",
                '    grade_band: "9"',
                "    weight: W2W1STU",
                "    replicate_weight_prefix: W2W1STU",
                "    replicate_weight_count: 2",
                "    replicate_weight_digits: 3",
                "    replicate_method: brr",
                '  "2":',
                "    label: spring_2012_grade_11",
                '    grade_band: "11"',
                "    weight: W2W1STU",
                "    replicate_weight_prefix: W2W1STU",
                "    replicate_weight_count: 2",
                "    replicate_weight_digits: 3",
                "    replicate_method: brr",
                "traits:",
                "  math_achievement:",
                '    "1": X1TXMSCR',
                '    "2": X2TXMSCR',
                "value_maps:",
                "  sex:",
                "    male: [1]",
                "    female: [2]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_registry()
    adapter = NCESSchoolAdapter(registry.get_dataset("hsls_2009"), raw_path=zip_path, mapping_path=mapping_path)
    normalized = adapter.to_long_person_trait().data.sort_values(["person_id", "cycle_or_wave"]).reset_index(drop=True)

    assert len(normalized) == 8
    assert set(normalized["trait_id"].unique()) == {"math_achievement"}
    assert set(normalized["cycle_or_wave"].unique()) == {"fall_2009_grade_9", "spring_2012_grade_11"}
    assert set(normalized["grade_or_age_band"].unique()) == {"9", "11"}
    assert set(normalized["sex_observed"].unique()) == {"male", "female"}
    assert normalized["country"].dropna().unique().tolist() == ["United States"]
    assert normalized["design_strata"].dropna().nunique() == 2
    assert normalized["design_psu"].dropna().nunique() == 4
    assert {"replicate_weight_001", "replicate_weight_002"}.issubset(normalized.columns)
    assert (normalized["replicate_method"] == "brr").all()
