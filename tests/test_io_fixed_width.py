from __future__ import annotations

from pathlib import Path

from sexvary.io import parse_sas_input_columns, read_fixed_width_from_sas_input


def test_parse_sas_input_columns_supports_position_list_input(tmp_path: Path):
    sas_path = tmp_path / "sample.sas"
    sas_path.write_text(
        "\n".join(
            [
                "DATA SAMPLE;",
                "INPUT",
                "      ID 1 - 3   AGE 4 - 5   str NAME 6 - 10",
                ";",
                "RUN;",
            ]
        ),
        encoding="latin-1",
    )
    specs = parse_sas_input_columns(sas_path)
    assert [spec.name for spec in specs] == ["ID", "AGE", "NAME"]
    assert specs[2].is_string is True


def test_read_fixed_width_from_sas_input_reads_position_list_input(tmp_path: Path):
    sas_path = tmp_path / "sample.sas"
    sas_path.write_text(
        "\n".join(
            [
                "DATA SAMPLE;",
                "INPUT",
                "      ID 1 - 3   AGE 4 - 5   str NAME 6 - 10",
                ";",
                "RUN;",
            ]
        ),
        encoding="latin-1",
    )
    data_path = tmp_path / "sample.txt"
    data_path.write_text("00112ALICE\n00209BOB  \n", encoding="latin-1")

    df = read_fixed_width_from_sas_input(data_path, sas_path=sas_path)

    assert df.columns.tolist() == ["ID", "AGE", "NAME"]
    assert df["ID"].tolist() == [1, 2]
    assert df["AGE"].tolist() == [12, 9]
    assert df["NAME"].astype(str).str.strip().tolist() == ["ALICE", "BOB"]
