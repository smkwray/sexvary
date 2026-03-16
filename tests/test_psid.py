from __future__ import annotations

from pathlib import Path
import zipfile

from sexvary.adapters.psid import PSIDAdapter
from sexvary.config import build_registry


def _format_fixed_width_row(values: list[tuple[int, int, object]]) -> str:
    width = max(end for _, end, _ in values)
    chars = [" "] * width
    for start, end, value in values:
        text = str(value).rjust(end - start + 1)
        chars[start - 1 : end] = list(text)
    return "".join(chars)


def _write_member_zip(zip_path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for member, text in members.items():
            zf.writestr(member, text)


def test_psid_adapter_extracts_cds_wave(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_member_zip(
        raw_dir / "CDS2014.zip",
        {
            "2014/DEMOG14.sas": "\n".join(
                [
                    "DATA DEMOG14;",
                    "INPUT",
                    "      X14YRID 1 - 5   X14CYPSN 6 - 7   X14CHWGT 8 - 14   X14R01SEX 15 - 15   X14R01AGE 16 - 17   X14R02SEX 18 - 18   X14R02AGE 19 - 20",
                    ";",
                    "RUN;",
                ]
            ),
            "2014/DEMOG14.txt": "\n".join(
                [
                    _format_fixed_width_row([(1, 5, 10001), (6, 7, 1), (8, 14, 1200000), (15, 15, 1), (16, 17, 10), (18, 18, 2), (19, 20, 11)]),
                    _format_fixed_width_row([(1, 5, 10001), (6, 7, 2), (8, 14, 1200000), (15, 15, 1), (16, 17, 10), (18, 18, 2), (19, 20, 11)]),
                ]
            ),
            "2014/ASSESS14.sas": "\n".join(
                [
                    "DATA ASSESS14;",
                    "INPUT",
                    "      A14YRID 1 - 5   A14CYPSN 6 - 7   A14IWAGE 8 - 10   A14AGEX 11 - 12   A14LWSS 13 - 15   A14PCSS 16 - 18   A14BRSS 19 - 21   A14APSS 22 - 24   A14MRSS 25 - 27",
                    ";",
                    "RUN;",
                ]
            ),
            "2014/ASSESS14.txt": "\n".join(
                [
                    _format_fixed_width_row([(1, 5, 10001), (6, 7, 1), (8, 10, 120), (11, 12, 10), (13, 15, 110), (16, 18, 105), (19, 21, 108), (22, 24, 112), (25, 27, 109)]),
                    _format_fixed_width_row([(1, 5, 10001), (6, 7, 2), (8, 10, 132), (11, 12, 11), (13, 15, 103), (16, 18, 101), (19, 21, 102), (22, 24, 107), (25, 27, 104)]),
                ]
            ),
            "2014/PCGCHILD14.sas": "\n".join(
                [
                    "DATA PCGCHILD14;",
                    "INPUT",
                    "      P14YRID 1 - 5   P14CYPSN 6 - 7   P14CHGRADE 8 - 9   P14CHAGE 10 - 11   P14HW1CM 12 - 16   P14HW2KG 17 - 21   P14BMI 22 - 25",
                    ";",
                    "RUN;",
                ]
            ),
            "2014/PCGCHILD14.txt": "\n".join(
                [
                    _format_fixed_width_row([(1, 5, 10001), (6, 7, 1), (8, 9, 5), (10, 11, 10), (12, 16, 145), (17, 21, 40), (22, 25, 19)]),
                    _format_fixed_width_row([(1, 5, 10001), (6, 7, 2), (8, 9, 6), (10, 11, 11), (12, 16, 147), (17, 21, 41), (22, 25, 18)]),
                ]
            ),
        },
    )

    registry = build_registry()
    spec = registry.get_dataset("psid_cds_tas")
    normalized = PSIDAdapter(spec, raw_path=raw_dir).to_long_person_trait()

    assert normalized.dataset_id == "psid_cds_tas"
    assert set(normalized.provenance["included_waves"]) == {"cds_2014"}
    assert {"woodcock_johnson:letter_word", "height_cm", "weight_kg", "bmi"} <= set(normalized.data["trait_id"].unique())
    assert set(normalized.data["sex_observed"].unique()) == {"male", "female"}
    assert normalized.data["person_id"].nunique() == 2


def test_psid_adapter_extracts_tas_wave(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_member_zip(
        raw_dir / "TA2023.zip",
        {
            "TA2023.sas": "\n".join(
                [
                    "DATA TA2023;",
                    "INPUT",
                    "      TA230003 1 - 5   TA230004 6 - 7   TA230171 8 - 8   TA232404 9 - 17   TA230997 18 - 20   TA230998 21 - 23   TA230999 24 - 25   TA232116 26 - 28   TA232118 29 - 29   TA232119 30 - 31",
                    ";",
                    "RUN;",
                ]
            ),
            "TA2023.txt": "\n".join(
                [
                    _format_fixed_width_row([(1, 5, 20001), (6, 7, 1), (8, 8, 1), (9, 17, 12345), (18, 20, 650), (21, 23, 680), (24, 25, 30), (26, 28, 180), (29, 29, 6), (30, 31, 0)]),
                    _format_fixed_width_row([(1, 5, 20001), (6, 7, 2), (8, 8, 2), (9, 17, 23456), (18, 20, 600), (21, 23, 620), (24, 25, 27), (26, 28, 150), (29, 29, 5), (30, 31, 6)]),
                ]
            ),
        },
    )

    registry = build_registry()
    spec = registry.get_dataset("psid_cds_tas")
    normalized = PSIDAdapter(spec, raw_path=raw_dir).to_long_person_trait()

    assert normalized.dataset_id == "psid_cds_tas"
    assert set(normalized.provenance["included_waves"]) == {"tas_2023"}
    assert {"sat_critical_reading", "sat_math", "act_composite", "height_cm", "weight_kg", "bmi"} <= set(
        normalized.data["trait_id"].unique()
    )
    assert set(normalized.data["grade_or_age_band"].dropna().unique()) == {"all_ages"}
    assert set(normalized.data["sex_observed"].unique()) == {"male", "female"}
    assert normalized.data["person_id"].nunique() == 2
    assert set(normalized.data["weight_source"].dropna().unique()) == {"TA232404"}
    assert set(normalized.data["weight_primary_source"].dropna().unique()) == {"TA232404"}


def test_psid_adapter_uses_tas_weight_fallback(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_member_zip(
        raw_dir / "TA2019.zip",
        {
            "TA2019.sas": "\n".join(
                [
                    "DATA TA2019;",
                    "INPUT",
                    "      TA190003 1 - 5   TA190004 6 - 7   TA190180 8 - 8   TA192199 9 - 17   TA192202 18 - 24   TA190924 25 - 27   TA190925 28 - 30   TA190926 31 - 32   TA191949 33 - 35   TA191951 36 - 36   TA191952 37 - 38",
                    ";",
                    "RUN;",
                ]
            ),
            "TA2019.txt": "\n".join(
                [
                    _format_fixed_width_row([(1, 5, 30001), (6, 7, 1), (8, 8, 1), (9, 17, 0), (18, 24, 1234567), (25, 27, 650), (28, 30, 660), (31, 32, 29), (33, 35, 180), (36, 36, 6), (37, 38, 0)]),
                    _format_fixed_width_row([(1, 5, 30001), (6, 7, 2), (8, 8, 2), (9, 17, 0), (18, 24, 7654321), (25, 27, 600), (28, 30, 610), (31, 32, 27), (33, 35, 150), (36, 36, 5), (37, 38, 6)]),
                ]
            ),
        },
    )

    registry = build_registry()
    spec = registry.get_dataset("psid_cds_tas")
    normalized = PSIDAdapter(spec, raw_path=raw_dir).to_long_person_trait()

    assert set(normalized.provenance["included_waves"]) == {"tas_2019"}
    assert set(normalized.data["weight_source"].dropna().unique()) == {"TA192202"}
    assert normalized.data["weight_primary_source"].dropna().empty
