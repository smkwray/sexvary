from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile
import importlib.util

import numpy as np
import pandas as pd

from sexvary.adapters.hrs import HRSAdapter
from sexvary.config import build_registry


_RUN_HRS_SPEC = importlib.util.spec_from_file_location(
    "run_hrs_pipeline",
    Path(__file__).resolve().parents[1] / "scripts" / "run_hrs_pipeline.py",
)
assert _RUN_HRS_SPEC is not None and _RUN_HRS_SPEC.loader is not None
_RUN_HRS_MODULE = importlib.util.module_from_spec(_RUN_HRS_SPEC)
_RUN_HRS_SPEC.loader.exec_module(_RUN_HRS_MODULE)
_build_robustness_comparison = _RUN_HRS_MODULE._build_robustness_comparison
_write_hrs_report = _RUN_HRS_MODULE._write_hrs_report


def _write_stata_bytes(df: pd.DataFrame) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".dta") as tmp:
        df.to_stata(tmp.name, write_index=False)
        return Path(tmp.name).read_bytes()


def _write_member_zip(zip_path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for member, payload in members.items():
            zf.writestr(member, payload)


def test_hrs_adapter_extracts_tracker_weighted_waves(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    tracker = pd.DataFrame(
        {
            "HHID": ["010001", "010002"],
            "PN": ["010", "020"],
            "SEX": [1, 2],
            "STRATUM": [1, 2],
            "QAGE": [68, 70],
            "RAGE": [70, 72],
            "SAGE": [72, 74],
            "QWGTR": [1500.0, 1800.0],
            "RWGTR": [1600.0, 1900.0],
            "SWGTR": [1700.0, 2000.0],
        }
    )
    _write_member_zip(raw_dir / "trk2022v1.zip", {"trk2022tr_r.dta": _write_stata_bytes(tracker)})

    wave_2018 = pd.DataFrame(
        {
            "hhid": ["010001", "010002"],
            "pn": ["010", "020"],
            "QD174": [6, 4],
            "QD174W": [np.nan, np.nan],
            "QD184": [5, 3],
            "QD184W": [np.nan, np.nan],
            "QD170": [10, 7],
            "QD142": [93, 93],
            "QD143": [86, 80],
            "QD144": [79, 73],
            "QD145": [72, 66],
            "QD146": [65, 59],
        }
    )
    wave_2022 = pd.DataFrame(
        {
            "HHID": ["010001", "010002"],
            "PN": ["010", "020"],
            "SD174": [5, np.nan],
            "SD174W": [np.nan, 7],
            "SD184": [4, np.nan],
            "SD184W": [np.nan, 6],
            "SD170": [9, 11],
            "SD142": [93, 93],
            "SD143": [86, 86],
            "SD144": [79, 79],
            "SD145": [72, 72],
            "SD146": [65, 60],
            "SNSSCORE": [540, 996],
        }
    )

    _write_member_zip(raw_dir / "h18core.zip", {"h18sta.zip": _write_member_zip_bytes({"h18d_r.dta": _write_stata_bytes(wave_2018)})})
    _write_member_zip(raw_dir / "h22core.zip", {"H22sta.zip": _write_member_zip_bytes({"H22D_R.dta": _write_stata_bytes(wave_2022)})})

    registry = build_registry()
    spec = registry.get_dataset("hrs_public")
    normalized = HRSAdapter(spec, raw_path=raw_dir).to_long_person_trait()

    assert normalized.dataset_id == "hrs_public"
    assert set(normalized.provenance["included_waves"]) == {"hrs_2018", "hrs_2022"}
    assert {"immediate_recall", "delayed_recall", "serial_7s", "total_cognition", "numeracy"} <= set(
        normalized.data["trait_id"].unique()
    )
    assert set(normalized.data["sex_observed"].unique()) == {"male", "female"}
    assert set(normalized.data["weight_source"].dropna().unique()) == {"QWGTR", "SWGTR"}
    assert set(normalized.data["design_inference_label"].dropna().unique()) == {"approximate_household_cluster_bootstrap"}
    assert set(normalized.data["design_strata"].dropna().unique()) == {"1", "2"}

    hrs_2018 = normalized.data[normalized.data["cycle_or_wave"] == "HRS 2018"]
    assert "numeracy" not in set(hrs_2018["trait_id"].unique())

    serial = normalized.data[(normalized.data["cycle_or_wave"] == "HRS 2022") & (normalized.data["trait_id"] == "serial_7s")]
    assert set(serial["score_raw"].tolist()) == {4.0, 5.0}

    numeracy = normalized.data[(normalized.data["cycle_or_wave"] == "HRS 2022") & (normalized.data["trait_id"] == "numeracy")]
    assert numeracy["score_raw"].notna().sum() == 1


def _write_member_zip_bytes(members: dict[str, bytes]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zf:
            for member, payload in members.items():
                zf.writestr(member, payload)
        return Path(tmp.name).read_bytes()


def test_hrs_robustness_comparison_handles_empty_baseline() -> None:
    comparison, summary = _build_robustness_comparison(
        {
            "baseline_weighted": pd.DataFrame(),
            "unweighted": pd.DataFrame(),
        }
    )

    assert comparison.empty
    assert summary.empty


def test_hrs_report_handles_empty_estimates(tmp_path: Path) -> None:
    report_path = _write_hrs_report(
        out_dir=tmp_path,
        estimates=pd.DataFrame(columns=["evidence_status", "inference_method"]),
        robustness_summary=pd.DataFrame(),
    )

    text = report_path.read_text(encoding="utf-8")
    assert "No HRS estimate rows were generated." in text
