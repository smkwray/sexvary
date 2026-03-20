import json
from pathlib import Path

from sexvary.orchestration import (
    build_selection_records,
    discover_pipeline_availability,
    PipelineSelectionRecord,
    PipelineRunRecord,
    discover_pipeline_invocations,
    write_run_manifest,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub\n", encoding="utf-8")
    return path


def test_discover_pipeline_invocations_finds_available_datasets(tmp_path: Path):
    _touch(tmp_path / "data" / "local" / "nlsy79.csv")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_paths.yaml").write_text(
        "local_datasets:\n  nlsy79_main:\n    path: data/local/nlsy79.csv\n",
        encoding="utf-8",
    )
    _touch(tmp_path / "data" / "raw" / "external" / "piaac_cycle2" / "prgusap2.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "pisa_2022" / "CY08MSP_STU_QQQ.SAV")
    _touch(tmp_path / "data" / "raw" / "external" / "nhanes_2011_2023" / "DEMO_G.xpt")
    _touch(tmp_path / "data" / "raw" / "external" / "nnyfs_2012" / "Y_DEMO.xpt")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_core" / "cpp_clean_v1.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_core" / "cpp_cognitive_scores.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_core" / "cpp_g_factors.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_core" / "cpp_weights.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_growth" / "cpp_growth_trajectories.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_growth" / "cpp_birthweight_zscores.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_growth" / "cpp_clean_v1.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "cpp_growth" / "cpp_weights.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "psid_cds_tas" / "psid_stub.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "timss_2019" / "asgusam7.sav")
    _touch(tmp_path / "data" / "raw" / "external" / "timss_2023" / "asgusam8.sav")
    _touch(tmp_path / "data" / "raw" / "external" / "pirls_2021" / "ASGAUSR5.sav")
    _touch(tmp_path / "data" / "raw" / "external" / "icils_2023" / "BSGUSAI3.sav")
    _touch(tmp_path / "data" / "raw" / "external" / "hrs_public" / "h22core.zip")
    _touch(tmp_path / "data" / "raw" / "external" / "hrs_public" / "trk2022v1.zip")
    _touch(tmp_path / "data" / "raw" / "external" / "ecls_k_2011" / "childK5p.dat")
    _touch(tmp_path / "data" / "raw" / "external" / "hsls_2009" / "hsls.csv")

    invocations = discover_pipeline_invocations(tmp_path, python_executable="/usr/bin/python3")
    assert [item.pipeline_id for item in invocations] == [
        "local_nlsy",
        "piaac_cycle2",
        "pisa_2022",
        "nhanes_2011_2023",
        "nnyfs_2012",
        "cpp_core",
        "cpp_growth",
        "psid_cds_tas",
        "hrs_public",
        "timss_2019",
        "timss_2023",
        "pirls_2021",
        "icils_2023",
        "ecls_k_2011",
        "hsls_2009",
    ]


def test_discover_pipeline_invocations_accepts_string_local_path_entries(tmp_path: Path):
    _touch(tmp_path / "data" / "local" / "nlsy79.csv")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_paths.yaml").write_text(
        "local_datasets:\n  nlsy79_main: data/local/nlsy79.csv\n",
        encoding="utf-8",
    )
    invocations = discover_pipeline_invocations(tmp_path, python_executable="/usr/bin/python3")
    assert [item.pipeline_id for item in invocations] == ["local_nlsy"]


def test_filter_invocations_respects_include_and_exclude(tmp_path: Path):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_paths.yaml").write_text("local_datasets: {}\n", encoding="utf-8")
    _touch(tmp_path / "data" / "raw" / "external" / "piaac_cycle2" / "prgusap2.csv")
    _touch(tmp_path / "data" / "raw" / "external" / "pisa_2022" / "CY08MSP_STU_QQQ.SAV")

    availability = discover_pipeline_availability(tmp_path, python_executable="/usr/bin/python3")
    selected, selection_records = build_selection_records(
        availability,
        include={"piaac_cycle2", "pisa_2022"},
        exclude={"pisa_2022"},
    )
    assert [item.pipeline_id for item in selected] == ["piaac_cycle2"]
    statuses = {item.pipeline_id: item.selection_status for item in selection_records}
    assert statuses["piaac_cycle2"] == "selected"
    assert statuses["pisa_2022"] == "excluded"


def test_build_selection_records_marks_missing_inputs(tmp_path: Path):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_paths.yaml").write_text("local_datasets: {}\n", encoding="utf-8")
    _touch(tmp_path / "data" / "raw" / "external" / "piaac_cycle2" / "prgusap2.csv")

    availability = discover_pipeline_availability(tmp_path, python_executable="/usr/bin/python3")
    selected, selection_records = build_selection_records(availability)

    assert [item.pipeline_id for item in selected] == ["piaac_cycle2"]
    statuses = {item.pipeline_id: item.selection_status for item in selection_records}
    assert statuses["piaac_cycle2"] == "selected"
    assert statuses["pisa_2022"] == "missing_input"
    assert statuses["nnyfs_2012"] == "missing_input"
    assert statuses["cpp_core"] == "missing_input"
    assert statuses["cpp_growth"] == "missing_input"
    assert statuses["hrs_public"] == "missing_input"
    assert statuses["timss_2019"] == "missing_input"


def test_discover_pipeline_availability_reads_shared_data_root_layout(tmp_path: Path):
    shared_root = tmp_path.parent / "data"
    _touch(shared_root / "sources" / "cdc" / "nnyfs" / "2012" / "Y_DEMO.xpt")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_paths.yaml").write_text("local_datasets: {}\n", encoding="utf-8")

    availability = discover_pipeline_availability(tmp_path, python_executable="/usr/bin/python3")
    statuses = {item.pipeline_id: item.status for item in availability}
    reasons = {item.pipeline_id: item.reason for item in availability}

    assert statuses["nnyfs_2012"] == "runnable"
    assert "sources/cdc/nnyfs/2012" in (reasons["nnyfs_2012"] or "")


def test_discover_pipeline_availability_cpp_requires_complete_file_sets(tmp_path: Path):
    shared_root = tmp_path.parent / "data"
    cpp_dir = shared_root / "sources" / "nih" / "cpp" / "release_v3_2"
    _touch(cpp_dir / "cpp_clean_v1.csv")
    _touch(cpp_dir / "cpp_cognitive_scores.csv")
    _touch(cpp_dir / "cpp_g_factors.csv")
    _touch(cpp_dir / "cpp_weights.csv")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_paths.yaml").write_text("local_datasets: {}\n", encoding="utf-8")

    availability = discover_pipeline_availability(tmp_path, python_executable="/usr/bin/python3")
    statuses = {item.pipeline_id: item.status for item in availability}
    reasons = {item.pipeline_id: item.reason for item in availability}

    assert statuses["cpp_core"] == "runnable"
    assert statuses["cpp_growth"] == "missing_input"
    assert "sources/nih/cpp/release_v3_2" in (reasons["cpp_core"] or "")
    assert "cpp_growth_trajectories.csv" in (reasons["cpp_growth"] or "")


def test_write_run_manifest_creates_latest_files(tmp_path: Path):
    record = PipelineRunRecord(
        pipeline_id="piaac_cycle2",
        label="PIAAC cycle 2",
        command=["python", "scripts/run_piaac_pipeline.py"],
        output_dir="results/piaac_cycle2",
        status="ok",
        returncode=0,
        started_utc="2026-03-15T00:00:00+00:00",
        finished_utc="2026-03-15T00:00:02+00:00",
        duration_seconds=2.0,
        stdout_tail="done",
        stderr_tail="",
        notes="prgusap2.csv",
    )
    json_path, md_path = write_run_manifest(
        root=tmp_path,
        python_executable="/usr/bin/python3",
        selected_invocations=[],
        selection_records=[],
        run_records=[record],
        compare_record=None,
        dry_run=False,
    )
    assert json_path.exists()
    assert md_path.exists()
    assert (tmp_path / "results" / "run_manifests" / "backend_run_latest.json").exists()
    assert (tmp_path / "results" / "run_manifests" / "backend_run_latest.md").exists()


def test_write_run_manifest_records_selection_statuses(tmp_path: Path):
    record = PipelineRunRecord(
        pipeline_id="piaac_cycle2",
        label="PIAAC cycle 2",
        command=["python", "scripts/run_piaac_pipeline.py"],
        output_dir="results/piaac_cycle2",
        status="ok",
        returncode=0,
        started_utc="2026-03-15T00:00:00+00:00",
        finished_utc="2026-03-15T00:00:02+00:00",
        duration_seconds=2.0,
        stdout_tail="done",
        stderr_tail="",
        notes="prgusap2.csv",
    )
    _, md_path = write_run_manifest(
        root=tmp_path,
        python_executable="/usr/bin/python3",
        selected_invocations=[],
        selection_records=[
            PipelineSelectionRecord(
                pipeline_id="piaac_cycle2",
                label="PIAAC cycle 2",
                selection_status="selected",
                reason="prgusap2.csv",
                output_dir="results/piaac_cycle2",
            ),
            PipelineSelectionRecord(
                pipeline_id="timss_2019",
                label="TIMSS 2019",
                selection_status="missing_input",
                reason="No supported raw files found under data/raw/external/timss_2019/.",
                output_dir=None,
            ),
        ],
        run_records=[record],
        compare_record=None,
        dry_run=False,
    )

    payload = json.loads((tmp_path / "results" / "run_manifests" / "backend_run_latest.json").read_text())
    assert payload["pipeline_selection"][1]["pipeline_id"] == "timss_2019"
    assert payload["pipeline_selection"][1]["selection_status"] == "missing_input"
    assert "## Pipeline selection" in md_path.read_text()
