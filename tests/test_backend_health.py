import json
import os
from pathlib import Path

from sexvary.backend_health import (
    availability_payload,
    availability_to_frame,
    build_restore_checklist,
    compare_availability,
    load_latest_backend_manifest,
    parse_restore_commands,
    summarize_backend_manifest,
)
from sexvary.orchestration import PipelineAvailability, PipelineInvocation


def test_load_latest_backend_manifest_prefers_executed_run(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "results" / "run_manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "backend_run_older.json").write_text(
        json.dumps({"dry_run": False, "finished_utc": "2026-03-15T23:11:58+00:00", "hostname": "exec-host"}),
        encoding="utf-8",
    )
    (manifest_dir / "backend_run_newer.json").write_text(
        json.dumps({"dry_run": True, "finished_utc": "2026-03-16T11:30:01+00:00", "hostname": "dry-host"}),
        encoding="utf-8",
    )
    older = manifest_dir / "backend_run_older.json"
    newer = manifest_dir / "backend_run_newer.json"
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    manifest = load_latest_backend_manifest(tmp_path, prefer_executed=True)
    assert manifest is not None
    assert manifest["hostname"] == "exec-host"


def test_summarize_backend_manifest_uses_pipeline_selection_when_present() -> None:
    summary = summarize_backend_manifest(
        {
            "hostname": "host-a",
            "dry_run": False,
            "pipeline_selection": [
                {"selection_status": "selected"},
                {"selection_status": "selected"},
                {"selection_status": "missing_input"},
                {"selection_status": "excluded"},
            ],
            "pipeline_runs": [{"status": "ok"}, {"status": "failed"}],
            "compare_run": {"status": "ok"},
        },
        environment="local",
    )
    assert summary["selected"] == 2
    assert summary["missing_input"] == 1
    assert summary["excluded"] == 1
    assert summary["failed_runs"] == 1


def test_compare_availability_surfaces_status_mismatches() -> None:
    invocation = PipelineInvocation(
        pipeline_id="timss_2019",
        label="TIMSS 2019",
        command=["python", "scripts/run_timss_pipeline.py"],
        output_dir="results/timss_2019",
        notes=None,
    )
    local = [
        PipelineAvailability(
            pipeline_id="timss_2019",
            label="TIMSS 2019",
            status="missing_input",
            reason="missing local raw file",
            invocation=None,
            output_dir=None,
        )
    ]
    remote = [
        PipelineAvailability(
            pipeline_id="timss_2019",
            label="TIMSS 2019",
            status="runnable",
            reason="T19_G4_USA_SPSS.zip",
            invocation=invocation,
            output_dir="results/timss_2019",
        )
    ]

    comparison = compare_availability(
        availability_to_frame(local, environment="local"),
        availability_to_frame(remote, environment="remote"),
    )
    assert len(comparison) == 1
    assert comparison.loc[0, "status_match"] == False
    assert comparison.loc[0, "local_status"] == "missing_input"
    assert comparison.loc[0, "remote_status"] == "runnable"


def test_availability_payload_includes_manifest_summary() -> None:
    availability = [
        PipelineAvailability(
            pipeline_id="piaac_cycle2",
            label="PIAAC cycle 2",
            status="runnable",
            reason="prgusap2.csv",
            invocation=None,
            output_dir="results/piaac_cycle2",
        )
    ]
    payload = availability_payload(
        availability,
        environment="local",
        manifest={
            "hostname": "host-a",
            "dry_run": False,
            "selected_pipelines": [{"pipeline_id": "piaac_cycle2"}],
            "pipeline_runs": [],
            "compare_run": {"status": "ok"},
        },
    )
    assert payload["manifest_summary"]["selected"] == 1
    assert payload["manifest_summary"]["hostname"] == "host-a"


def test_build_restore_checklist_generates_rsync_for_remote_only_dataset(tmp_path: Path) -> None:
    comparison = compare_availability(
        availability_to_frame(
            [
                PipelineAvailability(
                    pipeline_id="timss_2019",
                    label="TIMSS 2019",
                    status="missing_input",
                    reason="missing local raw file",
                    invocation=None,
                    output_dir=None,
                )
            ],
            environment="local",
        ),
        availability_to_frame(
            [
                PipelineAvailability(
                    pipeline_id="timss_2019",
                    label="TIMSS 2019",
                    status="runnable",
                    reason="T19_G4_USA_SPSS.zip",
                    invocation=None,
                    output_dir="results/timss_2019",
                )
            ],
            environment="remote",
        ),
    )

    checklist = build_restore_checklist(
        comparison,
        remote_host="user@example",
        remote_project_dir="sexvary",
        local_project_dir=tmp_path,
    )
    assert len(checklist) == 1
    assert checklist.loc[0, "pipeline_id"] == "timss_2019"
    assert "rsync -az" in checklist.loc[0, "suggested_command"]
    assert "data/raw/external/timss_2019/" in checklist.loc[0, "suggested_command"]


def test_parse_restore_commands_filters_and_validates_rsync_only() -> None:
    checklist = build_restore_checklist(
        compare_availability(
            availability_to_frame(
                [
                    PipelineAvailability(
                        pipeline_id="timss_2019",
                        label="TIMSS 2019",
                        status="missing_input",
                        reason="missing local raw file",
                        invocation=None,
                        output_dir=None,
                    )
                ],
                environment="local",
            ),
            availability_to_frame(
                [
                    PipelineAvailability(
                        pipeline_id="timss_2019",
                        label="TIMSS 2019",
                        status="runnable",
                        reason="T19_G4_USA_SPSS.zip",
                        invocation=None,
                        output_dir="results/timss_2019",
                    )
                ],
                environment="remote",
            ),
        ),
        remote_host="user@example",
        remote_project_dir="sexvary",
        local_project_dir=Path("/tmp/project"),
    )

    commands = parse_restore_commands(checklist, pipeline_ids={"timss_2019"})
    assert len(commands) == 1
    assert commands[0]["pipeline_id"] == "timss_2019"
    assert commands[0]["argv"][0] == "rsync"
