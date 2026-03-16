from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shlex

import pandas as pd

from .orchestration import PipelineAvailability


def load_latest_backend_manifest(root: Path, *, prefer_executed: bool = True) -> dict[str, object] | None:
    manifest_dir = root / "results" / "run_manifests"
    if not manifest_dir.exists():
        return None

    candidates = sorted(
        manifest_dir.glob("backend_run_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    fallback: dict[str, object] | None = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if fallback is None:
            fallback = payload
        if not prefer_executed or not bool(payload.get("dry_run", False)):
            return payload
    return fallback


def summarize_backend_manifest(manifest: dict[str, object] | None, *, environment: str) -> dict[str, object]:
    if not manifest:
        return {
            "environment": environment,
            "manifest_present": False,
            "hostname": None,
            "dry_run": None,
            "selected": 0,
            "missing_input": 0,
            "excluded": 0,
            "not_selected": 0,
            "failed_runs": 0,
            "compare_failed": False,
        }

    selection = manifest.get("pipeline_selection", []) or []
    if not selection:
        selection = [
            {"selection_status": "selected"}
            for _ in (manifest.get("selected_pipelines", []) or [])
        ]

    counts = {"selected": 0, "missing_input": 0, "excluded": 0, "not_selected": 0}
    for row in selection:
        status = str(row.get("selection_status", "unknown"))
        if status in counts:
            counts[status] += 1

    runs = manifest.get("pipeline_runs", []) or []
    compare_run = manifest.get("compare_run") or {}
    return {
        "environment": environment,
        "manifest_present": True,
        "hostname": manifest.get("hostname"),
        "dry_run": bool(manifest.get("dry_run", False)),
        "selected": counts["selected"],
        "missing_input": counts["missing_input"],
        "excluded": counts["excluded"],
        "not_selected": counts["not_selected"],
        "failed_runs": sum(1 for row in runs if row.get("status") == "failed"),
        "compare_failed": bool(compare_run.get("status") == "failed"),
    }


def availability_to_frame(availability: list[PipelineAvailability], *, environment: str) -> pd.DataFrame:
    rows = [
        {
            "environment": environment,
            "pipeline_id": item.pipeline_id,
            "dataset_label": item.label,
            "status": item.status,
            "reason": item.reason,
            "output_dir": item.output_dir,
        }
        for item in availability
    ]
    return pd.DataFrame(rows).sort_values(["pipeline_id"], kind="stable").reset_index(drop=True)


def compare_availability(local_df: pd.DataFrame, remote_df: pd.DataFrame) -> pd.DataFrame:
    local = local_df.rename(
        columns={
            "status": "local_status",
            "reason": "local_reason",
            "output_dir": "local_output_dir",
            "dataset_label": "local_dataset_label",
        }
    )
    remote = remote_df.rename(
        columns={
            "status": "remote_status",
            "reason": "remote_reason",
            "output_dir": "remote_output_dir",
            "dataset_label": "remote_dataset_label",
        }
    )
    merged = local.merge(
        remote,
        on="pipeline_id",
        how="outer",
    )
    merged["dataset_label"] = merged["local_dataset_label"].fillna(merged["remote_dataset_label"])
    merged["status_match"] = merged["local_status"].fillna("") == merged["remote_status"].fillna("")
    return merged[
        [
            "pipeline_id",
            "dataset_label",
            "local_status",
            "remote_status",
            "status_match",
            "local_reason",
            "remote_reason",
            "local_output_dir",
            "remote_output_dir",
        ]
    ].sort_values(["pipeline_id"], kind="stable").reset_index(drop=True)


def manifest_summary_frame(*summaries: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(summaries))


def availability_payload(availability: list[PipelineAvailability], *, environment: str, manifest: dict[str, object] | None) -> dict[str, object]:
    return {
        "environment": environment,
        "availability": [asdict(item) for item in availability],
        "manifest_summary": summarize_backend_manifest(manifest, environment=environment),
    }


def build_restore_checklist(
    comparison_df: pd.DataFrame,
    *,
    remote_host: str,
    remote_project_dir: str,
    local_project_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in comparison_df.itertuples(index=False):
        if getattr(row, "local_status", None) == "missing_input" and getattr(row, "remote_status", None) == "runnable":
            pipeline_id = str(row.pipeline_id)
            remote_output_raw = getattr(row, "remote_output_dir", "")
            local_output_raw = getattr(row, "local_output_dir", "")
            remote_output_dir = (
                f"results/{pipeline_id}"
                if pd.isna(remote_output_raw) or not str(remote_output_raw).strip()
                else str(remote_output_raw)
            )
            local_output_dir = (
                ""
                if pd.isna(local_output_raw) or not str(local_output_raw).strip()
                else str(local_output_raw)
            )
            remote_data_dir = f"{remote_project_dir.rstrip('/')}/data/raw/external/{pipeline_id}/"
            local_data_dir = (local_project_dir / "data" / "raw" / "external" / pipeline_id).as_posix() + "/"
            rows.append(
                {
                    "pipeline_id": pipeline_id,
                    "dataset_label": str(row.dataset_label),
                    "issue": "restore_local_raw_inputs",
                    "remote_reason": str(getattr(row, "remote_reason", "") or ""),
                    "suggested_command": f"rsync -az '{remote_host}:{remote_data_dir}' '{local_data_dir}'",
                    "notes": (
                        f"Remote is runnable and local is missing input. "
                        + (
                            f"Local outputs already exist at {local_output_dir}; "
                            if local_output_dir
                            else "No local output directory is currently recorded; "
                        )
                        + f"remote outputs are under {remote_output_dir}."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["pipeline_id"], kind="stable").reset_index(drop=True) if rows else pd.DataFrame()


def parse_restore_commands(
    checklist_df: pd.DataFrame,
    *,
    pipeline_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    if checklist_df.empty:
        return []

    work = checklist_df.copy()
    if pipeline_ids:
        work = work[work["pipeline_id"].isin(pipeline_ids)].copy()
    commands: list[dict[str, object]] = []
    for row in work.itertuples(index=False):
        command = str(row.suggested_command).strip()
        argv = shlex.split(command)
        if not argv or argv[0] != "rsync":
            raise ValueError(f"Unsupported restore command for {row.pipeline_id}: {command}")
        commands.append(
            {
                "pipeline_id": str(row.pipeline_id),
                "dataset_label": str(row.dataset_label),
                "issue": str(row.issue),
                "command": command,
                "argv": argv,
            }
        )
    return commands
