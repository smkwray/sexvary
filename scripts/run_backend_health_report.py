#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from dataclasses import asdict
import json
import subprocess

import pandas as pd

from sexvary.backend_health import (
    availability_payload,
    availability_to_frame,
    build_restore_checklist,
    compare_availability,
    load_latest_backend_manifest,
    manifest_summary_frame,
)
from sexvary.orchestration import discover_pipeline_availability
from sexvary.reporting import markdown_table
from sexvary.utils import ensure_dir, project_root


def _remote_snapshot(remote_host: str, *, remote_project_dir: str, remote_python: str) -> dict[str, object]:
    remote_project_literal = json.dumps(remote_project_dir)
    remote_python_literal = json.dumps(remote_python)
    inline = f"""
import json
import sys
from pathlib import Path
ROOT = Path({remote_project_literal}).expanduser().resolve()
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from sexvary.orchestration import discover_pipeline_availability
availability = discover_pipeline_availability(ROOT, python_executable={remote_python_literal})

def load_latest_backend_manifest(root, prefer_executed=True):
    manifest_dir = root / 'results' / 'run_manifests'
    if not manifest_dir.exists():
        return None
    candidates = sorted(manifest_dir.glob('backend_run_*.json'), key=lambda path: path.stat().st_mtime, reverse=True)
    fallback = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if fallback is None:
            fallback = payload
        if not prefer_executed or not bool(payload.get('dry_run', False)):
            return payload
    return fallback

def summarize_backend_manifest(manifest, environment):
    if not manifest:
        return {{
            'environment': environment,
            'manifest_present': False,
            'hostname': None,
            'dry_run': None,
            'selected': 0,
            'missing_input': 0,
            'excluded': 0,
            'not_selected': 0,
            'failed_runs': 0,
            'compare_failed': False,
        }}
    selection = manifest.get('pipeline_selection', []) or []
    if not selection:
        selection = [{{'selection_status': 'selected'}} for _ in (manifest.get('selected_pipelines', []) or [])]
    counts = {{'selected': 0, 'missing_input': 0, 'excluded': 0, 'not_selected': 0}}
    for row in selection:
        status = str(row.get('selection_status', 'unknown'))
        if status in counts:
            counts[status] += 1
    runs = manifest.get('pipeline_runs', []) or []
    compare_run = manifest.get('compare_run') or {{}}
    return {{
        'environment': environment,
        'manifest_present': True,
        'hostname': manifest.get('hostname'),
        'dry_run': bool(manifest.get('dry_run', False)),
        'selected': counts['selected'],
        'missing_input': counts['missing_input'],
        'excluded': counts['excluded'],
        'not_selected': counts['not_selected'],
        'failed_runs': sum(1 for row in runs if row.get('status') == 'failed'),
        'compare_failed': bool(compare_run.get('status') == 'failed'),
    }}

manifest = load_latest_backend_manifest(ROOT, prefer_executed=True)
print(json.dumps({{
    'environment': 'remote',
    'availability': [
        {{
            'pipeline_id': item.pipeline_id,
            'label': item.label,
            'status': item.status,
            'reason': item.reason,
            'output_dir': item.output_dir,
        }}
        for item in availability
    ],
    'manifest_summary': summarize_backend_manifest(manifest, environment='remote'),
}}))
"""
    completed = subprocess.run(
        ["ssh", remote_host, f"cd {remote_project_dir} && {remote_python} - <<'PY'\n{inline}\nPY"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Remote health snapshot failed.")
    return json.loads(completed.stdout)


def _write_report(
    *,
    manifest_summary: pd.DataFrame,
    availability_comparison: pd.DataFrame,
    restore_checklist: pd.DataFrame,
    output_path: Path,
) -> Path:
    lines = [
        "# Backend Health Report",
        "",
        "This report compares current backend rerunnability and latest executed-manifest state across environments.",
        "",
        "## Manifest summary",
        "",
        markdown_table(manifest_summary),
        "",
        "## Availability comparison",
        "",
        markdown_table(availability_comparison),
        "",
    ]

    mismatches = availability_comparison[~availability_comparison["status_match"]].copy()
    if not mismatches.empty:
        lines.extend(
            [
                "## Availability mismatches",
                "",
                markdown_table(mismatches),
                "",
            ]
        )

    if not restore_checklist.empty:
        lines.extend(
            [
                "## Restore checklist",
                "",
                "These commands are generated from local-vs-remote availability mismatches.",
                "",
                markdown_table(restore_checklist),
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local-vs-remote backend health report.")
    parser.add_argument("--remote-host", required=True, help="Remote SSH target, e.g. user@host.")
    parser.add_argument("--remote-project-dir", default="sexvary", help="Remote project directory.")
    parser.add_argument("--remote-python", default="python3", help="Remote Python executable.")
    parser.add_argument("--output-root", default="results", help="Output root relative to the project root.")
    args = parser.parse_args()

    root = project_root(__file__)

    local_availability = discover_pipeline_availability(root)
    local_manifest = load_latest_backend_manifest(root, prefer_executed=True)
    local_payload = availability_payload(local_availability, environment="local", manifest=local_manifest)
    remote_payload = _remote_snapshot(
        args.remote_host,
        remote_project_dir=args.remote_project_dir,
        remote_python=args.remote_python,
    )

    tables_dir = ensure_dir(root / args.output_root / "tables")
    reports_dir = ensure_dir(root / args.output_root / "reports")

    local_availability_df = availability_to_frame(local_availability, environment="local")
    remote_availability_df = pd.DataFrame(
        [
            {
                "environment": "remote",
                "pipeline_id": row["pipeline_id"],
                "dataset_label": row["label"],
                "status": row["status"],
                "reason": row.get("reason"),
                "output_dir": row.get("output_dir"),
            }
            for row in remote_payload["availability"]
        ]
    ).sort_values(["pipeline_id"], kind="stable").reset_index(drop=True)

    comparison_df = compare_availability(local_availability_df, remote_availability_df)
    manifest_summary_df = manifest_summary_frame(
        local_payload["manifest_summary"],
        remote_payload["manifest_summary"],
    )
    restore_checklist_df = build_restore_checklist(
        comparison_df,
        remote_host=args.remote_host,
        remote_project_dir=args.remote_project_dir,
        local_project_dir=root,
    )

    local_availability_path = tables_dir / "backend_health_local_availability.csv"
    remote_availability_path = tables_dir / "backend_health_remote_availability.csv"
    comparison_path = tables_dir / "backend_health_availability_comparison.csv"
    manifest_summary_path = tables_dir / "backend_health_manifest_summary.csv"
    restore_checklist_path = tables_dir / "backend_health_restore_checklist.csv"
    report_path = reports_dir / "backend_health.md"

    local_availability_df.to_csv(local_availability_path, index=False)
    remote_availability_df.to_csv(remote_availability_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    manifest_summary_df.to_csv(manifest_summary_path, index=False)
    restore_checklist_df.to_csv(restore_checklist_path, index=False)
    _write_report(
        manifest_summary=manifest_summary_df,
        availability_comparison=comparison_df,
        restore_checklist=restore_checklist_df,
        output_path=report_path,
    )

    for path in [
        local_availability_path,
        remote_availability_path,
        comparison_path,
        manifest_summary_path,
        restore_checklist_path,
        report_path,
    ]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
