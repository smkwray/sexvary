#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.orchestration import (
    PipelineInvocation,
    PipelineRunRecord,
    build_selection_records,
    discover_pipeline_availability,
    finalize_run_record,
    run_pipeline_invocation,
    write_run_manifest,
)
from sexvary.utils import project_root, utc_timestamp


def _dry_run_record(invocation: PipelineInvocation) -> PipelineRunRecord:
    now = utc_timestamp()
    return PipelineRunRecord(
        pipeline_id=invocation.pipeline_id,
        label=invocation.label,
        command=invocation.command,
        output_dir=invocation.output_dir,
        status="dry_run",
        returncode=0,
        started_utc=now,
        finished_utc=now,
        duration_seconds=0.0,
        stdout_tail="",
        stderr_tail="",
        notes=invocation.notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all available backend pipelines and rebuild combined outputs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child pipeline runs.")
    parser.add_argument("--include", action="append", help="Optional pipeline id(s) to include.")
    parser.add_argument("--exclude", action="append", help="Optional pipeline id(s) to exclude.")
    parser.add_argument("--skip-compare", action="store_true", help="Skip the final cross-dataset comparison rebuild.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if any pipeline fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run and write a manifest without executing pipelines.")
    parser.add_argument("--manifest-path", help="Optional explicit manifest JSON output path.")
    args = parser.parse_args()

    root = project_root(__file__)
    availability = discover_pipeline_availability(root, python_executable=args.python)
    selected, selection_records = build_selection_records(
        availability,
        include=set(args.include or []),
        exclude=set(args.exclude or []),
    )
    if not selected and not args.skip_compare:
        raise SystemExit("No runnable pipelines were detected after filtering.")

    for record in selection_records:
        if record.selection_status != "selected":
            reason = f" ({record.reason})" if record.reason else ""
            print(f"[backend] skip {record.pipeline_id}: {record.selection_status}{reason}")

    run_records: list[PipelineRunRecord] = []
    halted_on_error = False
    for invocation in selected:
        print(f"[backend] {invocation.pipeline_id}: {' '.join(invocation.command)}")
        if args.dry_run:
            run_records.append(_dry_run_record(invocation))
            continue
        record = finalize_run_record(run_pipeline_invocation(invocation, root=root))
        run_records.append(record)
        print(record.stdout_tail.strip())
        if record.status != "ok":
            if record.stderr_tail.strip():
                print(record.stderr_tail.strip(), file=sys.stderr)
            if args.stop_on_error:
                halted_on_error = True
                break

    compare_record: PipelineRunRecord | None = None
    if not args.skip_compare and not halted_on_error:
        compare_invocation = PipelineInvocation(
            pipeline_id="compare_results",
            label="Cross-dataset comparison",
            command=[args.python, "scripts/run_cross_dataset_comparison.py"],
            output_dir="results",
            notes="Rebuild combined tables, figures, and reports.",
        )
        print(f"[backend] {compare_invocation.pipeline_id}: {' '.join(compare_invocation.command)}")
        if args.dry_run:
            compare_record = _dry_run_record(compare_invocation)
        else:
            compare_record = finalize_run_record(run_pipeline_invocation(compare_invocation, root=root))
            print(compare_record.stdout_tail.strip())
            if compare_record.status != "ok" and compare_record.stderr_tail.strip():
                print(compare_record.stderr_tail.strip(), file=sys.stderr)

    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    json_path, md_path = write_run_manifest(
        root=root,
        python_executable=args.python,
        selected_invocations=selected,
        selection_records=selection_records,
        run_records=run_records,
        compare_record=compare_record,
        dry_run=bool(args.dry_run),
        manifest_path=manifest_path,
    )
    print(f"[backend] manifest: {json_path}")
    print(f"[backend] summary: {md_path}")

    failed = [record for record in run_records if record.status == "failed"]
    if compare_record and compare_record.status == "failed":
        failed.append(compare_record)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
