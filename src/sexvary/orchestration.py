from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

from .config import load_local_paths
from .utils import ensure_dir, external_data_dirs, utc_timestamp


RAW_SUFFIXES = {".csv", ".sav", ".parquet", ".dta", ".dat", ".zip", ".xlsx", ".xpt"}
METADATA_BASENAMES = {"FILE_INVENTORY.csv", "SOURCE.md", "TRANSFORM_LOG.md"}
TIMSS_PREFERRED_PATTERNS = (
    re.compile(r"^asg.*a[mzb][78]\.(sav|csv|parquet|dta)$", re.IGNORECASE),
    re.compile(r"^bsg.*a[mzb][78]\.(sav|csv|parquet|dta)$", re.IGNORECASE),
)


@dataclass(frozen=True)
class PipelineInvocation:
    pipeline_id: str
    label: str
    command: list[str]
    output_dir: str
    notes: str | None = None


@dataclass(frozen=True)
class PipelineRunRecord:
    pipeline_id: str
    label: str
    command: list[str]
    output_dir: str
    status: str
    returncode: int
    started_utc: str
    finished_utc: str
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str
    notes: str | None = None


@dataclass(frozen=True)
class PipelineAvailability:
    pipeline_id: str
    label: str
    status: str
    reason: str | None = None
    invocation: PipelineInvocation | None = None
    output_dir: str | None = None


@dataclass(frozen=True)
class PipelineSelectionRecord:
    pipeline_id: str
    label: str
    selection_status: str
    reason: str | None = None
    output_dir: str | None = None


def _supported_files(data_dir: Path, *, patterns: Iterable[str] | None = None) -> list[Path]:
    if not data_dir.exists():
        return []
    if patterns is None:
        return [
            path
            for path in sorted(data_dir.iterdir())
            if path.is_file() and path.name not in METADATA_BASENAMES and path.suffix.lower() in RAW_SUFFIXES
        ]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(data_dir.glob(pattern)))
    return [path for path in out if path.is_file() and path.name not in METADATA_BASENAMES]


def _local_nlsy_available(root: Path) -> tuple[bool, str | None]:
    local_paths = load_local_paths(root, missing_ok=True).get("local_datasets", {}) or {}
    existing = []
    for dataset_id, entry in local_paths.items():
        raw_path = entry.get("path") if isinstance(entry, dict) else entry if isinstance(entry, str) else None
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        if candidate.exists():
            existing.append(dataset_id)
    if not existing:
        return False, None
    return True, f"{len(existing)} configured local dataset(s): {', '.join(sorted(existing))}"


def _external_available(root: Path, dataset_id: str, *, patterns: Iterable[str] | None = None) -> tuple[bool, str | None]:
    searched = external_data_dirs(dataset_id, root)
    for data_dir in searched:
        files = _supported_files(data_dir, patterns=patterns)
        if files:
            return True, f"{data_dir}: {', '.join(path.name for path in files[:3])}"
    return False, "Searched: " + ", ".join(str(path) for path in searched)


def _hrs_available(root: Path) -> tuple[bool, str | None]:
    searched = external_data_dirs("hrs_public", root)
    for data_dir in searched:
        core_archives = sorted(path for path in data_dir.glob("*core.zip") if path.is_file() and path.name not in METADATA_BASENAMES)
        tracker_archives = sorted(path for path in data_dir.glob("trk*.zip") if path.is_file() and path.name not in METADATA_BASENAMES)
        if core_archives and tracker_archives:
            notes = ", ".join(path.name for path in (core_archives[:2] + tracker_archives[:1]))
            return True, f"{data_dir}: {notes}"
    return False, "Searched: " + ", ".join(str(path) for path in searched)

def discover_pipeline_availability(root: Path, *, python_executable: str | None = None) -> list[PipelineAvailability]:
    python = python_executable or sys.executable
    availability: list[PipelineAvailability] = []

    local_nlsy_ok, local_nlsy_notes = _local_nlsy_available(root)
    if local_nlsy_ok:
        invocation = PipelineInvocation(
            pipeline_id="local_nlsy",
            label="Local NLSY",
            command=[python, "scripts/run_local_nlsy_pipeline.py"],
            output_dir="results/local_nlsy",
            notes=local_nlsy_notes,
        )
        availability.append(
            PipelineAvailability(
                pipeline_id="local_nlsy",
                label="Local NLSY",
                status="runnable",
                reason=local_nlsy_notes,
                invocation=invocation,
                output_dir=invocation.output_dir,
            )
        )
    else:
        availability.append(
            PipelineAvailability(
                pipeline_id="local_nlsy",
                label="Local NLSY",
                status="missing_input",
                reason="No configured local NLSY dataset paths resolved to existing files.",
            )
        )

    for dataset_id, script_name, label in (
        ("piaac_cycle2", "run_piaac_pipeline.py", "PIAAC cycle 2"),
        ("pisa_2022", "run_pisa_pipeline.py", "PISA 2022"),
        ("nhanes_2011_2023", "run_nhanes_pipeline.py", "NHANES selected cycles"),
        ("nnyfs_2012", "run_nnyfs_pipeline.py", "NNYFS 2012"),
        ("psid_cds_tas", "run_psid_pipeline.py", "PSID CDS / TAS"),
    ):
        available, notes = _external_available(root, dataset_id)
        if available:
            invocation = PipelineInvocation(
                pipeline_id=dataset_id,
                label=label,
                command=[python, f"scripts/{script_name}"],
                output_dir=f"results/{dataset_id}",
                notes=notes,
            )
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="runnable",
                    reason=notes,
                    invocation=invocation,
                    output_dir=invocation.output_dir,
                )
            )
        else:
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="missing_input",
                    reason=notes,
                )
            )

    hrs_available, hrs_notes = _hrs_available(root)
    if hrs_available:
        invocation = PipelineInvocation(
            pipeline_id="hrs_public",
            label="HRS public",
            command=[python, "scripts/run_hrs_pipeline.py"],
            output_dir="results/hrs_public",
            notes=hrs_notes,
        )
        availability.append(
            PipelineAvailability(
                pipeline_id="hrs_public",
                label="HRS public",
                status="runnable",
                reason=hrs_notes,
                invocation=invocation,
                output_dir=invocation.output_dir,
            )
        )
    else:
        availability.append(
            PipelineAvailability(
                pipeline_id="hrs_public",
                label="HRS public",
                status="missing_input",
                reason=hrs_notes,
            )
        )

    for dataset_id, label in (("timss_2019", "TIMSS 2019"), ("timss_2023", "TIMSS 2023")):
        timss_available, timss_notes = _external_available(root, dataset_id)
        if timss_available:
            invocation = PipelineInvocation(
                pipeline_id=dataset_id,
                label=label,
                command=[python, "scripts/run_timss_pipeline.py", "--dataset-id", dataset_id],
                output_dir=f"results/{dataset_id}",
                notes=timss_notes,
            )
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="runnable",
                    reason=timss_notes,
                    invocation=invocation,
                    output_dir=invocation.output_dir,
                )
            )
        else:
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="missing_input",
                    reason=timss_notes,
                )
            )

    for dataset_id, script_name, label in (
        ("pirls_2021", "run_pirls_pipeline.py", "PIRLS 2021"),
        ("icils_2023", "run_icils_pipeline.py", "ICILS 2023"),
    ):
        available, notes = _external_available(root, dataset_id)
        if available:
            invocation = PipelineInvocation(
                pipeline_id=dataset_id,
                label=label,
                command=[python, f"scripts/{script_name}"],
                output_dir=f"results/{dataset_id}",
                notes=notes,
            )
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="runnable",
                    reason=notes,
                    invocation=invocation,
                    output_dir=invocation.output_dir,
                )
            )
        else:
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="missing_input",
                    reason=notes,
                )
            )

    for dataset_id, label in (("ecls_k_2011", "ECLS-K:2011"), ("hsls_2009", "HSLS:09")):
        available, notes = _external_available(root, dataset_id, patterns=("*.dat", "*.zip", "*.sav", "*.csv", "*.parquet", "*.dta", "*.xlsx"))
        if available:
            invocation = PipelineInvocation(
                pipeline_id=dataset_id,
                label=label,
                command=[python, "scripts/run_nces_school_pipeline.py", "--dataset-id", dataset_id],
                output_dir=f"results/{dataset_id}",
                notes=notes,
            )
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="runnable",
                    reason=notes,
                    invocation=invocation,
                    output_dir=invocation.output_dir,
                )
            )
        else:
            availability.append(
                PipelineAvailability(
                    pipeline_id=dataset_id,
                    label=label,
                    status="missing_input",
                    reason=notes,
                )
            )

    return availability


def discover_pipeline_invocations(root: Path, *, python_executable: str | None = None) -> list[PipelineInvocation]:
    return [
        item.invocation
        for item in discover_pipeline_availability(root, python_executable=python_executable)
        if item.status == "runnable" and item.invocation is not None
    ]


def build_selection_records(
    availability: list[PipelineAvailability],
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> tuple[list[PipelineInvocation], list[PipelineSelectionRecord]]:
    selected: list[PipelineInvocation] = []
    records: list[PipelineSelectionRecord] = []

    for item in availability:
        if item.status != "runnable" or item.invocation is None:
            records.append(
                PipelineSelectionRecord(
                    pipeline_id=item.pipeline_id,
                    label=item.label,
                    selection_status="missing_input",
                    reason=item.reason,
                    output_dir=item.output_dir,
                )
            )
            continue
        if include and item.pipeline_id not in include:
            records.append(
                PipelineSelectionRecord(
                    pipeline_id=item.pipeline_id,
                    label=item.label,
                    selection_status="not_selected",
                    reason="Not requested by include filter.",
                    output_dir=item.output_dir,
                )
            )
            continue
        if exclude and item.pipeline_id in exclude:
            records.append(
                PipelineSelectionRecord(
                    pipeline_id=item.pipeline_id,
                    label=item.label,
                    selection_status="excluded",
                    reason="Excluded by filter.",
                    output_dir=item.output_dir,
                )
            )
            continue
        selected.append(item.invocation)
        records.append(
            PipelineSelectionRecord(
                pipeline_id=item.pipeline_id,
                label=item.label,
                selection_status="selected",
                reason=item.reason,
                output_dir=item.output_dir,
            )
        )

    return selected, records


def filter_invocations(
    invocations: list[PipelineInvocation],
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[PipelineInvocation]:
    out = invocations
    if include:
        out = [item for item in out if item.pipeline_id in include]
    if exclude:
        out = [item for item in out if item.pipeline_id not in exclude]
    return out


def run_pipeline_invocation(invocation: PipelineInvocation, *, root: Path) -> PipelineRunRecord:
    started = utc_timestamp()
    completed = subprocess.run(
        invocation.command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    finished = utc_timestamp()
    status = "ok" if completed.returncode == 0 else "failed"
    return PipelineRunRecord(
        pipeline_id=invocation.pipeline_id,
        label=invocation.label,
        command=invocation.command,
        output_dir=invocation.output_dir,
        status=status,
        returncode=int(completed.returncode),
        started_utc=started,
        finished_utc=finished,
        duration_seconds=max(0.0, (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()),
        stdout_tail=completed.stdout[-4000:],
        stderr_tail=completed.stderr[-4000:],
        notes=invocation.notes,
    )

def finalize_run_record(record: PipelineRunRecord) -> PipelineRunRecord:
    return record


def write_run_manifest(
    *,
    root: Path,
    python_executable: str,
    selected_invocations: list[PipelineInvocation],
    selection_records: list[PipelineSelectionRecord],
    run_records: list[PipelineRunRecord],
    compare_record: PipelineRunRecord | None,
    dry_run: bool,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    manifest_dir = ensure_dir(root / "results" / "run_manifests")
    timestamp_slug = utc_timestamp().replace(":", "").replace("+00:00", "Z")
    json_path = manifest_path or (manifest_dir / f"backend_run_{timestamp_slug}.json")
    md_path = json_path.with_suffix(".md")

    payload = {
        "started_utc": run_records[0].started_utc if run_records else utc_timestamp(),
        "finished_utc": compare_record.finished_utc if compare_record else (run_records[-1].finished_utc if run_records else utc_timestamp()),
        "project_root": str(root),
        "hostname": platform.node(),
        "python_executable": python_executable,
        "dry_run": dry_run,
        "selected_pipelines": [asdict(item) for item in selected_invocations],
        "pipeline_selection": [asdict(item) for item in selection_records],
        "pipeline_runs": [asdict(item) for item in run_records],
        "compare_run": asdict(compare_record) if compare_record else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Backend run manifest",
        "",
        f"- Started UTC: `{payload['started_utc']}`",
        f"- Finished UTC: `{payload['finished_utc']}`",
        f"- Host: `{payload['hostname']}`",
        f"- Python: `{python_executable}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Pipeline selection",
        "",
        "| pipeline_id | selection_status | output_dir | reason |",
        "| --- | --- | --- | --- |",
    ]
    for record in selection_records:
        lines.append(
            f"| {record.pipeline_id} | {record.selection_status} | {record.output_dir or ''} | {(record.reason or '').replace('|', '/')} |"
        )

    lines.extend(
        [
        "",
        "## Pipelines",
        "",
        "| pipeline_id | status | returncode | output_dir | notes |",
        "| --- | --- | ---: | --- | --- |",
    ])
    for record in run_records:
        lines.append(
            f"| {record.pipeline_id} | {record.status} | {record.returncode} | {record.output_dir} | {(record.notes or '').replace('|', '/')} |"
        )
    if compare_record:
        lines.extend(
            [
                "",
                "## Comparison rebuild",
                "",
                "| pipeline_id | status | returncode | output_dir |",
                "| --- | --- | ---: | --- |",
                f"| {compare_record.pipeline_id} | {compare_record.status} | {compare_record.returncode} | {compare_record.output_dir} |",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest_json = manifest_dir / "backend_run_latest.json"
    latest_md = manifest_dir / "backend_run_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path
