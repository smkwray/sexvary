#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import subprocess

import pandas as pd
from pandas.errors import EmptyDataError

from sexvary.utils import project_root


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"[sync-health] {' '.join(command)}")
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    return int(completed.returncode)


def _restore_count(checklist_path: Path) -> int:
    if not checklist_path.exists():
        return 0
    try:
        df = pd.read_csv(checklist_path)
    except EmptyDataError:
        return 0
    return int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backend health, apply restores, and rerun backend health.")
    parser.add_argument("--remote-host", required=True, help="Remote SSH target, e.g. user@host.")
    parser.add_argument("--remote-project-dir", default="sexvary", help="Remote project directory.")
    parser.add_argument("--remote-python", default="python3", help="Remote Python executable.")
    parser.add_argument("--output-root", default="results", help="Output root relative to the project root.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    root = project_root(__file__)
    checklist_path = root / args.output_root / "tables" / "backend_health_restore_checklist.csv"

    health_command = [
        sys.executable,
        "scripts/run_backend_health_report.py",
        "--remote-host",
        args.remote_host,
        "--remote-project-dir",
        args.remote_project_dir,
        "--remote-python",
        args.remote_python,
        "--output-root",
        args.output_root,
    ]
    restore_command = [
        sys.executable,
        "scripts/run_backend_restore.py",
        "--checklist-path",
        str(checklist_path),
    ]

    if _run(health_command, cwd=root, dry_run=args.dry_run) != 0:
        raise SystemExit(1)

    restore_rows = _restore_count(checklist_path)
    print(f"[sync-health] restore actions: {restore_rows}")
    if restore_rows == 0:
        print("[sync-health] no restore actions needed")
        return

    if _run(restore_command, cwd=root, dry_run=args.dry_run) != 0:
        raise SystemExit(1)

    if _run(health_command, cwd=root, dry_run=args.dry_run) != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
