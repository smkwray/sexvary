#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sexvary.config import build_registry, load_local_paths, resolve_local_dataset_path
from sexvary.utils import project_root


def main() -> None:
    root = project_root(__file__)
    registry = build_registry(root)

    print(f"Project root: {root}")
    print(f"Datasets: {len(registry.datasets)}")
    print(f"Traits: {len(registry.traits)}")

    # Validate required top-level files/directories.
    required = [
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "config",
        root / "docs",
        root / "src",
        root / "scripts",
        root / "tests",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required project paths: {missing}")

    local = load_local_paths(root, missing_ok=True)
    local_datasets = local.get("local_datasets", {})
    print(f"Registered local datasets: {len(local_datasets)}")

    for dataset_id, path in local_datasets.items():
        resolved = resolve_local_dataset_path(path, root)
        status = "exists" if resolved.exists() else "missing"
        print(f"  - {dataset_id}: {path} -> {resolved} [{status}]")

    print("Validation OK")


if __name__ == "__main__":
    main()
