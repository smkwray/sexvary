#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import yaml

from sexvary.config import build_registry, normalize_local_dataset_path
from sexvary.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a local NLSY dataset path.")
    parser.add_argument("--dataset-id", required=True, help="Dataset id, e.g. nlsy79_main")
    parser.add_argument("--path", required=True, help="Absolute or relative path to the local file")
    parser.add_argument("--config-path", default="config/local_paths.yaml", help="Path to local paths yaml")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    spec = registry.get_dataset(args.dataset_id)
    if not spec.is_user_local:
        raise SystemExit(f"{args.dataset_id} is not configured as a user-local dataset.")

    config_path = root / args.config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    data.setdefault("local_datasets", {})
    data.setdefault("notes", {})
    data["local_datasets"][args.dataset_id] = normalize_local_dataset_path(args.path, root)
    if args.dataset_id not in data["notes"]:
        data["notes"][args.dataset_id] = f"Registered by script for {args.dataset_id}."

    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Registered {args.dataset_id} -> {data['local_datasets'][args.dataset_id]}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
