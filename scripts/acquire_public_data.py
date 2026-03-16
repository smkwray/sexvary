#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from sexvary.config import build_registry
from sexvary.utils import canonical_external_data_dir, ensure_dir, project_root, shared_data_root, utc_timestamp


def write_source_note(base_dir: Path, spec) -> Path:
    path = base_dir / "SOURCE.md"
    lines = [
        f"# {spec.name}",
        "",
        f"- dataset_id: `{spec.id}`",
        f"- access_mode: `{spec.access_mode}`",
        f"- automation: `{spec.automation}`",
        f"- official_url: {spec.official_url}",
    ]
    if spec.methodology_url:
        lines.append(f"- methodology_url: {spec.methodology_url}")
    if spec.design_type:
        lines.append(f"- design_type: `{spec.design_type}`")
    lines.extend(
        [
            f"- generated_utc: {utc_timestamp()}",
            "",
            "## Notes",
            spec.notes or "_No additional notes provided._",
            "",
        ]
    )
    if spec.caveats:
        lines.append("## Caveats")
        lines.extend([f"- {item}" for item in spec.caveats])
        lines.append("")
    lines.extend(
        [
            "## Manual acquisition status",
            "- [ ] Open official page",
            "- [ ] Download public-use files or register if required",
            "- [ ] Save archive or extracted files into this folder",
            "- [ ] Create FILE_INVENTORY.csv",
            "- [ ] Record any manual transformations in TRANSFORM_LOG.md",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_checklist(root: Path, registry) -> Path:
    path = root / "docs" / "PUBLIC_DATA_CHECKLIST.md"
    lines = [
        "# Public data acquisition checklist",
        "",
        f"Generated: {utc_timestamp()}",
        "",
        "| Dataset | Tier | Access mode | Automation | Folder | Status |",
        "|---|---:|---|---|---|---|",
    ]
    for spec in registry.external_datasets():
        canonical = canonical_external_data_dir(spec.id, root)
        folder = str(canonical) if canonical is not None else f"data/raw/external/{spec.id}"
        lines.append(
            f"| {spec.name} | {spec.tier} | {spec.access_mode} | {spec.automation} | `{folder}` | pending |"
        )
    lines.append("")
    lines.append("Portal-based datasets may require a human login or click-through agreement.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create external dataset acquisition folders and source notes.")
    parser.add_argument("--write-checklist", action="store_true", help="Also write docs/PUBLIC_DATA_CHECKLIST.md")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    shared_root = shared_data_root(root)

    for spec in registry.external_datasets():
        canonical = canonical_external_data_dir(spec.id, root)
        base_dir = ensure_dir(canonical if canonical is not None else (root / "data" / "raw" / "external" / spec.id))
        write_source_note(base_dir, spec)
        inventory = base_dir / "FILE_INVENTORY.csv"
        if not inventory.exists():
            inventory.write_text("filename,notes\n", encoding="utf-8")
        transform_log = base_dir / "TRANSFORM_LOG.md"
        if not transform_log.exists():
            transform_log.write_text(
                "# Transform log\n\nRecord any manual extraction, recoding, or merging steps here.\n",
                encoding="utf-8",
            )

    if args.write_checklist:
        checklist_path = write_checklist(root, registry)
        print(f"Wrote checklist: {checklist_path}")

    print(f"Prepared {len(registry.external_datasets())} external dataset folders under {shared_root / 'sources'}")


if __name__ == "__main__":
    main()
