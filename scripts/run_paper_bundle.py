#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sexvary.paper_bundle import build_paper_bundle
from sexvary.utils import ensure_dir, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical manuscript-facing paper bundle.")
    parser.add_argument(
        "--output-dir",
        default="results/paper_bundle",
        help="Output directory relative to the project root.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Do not rerun the comparison/report build before packaging.",
    )
    args = parser.parse_args()

    root = project_root(__file__)
    if not args.skip_compare:
        subprocess.run([sys.executable, str(root / "scripts" / "run_cross_dataset_comparison.py")], check=True)

    output_dir = ensure_dir(root / args.output_dir)
    outputs = build_paper_bundle(root, output_dir)
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
