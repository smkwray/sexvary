from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FILES = [
    "cpp_clean_v1.csv",
    "cpp_cognitive_scores.csv",
    "cpp_g_factors.csv",
    "cpp_weights.csv",
    "cpp_growth_trajectories.csv",
    "cpp_birthweight_zscores.csv",
    "cpp_kinship_links.csv",
    "cpp_twin_zygosity.csv",
]


DEFAULT_KEYWORDS = [
    "case",
    "mother",
    "sex",
    "site",
    "center",
    "wisc",
    "sb",
    "stanford",
    "wrat",
    "digit",
    "memory",
    "g",
    "weight",
    "height",
    "birth",
    "gest",
]


def read_header(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def keyword_hits(columns: Iterable[str], keywords: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in keywords:
        hits = [col for col in columns if key.lower() in col.lower()]
        if hits:
            out[key] = hits
    return out


def build_report(raw_dir: Path, output_dir: Path, keywords: list[str]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, object]] = {}

    for filename in DEFAULT_FILES:
        path = raw_dir / filename
        if not path.exists():
            manifest[filename] = {"exists": False}
            continue
        cols = read_header(path)
        manifest[filename] = {
            "exists": True,
            "n_columns": len(cols),
            "columns": cols,
            "keyword_hits": keyword_hits(cols, keywords),
        }

    json_path = output_dir / "cpp_header_inventory.json"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# CPP header inspection report",
        "",
        f"Raw directory: `{raw_dir}`",
        "",
    ]

    for filename, payload in manifest.items():
        lines.append(f"## {filename}")
        if not payload.get("exists"):
            lines.append("")
            lines.append("File not found.")
            lines.append("")
            continue

        n_columns = payload["n_columns"]
        hits = payload["keyword_hits"]
        columns = payload["columns"]

        lines.append("")
        lines.append(f"- Column count: {n_columns}")
        lines.append("")
        if hits:
            lines.append("### Keyword hits")
            lines.append("")
            for key, values in hits.items():
                lines.append(f"- `{key}`: {', '.join(f'`{v}`' for v in values)}")
            lines.append("")
        else:
            lines.append("No keyword hits for the requested keywords.")
            lines.append("")

        lines.append("### First 60 columns")
        lines.append("")
        preview = columns[:60]
        lines.append(", ".join(f"`{c}`" for c in preview))
        lines.append("")

    md_path = output_dir / "cpp_header_inventory.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CPP CSV headers and emit a candidate-column report.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing downloaded CPP files.")
    parser.add_argument(
        "--output-dir",
        default="results/cpp_header_inventory",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_KEYWORDS,
        help="Keywords used to find likely ID, wave, score, and weight columns.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    json_path, md_path = build_report(raw_dir, output_dir, list(args.keywords))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
