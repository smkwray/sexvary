from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sexvary.adapters import CPPAdapter
from sexvary.config import build_registry
from sexvary.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the CPPAdapter on real downloaded files.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--mode", choices=["cpp_core", "cpp_growth"], default="cpp_core")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    root = project_root(__file__)
    registry = build_registry(root)
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    mapping = Path(args.mapping).expanduser().resolve()
    spec = registry.get_dataset(args.mode)

    adapter = CPPAdapter(
        dataset_spec=spec,
        raw_path=raw_dir,
        mapping_path=mapping,
        mode=args.mode,
    )
    normalized = adapter.to_long_person_trait()
    df = normalized.data

    if df.empty:
        raise SystemExit("Adapter emitted zero rows.")

    print("Rows:", len(df))
    print("Traits:", sorted(df["trait_id"].dropna().astype(str).unique().tolist()))
    print("Waves:", sorted(df["cycle_or_wave"].dropna().astype(str).unique().tolist()))
    print()
    print(df.groupby(["cycle_or_wave", "trait_id"], dropna=False).size().reset_index(name="n"))

    if args.output_csv:
        out = Path(args.output_csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
