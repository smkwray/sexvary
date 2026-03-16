from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import pyreadstat


SAS_INPUT_LINE_RE = re.compile(r"^\s*@(?P<start>\d+)\s+(?P<name>[A-Za-z0-9_]+)\s+(?P<format>\$?\d+(?:\.\d*)?)\s*$")
FIXED_WIDTH_POSITION_RE = re.compile(
    r"(?:(?P<type>str)\s+)?(?P<name>[A-Za-z0-9_]+)\s+(?P<start>\d+)\s*-\s*(?P<end>\d+)"
)


@dataclass(frozen=True)
class FixedWidthColumnSpec:
    name: str
    start: int
    end: int
    is_string: bool


def _detect_csv_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(8192)
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return ","
    return dialect.delimiter


def parse_sas_input_columns(path: str | Path) -> list[FixedWidthColumnSpec]:
    text = Path(path).read_text(encoding="latin-1")
    in_input_block = False
    specs: list[FixedWidthColumnSpec] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not in_input_block:
            if re.match(r"^\s*input\s*$", line, re.IGNORECASE):
                in_input_block = True
            continue
        if line.strip() == ";":
            break
        match = SAS_INPUT_LINE_RE.match(line)
        if match is not None:
            start = int(match.group("start")) - 1
            fmt = match.group("format")
            is_string = fmt.startswith("$")
            width = int(fmt.lstrip("$").split(".", 1)[0])
            specs.append(
                FixedWidthColumnSpec(
                    name=match.group("name"),
                    start=start,
                    end=start + width,
                    is_string=is_string,
                )
            )
            continue
        for position_match in FIXED_WIDTH_POSITION_RE.finditer(line):
            start = int(position_match.group("start")) - 1
            end = int(position_match.group("end"))
            specs.append(
                FixedWidthColumnSpec(
                    name=position_match.group("name"),
                    start=start,
                    end=end,
                    is_string=position_match.group("type") == "str",
                )
            )
    if not specs:
        raise ValueError(f"No SAS input block columns found in {path}.")
    deduped: list[FixedWidthColumnSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def read_fixed_width_from_sas_input(
    data_path: str | Path,
    *,
    sas_path: str | Path,
    usecols: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    specs = parse_sas_input_columns(sas_path)
    if usecols is not None:
        wanted = set(usecols)
        specs = [spec for spec in specs if spec.name in wanted]
        missing = sorted(wanted - {spec.name for spec in specs})
        if missing:
            raise KeyError(f"Requested SAS input columns were not found in {sas_path}: {missing}")
    colspecs = [(spec.start, spec.end) for spec in specs]
    names = [spec.name for spec in specs]
    dtype = {spec.name: "string" for spec in specs if spec.is_string}
    return pd.read_fwf(data_path, colspecs=colspecs, names=names, dtype=dtype or None, **kwargs)


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if "sep" not in kwargs and "delimiter" not in kwargs:
            kwargs["sep"] = _detect_csv_delimiter(path)
        kwargs.setdefault("low_memory", False)
        return pd.read_csv(path, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, **kwargs)
    if suffix == ".dta":
        return pd.read_stata(path, **kwargs)
    if suffix == ".sav":
        return pd.read_spss(path, **kwargs)
    if suffix == ".xpt":
        kwargs.pop("encoding", None)
        df, _ = pyreadstat.read_xport(path, encoding="LATIN1")
        return df
    if suffix == ".sas7bdat":
        return pd.read_sas(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported file type for read_table: {path.suffix}")


def write_table(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file type for write_table: {path.suffix}")
    return path
