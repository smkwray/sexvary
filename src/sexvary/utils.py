from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re


SHARED_DATA_ROOT_ENV = "PROJ_SHARED_DATA_ROOT"
METADATA_BASENAMES = {"FILE_INVENTORY.csv", "SOURCE.md", "TRANSFORM_LOG.md"}

CANONICAL_EXTERNAL_DATASET_PATHS = {
    "piaac_cycle2": ("oecd", "piaac", "cycle2"),
    "pisa_2022": ("oecd", "pisa", "2022"),
    "timss_2019": ("iea", "timss", "2019"),
    "timss_2023": ("iea", "timss", "2023"),
    "pirls_2021": ("iea", "pirls", "2021"),
    "icils_2023": ("iea", "icils", "2023"),
    "nhanes_2011_2023": ("cdc", "nhanes", "selected_cycles_2011_2023"),
    "nnyfs_2012": ("cdc", "nnyfs", "2012"),
    "ecls_k_2011": ("nces", "ecls_k", "2011"),
    "hsls_2009": ("nces", "hsls", "2009"),
    "els_2002": ("nces", "els", "2002"),
    "psid_cds_tas": ("umich", "psid_cds_tas", "public"),
    "add_health_public": ("unc", "add_health", "public_use"),
    "hrs_public": ("umich", "hrs", "public"),
    "midus_public": ("uw_madison", "midus", "public"),
}


def project_root(start: str | Path | None = None) -> Path:
    if start is None:
        return Path(__file__).resolve().parents[2]
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "config").exists():
            return candidate
        if candidate == path and (candidate / "config").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root from the supplied path.")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def shared_data_root(start: str | Path | None = None) -> Path:
    configured = os.environ.get(SHARED_DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    root = project_root(start)
    sibling = (root.parent / "data").resolve()
    if sibling.exists():
        return sibling
    return (root / "data").resolve()


def canonical_external_data_dir(dataset_id: str, start: str | Path | None = None) -> Path | None:
    parts = CANONICAL_EXTERNAL_DATASET_PATHS.get(dataset_id)
    if parts is None:
        return None
    return shared_data_root(start) / "sources" / parts[0] / parts[1] / parts[2]


def legacy_external_data_dir(dataset_id: str, start: str | Path | None = None) -> Path:
    return project_root(start) / "data" / "raw" / "external" / dataset_id


def external_data_dirs(dataset_id: str, start: str | Path | None = None) -> list[Path]:
    dirs: list[Path] = []
    canonical = canonical_external_data_dir(dataset_id, start)
    if canonical is not None:
        dirs.append(canonical)
    legacy = legacy_external_data_dir(dataset_id, start)
    if legacy not in dirs:
        dirs.append(legacy)
    return dirs


def first_existing_external_data_dir(dataset_id: str, start: str | Path | None = None) -> Path | None:
    for path in external_data_dirs(dataset_id, start):
        if path.exists():
            return path
    return None


def existing_external_dataset_files(
    dataset_id: str,
    *,
    start: str | Path | None = None,
    patterns: tuple[str, ...],
) -> tuple[Path | None, list[Path]]:
    for data_dir in external_data_dirs(dataset_id, start):
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(data_dir.glob(pattern)))
        files = [path for path in files if path.is_file() and path.name not in METADATA_BASENAMES]
        if files:
            return data_dir, files
    return None, []


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")
