from __future__ import annotations

from pathlib import Path
import re
import shutil
import zipfile


def extract_matching_zip_members(
    zip_paths: list[Path],
    *,
    member_patterns: tuple[re.Pattern[str], ...],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as handle:
            for member in handle.namelist():
                if member.endswith("/"):
                    continue
                basename = Path(member).name
                if not any(pattern.match(basename) for pattern in member_patterns):
                    continue
                target = output_dir / basename
                if not target.exists():
                    with handle.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted.append(target)
    return sorted(dict.fromkeys(extracted))
