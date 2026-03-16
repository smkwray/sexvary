from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .timss import TIMSSAdapter


class PIRLSAdapter(TIMSSAdapter):
    TRAIT_PATTERNS = {
        ("4", "reading_achievement"): r"^ASRREA(\d{2})$",
        ("4", "reading_informational"): r"^ASRINF(\d{2})$",
        ("4", "reading_literary"): r"^ASRLIT(\d{2})$",
    }

    def __init__(
        self,
        dataset_spec,
        raw_path: str | Path | Iterable[str | Path],
        *,
        country_ids: list[str] | None = None,
        grades: list[str] | None = None,
        traits: list[str] | None = None,
    ):
        super().__init__(
            dataset_spec,
            raw_path,
            country_ids=country_ids,
            grades=grades,
            traits=traits,
            trait_patterns=self.TRAIT_PATTERNS,
            cycle_label_template="pirls2021_grade{grade}",
        )
