from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import DatasetSpec


@dataclass(frozen=True)
class NormalizedTraitFrame:
    dataset_id: str
    data: pd.DataFrame
    provenance: dict


class BaseAdapter:
    """Base class for dataset-specific adapters.

    Adapters should convert raw data into a standard long person-trait table with
    at least these columns:

    - source_id
    - dataset_id
    - cycle_or_wave
    - country
    - grade_or_age_band
    - person_id
    - sex_observed
    - age
    - trait_id
    - score_raw
    - weight_main

    Optional standardized survey columns can also be carried through when available:

    - pv_index
    - variance_method / replicate_method
    - fay_factor
    - n_replicates
    - dataset-specific replicate weight columns
    - design_strata / design_psu
    """

    def __init__(self, dataset_spec: DatasetSpec, raw_path: str | Path):
        self.dataset_spec = dataset_spec
        self.raw_path = Path(raw_path)

    def load_raw(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_long_person_trait(self) -> NormalizedTraitFrame:
        raise NotImplementedError
