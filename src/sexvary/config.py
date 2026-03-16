from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from .utils import project_root


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    tier: int
    priority: str
    access_mode: str
    automation: str
    official_url: str
    methodology_url: str | None = None
    target_population: str | None = None
    domains: list[str] = field(default_factory=list)
    candidate_traits: list[str] = field(default_factory=list)
    design_type: str | None = None
    expected_local_subdir: str | None = None
    notes: str | None = None
    caveats: list[str] = field(default_factory=list)

    @property
    def is_user_local(self) -> bool:
        return self.access_mode == "user_local"

    @property
    def is_external(self) -> bool:
        return not self.is_user_local


@dataclass(frozen=True)
class TraitSpec:
    id: str
    label: str
    family: str
    scale_type: str
    priority: str
    recommended_metrics: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class Registry:
    root: Path
    datasets: dict[str, DatasetSpec]
    traits: dict[str, TraitSpec]
    analysis_config: dict[str, Any]

    def get_dataset(self, dataset_id: str) -> DatasetSpec:
        try:
            return self.datasets[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset id: {dataset_id}") from exc

    def get_trait(self, trait_id: str) -> TraitSpec:
        try:
            return self.traits[trait_id]
        except KeyError as exc:
            raise KeyError(f"Unknown trait id: {trait_id}") from exc

    def dataset_ids(self) -> list[str]:
        return list(self.datasets)

    def trait_ids(self) -> list[str]:
        return list(self.traits)

    def external_datasets(self) -> list[DatasetSpec]:
        return [spec for spec in self.datasets.values() if spec.is_external]

    def local_datasets(self) -> list[DatasetSpec]:
        return [spec for spec in self.datasets.values() if spec.is_user_local]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML file must contain a dictionary at top level: {path}")
    return data


def _config_dir(root: str | Path | None = None) -> Path:
    if root is None:
        root = project_root()
    return Path(root) / "config"


def load_datasets(root: str | Path | None = None) -> list[DatasetSpec]:
    data = _load_yaml(_config_dir(root) / "datasets.yaml")
    items = data.get("datasets", [])
    if not isinstance(items, list):
        raise TypeError("config/datasets.yaml must define a list under `datasets`.")
    out: list[DatasetSpec] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise TypeError("Each dataset entry must be a mapping.")
        out.append(DatasetSpec(**raw))
    return out


def load_traits(root: str | Path | None = None) -> list[TraitSpec]:
    data = _load_yaml(_config_dir(root) / "traits.yaml")
    items = data.get("traits", [])
    if not isinstance(items, list):
        raise TypeError("config/traits.yaml must define a list under `traits`.")
    out: list[TraitSpec] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise TypeError("Each trait entry must be a mapping.")
        out.append(TraitSpec(**raw))
    return out


def load_analysis_config(root: str | Path | None = None) -> dict[str, Any]:
    return _load_yaml(_config_dir(root) / "analysis.yaml")


def load_local_paths(root: str | Path | None = None, *, missing_ok: bool = True) -> dict[str, Any]:
    path = _config_dir(root) / "local_paths.yaml"
    if not path.exists():
        if missing_ok:
            return {"local_datasets": {}, "notes": {}}
        raise FileNotFoundError(f"Local paths file not found: {path}")
    return _load_yaml(path)


def resolve_local_dataset_path(path: str | Path, root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    base = project_root(root) if root is not None else project_root()
    return (base / candidate).resolve()


def normalize_local_dataset_path(path: str | Path, root: str | Path | None = None) -> str:
    base = project_root(root) if root is not None else project_root()
    resolved = Path(path).expanduser().resolve()
    try:
        relative = os.path.relpath(resolved, start=base)
    except ValueError:
        return str(resolved)
    return Path(relative).as_posix()


def build_registry(root: str | Path | None = None) -> Registry:
    root_path = project_root(root) if root is not None else project_root()
    datasets = load_datasets(root_path)
    traits = load_traits(root_path)
    analysis = load_analysis_config(root_path)

    dataset_map = {item.id: item for item in datasets}
    if len(dataset_map) != len(datasets):
        raise ValueError("Duplicate dataset ids found in config/datasets.yaml.")

    trait_map = {item.id: item for item in traits}
    if len(trait_map) != len(traits):
        raise ValueError("Duplicate trait ids found in config/traits.yaml.")

    return Registry(root=root_path, datasets=dataset_map, traits=trait_map, analysis_config=analysis)
