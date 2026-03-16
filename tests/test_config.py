from pathlib import Path

from sexvary.config import normalize_local_dataset_path, resolve_local_dataset_path


def test_resolve_local_dataset_path_uses_project_root_for_relative_paths():
    root = Path(__file__).resolve().parents[1]
    resolved = resolve_local_dataset_path("data/raw/nlsy/example.csv", root)
    assert resolved == (root / "data/raw/nlsy/example.csv").resolve()


def test_normalize_local_dataset_path_prefers_repo_relative_paths():
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "external" / "data" / "processed" / "nlsy79_cfa.csv"
    normalized = normalize_local_dataset_path(sibling, root)
    assert normalized == "../external/data/processed/nlsy79_cfa.csv"
