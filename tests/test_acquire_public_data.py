from __future__ import annotations

from pathlib import Path
import importlib.util

from sexvary.config import build_registry


_ACQUIRE_SPEC = importlib.util.spec_from_file_location(
    "acquire_public_data",
    Path(__file__).resolve().parents[1] / "scripts" / "acquire_public_data.py",
)
assert _ACQUIRE_SPEC is not None and _ACQUIRE_SPEC.loader is not None
_ACQUIRE_MODULE = importlib.util.module_from_spec(_ACQUIRE_SPEC)
_ACQUIRE_SPEC.loader.exec_module(_ACQUIRE_MODULE)
write_source_note = _ACQUIRE_MODULE.write_source_note


def test_write_source_note_handles_multiple_dataset_specs_in_one_directory(tmp_path: Path):
    registry = build_registry()
    path = write_source_note(tmp_path, [registry.get_dataset("cpp_core"), registry.get_dataset("cpp_growth")])
    text = path.read_text(encoding="utf-8")

    assert path == tmp_path / "SOURCE.md"
    assert "dataset_ids: `cpp_core`, `cpp_growth`" in text
    assert "## CPP core cognition" in text
    assert "## CPP growth trajectories" in text
