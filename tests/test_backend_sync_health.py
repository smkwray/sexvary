from pathlib import Path

from scripts.run_backend_sync_health import _restore_count


def test_restore_count_is_zero_for_missing_or_empty_checklist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    assert _restore_count(missing) == 0

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert _restore_count(empty) == 0


def test_restore_count_reads_nonempty_checklist(tmp_path: Path) -> None:
    checklist = tmp_path / "checklist.csv"
    checklist.write_text(
        "pipeline_id,dataset_label,issue,remote_reason,suggested_command,notes\n"
        "timss_2019,TIMSS 2019,restore_local_raw_inputs,files,rsync -az 'a' 'b',note\n",
        encoding="utf-8",
    )
    assert _restore_count(checklist) == 1
