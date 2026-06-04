from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_text,
)


def test_atomic_write_directory_cleans_temp_path_after_failure(tmp_path: Path) -> None:
    final_root = tmp_path / "reports" / "private_real" / "a_share" / "run"

    def failing_writer(temp_root: Path) -> None:
        (temp_root / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_writer_failure")

    with pytest.raises(RuntimeError, match="simulated_writer_failure"):
        atomic_write_directory(final_root, failing_writer)

    assert not final_root.exists()
    assert list(final_root.parent.glob(".run.tmp-*")) == []


def test_atomic_write_text_creates_nested_temp_parent(tmp_path: Path) -> None:
    final_path = tmp_path / "root" / "year=2014" / "part.txt"

    atomic_write_text(final_path, "ok\n", encoding="utf-8")

    assert final_path.read_text(encoding="utf-8") == "ok\n"
