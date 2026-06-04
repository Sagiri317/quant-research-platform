from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_parquet,
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


def test_atomic_file_writers_reject_existing_targets_without_overwrite(
    tmp_path: Path,
) -> None:
    targets = {
        "json": tmp_path / "manifest.json",
        "csv": tmp_path / "table.csv",
        "parquet": tmp_path / "table.parquet",
    }
    for path in targets.values():
        path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_json(targets["json"], {"ok": True})
    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_csv(pd.DataFrame({"x": [1]}), targets["csv"], index=False)
    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_parquet(pd.DataFrame({"x": [1]}), targets["parquet"])

    for path in targets.values():
        assert path.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_atomic_csv_writer_cleans_temp_file_after_mid_write_failure(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "nested" / "table.csv"

    class FailingCsvFrame:
        def to_csv(self, path: Path, **_kwargs: object) -> None:
            path.write_text("partial\n", encoding="utf-8")
            raise RuntimeError("simulated_csv_failure")

    with pytest.raises(RuntimeError, match="simulated_csv_failure"):
        atomic_write_csv(FailingCsvFrame(), final_path, index=False)

    assert not final_path.exists()
    assert list(final_path.parent.glob(".table.csv.tmp-*")) == []


def test_atomic_parquet_writer_cleans_temp_file_after_mid_write_failure(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "nested" / "table.parquet"

    class FailingParquetFrame:
        def to_parquet(self, path: Path, **_kwargs: object) -> None:
            path.write_text("partial\n", encoding="utf-8")
            raise RuntimeError("simulated_parquet_failure")

    with pytest.raises(RuntimeError, match="simulated_parquet_failure"):
        atomic_write_parquet(FailingParquetFrame(), final_path)

    assert not final_path.exists()
    assert list(final_path.parent.glob(".table.parquet.tmp-*")) == []
