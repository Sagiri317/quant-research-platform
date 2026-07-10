from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage.atomic as atomic
from quant_research_platform.storage._scratch_ownership import owned_scratch_name
from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
)


class _FixedUuid:
    hex = "c" * 32


def test_atomic_write_directory_cleans_temp_path_after_failure(tmp_path: Path) -> None:
    final_root = tmp_path / "outputs" / "local" / "a_share" / "bundle"

    def failing_writer(temp_root: Path) -> None:
        (temp_root / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_writer_failure")

    with pytest.raises(RuntimeError, match="simulated_writer_failure"):
        atomic_write_directory(final_root, failing_writer)

    assert not final_root.exists()
    assert list(final_root.parent.glob(".tmp-*")) == []


def test_atomic_write_directory_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "outputs" / "local" / "a_share" / "bundle"
    original_rename = Path.rename
    calls = 0

    def flaky_rename(self: Path, target: Path) -> Path:
        nonlocal calls
        if (
            self.name.startswith(".bundle.tmp-")
            and Path(target) == final_root
            and calls == 0
        ):
            calls += 1
            raise PermissionError(5, "Access is denied")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    atomic_write_directory(
        final_root,
        lambda temp_root: (temp_root / "done.txt").write_text("ok\n", encoding="utf-8"),
    )

    assert calls == 1
    assert (final_root / "done.txt").read_text(encoding="utf-8") == "ok\n"
    assert list(final_root.parent.glob(".tmp-*")) == []


def test_atomic_write_directory_does_not_clean_unclaimed_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"
    live_temp = final_root.with_name(
        owned_scratch_name(".bundle.tmp-", "c" * 12)
    )
    live_temp.mkdir()
    (live_temp / "live.txt").write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(atomic, "uuid4", lambda: _FixedUuid())

    with pytest.raises(FileExistsError):
        atomic_write_directory(final_root, lambda _root: None)

    assert (live_temp / "live.txt").read_text(encoding="utf-8") == "live\n"


def test_atomic_write_directory_recovers_dead_owner_when_final_exists(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "bundle"
    final_root.mkdir()
    stale = final_root.with_name(".bundle.tmp-deadbeef")
    stale.mkdir()
    (stale / "partial.bin").write_bytes(b"partial")

    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_directory(final_root, lambda _root: None)

    assert not stale.exists()


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
    }
    for path in targets.values():
        path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_json(targets["json"], {"ok": True})
    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_csv(pd.DataFrame({"x": [1]}), targets["csv"], index=False)

    for path in targets.values():
        assert path.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.glob(".*.tmp-*")) == []


@pytest.mark.parametrize("kind", ["text", "csv"])
def test_atomic_file_writers_do_not_overwrite_unclaimed_collision(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "output.txt"
    live_temp = final_path.with_name(f".tmp-{'c' * 12}")
    live_temp.write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(atomic, "uuid4", lambda: _FixedUuid())

    with pytest.raises(FileExistsError):
        if kind == "text":
            atomic_write_text(final_path, "new\n")
        else:
            atomic_write_csv(pd.DataFrame({"x": [1]}), final_path, index=False)

    assert live_temp.read_text(encoding="utf-8") == "live\n"
    assert not final_path.exists()


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
