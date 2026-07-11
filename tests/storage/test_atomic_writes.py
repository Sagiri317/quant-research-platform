from __future__ import annotations

import os
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
from quant_research_platform.storage.errors import OutputAlreadyExistsError


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
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    assert list(final_root.parent.glob(f"{scratch_prefix}*")) == []


@pytest.mark.parametrize("cancellation", [KeyboardInterrupt, SystemExit])
def test_atomic_write_directory_cleans_claimed_paths_after_control_flow_cancellation(
    tmp_path: Path,
    cancellation: type[BaseException],
) -> None:
    final_root = tmp_path / "bundle"

    def cancelling_writer(temp_root: Path) -> None:
        (temp_root / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise cancellation

    with pytest.raises(cancellation):
        atomic_write_directory(final_root, cancelling_writer)

    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    assert list(tmp_path.glob(f"{scratch_prefix}*")) == []


@pytest.mark.parametrize("writer_kind", ["text", "csv"])
def test_atomic_file_writers_clean_claimed_temp_after_control_flow_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_kind: str,
) -> None:
    final_path = tmp_path / f"output.{writer_kind}"
    monkeypatch.setattr(
        atomic,
        "publish_no_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        if writer_kind == "text":
            atomic_write_text(final_path, "payload\n")
        else:
            atomic_write_csv(pd.DataFrame({"value": [1]}), final_path, index=False)

    assert tuple(tmp_path.iterdir()) == ()


def test_atomic_directory_supports_long_final_component(tmp_path: Path) -> None:
    component_length = min(225, 240 - len(str(tmp_path)) - 1)
    assert component_length > 32
    final_root = tmp_path / ("x" * component_length)

    atomic_write_directory(
        final_root,
        lambda stage: (stage / "done").write_text("ok", encoding="utf-8"),
    )

    assert (final_root / "done").read_text(encoding="utf-8") == "ok"


def test_atomic_directory_recovery_is_glob_safe_for_target_name(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "[x]"
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stale = tmp_path / f"{scratch_prefix}qzzzzzz-stale"
    stale.mkdir()
    Path(f"{stale}.target").write_text(
        atomic._atomic_target_identity(final_root),
        encoding="utf-8",
    )

    atomic_write_directory(
        final_root,
        lambda stage: (stage / "done").write_text("ok", encoding="utf-8"),
    )

    assert not stale.exists()
    assert (final_root / "done").read_text(encoding="utf-8") == "ok"


@pytest.mark.parametrize("marker_payload", ["", "truncated-target"])
def test_atomic_directory_recovers_empty_stale_stage_with_invalid_marker(
    tmp_path: Path,
    marker_payload: str,
) -> None:
    final_root = tmp_path / "bundle"
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stale = tmp_path / f"{scratch_prefix}qzzzzzz-stale"
    stale.mkdir()
    stale_marker = Path(f"{stale}.target")
    stale_marker.write_text(marker_payload, encoding="utf-8")

    atomic_write_directory(
        final_root,
        lambda stage: (stage / "done").write_text("ok", encoding="utf-8"),
    )

    assert not stale.exists()
    assert not stale_marker.exists()
    assert (final_root / "done").read_text(encoding="utf-8") == "ok"


@pytest.mark.parametrize("marker_payload", ["", "truncated-target"])
def test_atomic_directory_recovers_orphan_stale_marker(
    tmp_path: Path,
    marker_payload: str,
) -> None:
    final_root = tmp_path / "bundle"
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stale = tmp_path / f"{scratch_prefix}qzzzzzz-stale"
    stale_marker = Path(f"{stale}.target")
    stale_marker.write_text(marker_payload, encoding="utf-8")

    atomic_write_directory(
        final_root,
        lambda stage: (stage / "done").write_text("ok", encoding="utf-8"),
    )

    assert not stale_marker.exists()


def test_atomic_write_directory_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "outputs" / "local" / "a_share" / "bundle"
    original_rename = Path.rename
    calls = 0
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)

    def flaky_rename(self: Path, target: Path) -> Path:
        nonlocal calls
        if (
            self.name.startswith(scratch_prefix)
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
    assert list(final_root.parent.glob(f"{scratch_prefix}*")) == []


def test_atomic_write_directory_does_not_clean_unclaimed_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    live_temp = final_root.with_name(
        owned_scratch_name(scratch_prefix, "c" * 12)
    )
    live_temp.mkdir()
    (live_temp / "live.txt").write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(atomic, "uuid4", lambda: _FixedUuid())

    with pytest.raises(FileExistsError):
        atomic_write_directory(final_root, lambda _root: None)

    assert (live_temp / "live.txt").read_text(encoding="utf-8") == "live\n"


def test_atomic_directory_preserves_unclaimed_scratch_directory(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "bundle"
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    unclaimed = tmp_path / f"{scratch_prefix}user-data"
    unclaimed.mkdir()
    sentinel = unclaimed / "sentinel.txt"
    sentinel.write_text("unclaimed\n", encoding="utf-8")

    atomic_write_directory(
        final_root,
        lambda stage: (stage / "done").write_text("ok", encoding="utf-8"),
    )

    assert sentinel.read_text(encoding="utf-8") == "unclaimed\n"


def test_atomic_directory_preserves_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"

    def racing_publish(source: Path, target: Path) -> None:
        del source
        target.mkdir()
        (target / "winner.txt").write_text("concurrent-winner\n", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(atomic, "publish_no_replace", racing_publish)

    with pytest.raises(OutputAlreadyExistsError):
        atomic_write_directory(
            final_root,
            lambda stage: (stage / "candidate.txt").write_text(
                "candidate\n",
                encoding="utf-8",
            ),
        )

    assert (final_root / "winner.txt").read_text(encoding="utf-8") == (
        "concurrent-winner\n"
    )


def test_atomic_write_directory_recovers_dead_owner_when_final_exists(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "bundle"
    final_root.mkdir()
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stale = final_root.with_name(f"{scratch_prefix}qzzzzzz-stale")
    stale.mkdir()
    (stale / "partial.bin").write_bytes(b"partial")
    Path(f"{stale}.target").write_text(
        atomic._atomic_target_identity(final_root),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_directory(final_root, lambda _root: None)

    assert not stale.exists()


def test_atomic_directory_recovery_does_not_cross_exact_target_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "artifact-138970"
    second = tmp_path / "artifact-147165"
    first_prefix = atomic._atomic_directory_scratch_prefix(first)
    second_prefix = atomic._atomic_directory_scratch_prefix(second)
    assert first_prefix == second_prefix
    second_scratch = second.with_name(f"{second_prefix}qzzzzzz-stale")
    second_scratch.mkdir()
    (second_scratch / "partial.bin").write_bytes(b"partial")
    Path(f"{second_scratch}.target").write_text(
        os.path.normcase(str(second.resolve())),
        encoding="utf-8",
    )

    atomic_write_directory(first, lambda root: (root / "done").write_text("ok"))

    assert second_scratch.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystem alias contract")
def test_atomic_directory_target_identity_normalizes_windows_case_aliases(
    tmp_path: Path,
) -> None:
    upper = tmp_path / "Bundle"
    lower = tmp_path / "bundle"

    assert atomic._atomic_directory_scratch_prefix(
        upper
    ) == atomic._atomic_directory_scratch_prefix(lower)


def test_atomic_directory_marker_cleanup_does_not_reverse_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"
    original_unlink = Path.unlink

    def deny_marker_unlink(self: Path, *args, **kwargs):  # noqa: ANN001
        if self.name.endswith(".target"):
            raise PermissionError(5, "Access is denied")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_marker_unlink)

    atomic_write_directory(
        final_root,
        lambda root: (root / "done.txt").write_text("ok\n", encoding="utf-8"),
    )

    assert (final_root / "done.txt").read_text(encoding="utf-8") == "ok\n"


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
    assert list(tmp_path.glob(".tmp-*")) == []


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


def test_publish_file_no_replace_preserves_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "candidate.tmp"
    target = tmp_path / "output.txt"
    source.write_text("candidate\n", encoding="utf-8")

    def racing_rename(source_path: Path, target_path: Path) -> None:
        del source_path
        Path(target_path).write_text("concurrent-winner\n", encoding="utf-8")
        raise FileExistsError(target_path)

    monkeypatch.setattr(atomic, "_rename_no_replace", racing_rename)

    with pytest.raises(OutputAlreadyExistsError):
        atomic.publish_no_replace(source, target)

    assert target.read_text(encoding="utf-8") == "concurrent-winner\n"
    assert source.read_text(encoding="utf-8") == "candidate\n"


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
    assert list(final_path.parent.glob(".tmp-*")) == []
