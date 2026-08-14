from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage.atomic as atomic
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage._scratch_ownership import owned_scratch_name
from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from tests.helpers.runtime_resources import capture_test_policy


class _FixedUuid:
    hex = "c" * 32


_NAMESPACE_PRIMARY_TYPES = (KeyboardInterrupt,)


def test_atomic_publish_rejects_failed_active_rss_checkpoint(tmp_path: Path) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024,
        rss_bytes=100,
        explicit_cap_bytes=600,
    )
    measurement = policy._measurement  # noqa: SLF001

    def failed_checkpoint() -> dict[str, int | float | str]:
        payload = measurement._payload()  # type: ignore[attr-defined]  # noqa: SLF001
        payload["sampled_peak_rss_bytes"] = 701
        payload["peak_incremental_rss_bytes"] = 601
        payload["status"] = "FAIL"
        return payload

    measurement.checkpoint_evidence = failed_checkpoint  # type: ignore[method-assign]
    target = tmp_path / "result.txt"

    with pytest.raises(MemoryError, match="runtime_rss_budget_exceeded"):
        with runtime_resource_policy_scope(policy):
            atomic_write_text(target, "result\n")

    assert not target.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_atomic_publish_has_no_rejection_after_irreversible_rename(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024,
        rss_bytes=100,
        explicit_cap_bytes=600,
    )
    measurement = policy._measurement  # noqa: SLF001
    checkpoints = 0

    def reject_after_rename() -> dict[str, int | float | str]:
        nonlocal checkpoints
        checkpoints += 1
        payload = measurement._payload()  # type: ignore[attr-defined]  # noqa: SLF001
        if checkpoints == 2:
            payload["sampled_peak_rss_bytes"] = 701
            payload["peak_incremental_rss_bytes"] = 601
            payload["status"] = "FAIL"
        return payload

    measurement.checkpoint_evidence = reject_after_rename  # type: ignore[method-assign]
    target = tmp_path / "result.txt"

    with runtime_resource_policy_scope(policy):
        atomic_write_text(target, "result\n")

    assert checkpoints == 2
    assert target.read_text(encoding="utf-8") == "result\n"


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


def test_atomic_target_marker_rejects_hardlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "external-target"
    external.write_text("outside", encoding="utf-8")
    marker = tmp_path / "stage.target"
    try:
        os.link(external, marker)
    except OSError:
        pytest.skip("hardlink_not_available")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardlinked_marker_opened")
        ),
    )

    with pytest.raises(OSError, match="invalid_atomic_target_marker"):
        atomic._read_atomic_target_marker(marker)


def test_atomic_cleanup_preserves_nonempty_stage_with_hardlinked_marker(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "bundle"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stale = tmp_path / f"{prefix}qzzzzzz-stale"
    stale.mkdir()
    (stale / "partial").write_text("keep", encoding="utf-8")
    external = tmp_path / "external-target"
    external.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")
    marker = Path(f"{stale}.target")
    try:
        os.link(external, marker)
    except OSError:
        pytest.skip("hardlink_not_available")

    atomic._cleanup_stale_atomic_directories(
        final_root,
        parent_identity=final_root.parent.lstat(),
        prefix=prefix,
        target_identity=atomic._atomic_target_identity(final_root),
    )

    assert stale.is_dir()
    assert marker.is_file()
    assert external.is_file()


def test_atomic_stale_sweep_rejects_caller_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    final_root = parent / "bundle"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    parent_identity = parent.lstat()
    displaced = tmp_path / "displaced"
    parent.rename(displaced)
    parent.mkdir()
    stale = parent / f"{prefix}qzzzzzz-stale"
    stale.mkdir()
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=parent_identity,
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert stale.is_dir()


@pytest.mark.parametrize("primary_type", _NAMESPACE_PRIMARY_TYPES)
@pytest.mark.parametrize("remove_case", ("stage", "attached-marker", "orphan-marker"))
def test_atomic_stale_sweep_preserves_remove_primary_when_parent_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
    remove_case: str,
) -> None:
    final_root = tmp_path / "bundle"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stage = tmp_path / f"{prefix}qzzzzzz-stale"
    marker = Path(f"{stage}.target")
    if remove_case != "orphan-marker":
        stage.mkdir()
    if remove_case != "stage":
        marker.write_text(
            atomic._atomic_target_identity(final_root),
            encoding="utf-8",
        )
    primary = primary_type("remove_primary")
    real_remove = atomic.remove_owned_scratch
    real_matches = atomic.require_owned_scratch.__globals__["scratch_identity_matches"]
    removal_started = False
    post_remove_probe_hits = 0

    def remove(path: Path, identity: os.stat_result) -> None:
        nonlocal removal_started
        if (remove_case == "stage" and path == stage) or (
            remove_case != "stage" and path == marker
        ):
            removal_started = True
            raise primary
        real_remove(path, identity)

    def match_parent(*args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        nonlocal post_remove_probe_hits
        if removal_started:
            post_remove_probe_hits += 1
            raise ValueError("parent_probe_secondary")
        return real_matches(*args, **kwargs)

    monkeypatch.setattr(
        atomic,
        "scratch_owner_is_stale",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(atomic, "remove_owned_scratch", remove)
    monkeypatch.setattr(
        atomic,
        "scratch_identity_matches",
        match_parent,
    )
    monkeypatch.setitem(
        atomic.require_owned_scratch.__globals__,
        "scratch_identity_matches",
        match_parent,
    )

    caught: BaseException | None = None
    try:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )
    except BaseException as error:
        caught = error

    assert caught is primary
    assert post_remove_probe_hits == 1


def test_atomic_write_directory_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "outputs" / "local" / "a_share" / "bundle"
    original_rename = atomic.move_path_no_replace
    calls = 0
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)

    def flaky_rename(source: Path, target: Path) -> None:
        nonlocal calls
        if (
            source.name.startswith(scratch_prefix)
            and Path(target) == final_root
            and calls == 0
        ):
            calls += 1
            raise PermissionError(5, "Access is denied")
        original_rename(source, target)

    monkeypatch.setattr(atomic, "move_path_no_replace", flaky_rename)
    atomic_write_directory(
        final_root,
        lambda temp_root: (temp_root / "done.txt").write_text("ok\n", encoding="utf-8"),
    )

    assert calls == 1
    assert (final_root / "done.txt").read_text(encoding="utf-8") == "ok\n"
    assert list(final_root.parent.glob(f"{scratch_prefix}*")) == []


def test_publish_sleep_secondary_preserves_move_permission_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_text("owned", encoding="utf-8")
    primary = PermissionError(5, "Access is denied")
    secondary = KeyboardInterrupt("sleep_secondary")

    monkeypatch.setattr(
        atomic,
        "move_path_no_replace",
        lambda _source, _target: (_ for _ in ()).throw(primary),
    )
    monkeypatch.setattr(
        atomic.time,
        "sleep",
        lambda _delay: (_ for _ in ()).throw(secondary),
    )

    caught: BaseException | None = None
    try:
        atomic.publish_no_replace(source, target, source_identity=source.lstat())
    except BaseException as error:
        caught = error

    assert caught is primary


@pytest.mark.parametrize(
    ("kind", "primary_type", "expects_error"),
    (
        ("file", OSError, False),
        ("directory", KeyboardInterrupt, True),
    ),
)
def test_publish_reconciles_post_commit_namespace_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    primary_type: type[BaseException],
    expects_error: bool,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    if kind == "directory":
        source.mkdir()
        (source / "payload").write_text("owned", encoding="utf-8")
    else:
        source.write_text("owned", encoding="utf-8")
    identity = source.lstat()
    primary = primary_type("post_commit")
    real_move = atomic.move_path_no_replace

    def commit_then_raise(source_path: Path, target_path: Path) -> None:
        real_move(source_path, target_path)
        raise primary

    monkeypatch.setattr(atomic, "move_path_no_replace", commit_then_raise)
    caught: BaseException | None = None
    try:
        atomic.publish_no_replace(
            source,
            target,
            source_identity=identity,
        )
    except BaseException as error:
        caught = error

    expected_error = primary if expects_error else None
    payload = target / "payload" if kind == "directory" else target
    assert {
        "caught": caught,
        "source_exists": source.exists(),
        "target_payload": payload.read_text(encoding="utf-8"),
    } == {
        "caught": expected_error,
        "source_exists": False,
        "target_payload": "owned",
    }


def test_publish_fails_closed_when_source_and_target_share_claimed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_text("owned", encoding="utf-8")
    os.link(source, target)
    identity = source.lstat()
    move_hits = 0

    def no_op(_source: Path, _target: Path) -> None:
        nonlocal move_hits
        move_hits += 1

    monkeypatch.setattr(
        atomic,
        "move_path_no_replace",
        no_op,
    )

    with pytest.raises(ValueError, match="storage_scratch_identity_changed") as caught:
        atomic.publish_no_replace(source, target, source_identity=identity)

    assert {
        "source_exact": os.path.samestat(source.lstat(), identity),
        "target_exact": os.path.samestat(target.lstat(), identity),
        "payloads": (source.read_text(), target.read_text()),
        "move_hits": move_hits,
        "notes": getattr(caught.value, "__notes__", []),
    } == {
        "source_exact": True,
        "target_exact": True,
        "payloads": ("owned", "owned"),
        "move_hits": 1,
        "notes": [],
    }


@pytest.mark.parametrize("move_outcome", ("normal", "error"))
def test_publish_cleans_confirmed_target_when_source_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    move_outcome: str,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_text("owned", encoding="utf-8")
    identity = source.lstat()
    move_primary = RuntimeError("move_primary") if move_outcome == "error" else None
    source_probe_primary = RuntimeError("source_probe_primary")
    real_move = atomic.move_path_no_replace
    real_matches = atomic.scratch_identity_matches
    probe_order: list[Path] = []

    def move(source_path: Path, target_path: Path) -> None:
        real_move(source_path, target_path)
        if move_primary is not None:
            raise move_primary

    def fail_source_probe(path: Path, claimed: os.stat_result) -> bool:
        probe_order.append(path)
        if path == source:
            raise source_probe_primary
        return real_matches(path, claimed)

    monkeypatch.setattr(atomic, "move_path_no_replace", move)
    monkeypatch.setattr(atomic, "scratch_identity_matches", fail_source_probe)

    expected_primary = move_primary or source_probe_primary
    with pytest.raises(type(expected_primary)) as caught:
        atomic.publish_no_replace(source, target, source_identity=identity)

    assert {
        "caught": caught.value,
        "probe_order": probe_order,
        "source_exists": source.exists(),
        "target_exists": target.exists(),
    } == {
        "caught": expected_primary,
        "probe_order": [target, source],
        "source_exists": False,
        "target_exists": False,
    }


@pytest.mark.parametrize("move_outcome", ("normal", "error"))
def test_publish_observes_source_after_target_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    move_outcome: str,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_text("owned", encoding="utf-8")
    identity = source.lstat()
    move_primary = OSError("move_primary") if move_outcome == "error" else None
    target_probe_primary = RuntimeError("target_probe_primary")
    real_move = atomic.move_path_no_replace
    real_matches = atomic.scratch_identity_matches
    probe_order: list[Path] = []

    def move(source_path: Path, target_path: Path) -> None:
        real_move(source_path, target_path)
        if move_primary is not None:
            raise move_primary

    def fail_target_probe(path: Path, claimed: os.stat_result) -> bool:
        probe_order.append(path)
        if path == target:
            raise target_probe_primary
        return real_matches(path, claimed)

    monkeypatch.setattr(atomic, "move_path_no_replace", move)
    monkeypatch.setattr(atomic, "scratch_identity_matches", fail_target_probe)

    expected_primary = move_primary or target_probe_primary
    with pytest.raises(type(expected_primary)) as caught:
        atomic.publish_no_replace(source, target, source_identity=identity)

    assert {
        "caught": caught.value,
        "probe_order": probe_order,
        "source_exists": source.exists(),
        "target_exact": os.path.samestat(target.lstat(), identity),
    } == {
        "caught": expected_primary,
        "probe_order": [target, source],
        "source_exists": False,
        "target_exact": True,
    }


@pytest.mark.parametrize("kind", ["text", "directory"])
def test_atomic_publish_does_not_retry_replaced_claimed_source(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / ("bundle" if kind == "directory" else "output")
    original_move = atomic.move_path_no_replace
    original_source: Path | None = None
    foreign_source: Path | None = None
    calls = 0

    def replace_then_retry(source: Path, target: Path) -> None:
        nonlocal calls, foreign_source, original_source
        calls += 1
        if calls == 1:
            original_source = source.with_name(f"{source.name}.original")
            source.rename(original_source)
            if source.suffix or kind != "directory":
                source.write_text("foreign\n", encoding="utf-8")
            else:
                source.mkdir()
                (source / "foreign.txt").write_text("foreign\n", encoding="utf-8")
            foreign_source = source
            raise PermissionError(5, "Access is denied")
        original_move(source, target)

    monkeypatch.setattr(atomic, "move_path_no_replace", replace_then_retry)
    monkeypatch.setattr(atomic.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError) as caught:
        if kind == "text":
            atomic_write_text(final_path, "original\n")
        elif kind == "csv":
            atomic_write_csv(pd.DataFrame({"value": [1]}), final_path, index=False)
        else:
            atomic_write_directory(
                final_path,
                lambda stage: (stage / "original.txt").write_text(
                    "original\n", encoding="utf-8"
                ),
            )

    assert caught.value.args == (5, "Access is denied")
    assert calls == 1
    assert original_source is not None and original_source.exists()
    assert foreign_source is not None and foreign_source.exists()
    assert not final_path.exists()


@pytest.mark.parametrize("kind", ["text", "directory"])
def test_atomic_publish_rejects_target_replacement_after_move(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / ("bundle" if kind == "directory" else "output")
    original_move = atomic.move_path_no_replace
    published_original = final_path.with_name(f"{final_path.name}.original")

    def replace_target(source: Path, target: Path) -> None:
        original_move(source, target)
        Path(target).rename(published_original)
        if kind == "directory":
            Path(target).mkdir()
            (Path(target) / "foreign.txt").write_text("foreign\n", encoding="utf-8")
        else:
            Path(target).write_text("foreign\n", encoding="utf-8")

    monkeypatch.setattr(atomic, "move_path_no_replace", replace_target)

    with pytest.raises(ValueError, match="storage_scratch_identity_changed"):
        if kind == "text":
            atomic_write_text(final_path, "original\n")
        elif kind == "csv":
            atomic_write_csv(pd.DataFrame({"value": [1]}), final_path, index=False)
        else:
            atomic_write_directory(
                final_path,
                lambda stage: (stage / "original.txt").write_text(
                    "original\n", encoding="utf-8"
                ),
            )

    assert published_original.exists()
    assert final_path.exists()


def test_atomic_directory_collision_probe_preserves_file_exists_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"
    primary = FileExistsError("publish_collision")
    secondary = KeyboardInterrupt("classification_probe")
    publish_hit = False

    def collide(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        nonlocal publish_hit
        publish_hit = True
        raise primary

    real_exists = Path.exists

    def fail_probe(path: Path) -> bool:
        if publish_hit and path == final_root:
            raise secondary
        return real_exists(path)

    monkeypatch.setattr(atomic, "publish_no_replace", collide)
    monkeypatch.setattr(Path, "exists", fail_probe)

    caught: BaseException | None = None
    try:
        atomic_write_directory(final_root, lambda _stage: None)
    except BaseException as error:
        caught = error

    assert publish_hit
    assert caught is primary
    assert primary.__notes__ == [
        "atomic target classification also failed: KeyboardInterrupt"
    ]


def test_publish_collision_probe_preserves_file_exists_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_text("source", encoding="utf-8")
    primary = FileExistsError("publish_collision")
    secondary = KeyboardInterrupt("classification_probe")
    move_hit = False

    def collide(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        nonlocal move_hit
        move_hit = True
        raise primary

    real_exists = Path.exists

    def fail_probe(path: Path) -> bool:
        if move_hit and path == target:
            raise secondary
        return real_exists(path)

    monkeypatch.setattr(atomic, "move_path_no_replace", collide)
    monkeypatch.setattr(Path, "exists", fail_probe)

    caught: BaseException | None = None
    try:
        atomic.publish_no_replace(source, target, source_identity=source.lstat())
    except BaseException as error:
        caught = error

    assert move_hit
    assert caught is primary
    assert primary.__notes__ == [
        "atomic target classification also failed: KeyboardInterrupt"
    ]


def test_atomic_write_directory_does_not_clean_unclaimed_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"
    scratch_prefix = atomic._atomic_directory_scratch_prefix(final_root)
    live_temp = final_root.with_name(owned_scratch_name(scratch_prefix, "c" * 8))
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

    def racing_publish(
        source: Path,
        target: Path,
        *,
        source_identity: os.stat_result,
    ) -> None:
        assert os.path.samestat(source.lstat(), source_identity)
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


def test_atomic_directory_marker_cleanup_failure_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "bundle"
    original_remove = atomic.remove_owned_scratch

    def deny_marker_cleanup(path: Path, identity: os.stat_result) -> None:
        if path.name.endswith(".target"):
            raise PermissionError(5, "Access is denied")
        original_remove(path, identity)

    monkeypatch.setattr(atomic, "remove_owned_scratch", deny_marker_cleanup)

    with pytest.raises(PermissionError, match="Access is denied"):
        atomic_write_directory(
            final_root,
            lambda root: (root / "done.txt").write_text("ok\n", encoding="utf-8"),
        )

    assert not final_root.exists()
    assert list(tmp_path.glob(".a-*.target"))


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
    assert list(tmp_path.glob(".f-*")) == []


@pytest.mark.parametrize("kind", ["text", "csv"])
def test_atomic_file_writers_do_not_overwrite_unclaimed_collision(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "output.txt"
    live_temp = final_path.with_name(owned_scratch_name(".f-", "c" * 8))
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

    monkeypatch.setattr(atomic, "move_path_no_replace", racing_rename)

    with pytest.raises(OutputAlreadyExistsError):
        atomic.publish_no_replace(source, target, source_identity=source.lstat())

    assert target.read_text(encoding="utf-8") == "concurrent-winner\n"
    assert source.read_text(encoding="utf-8") == "candidate\n"


def test_atomic_csv_writer_cleans_temp_file_after_mid_write_failure(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "nested" / "table.csv"

    class FailingCsvFrame:
        def to_csv(self, sink, **_kwargs: object) -> None:  # noqa: ANN001
            sink.write("partial\n")
            raise RuntimeError("simulated_csv_failure")

    with pytest.raises(RuntimeError, match="simulated_csv_failure"):
        atomic_write_csv(FailingCsvFrame(), final_path, index=False)

    assert not final_path.exists()
    assert list(final_path.parent.glob(".f-*")) == []
