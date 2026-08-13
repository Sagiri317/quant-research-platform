from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.storage._dataset_comparison as comparison
import quant_research_platform.storage._dataset_fingerprinting as fingerprinting
import quant_research_platform.storage._dataset_inspection as inspection
import quant_research_platform.storage._dataset_key_tracking as key_owner
import quant_research_platform.storage._dataset_paths as dataset_paths
import quant_research_platform.storage._dataset_physical as physical
import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_writing as dataset_writing
from quant_research_platform.artifacts import _parquet_writers as parquet_writers
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage import (
    _dataset_inventory as inventory_owner,
)
from quant_research_platform.storage import _scratch_ownership as scratch_owner
from quant_research_platform.storage import (
    atomic,
    manifests,
)
from quant_research_platform.storage._dataset_types import DatasetReadMode
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)

_PRIMARY_EXCEPTION_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)

_STALE_SWEEP_FAILURE_TYPES = (
    OSError,
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
)

_NAMESPACE_PRIMARY_TYPES = (OSError, KeyboardInterrupt)


def test_owned_file_stale_sweep_attempts_all_and_preserves_first_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = ".f-"
    stale_first = tmp_path / scratch_owner.owned_scratch_name(prefix, "first")
    stale_second = tmp_path / scratch_owner.owned_scratch_name(prefix, "second")
    live = tmp_path / scratch_owner.owned_scratch_name(prefix, "live")
    malformed = tmp_path / f"{prefix}malformed"
    wrong_kind = tmp_path / scratch_owner.owned_scratch_name(prefix, "directory")
    for path in (stale_first, stale_second, live, malformed):
        path.write_bytes(b"owned")
    wrong_kind.mkdir()
    primary = OSError("first_cleanup_failure")
    real_remove = scratch_owner.remove_owned_scratch
    remove_hits: list[Path] = []

    def remove(path: Path, identity: os.stat_result, **kwargs) -> None:  # noqa: ANN003
        remove_hits.append(path)
        if path == stale_first:
            raise primary
        real_remove(path, identity, **kwargs)

    monkeypatch.setattr(
        scratch_owner,
        "scratch_owner_is_stale",
        lambda path, *, prefix: Path(path) in {stale_first, stale_second},
    )
    monkeypatch.setattr(scratch_owner, "remove_owned_scratch", remove)

    with pytest.raises(OSError) as caught:
        scratch_owner.cleanup_stale_owned_scratch(
            tmp_path,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            directory=False,
        )

    assert caught.value is primary
    assert remove_hits == [stale_first, stale_second]
    assert stale_first.is_file()
    assert not stale_second.exists()
    assert live.is_file()
    assert malformed.is_file()
    assert wrong_kind.is_dir()


@pytest.mark.parametrize("primary_type", _NAMESPACE_PRIMARY_TYPES)
def test_owned_stale_sweep_preserves_remove_primary_when_parent_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    prefix = ".f-"
    candidate = tmp_path / scratch_owner.owned_scratch_name(prefix, "candidate")
    candidate.write_bytes(b"owned")
    primary = primary_type("remove_primary")
    real_matches = scratch_owner.scratch_identity_matches
    removal_started = False
    post_remove_probe_hits = 0

    def remove(*_args, **_kwargs) -> None:
        nonlocal removal_started
        removal_started = True
        raise primary

    def match_parent(*args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        nonlocal post_remove_probe_hits
        if removal_started:
            post_remove_probe_hits += 1
            raise ValueError("parent_probe_secondary")
        return real_matches(*args, **kwargs)

    monkeypatch.setattr(
        scratch_owner,
        "scratch_owner_is_stale",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        scratch_owner,
        "remove_owned_scratch",
        remove,
    )
    monkeypatch.setattr(
        scratch_owner,
        "scratch_identity_matches",
        match_parent,
    )

    caught: BaseException | None = None
    try:
        scratch_owner.cleanup_stale_owned_scratch(
            tmp_path,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            directory=False,
        )
    except BaseException as error:
        caught = error

    assert caught is primary
    assert post_remove_probe_hits == 1


def test_owned_file_stale_sweep_captures_identity_before_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = ".f-"
    candidate = tmp_path / scratch_owner.owned_scratch_name(prefix, "candidate")
    candidate.write_bytes(b"owned")
    liveness_hits = 0

    def replace_during_liveness(path: Path, *, prefix: str) -> bool:
        nonlocal liveness_hits
        del prefix
        liveness_hits += 1
        Path(path).unlink()
        Path(path).write_bytes(b"foreign")
        return True

    monkeypatch.setattr(
        scratch_owner,
        "scratch_owner_is_stale",
        replace_during_liveness,
    )

    with pytest.raises(ValueError, match="storage_scratch_identity_changed"):
        scratch_owner.cleanup_stale_owned_scratch(
            tmp_path,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            directory=False,
        )

    assert liveness_hits == 1
    assert candidate.read_bytes() == b"foreign"


def test_owned_stale_sweep_binds_caller_parent_and_checks_each_delete_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_identity = parent.lstat()
    displaced = tmp_path / "displaced"
    parent.rename(displaced)
    parent.mkdir()
    candidate = parent / scratch_owner.owned_scratch_name(".f-", "replacement")
    candidate.write_bytes(b"foreign")
    checks: list[Path] = []
    real_require = scratch_owner.require_owned_scratch

    def require(path: Path, identity: os.stat_result) -> None:
        checks.append(Path(path))
        real_require(path, identity)

    monkeypatch.setattr(scratch_owner, "require_owned_scratch", require)
    monkeypatch.setattr(
        scratch_owner,
        "scratch_owner_is_stale",
        lambda *_args, **_kwargs: True,
    )
    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        scratch_owner.cleanup_stale_owned_scratch(
            parent,
            parent_identity=parent_identity,
            prefix=".f-",
            directory=False,
        )

    assert checks == [parent]
    assert candidate.read_bytes() == b"foreign"


def test_owned_file_stale_sweep_recovers_hard_kill_abandon(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        (
            sys.executable,
            "-B",
            "-c",
            (
                "import os,sys; from pathlib import Path; "
                "from quant_research_platform.storage._scratch_ownership "
                "import open_owned_scratch_file,owned_scratch_name; "
                "p=Path(sys.argv[1])/owned_scratch_name('.f-','abandon'); "
                "h,_=open_owned_scratch_file(p); h.write(b'owned'); "
                "h.flush(); os._exit(0)"
            ),
            str(tmp_path),
        ),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert len(tuple(tmp_path.glob(".f-*"))) == 1

    scratch_owner.cleanup_stale_owned_scratch(
        tmp_path,
        parent_identity=tmp_path.lstat(),
        prefix=".f-",
        directory=False,
    )

    assert not tuple(tmp_path.glob(".f-*"))


def test_shared_path_move_is_fail_if_destination_exists(tmp_path: Path) -> None:
    shared = __import__(
        "quant_research_platform.storage._scratch_ownership", fromlist=["*"]
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    with pytest.raises(FileExistsError):
        shared.move_path_no_replace(source, destination)

    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "destination"


@pytest.mark.parametrize(
    "owner_kind",
    ("atomic-file", "atomic-directory", "key-scratch", "inventory-scratch"),
)
def test_creator_rechecks_parent_reference_before_payload_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_kind: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    original_parent = tmp_path / "original-parent"
    injection_hits: list[Path] = []
    payload_hits: list[str] = []
    real_mkdir = os.mkdir
    real_open = os.open

    def replace_parent_once() -> None:
        if injection_hits:
            return
        parent.rename(original_parent)
        parent.mkdir()
        injection_hits.append(parent)

    if owner_kind == "atomic-file":
        real_open_exclusive = scratch_owner.open_owned_scratch_file

        class HandleProbe:
            def __init__(self, handle) -> None:  # noqa: ANN001
                self.handle = handle

            def write(self, value):  # noqa: ANN001, ANN202
                payload_hits.append("write")
                return self.handle.write(value)

            def __getattr__(self, name: str):  # noqa: ANN204
                return getattr(self.handle, name)

        def open_candidate(path, *args, **kwargs):  # noqa: ANN001, ANN202
            if Path(path).parent == parent:
                replace_parent_once()
            return real_open(path, *args, **kwargs)

        def open_exclusive(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            handle, identity = real_open_exclusive(*args, **kwargs)
            return HandleProbe(handle), identity

        monkeypatch.setattr(atomic.os, "open", open_candidate)
        monkeypatch.setattr(
            scratch_owner,
            "open_owned_scratch_file",
            open_exclusive,
        )
        operation = lambda: atomic.atomic_write_text(  # noqa: E731
            parent / "value.txt", "payload"
        )
    elif owner_kind == "atomic-directory":

        def mkdir_candidate(path, *args, **kwargs):  # noqa: ANN001, ANN202
            candidate = Path(path)
            if candidate.parent == parent and candidate != parent:
                replace_parent_once()
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(atomic.os, "mkdir", mkdir_candidate)
        operation = lambda: atomic.atomic_write_directory(  # noqa: E731
            parent / "tree", lambda _root: payload_hits.append("writer")
        )
    else:
        owner_module = key_owner if owner_kind == "key-scratch" else inventory_owner

        def mkdir_candidate(path, *args, **kwargs):  # noqa: ANN001, ANN202
            candidate = Path(path)
            if candidate.parent == parent and candidate != parent:
                replace_parent_once()
            return real_mkdir(path, *args, **kwargs)

        def connect(*_args, **_kwargs):  # noqa: ANN202
            payload_hits.append("sqlite-connect")
            raise RuntimeError("payload_use_hit")

        monkeypatch.setattr(owner_module.os, "mkdir", mkdir_candidate)
        monkeypatch.setattr(owner_module.sqlite3, "connect", connect)
        if owner_kind == "key-scratch":
            operation = lambda: key_owner.ExactPrimaryKeyIndex(  # noqa: E731
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=parent,
            ).__enter__()
        else:
            operation = lambda: inventory_owner.DatasetSessionInventory(  # noqa: E731
                data_root=tmp_path / "data",
                scratch_parent=parent,
                task_memory_budget_bytes=1024,
            )

    caught: BaseException | None = None
    try:
        operation()
    except BaseException as error:
        caught = error

    assert len(injection_hits) == 1
    assert payload_hits == []
    assert isinstance(caught, ValueError)
    assert not tuple(original_parent.iterdir())
    survivors = tuple(parent.iterdir())
    if owner_kind == "atomic-file":
        assert survivors == ()
    else:
        assert len(survivors) == 1
        assert survivors[0].is_dir()


@pytest.mark.parametrize(
    ("owner_module", "prefix", "open_owner"),
    (
        (
            key_owner,
            ".k-",
            lambda scratch, root: key_owner.ExactPrimaryKeyIndex(
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=scratch,
            ).__enter__(),
        ),
        (
            inventory_owner,
            ".i-",
            lambda scratch, root: inventory_owner.DatasetSessionInventory(
                data_root=root / "data",
                scratch_parent=scratch,
                task_memory_budget_bytes=1024,
            ),
        ),
    ),
    ids=("exact-primary-key-index", "dataset-session-inventory"),
)
def test_sqlite_owner_binds_operation_parent_before_stale_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_module: object,
    prefix: str,
    open_owner,
) -> None:
    parent = tmp_path / "scratch"
    parent.mkdir()
    displaced = tmp_path / "displaced"
    cleanup_name = "cleanup_stale_owned_scratch"
    real_cleanup = getattr(owner_module, cleanup_name)
    candidate: Path | None = None
    connect_hits = 0

    def replace_then_cleanup(root: Path, **kwargs) -> None:  # noqa: ANN003
        nonlocal candidate
        Path(root).rename(displaced)
        Path(root).mkdir()
        candidate = Path(root) / f"{prefix}qzzzzzz-replacement"
        candidate.mkdir()
        return real_cleanup(root, **kwargs)

    def connect(*_args, **_kwargs):  # noqa: ANN202
        nonlocal connect_hits
        connect_hits += 1
        raise AssertionError("sqlite connect must not run")

    monkeypatch.setattr(owner_module, cleanup_name, replace_then_cleanup)
    monkeypatch.setitem(
        real_cleanup.__globals__,
        "scratch_owner_is_stale",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(owner_module.sqlite3, "connect", connect)

    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        open_owner(parent, tmp_path)

    assert connect_hits == 0
    assert candidate is not None and candidate.is_dir()


def test_write_and_scratch_owners_reject_aliased_parent_before_access(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    operations = (
        lambda: atomic.atomic_write_text(alias / "text.txt", "payload"),
        lambda: atomic.atomic_write_csv(
            pd.DataFrame({"value": [1]}), alias / "frame.csv", index=False
        ),
        lambda: atomic.atomic_write_directory(alias / "tree", lambda _root: None),
        lambda: manifests.replace_qrp_json(
            alias / "manifest.json", {"schema_id": "qrp-test"}
        ),
        lambda: key_owner.ExactPrimaryKeyIndex(
            primary_key=("value",),
            timestamp_columns=(),
            scratch_parent=alias,
        ).__enter__(),
        lambda: inventory_owner.DatasetSessionInventory(
            data_root=tmp_path / "data",
            scratch_parent=alias,
            task_memory_budget_bytes=1024,
        ),
    )

    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    assert not tuple(target.iterdir())


@pytest.mark.parametrize(
    ("owner_module", "open_owner"),
    (
        (
            key_owner,
            lambda scratch: key_owner.ExactPrimaryKeyIndex(
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=scratch,
            ).__enter__(),
        ),
        (
            inventory_owner,
            lambda scratch: inventory_owner.DatasetSessionInventory(
                data_root=scratch.parent / "data",
                scratch_parent=scratch,
                task_memory_budget_bytes=1024,
            ),
        ),
    ),
    ids=["exact-primary-key-index", "dataset-session-inventory"],
)
def test_scratch_owner_cleans_retained_temp_after_post_create_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_module: object,
    open_owner,
) -> None:
    scratch = tmp_path / "scratch"
    created: list[Path] = []
    gate_failure = KeyboardInterrupt("post_create_gate")

    post_create_calls = 0
    real_gate = owner_module.require_unaliased_path

    def gate(path: Path, *, error: str) -> Path:
        nonlocal post_create_calls
        candidate = Path(path)
        if candidate.parent == scratch and candidate.name.startswith("."):
            if not created:
                created.append(candidate)
            post_create_calls += 1
            if post_create_calls == 1:
                raise gate_failure
        return real_gate(candidate, error=error)

    monkeypatch.setattr(owner_module, "require_unaliased_path", gate)

    with pytest.raises(KeyboardInterrupt) as caught:
        open_owner(scratch)

    assert caught.value is gate_failure
    assert len(created) == 1
    assert not created[0].exists()


@pytest.mark.parametrize(
    ("owner_module", "open_owner"),
    (
        (
            key_owner,
            lambda scratch: key_owner.ExactPrimaryKeyIndex(
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=scratch,
            ).__enter__(),
        ),
        (
            inventory_owner,
            lambda scratch: inventory_owner.DatasetSessionInventory(
                data_root=scratch.parent / "data",
                scratch_parent=scratch,
                task_memory_budget_bytes=1024,
            ),
        ),
    ),
    ids=("exact-primary-key-index", "dataset-session-inventory"),
)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_scratch_owner_cleans_when_database_constructor_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_module: object,
    open_owner,
    primary_type: type[BaseException],
) -> None:
    scratch = tmp_path / "scratch"
    primary = primary_type("constructor_primary")
    created: list[Path] = []
    real_create = owner_module.create_owned_scratch

    def record_create(path: Path) -> os.stat_result:
        identity = real_create(path)
        created.append(path)
        return identity

    monkeypatch.setattr(owner_module, "create_owned_scratch", record_create)

    def fail_connect(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(owner_module.sqlite3, "connect", fail_connect)

    with pytest.raises(primary_type) as caught:
        open_owner(scratch)

    assert caught.value is primary
    assert len(created) == 1
    assert not created[0].exists()


def _run_file_creator(owner_name: str, tmp_path: Path) -> object:
    if owner_name == "atomic-text":
        return atomic.atomic_write_text(tmp_path / "value.txt", "value")
    if owner_name == "atomic-csv":
        return atomic.atomic_write_csv(
            pd.DataFrame({"value": [1]}), tmp_path / "value.csv", index=False
        )
    if owner_name in {"atomic-directory", "atomic-target-marker"}:
        return atomic.atomic_write_directory(tmp_path / "tree", lambda _root: None)
    if owner_name == "manifest-replace":
        return manifests.replace_qrp_json(
            tmp_path / "manifest.json", {"schema_id": "qrp-test"}
        )
    if owner_name == "chunked-parquet":
        return parquet_writers.write_parquet_chunks(
            (),
            tmp_path / "prediction.parquet",
            required=("timestamp",),
            key_columns=("timestamp",),
            compression="zstd",
        )
    return parquet_writers.write_parquet(
        pd.DataFrame({"value": [1]}), tmp_path / "value.parquet"
    )


def _file_creator_module(owner_name: str):  # noqa: ANN202
    del owner_name
    return scratch_owner


def _file_writer_module(owner_name: str):  # noqa: ANN202
    if owner_name in {"atomic-text", "atomic-csv", "atomic-target-marker"}:
        return atomic
    if owner_name == "manifest-replace":
        return manifests
    return parquet_writers


@pytest.mark.parametrize(
    "owner_name",
    (
        "atomic-text",
        "manifest-replace",
        "single-parquet",
    ),
)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_writer_parent_identity_primary_survives_handle_close_secondary(
    owner_name: str,
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _file_creator_module(owner_name)
    writer_module = _file_writer_module(owner_name)
    primary = primary_type("parent_identity_primary")
    secondary = RuntimeError("handle_close_secondary")
    identity_hits: list[tuple[Path, os.stat_result]] = []

    class Handle:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise secondary

    handle = Handle()

    def fail_parent_identity(path: Path, identity: os.stat_result) -> None:
        identity_hits.append((path, identity))
        raise primary

    monkeypatch.setattr(
        module,
        "open_owned_scratch_file",
        lambda *_args, **_kwargs: (handle, None),
    )
    monkeypatch.setattr(
        writer_module,
        "require_owned_scratch",
        fail_parent_identity,
    )

    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    assert caught is primary
    assert len(identity_hits) == 1
    assert identity_hits[0][0] == tmp_path
    assert handle.close_calls == 1


def _file_creator_cleanup(owner_name: str):  # noqa: ANN202
    del owner_name
    return scratch_owner, "remove_owned_scratch"


_FILE_CREATOR_NAMES = ("atomic-text",)


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
@pytest.mark.parametrize("wrapper_closes_descriptor", (False, True))
def test_file_creator_owns_descriptor_when_fdopen_handoff_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
    wrapper_closes_descriptor: bool,
) -> None:
    module = _file_creator_module(owner_name)
    primary = primary_type(f"{owner_name}_fdopen_handoff")
    real_close = os.close
    real_fstat = os.fstat
    descriptors: list[int] = []

    def fail_fdopen(descriptor: int, *_args, **_kwargs):  # noqa: ANN202
        descriptors.append(descriptor)
        if wrapper_closes_descriptor:
            real_close(descriptor)
        raise primary

    monkeypatch.setattr(module.os, "fdopen", fail_fdopen)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    try:
        real_fstat(descriptor)
    except OSError:
        descriptor_closed = True
    else:
        descriptor_closed = False
        real_close(descriptor)
    actual = {
        "primary": caught is primary,
        "descriptor_closed": descriptor_closed,
        "survivors": tuple(tmp_path.iterdir()),
    }
    expected = {
        "primary": True,
        "descriptor_closed": True,
        "survivors": (),
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_creator_fdopen_handoff_preserves_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    module, cleanup_name = _file_creator_cleanup(owner_name)
    primary = primary_type(f"{owner_name}_fdopen_handoff")
    real_open = os.open
    real_close = os.close
    real_cleanup = getattr(module, cleanup_name)
    opened_paths: list[Path] = []
    cleanup_identities: list[os.stat_result] = []

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        opened_paths.append(Path(path))
        return descriptor

    def replace_then_fail(descriptor: int, *_args, **_kwargs):  # noqa: ANN202
        assert len(opened_paths) == 1
        real_close(descriptor)
        path = opened_paths[0]
        path.unlink()
        replacement = real_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            os.write(replacement, b"foreign")
        finally:
            real_close(replacement)
        raise primary

    def record_cleanup(path: Path, identity: os.stat_result) -> None:
        if path == opened_paths[0]:
            cleanup_identities.append(identity)
        real_cleanup(path, identity)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fdopen", replace_then_fail)
    monkeypatch.setattr(module, cleanup_name, record_cleanup)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    assert len(opened_paths) == 1
    actual = {
        "primary": caught is primary,
        "cleanup_hits": len(cleanup_identities),
        "replacement": opened_paths[0].read_bytes(),
    }
    expected = {
        "primary": True,
        "cleanup_hits": 1,
        "replacement": b"foreign",
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_creator_fdopen_handoff_preserves_primary_over_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    module, cleanup_name = _file_creator_cleanup(owner_name)
    primary = primary_type(f"{owner_name}_fdopen_handoff")
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    real_cleanup = getattr(module, cleanup_name)
    descriptors: list[int] = []
    opened_paths: list[Path] = []
    close_hits = 0
    cleanup_hits = 0

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        opened_paths.append(Path(path))
        return descriptor

    def fail_fdopen(descriptor: int, *_args, **_kwargs):  # noqa: ANN202
        descriptors.append(descriptor)
        raise primary

    def fail_close(descriptor: int) -> None:
        nonlocal close_hits
        close_hits += 1
        real_close(descriptor)
        raise OSError("descriptor_close_secondary")

    def fail_cleanup(path: Path, identity: os.stat_result) -> None:
        nonlocal cleanup_hits
        if path == opened_paths[0]:
            cleanup_hits += 1
            raise OSError("path_cleanup_secondary")
        real_cleanup(path, identity)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(module.os, "close", fail_close)
    monkeypatch.setattr(module, cleanup_name, fail_cleanup)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    assert len(descriptors) == 1
    try:
        real_fstat(descriptors[0])
    except OSError:
        descriptor_closed = True
    else:
        descriptor_closed = False
        real_close(descriptors[0])
    actual = {
        "primary": caught is primary,
        "descriptor_closed": descriptor_closed,
        "close_hits": close_hits,
        "cleanup_hits": cleanup_hits,
    }
    expected = {
        "primary": True,
        "descriptor_closed": True,
        "close_hits": 1,
        "cleanup_hits": 1,
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
def test_file_creator_preserves_os_open_collision_before_fdopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
) -> None:
    module = _file_creator_module(owner_name)
    real_open = os.open
    real_close = os.close
    open_hits = 0
    fdopen_hits = 0
    collisions: list[Path] = []

    def collide(path, _flags, *_args, **_kwargs):  # noqa: ANN001, ANN202
        nonlocal open_hits
        open_hits += 1
        collision = Path(path)
        descriptor = real_open(collision, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            os.write(descriptor, b"foreign")
        finally:
            real_close(descriptor)
        collisions.append(collision)
        raise FileExistsError(collision)

    def unexpected_fdopen(*_args, **_kwargs):  # noqa: ANN202
        nonlocal fdopen_hits
        fdopen_hits += 1
        raise AssertionError("fdopen_after_open_collision")

    monkeypatch.setattr(module.os, "open", collide)
    monkeypatch.setattr(module.os, "fdopen", unexpected_fdopen)
    with pytest.raises(FileExistsError):
        _run_file_creator(owner_name, tmp_path)

    assert len(collisions) == 1
    actual = {
        "open_hits": open_hits,
        "fdopen_hits": fdopen_hits,
        "collision": collisions[0].read_bytes(),
    }
    expected = {
        "open_hits": 1,
        "fdopen_hits": 0,
        "collision": b"foreign",
    }
    assert actual == expected


@pytest.mark.parametrize(
    "owner_name",
    _FILE_CREATOR_NAMES,
)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_creator_cleans_when_identity_capture_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type(f"{owner_name}_identity_capture")
    real_fstat = os.fstat
    calls = 0
    hits = 0

    def interrupt_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls, hits
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 1:
            hits += 1
            raise primary
        return metadata

    monkeypatch.setattr(os, "fstat", interrupt_fstat)

    with pytest.raises(primary_type) as caught:
        _run_file_creator(owner_name, tmp_path)

    assert calls == 2
    assert hits == 1
    assert caught.value is primary
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    "owner_name",
    ("atomic-text", "manifest-replace", "single-parquet"),
)
@pytest.mark.parametrize("primary_type", _NAMESPACE_PRIMARY_TYPES)
def test_file_creator_unknown_identity_is_recovered_by_later_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    module = _file_creator_module(owner_name)
    primary = primary_type(f"{owner_name}_identity_capture")
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    descriptors: list[int] = []
    opened_paths: list[Path] = []
    fstat_hits = 0
    close_hits = 0

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        opened_paths.append(Path(path))
        return descriptor

    def fail_fstat(_descriptor: int) -> os.stat_result:
        nonlocal fstat_hits
        fstat_hits += 1
        raise primary

    def record_close(descriptor: int) -> None:
        nonlocal close_hits
        close_hits += 1
        real_close(descriptor)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fstat", fail_fstat)
    monkeypatch.setattr(module.os, "close", record_close)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    try:
        real_fstat(descriptor)
    except OSError:
        descriptor_closed = True
    else:
        descriptor_closed = False
        real_close(descriptor)
    path_survived_without_identity = opened_paths[0].exists()
    orphan = opened_paths[0]
    monkeypatch.setattr(module.os, "fstat", real_fstat)
    monkeypatch.setattr(
        scratch_owner,
        "scratch_owner_is_stale",
        lambda path, *, prefix: Path(path) == orphan,
    )
    _run_file_creator(owner_name, tmp_path)
    actual = {
        "primary": caught is primary,
        "fstat_hits": fstat_hits,
        "close_hits": close_hits,
        "descriptor_closed": descriptor_closed,
        "path_survived_without_identity": path_survived_without_identity,
        "later_operation_removed_orphan": not orphan.exists(),
    }
    expected = {
        "primary": True,
        "fstat_hits": 2,
        "close_hits": 1,
        "descriptor_closed": True,
        "path_survived_without_identity": True,
        "later_operation_removed_orphan": True,
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_creator_preserves_primary_when_descriptor_recheck_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    module, cleanup_name = _file_creator_cleanup(owner_name)
    primary = primary_type(f"{owner_name}_wrapper_primary")
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    real_cleanup = getattr(module, cleanup_name)
    descriptors: list[int] = []
    opened_paths: list[Path] = []
    fstat_hits = 0
    close_hits = 0
    cleanup_hits = 0

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        opened_paths.append(Path(path))
        return descriptor

    def fail_second_fstat(descriptor: int) -> os.stat_result:
        nonlocal fstat_hits
        fstat_hits += 1
        if fstat_hits == 2:
            raise OSError("descriptor_recheck_secondary")
        return real_fstat(descriptor)

    def fail_fdopen(*_args, **_kwargs):  # noqa: ANN202
        raise primary

    def record_close(descriptor: int) -> None:
        nonlocal close_hits
        close_hits += 1
        real_close(descriptor)

    def record_cleanup(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal cleanup_hits
        cleanup_hits += 1
        return real_cleanup(*args, **kwargs)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fstat", fail_second_fstat)
    monkeypatch.setattr(module.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(module.os, "close", record_close)
    monkeypatch.setattr(module, cleanup_name, record_cleanup)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    descriptor = descriptors[0]
    descriptor_open = True
    try:
        real_fstat(descriptor)
    except OSError:
        descriptor_open = False
    if descriptor_open:
        real_close(descriptor)
    if opened_paths[0].exists():
        opened_paths[0].unlink()
    actual = {
        "primary": caught is primary,
        "fstat_hits": fstat_hits,
        "close_hits": close_hits,
        "cleanup_hits": cleanup_hits,
        "descriptor_safely_retained": descriptor_open,
    }
    expected = {
        "primary": True,
        "fstat_hits": 2,
        "close_hits": 0,
        "cleanup_hits": 1,
        "descriptor_safely_retained": True,
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_creator_does_not_close_reused_foreign_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    module = _file_creator_module(owner_name)
    primary = primary_type(f"{owner_name}_wrapper_primary")
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    descriptors: list[int] = []
    opened_paths: list[Path] = []
    close_hits = 0
    foreign_path = tmp_path / "foreign.bin"

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        opened_paths.append(Path(path))
        return descriptor

    def reuse_then_fail(descriptor: int, *_args, **_kwargs):  # noqa: ANN202
        real_close(descriptor)
        foreign_descriptor = real_open(
            foreign_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
        )
        if foreign_descriptor != descriptor:
            os.dup2(foreign_descriptor, descriptor)
            real_close(foreign_descriptor)
        os.write(descriptor, b"foreign")
        raise primary

    def record_close(descriptor: int) -> None:
        nonlocal close_hits
        close_hits += 1
        real_close(descriptor)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fdopen", reuse_then_fail)
    monkeypatch.setattr(module.os, "close", record_close)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    descriptor = descriptors[0]
    descriptor_open = True
    try:
        real_fstat(descriptor)
    except OSError:
        descriptor_open = False
    if descriptor_open:
        os.lseek(descriptor, 0, os.SEEK_SET)
        foreign_payload = os.read(descriptor, len(b"foreign"))
        real_close(descriptor)
    else:
        foreign_payload = b""
    actual = {
        "primary": caught is primary,
        "helper_close_hits": close_hits,
        "foreign_descriptor_open": descriptor_open,
        "foreign_payload": foreign_payload,
        "own_path_removed": not opened_paths[0].exists(),
    }
    expected = {
        "primary": True,
        "helper_close_hits": 0,
        "foreign_descriptor_open": True,
        "foreign_payload": b"foreign",
        "own_path_removed": True,
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
@pytest.mark.parametrize("close_timing", ("before", "after"))
def test_file_creator_does_not_retry_ambiguous_descriptor_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
    close_timing: str,
) -> None:
    module, cleanup_name = _file_creator_cleanup(owner_name)
    primary = primary_type(f"{owner_name}_wrapper_primary")
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    real_cleanup = getattr(module, cleanup_name)
    descriptors: list[int] = []
    opened_paths: list[Path] = []
    close_hits = 0
    cleanup_hits = 0

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        opened_paths.append(Path(path))
        return descriptor

    def fail_fdopen(*_args, **_kwargs):  # noqa: ANN202
        raise primary

    def fail_close(descriptor: int) -> None:
        nonlocal close_hits
        close_hits += 1
        if close_timing == "after":
            real_close(descriptor)
        raise OSError(f"descriptor_close_{close_timing}")

    def cleanup(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal cleanup_hits
        cleanup_hits += 1
        if close_timing == "before":
            raise OSError("path_cleanup_blocked_with_open_descriptor")
        return real_cleanup(*args, **kwargs)

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(module.os, "close", fail_close)
    monkeypatch.setattr(module, cleanup_name, cleanup)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc

    descriptor = descriptors[0]
    descriptor_open = True
    try:
        real_fstat(descriptor)
    except OSError:
        descriptor_open = False
    path_survived = opened_paths[0].exists()
    if descriptor_open:
        real_close(descriptor)
    if path_survived:
        opened_paths[0].unlink()
    actual = {
        "primary": caught is primary,
        "close_hits": close_hits,
        "cleanup_hits": cleanup_hits,
        "descriptor_safely_retained": descriptor_open,
        "path_safely_retained": path_survived,
    }
    expected = {
        "primary": True,
        "close_hits": 1,
        "cleanup_hits": 1,
        "descriptor_safely_retained": close_timing == "before",
        "path_safely_retained": close_timing == "before",
    }
    assert actual == expected


@pytest.mark.parametrize("owner_name", _FILE_CREATOR_NAMES)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_file_creator_owns_handle_return_before_descriptor_state_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException],
) -> None:
    module = _file_creator_module(owner_name)
    helper = module.open_owned_scratch_file
    primary = primary_type(f"{owner_name}_post_handoff")
    real_open = os.open
    real_fdopen = os.fdopen
    real_fstat = os.fstat
    descriptors: list[int] = []
    opened_paths: list[Path] = []
    handles: list[object] = []
    handoff_returned = False
    trace_hits = 0

    def record_open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202
        descriptor = real_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        opened_paths.append(Path(path))
        return descriptor

    def record_fdopen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal handoff_returned
        handle = real_fdopen(*args, **kwargs)
        handles.append(handle)
        handoff_returned = True
        return handle

    def trace(frame, event, _arg):  # noqa: ANN001, ANN202
        nonlocal trace_hits
        if event == "line" and frame.f_code is helper.__code__ and handoff_returned:
            trace_hits += 1
            raise primary
        return trace

    monkeypatch.setattr(module.os, "open", record_open)
    monkeypatch.setattr(module.os, "fdopen", record_fdopen)
    previous_trace = sys.gettrace()
    sys.settrace(trace)
    caught: BaseException | None = None
    try:
        _run_file_creator(owner_name, tmp_path)
    except BaseException as exc:
        caught = exc
    finally:
        sys.settrace(previous_trace)
    assert sys.gettrace() is previous_trace

    descriptor = descriptors[0]
    try:
        real_fstat(descriptor)
    except OSError:
        descriptor_closed = True
    else:
        descriptor_closed = False
        os.close(descriptor)
    actual = {
        "primary": caught is primary,
        "trace_hits": trace_hits,
        "handle_closed": len(handles) == 1 and handles[0].closed,
        "descriptor_closed": descriptor_closed,
        "own_path_removed": not opened_paths[0].exists(),
    }
    expected = {
        "primary": True,
        "trace_hits": 1,
        "handle_closed": True,
        "descriptor_closed": True,
        "own_path_removed": True,
    }
    assert actual == expected


@pytest.mark.parametrize(
    "owner_name",
    (
        "atomic-text",
        "manifest-replace",
        "single-parquet",
    ),
)
@pytest.mark.parametrize(
    "primary_type",
    (None, RuntimeError, KeyboardInterrupt),
    ids=("identity-error", "exception", "keyboard-interrupt"),
)
def test_file_creator_does_not_delete_post_claim_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_name: str,
    primary_type: type[BaseException] | None,
) -> None:
    primary = primary_type("post_claim_primary") if primary_type is not None else None
    replacements: list[Path] = []
    hits = 0

    def replace_file(path: Path) -> None:
        path.unlink()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            os.write(descriptor, b"foreign")
        finally:
            os.close(descriptor)
        replacements.append(path)

    if owner_name in {"atomic-text", "atomic-csv", "atomic-target-marker"}:
        owner_module = atomic
        helper_name = "require_owned_scratch"
    elif owner_name == "manifest-replace":
        owner_module = manifests
        helper_name = "require_owned_scratch"
    else:
        owner_module = parquet_writers
        helper_name = "require_owned_scratch"
    real_require = getattr(owner_module, helper_name)

    def replace_before_check(path: Path, identity: os.stat_result) -> None:
        nonlocal hits
        if Path(path) == tmp_path:
            return real_require(path, identity)
        if owner_name == "atomic-target-marker" and path.suffix != ".target":
            return real_require(path, identity)
        if hits:
            return real_require(path, identity)
        hits += 1
        replace_file(path)
        if primary is not None:
            raise primary
        return real_require(path, identity)

    monkeypatch.setattr(owner_module, helper_name, replace_before_check)

    def operation() -> object:
        if owner_name == "chunked-parquet":
            return parquet_writers.write_parquet_chunks(
                (
                    pd.DataFrame(
                        {
                            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
                        }
                    ),
                ),
                tmp_path / "prediction.parquet",
                required=("timestamp",),
                key_columns=("timestamp",),
                compression="zstd",
            )
        return _run_file_creator(owner_name, tmp_path)

    expected_type = primary_type or ValueError
    with pytest.raises(expected_type) as caught:
        operation()

    assert hits == 1
    if primary is not None:
        assert caught.value is primary
    assert len(replacements) == 1
    assert replacements[0].read_bytes() == b"foreign"


@pytest.mark.parametrize(
    ("owner_module", "owner_type", "open_owner"),
    (
        (
            key_owner,
            key_owner.ExactPrimaryKeyIndex,
            lambda scratch: key_owner.ExactPrimaryKeyIndex(
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=scratch,
            ).__enter__(),
        ),
        (
            inventory_owner,
            inventory_owner.DatasetSessionInventory,
            lambda scratch: inventory_owner.DatasetSessionInventory(
                data_root=scratch.parent / "data",
                scratch_parent=scratch,
                task_memory_budget_bytes=1024,
            ),
        ),
    ),
    ids=("exact-primary-key-index", "dataset-session-inventory"),
)
def test_scratch_creator_is_cleanup_owned_before_instance_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_module: object,
    owner_type: type,
    open_owner,
) -> None:
    primary = SystemExit(9)
    created: list[Path] = []
    real_create = owner_module.create_owned_scratch

    def record_create(path: Path) -> os.stat_result:
        identity = real_create(path)
        created.append(path)
        return identity

    real_setattr = owner_type.__setattr__

    def interrupt_assignment(self, name: str, value: object) -> None:
        real_setattr(self, name, value)
        if name == "_temporary_directory" and value is not None:
            raise primary

    monkeypatch.setattr(owner_module, "create_owned_scratch", record_create)
    monkeypatch.setattr(owner_type, "__setattr__", interrupt_assignment)

    with pytest.raises(SystemExit) as caught:
        open_owner(tmp_path / "scratch")

    assert caught.value is primary
    assert len(created) == 1
    assert not created[0].exists()


@pytest.mark.parametrize(
    "open_owner",
    (
        lambda scratch: key_owner.ExactPrimaryKeyIndex(
            primary_key=("value",),
            timestamp_columns=(),
            scratch_parent=scratch,
        ).__enter__(),
        lambda scratch: inventory_owner.DatasetSessionInventory(
            data_root=scratch.parent / "data",
            scratch_parent=scratch,
            task_memory_budget_bytes=1024,
        ),
    ),
    ids=("exact-primary-key-index", "dataset-session-inventory"),
)
def test_scratch_owner_does_not_delete_post_claim_replacement(
    tmp_path: Path,
    open_owner,
) -> None:
    owner = open_owner(tmp_path / "scratch")
    scratch_path = (
        owner.scratch_path
        if hasattr(owner, "scratch_path")
        else owner.database_path.parent
    )
    connection = owner._connection
    assert connection is not None
    connection.close()
    owner._connection = None
    shutil.rmtree(scratch_path)
    scratch_path.mkdir()
    (scratch_path / "foreign.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(ValueError, match="storage_scratch_identity_changed"):
        owner.close()

    assert (scratch_path / "foreign.txt").read_text(encoding="utf-8") == "foreign"


def _directory_not_empty_error() -> OSError:
    error = OSError("directory_not_empty")
    error.winerror = 145
    return error


def test_non_directory_not_empty_cleanup_error_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "owned"
    path.mkdir()
    identity = path.lstat()
    primary = OSError("ordinary_cleanup_error")
    calls = 0

    def fail(_candidate, *, onexc=None) -> None:  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        raise primary

    monkeypatch.setattr(scratch_owner.shutil, "rmtree", fail)
    with pytest.raises(OSError) as caught:
        scratch_owner.remove_owned_scratch(path, identity)

    assert caught.value is primary
    assert calls == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only fallback boundary")
def test_posix_directory_not_empty_cleanup_error_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "owned"
    path.mkdir()
    identity = path.lstat()
    primary = _directory_not_empty_error()
    calls = 0

    def fail(_candidate, *, onexc=None) -> None:  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        raise primary

    monkeypatch.setattr(scratch_owner.shutil, "rmtree", fail)
    with pytest.raises(OSError) as caught:
        scratch_owner.remove_owned_scratch(path, identity)

    assert caught.value is primary
    assert calls == 1


@pytest.mark.parametrize("foreign_kind", ("file", "link"))
def test_scratch_create_rejects_observable_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_kind: str,
) -> None:
    path = tmp_path / "scratch"
    external = tmp_path / "external"
    external.mkdir()
    (external / "foreign.txt").write_text("foreign", encoding="utf-8")
    real_mkdir = os.mkdir
    mkdir_hits = 0

    def replace_after_mkdir(candidate, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal mkdir_hits
        result = real_mkdir(candidate, *args, **kwargs)
        if Path(candidate) != path:
            return result
        mkdir_hits += 1
        shutil.rmtree(path)
        if foreign_kind == "file":
            path.write_bytes(b"foreign")
            return result
        try:
            path.symlink_to(external, target_is_directory=True)
        except OSError as exc:
            if os.name != "nt" or not hasattr(Path, "is_junction"):
                pytest.skip(f"directory_reparse_not_available: {exc}")
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(path), str(external)],
                capture_output=True,
                text=True,
            )
            if created.returncode != 0:
                pytest.skip("directory_reparse_not_available")
        return result

    monkeypatch.setattr(os, "mkdir", replace_after_mkdir)
    caught: BaseException | None = None
    try:
        scratch_owner.create_owned_scratch(path)
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    foreign_preserved = (
        path.read_bytes() == b"foreign"
        if foreign_kind == "file" and path.is_file()
        else (path / "foreign.txt").read_text(encoding="utf-8") == "foreign"
        if path.exists()
        else False
    )
    actual = {
        "mkdir_hits": mkdir_hits,
        "error": None if caught is None else f"{type(caught).__name__}:{caught}",
        "foreign_preserved": foreign_preserved,
    }
    expected = {
        "mkdir_hits": 1,
        "error": "ValueError:storage_scratch_identity_changed",
        "foreign_preserved": True,
    }
    assert actual == expected


def test_scratch_create_rejects_parent_replacement_without_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    path = parent / "scratch"
    displaced = tmp_path / "displaced-parent"
    real_mkdir = os.mkdir
    mkdir_hits = 0

    def replace_parent_after_mkdir(candidate, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal mkdir_hits
        result = real_mkdir(candidate, *args, **kwargs)
        if Path(candidate) != path:
            return result
        mkdir_hits += 1
        parent.rename(displaced)
        real_mkdir(parent)
        real_mkdir(path)
        (path / "foreign.txt").write_text("foreign", encoding="utf-8")
        (displaced / "scratch" / "owned.txt").write_text("owned", encoding="utf-8")
        return result

    monkeypatch.setattr(os, "mkdir", replace_parent_after_mkdir)
    caught: BaseException | None = None
    try:
        scratch_owner.create_owned_scratch(path)
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "mkdir_hits": mkdir_hits,
        "error": None if caught is None else f"{type(caught).__name__}:{caught}",
        "foreign_preserved": (path / "foreign.txt").is_file(),
        "owned_survivor_preserved": (displaced / "scratch" / "owned.txt").is_file(),
    }
    expected = {
        "mkdir_hits": 1,
        "error": "ValueError:storage_scratch_parent_identity_changed",
        "foreign_preserved": True,
        "owned_survivor_preserved": True,
    }
    assert actual == expected


@pytest.mark.parametrize("caller", ("atomic", "key-index", "inventory"))
def test_directory_callers_stop_before_foreign_file_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller: str,
) -> None:
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    final_root = scratch_parent / "result"
    displaced = tmp_path / "displaced"
    created_paths: list[Path] = []
    side_effect_hits: list[Path] = []

    def replace_after_create(real_create, candidate: Path):  # noqa: ANN001, ANN202
        identity = real_create(candidate)
        created_paths.append(Path(candidate))
        scratch_parent.rename(displaced)
        scratch_parent.mkdir()
        Path(candidate).mkdir()
        (Path(candidate) / "foreign.txt").write_text("foreign", encoding="utf-8")
        return identity

    def connect(path, *_args, **_kwargs):  # noqa: ANN001, ANN202
        side_effect_hits.append(Path(path))
        raise RuntimeError("caller_side_effect_reached")

    if caller == "key-index":
        real_create = key_owner.create_owned_scratch
        monkeypatch.setattr(
            key_owner,
            "create_owned_scratch",
            lambda path: replace_after_create(real_create, path),
        )
        monkeypatch.setattr(key_owner.sqlite3, "connect", connect)
        operation = lambda: key_owner.ExactPrimaryKeyIndex(  # noqa: E731
            primary_key=("value",),
            timestamp_columns=(),
            scratch_parent=scratch_parent,
        ).__enter__()
    elif caller == "inventory":
        real_create = inventory_owner.create_owned_scratch
        monkeypatch.setattr(
            inventory_owner,
            "create_owned_scratch",
            lambda path: replace_after_create(real_create, path),
        )
        monkeypatch.setattr(inventory_owner.sqlite3, "connect", connect)
        operation = lambda: inventory_owner.DatasetSessionInventory(  # noqa: E731
            data_root=tmp_path / "data",
            scratch_parent=scratch_parent,
            task_memory_budget_bytes=1024,
        )
    else:
        real_create = atomic.create_owned_scratch
        monkeypatch.setattr(
            atomic,
            "create_owned_scratch",
            lambda path: replace_after_create(real_create, path),
        )
        operation = lambda: atomic.atomic_write_directory(  # noqa: E731
            final_root,
            lambda path: side_effect_hits.append(path),
        )

    caught: BaseException | None = None
    try:
        operation()
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "create_hits": len(created_paths),
        "side_effect_hits": len(side_effect_hits),
        "error": None if caught is None else f"{type(caught).__name__}:{caught}",
        "foreign_preserved": len(created_paths) == 1
        and (created_paths[0] / "foreign.txt").read_text(encoding="utf-8") == "foreign",
    }
    expected = {
        "create_hits": 1,
        "side_effect_hits": 0,
        "error": "ValueError:storage_scratch_identity_changed",
        "foreign_preserved": True,
    }
    assert actual == expected


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=megabytes * 1024 * 1024,
    )


def test_unaliased_path_checks_lexical_component_before_parent_traversal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alias = tmp_path / "alias"
    visited: list[Path] = []

    def is_alias(path: Path) -> bool:
        visited.append(path)
        return path == alias

    monkeypatch.setattr(dataset_paths, "path_is_filesystem_alias", is_alias)

    with pytest.raises(ValueError, match="invalid_path"):
        dataset_paths.require_unaliased_path(
            alias / ".." / "artifact",
            error="invalid_path",
        )

    assert alias in visited


def test_bounded_json_rejects_same_inode_hardlink_replacement_after_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    moved = tmp_path / "moved.json"
    path.write_text('{"schema_id":"test"}', encoding="utf-8")
    preserve_primary = manifests.scratch_owner.handle_preserving_primary

    @contextmanager
    def replace_after_read(handle):  # noqa: ANN001, ANN202
        with preserve_primary(handle) as opened:
            yield opened
        path.rename(moved)
        try:
            os.link(moved, path)
        except OSError as exc:
            pytest.skip(f"hardlink_not_available: {exc}")

    monkeypatch.setattr(
        manifests.scratch_owner,
        "handle_preserving_primary",
        replace_after_read,
    )

    with pytest.raises(
        ValueError,
        match="dataset_metadata_file_identity_changed",
    ):
        manifests.read_bounded_qrp_json(path)


def test_chunked_read_rejects_different_explicit_policy_at_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _policy(1)
    different = _policy(2)
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_after_policy_conflict"),
    )

    with runtime_resource_policy_scope(active):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            dataset_reading.read_dataset(
                Path("unused"),
                mode=DatasetReadMode.CHUNKED,
                resource_policy=different,
            )


def test_chunked_read_binds_explicit_outer_for_active_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    observed: list[RuntimeResourcePolicy] = []

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        observed.append(resource_policy)
        yield SimpleNamespace(resource_policy=resource_policy)

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter(()),
    )

    with runtime_resource_policy_scope(policy):
        chunks = dataset_reading.read_dataset(
            Path("unused"),
            mode=DatasetReadMode.CHUNKED,
            resource_policy=policy,
        )
        assert list(chunks) == []

    assert observed == [policy]


def test_chunked_read_rejects_delayed_consumption_after_outer_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_for_retired_policy"),
    )

    with runtime_resource_policy_scope(policy):
        chunks = dataset_reading.read_dataset(
            Path("unused"),
            mode=DatasetReadMode.CHUNKED,
            resource_policy=policy,
        )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        next(chunks)


def test_chunked_read_rejects_implicit_active_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_after_competing_root"),
    )

    with runtime_resource_policy_scope(_policy(1)):
        with pytest.raises(
            RuntimeError,
            match="runtime_resource_policy_competing_root",
        ):
            dataset_reading.read_dataset(
                Path("unused"),
                mode=DatasetReadMode.CHUNKED,
                resource_policy=None,
            )


@pytest.mark.parametrize(
    ("module", "owner", "body_name"),
    (
        (inspection, inspection.inspect_dataset, "inspect_dataset_session"),
        (
            fingerprinting,
            fingerprinting.fingerprint_dataset,
            "fingerprint_dataset_session",
        ),
    ),
)
def test_standalone_storage_root_preflights_before_admission_and_cleans_before_release(
    module,
    owner,
    body_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = capture_storage_test_policy(
        working_headroom_bytes=1024**2,
    )

    class Session:
        resource_policy = policy

        def __enter__(self) -> Session:
            events.append("session-enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("cleanup")

    @contextmanager
    def scope(_policy):  # noqa: ANN001, ANN202
        events.append("admission")
        try:
            yield policy
        finally:
            events.append("release")

    def preflight(*_args, **_kwargs):  # noqa: ANN202
        events.append("preflight")
        return Session()

    monkeypatch.setattr(
        module,
        "active_runtime_resource_policy",
        lambda: pytest.fail("implicit_policy_issued_before_metadata_preflight"),
        raising=False,
    )

    @contextmanager
    def preflight_without_early_snapshot(*_args, **kwargs):  # noqa: ANN202
        assert kwargs["resource_policy"] is None
        session = preflight()
        with scope(policy):
            with session:
                yield session

    monkeypatch.setattr(
        module,
        "open_dataset_session",
        preflight_without_early_snapshot,
    )
    monkeypatch.setattr(
        module,
        body_name,
        lambda *_args, **_kwargs: events.append("body"),
    )

    owner(Path("dataset"), resource_policy=None)

    assert events == [
        "preflight",
        "admission",
        "session-enter",
        "body",
        "cleanup",
        "release",
    ]


@pytest.mark.parametrize(
    "mode",
    (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED),
)
def test_read_dataset_keeps_discovered_policy_active_through_session_exit(
    mode: DatasetReadMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    competing = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    class Session:
        resource_policy = policy

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            assert current_runtime_resource_policy() is policy
            with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
                with runtime_resource_policy_scope(competing):
                    pass

    @contextmanager
    def open_session(*_args, **_kwargs):  # noqa: ANN202
        with runtime_resource_policy_scope(policy):
            with Session() as session:
                yield session

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "materialize_dataset_session",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(()),
    )

    result = dataset_reading.read_dataset(
        Path("dataset"),
        mode=mode,
        resource_policy=None,
    )
    if mode is DatasetReadMode.CHUNKED:
        list(result)


class _FailureBetweenValues:
    def __init__(
        self,
        first: Path,
        failure: BaseException,
        later: Path,
    ) -> None:
        self._values = iter((first, failure, later))

    def __iter__(self) -> _FailureBetweenValues:
        return self

    def __next__(self) -> Path:
        value = next(self._values)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.mark.parametrize(
    ("owner", "later_name"),
    (
        (inspection.inspect_dataset_session, "_inspection_runtime_metadata"),
        (fingerprinting.fingerprint_dataset_session, "iter_dataset_file_batches"),
    ),
)
def test_session_owner_rejects_policy_mismatch_before_storage_work(
    monkeypatch: pytest.MonkeyPatch,
    owner,
    later_name: str,
) -> None:
    session_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    explicit_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    module = (
        inspection if owner is inspection.inspect_dataset_session else fingerprinting
    )
    monkeypatch.setattr(
        module,
        later_name,
        lambda *_args, **_kwargs: pytest.fail("storage work reached"),
    )
    context = SimpleNamespace(resource_policy=session_policy)

    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        owner(context, resource_policy=explicit_policy)


class _BodyAndCloseFailingHandle:
    def __init__(self, primary: BaseException) -> None:
        self.primary = primary

    def __enter__(self) -> _BodyAndCloseFailingHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def fileno(self) -> int:
        return 1

    def read(self, *_args: object) -> bytes:
        raise self.primary

    def write(self, *_args: object) -> int:
        raise self.primary

    def flush(self) -> None:
        pass

    def close(self) -> None:
        raise OSError("close_secondary")


def _inventory_with_failing_connection(
    tmp_path: Path,
) -> inventory_owner.DatasetSessionInventory:
    class FailingConnection:
        def close(self) -> None:
            raise RuntimeError("close_failed")

    inventory = inventory_owner.DatasetSessionInventory(
        data_root=tmp_path / "data",
        scratch_parent=tmp_path / "scratch",
        task_memory_budget_bytes=1024,
    )
    assert inventory._connection is not None
    inventory._connection.close()
    inventory._connection = FailingConnection()  # type: ignore[assignment]
    return inventory


def test_inventory_exit_preserves_generator_exit_primary(tmp_path: Path) -> None:
    inventory = _inventory_with_failing_connection(tmp_path)

    with pytest.raises(GeneratorExit):
        with inventory:
            raise GeneratorExit()


def test_inventory_exit_propagates_cleanup_failure_without_primary(
    tmp_path: Path,
) -> None:
    inventory = _inventory_with_failing_connection(tmp_path)

    with pytest.raises(RuntimeError, match="close_failed"):
        with inventory:
            pass


def _patch_open_session_metadata_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, str], dict[str, str]]:
    spec = {"schema_id": "qrp-dataset-spec"}
    manifest = {"schema_id": "qrp-dataset-manifest"}
    monkeypatch.setattr(dataset_reading, "require_artifact_root", Path)
    monkeypatch.setattr(
        dataset_reading,
        "require_unaliased_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    monkeypatch.setattr(Path, "is_dir", lambda _self: True)
    monkeypatch.setattr(
        dataset_reading,
        "dataset_file_identity",
        lambda _path: physical.DatasetFileIdentity(1, 2, 2, 3),
    )
    monkeypatch.setattr(
        dataset_reading,
        "read_bounded_qrp_json",
        lambda path, **_kwargs: (
            b"{}",
            spec if Path(path).name == "spec.json" else manifest,
        ),
    )
    monkeypatch.setattr(
        dataset_reading,
        "validate_dataset_spec_payload",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "validate_dataset_manifest_payload",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "validate_dataset_manifest_consumable",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "manifest_has_runtime_metadata",
        lambda *_a, **_k: True,
    )
    return spec, manifest


def test_open_session_construction_cleanup_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    class Inventory:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs: object) -> None:
            assert current_runtime_resource_policy() is policy

        def discover(self, _spec: object) -> None:
            raise GeneratorExit()

        def close(self) -> None:
            raise RuntimeError("close_failed")

    _patch_open_session_metadata_preflight(monkeypatch)
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)

    with pytest.raises(GeneratorExit):
        with dataset_reading.open_dataset_session(
            Path("dataset"),
            columns=None,
            time_window=None,
            resource_policy=policy,
        ):
            pass


def test_open_session_keeps_exact_policy_active_through_owned_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    events: list[str] = []
    candidate = Path("dataset/data/year=2026/part-000.parquet")

    def record(event: str) -> None:
        assert current_runtime_resource_policy() is policy
        events.append(event)

    class Inventory:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs: object) -> None:
            record("inventory-init")

        def discover(self, _spec: object) -> None:
            record("discover")

        def path_count(self, *, selected: bool) -> int:
            assert selected is False
            assert current_runtime_resource_policy() is policy
            return 1

        def iter_candidate_paths(self):  # noqa: ANN202
            record("candidate-iteration")
            yield candidate

        def observe(self, path: Path, _metadata: object, *, selected: bool) -> None:
            assert path == candidate
            assert selected
            record("observe")

        def close(self) -> None:
            record("inventory-close")

    class Session:
        def __init__(self, inventory: Inventory) -> None:
            self.inventory = inventory

        def __enter__(self) -> Session:
            record("session-enter")
            return self

        def __exit__(self, *_args: object) -> None:
            record("session-exit")
            self.inventory.close()

    def inspect(path: Path) -> object:
        assert path == candidate
        record("inspect")
        return object()

    def prune(
        _data_root: Path,
        paths: tuple[Path, ...],
        **_kwargs: object,
    ) -> tuple[Path, ...]:
        assert paths == (candidate,)
        assert current_runtime_resource_policy() is policy
        return paths

    def build(**kwargs: object) -> Session:
        record("build")
        return Session(kwargs["inventory"])  # type: ignore[arg-type]

    _patch_open_session_metadata_preflight(monkeypatch)
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(dataset_reading, "inspect_dataset_part", inspect)
    monkeypatch.setattr(dataset_reading, "prune_paths_for_time_window", prune)
    monkeypatch.setattr(dataset_reading, "_build_dataset_session", build)

    with dataset_reading.open_dataset_session(
        Path("dataset"),
        columns=None,
        time_window=None,
        resource_policy=policy,
    ) as session:
        assert isinstance(session, Session)
        record("body")

    assert events == [
        "inventory-init",
        "discover",
        "candidate-iteration",
        "inspect",
        "observe",
        "build",
        "session-enter",
        "body",
        "session-exit",
        "inventory-close",
    ]


def test_open_session_policy_entry_failure_precedes_inventory_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    primary = RuntimeError("runtime_resource_policy_concurrent_root")
    inventory_init_calls = 0

    class Inventory:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal inventory_init_calls
            inventory_init_calls += 1

    @contextmanager
    def failing_scope(current: RuntimeResourcePolicy):
        assert current is policy
        raise primary
        yield

    _patch_open_session_metadata_preflight(monkeypatch)
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(dataset_reading, "runtime_resource_policy_scope", failing_scope)

    with pytest.raises(RuntimeError) as raised:
        with dataset_reading.open_dataset_session(
            Path("dataset"),
            columns=None,
            time_window=None,
            resource_policy=policy,
        ):
            pytest.fail("session body reached after policy entry failure")

    assert raised.value is primary
    assert inventory_init_calls == 0


def test_opened_parquet_close_preserves_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def fileno(self) -> int:
            return 1

    class Parquet:
        def close(self) -> None:
            raise RuntimeError("close_failed")

    metadata = type(
        "Stat",
        (),
        {"st_dev": 1, "st_ino": 2, "st_size": 3, "st_mtime_ns": 4},
    )()
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: physical.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with pytest.raises(GeneratorExit):
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            raise GeneratorExit()


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_write_dataset_activates_and_closes_caller_iterator_under_policy(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_storage_test_policy(
        working_headroom_bytes=1024**2,
    )
    events: list[tuple[str, RuntimeResourcePolicy | None]] = []
    primary = primary_type("body_primary")

    class InputIterator:
        def __iter__(self) -> InputIterator:
            return self

        def __next__(self) -> pd.DataFrame:
            raise StopIteration

        def close(self) -> None:
            events.append(("close", current_runtime_resource_policy()))

    class CallerIterable:
        def __iter__(self) -> InputIterator:
            events.append(("iter", current_runtime_resource_policy()))
            return InputIterator()

    monkeypatch.setattr(dataset_writing, "validate_dataset_spec", lambda _spec: None)
    monkeypatch.setattr(
        dataset_writing,
        "validate_dataset_artifact_root",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        dataset_writing,
        "atomic_write_directory",
        lambda *_args: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(primary_type) as raised:
        dataset_writing.write_dataset(
            Path("dataset"),
            CallerIterable(),
            spec=object(),  # type: ignore[arg-type]
            resource_policy=policy,
        )

    assert raised.value is primary
    assert events == [("iter", policy), ("close", policy)]


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_opened_parquet_raw_handle_close_preserves_active_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("body_primary")

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return 1

        def close(self) -> None:
            raise OSError("raw_close_secondary")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: physical.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with pytest.raises(primary_type) as raised:
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            raise primary

    assert raised.value is primary


def test_opened_parquet_raw_handle_close_propagates_without_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return 1

        def close(self) -> None:
            raise OSError("raw_close_failed")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: physical.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with pytest.raises(OSError, match="raw_close_failed"):
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            pass


@pytest.mark.parametrize(
    "primary",
    (MemoryError(), KeyboardInterrupt(), SystemExit(), GeneratorExit()),
)
def test_atomic_directory_cleanup_preserves_active_primary(
    primary: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        atomic,
        "remove_owned_scratch",
        lambda _path, _identity: (_ for _ in ()).throw(RuntimeError("cleanup_failed")),
    )

    def writer(_path: Path) -> None:
        raise primary

    with pytest.raises(type(primary)):
        atomic.atomic_write_directory(tmp_path / "result", writer)


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_atomic_directory_attempts_all_cleanup_and_preserves_body_primary(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("body_primary")
    events: list[str] = []

    def remove_claimed(path: Path, _identity: os.stat_result) -> None:
        events.append("marker" if path.name.endswith(".target") else "stage")
        raise OSError("cleanup_secondary")

    monkeypatch.setattr(atomic, "remove_owned_scratch", remove_claimed)

    with pytest.raises(primary_type) as raised:
        atomic.atomic_write_directory(
            tmp_path / "result",
            lambda _path: (_ for _ in ()).throw(primary),
        )

    assert raised.value is primary
    assert events == ["stage", "marker"]


@pytest.mark.parametrize("orphan", (False, True), ids=("stale-stage", "orphan"))
def test_atomic_stale_marker_cleanup_failure_propagates_without_primary(
    orphan: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stage = final_root.with_name(f"{prefix}owner-stale")
    marker = Path(f"{stage}.target")
    if not orphan:
        stage.mkdir()
    marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    real_remove = atomic.remove_owned_scratch

    def remove_owned(path: Path, identity: os.stat_result) -> None:
        if path == marker:
            raise OSError("marker_cleanup_failed")
        real_remove(path, identity)

    monkeypatch.setattr(atomic, "remove_owned_scratch", remove_owned)

    with pytest.raises(OSError, match="marker_cleanup_failed"):
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert not stage.exists()
    assert marker.exists()


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_atomic_stale_sweep_attempts_independent_cleanup_and_preserves_first_failure(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    first_stage = final_root.with_name(f"{prefix}first-stage")
    later_stage = final_root.with_name(f"{prefix}later-stage")
    first_marker = Path(f"{first_stage}.target")
    later_marker = Path(f"{later_stage}.target")
    first_orphan = final_root.with_name(f"{prefix}first-orphan.target")
    later_orphan = final_root.with_name(f"{prefix}later-orphan.target")
    for stage in (first_stage, later_stage):
        stage.mkdir()
    for marker in (first_marker, later_marker, first_orphan, later_orphan):
        marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")
    primary = primary_type("first_cleanup_failure")
    events: list[tuple[str, Path]] = []

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter((first_stage, later_stage))
        assert pattern == f"{prefix}*.target"
        return iter((first_orphan, later_orphan))

    def remove_path(path: Path) -> None:
        events.append(("stage", path))
        if path == first_stage:
            raise primary

    def remove_marker(path: Path) -> None:
        events.append(("marker", path))
        if path == first_orphan:
            raise OSError("later_cleanup_failure")

    def remove_owned(path: Path, _identity: os.stat_result, **_kwargs: object) -> None:
        if path.name.endswith(".target"):
            remove_marker(path)
        else:
            remove_path(path)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(atomic, "remove_owned_scratch", remove_owned)

    with pytest.raises(primary_type) as raised:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert raised.value is primary
    assert events == [
        ("stage", first_stage),
        ("stage", later_stage),
        ("marker", later_marker),
        ("marker", first_orphan),
        ("marker", later_orphan),
    ]
    assert first_stage.exists()
    assert first_marker.exists()


@pytest.mark.parametrize("primary_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize(
    "later_failure_site",
    ("ownership", "iterdir", "exists"),
)
def test_atomic_stale_sweep_preserves_first_failure_across_later_probes(
    primary_type: type[BaseException],
    later_failure_site: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    first_stage = final_root.with_name(f"{prefix}first")
    probe_stage = final_root.with_name(f"{prefix}probe")
    later_stage = final_root.with_name(f"{prefix}later")
    first_marker = Path(f"{first_stage}.target")
    probe_marker = Path(f"{probe_stage}.target")
    later_marker = Path(f"{later_stage}.target")
    probe_orphan = final_root.with_name(f"{prefix}probe-orphan.target")
    later_orphan = final_root.with_name(f"{prefix}later-orphan.target")
    primary = primary_type("first_failure")
    later_failure = RuntimeError("later_probe_failure")
    removed_stages: list[Path] = []
    removed_markers: list[Path] = []
    for stage in (first_stage, probe_stage, later_stage):
        stage.mkdir()
    for marker in (
        first_marker,
        probe_marker,
        later_marker,
        probe_orphan,
        later_orphan,
    ):
        marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter((first_stage, probe_stage, later_stage))
        assert pattern == f"{prefix}*.target"
        if later_failure_site == "exists":
            return iter((probe_orphan, later_orphan))
        return iter(())

    def is_stale(path: Path, *, prefix: str) -> bool:
        assert prefix
        if later_failure_site == "ownership" and path == probe_stage:
            raise later_failure
        return True

    def iterdir(path: Path):
        if later_failure_site == "iterdir" and path == probe_stage:
            raise later_failure
        return iter(())

    def exists(path: Path) -> bool:
        if later_failure_site == "exists" and path == probe_orphan:
            raise later_failure
        return False

    def remove_path(path: Path) -> None:
        removed_stages.append(path)
        if path == first_stage:
            raise primary

    def remove_marker(path: Path) -> None:
        removed_markers.append(path)

    def remove_owned(path: Path, _identity: os.stat_result) -> None:
        if path.name.endswith(".target"):
            remove_marker(path)
        else:
            remove_path(path)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", is_stale)
    monkeypatch.setattr(
        atomic,
        "_read_atomic_target_marker",
        lambda _path: atomic._atomic_target_identity(final_root),
    )
    monkeypatch.setattr(atomic, "remove_owned_scratch", remove_owned)

    with pytest.raises(primary_type) as raised:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert raised.value is primary
    assert later_stage in removed_stages
    if later_failure_site != "exists":
        assert probe_stage not in removed_stages
    if later_failure_site == "exists":
        assert later_orphan in removed_markers


@pytest.mark.parametrize("primary_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize("prefix", (".i-", ".k-"))
@pytest.mark.parametrize("later_failure_site", ("glob", "lstat", "ownership"))
def test_sqlite_stale_sweep_attempts_later_owned_candidates_after_failure(
    prefix: str,
    later_failure_site: str,
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / f"{prefix}first"
    probe = tmp_path / f"{prefix}probe"
    later = tmp_path / f"{prefix}later"
    primary = primary_type("first_failure")
    later_failure = RuntimeError("later_probe_failure")
    removed: list[Path] = []
    real_lstat = Path.lstat
    for scratch in (first, probe, later):
        scratch.mkdir()

    def glob(_path: Path, pattern: str):
        assert pattern == f"{prefix}*"
        if later_failure_site == "glob":
            return _FailureBetweenValues(first, later_failure, later)
        return iter((first, probe, later))

    def lstat(path: Path) -> os.stat_result:
        if later_failure_site == "lstat" and path == probe:
            raise later_failure
        return real_lstat(path)

    def is_stale(path: Path, *, prefix: str) -> bool:
        assert prefix
        if later_failure_site == "ownership" and path == probe:
            raise later_failure
        return True

    def remove_owned(path: Path, _identity: os.stat_result, **_kwargs: object) -> None:
        removed.append(path)
        if path == first:
            raise primary

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(scratch_owner, "scratch_owner_is_stale", is_stale)
    monkeypatch.setattr(scratch_owner, "remove_owned_scratch", remove_owned)

    with pytest.raises(primary_type) as raised:
        scratch_owner.cleanup_stale_owned_scratch(
            tmp_path,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            directory=True,
        )

    assert raised.value is primary
    assert removed == [first, later]


def test_atomic_stale_stage_uses_identity_captured_before_stale_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stage = final_root.with_name(f"{prefix}stale")
    stage.mkdir()
    stale_hits: list[Path] = []

    def replace_during_stale_probe(path: Path, *, prefix: str) -> bool:
        assert prefix
        stale_hits.append(path)
        shutil.rmtree(path)
        path.mkdir()
        return True

    monkeypatch.setattr(atomic, "scratch_owner_is_stale", replace_during_stale_probe)

    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert {"stale_hits": stale_hits, "replacement_survived": stage.is_dir()} == {
        "stale_hits": [stage],
        "replacement_survived": True,
    }


def test_atomic_stale_marker_uses_one_captured_identity_for_read_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stage = final_root.with_name(f"{prefix}stale")
    marker = Path(f"{stage}.target")
    target_identity = atomic._atomic_target_identity(final_root)
    marker.write_text(target_identity, encoding="utf-8")
    read_hits: list[Path] = []

    def replace_during_read(path: Path) -> str:
        read_hits.append(path)
        path.unlink()
        path.write_text(target_identity, encoding="utf-8")
        return target_identity

    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(atomic, "_read_atomic_target_marker", replace_during_read)

    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=target_identity,
        )

    assert {"read_hits": read_hits, "replacement_survived": marker.is_file()} == {
        "read_hits": [marker],
        "replacement_survived": True,
    }


@pytest.mark.parametrize("prefix", (".i-", ".k-"))
def test_sqlite_stale_sweep_uses_identity_captured_before_stale_probe(
    prefix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / f"{prefix}stale"
    scratch.mkdir()
    stale_hits: list[Path] = []

    def replace_during_stale_probe(path: Path, *, prefix: str) -> bool:
        assert prefix
        stale_hits.append(path)
        shutil.rmtree(path)
        path.mkdir()
        return True

    monkeypatch.setattr(
        scratch_owner, "scratch_owner_is_stale", replace_during_stale_probe
    )

    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        scratch_owner.cleanup_stale_owned_scratch(
            tmp_path,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            directory=True,
        )

    assert {"stale_hits": stale_hits, "replacement_survived": scratch.is_dir()} == {
        "stale_hits": [scratch],
        "replacement_survived": True,
    }


@pytest.mark.parametrize("failure_type", _STALE_SWEEP_FAILURE_TYPES)
def test_atomic_stale_marker_probe_failure_propagates_after_attempting_later_stage(
    failure_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    unverified = final_root.with_name(f"{prefix}unverified")
    later = final_root.with_name(f"{prefix}later")
    unverified_marker = Path(f"{unverified}.target")
    later_marker = Path(f"{later}.target")
    failure = failure_type("marker_probe_failure")
    removed: list[Path] = []
    for stage in (unverified, later):
        stage.mkdir()
    for marker in (unverified_marker, later_marker):
        marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter((unverified, later))
        return iter(())

    real_lstat = Path.lstat

    def lstat(path: Path) -> os.stat_result:
        if path == unverified_marker:
            raise failure
        return real_lstat(path)

    def remove_owned(path: Path, _identity: os.stat_result) -> None:
        if not path.name.endswith(".target"):
            removed.append(path)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "iterdir", lambda _path: iter(()))
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(
        atomic,
        "_read_atomic_target_marker",
        lambda _path: atomic._atomic_target_identity(final_root),
    )
    monkeypatch.setattr(atomic, "remove_owned_scratch", remove_owned)

    expected_type = ValueError if failure_type is OSError else failure_type
    with pytest.raises(expected_type) as raised:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    if failure_type is OSError:
        assert raised.value.__cause__ is failure
    else:
        assert raised.value is failure
    assert removed == [later]


@pytest.mark.parametrize("marker_failure_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize("earlier_failure", (False, True))
def test_atomic_stale_marker_read_failure_preserves_first_and_attempts_later_stage(
    marker_failure_type: type[BaseException],
    earlier_failure: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    first = final_root.with_name(f"{prefix}first")
    unverified = final_root.with_name(f"{prefix}unverified")
    later = final_root.with_name(f"{prefix}later")
    first_marker = Path(f"{first}.target")
    unverified_marker = Path(f"{unverified}.target")
    later_marker = Path(f"{later}.target")
    primary = OSError("first_removal_failure")
    marker_failure = marker_failure_type("marker_read_failure")
    attempts: list[Path] = []
    stages = (first, unverified, later) if earlier_failure else (unverified, later)
    for stage in (first, unverified, later):
        stage.mkdir()
    for marker in (first_marker, unverified_marker, later_marker):
        marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter(stages)
        return iter(())

    def read_marker(path: Path) -> str:
        if path == unverified_marker:
            raise marker_failure
        assert path in {first_marker, later_marker}
        return atomic._atomic_target_identity(final_root)

    def remove_path(path: Path) -> None:
        attempts.append(path)
        if earlier_failure and path == first:
            raise primary

    def remove_owned(path: Path, _identity: os.stat_result) -> None:
        if not path.name.endswith(".target"):
            remove_path(path)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "iterdir", lambda _path: iter(()))
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(atomic, "_read_atomic_target_marker", read_marker)
    monkeypatch.setattr(atomic, "remove_owned_scratch", remove_owned)

    expected = primary if earlier_failure else marker_failure
    observed: BaseException | None = None
    try:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            parent_identity=final_root.parent.lstat(),
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )
    except BaseException as raised:
        observed = raised

    assert observed is expected
    assert unverified not in attempts
    assert later in attempts


@pytest.mark.parametrize("failure_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize("prefix", (".i-", ".k-"))
def test_sqlite_stale_ownership_probe_failure_propagates_after_later_cleanup(
    prefix: str,
    failure_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unverified = tmp_path / f"{prefix}unverified"
    later = tmp_path / f"{prefix}later"
    failure = failure_type("ownership_probe_failure")
    removed: list[Path] = []
    for scratch in (unverified, later):
        scratch.mkdir()

    def is_stale(path: Path, *, prefix: str) -> bool:
        assert prefix
        if path == unverified:
            raise failure
        return True

    monkeypatch.setattr(Path, "glob", lambda *_a, **_k: iter((unverified, later)))
    monkeypatch.setattr(scratch_owner, "scratch_owner_is_stale", is_stale)
    monkeypatch.setattr(
        scratch_owner,
        "remove_owned_scratch",
        lambda path, _identity, **_kwargs: removed.append(path),
    )

    with pytest.raises(failure_type) as raised:
        scratch_owner.cleanup_stale_owned_scratch(
            tmp_path,
            parent_identity=tmp_path.lstat(),
            prefix=prefix,
            directory=True,
        )

    assert raised.value is failure
    assert removed == [later]


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_bounded_manifest_read_preserves_body_primary_over_handle_close(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_id":"qrp-test"}', encoding="utf-8")
    metadata = path.stat()
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )
    monkeypatch.setattr(manifests.os, "fstat", lambda _fd: metadata)

    with pytest.raises(primary_type, match="body_primary"):
        manifests.read_bounded_qrp_json(path)


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_manifest_replace_preserves_body_primary_over_handle_close(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        scratch_owner,
        "open_owned_scratch_file",
        lambda *_args, **_kwargs: (
            _BodyAndCloseFailingHandle(primary),
            SimpleNamespace(st_dev=1, st_ino=2),
        ),
    )

    with pytest.raises(primary_type, match="body_primary"):
        manifests.replace_qrp_json(
            tmp_path / "manifest.json",
            {"schema_id": "qrp-test"},
        )


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
@pytest.mark.parametrize(
    "operation_name",
    ("text-write", "marker-write", "marker-read"),
)
def test_atomic_file_handles_preserve_body_primary_over_close(
    operation_name: str,
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker.target"
    if operation_name == "marker-read":
        marker.write_text("target", encoding="utf-8")
    primary = primary_type("body_primary")
    if operation_name == "marker-read":
        monkeypatch.setattr(
            Path,
            "open",
            lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
        )
    else:
        monkeypatch.setattr(
            scratch_owner,
            "open_owned_scratch_file",
            lambda *_args, **_kwargs: (
                _BodyAndCloseFailingHandle(primary),
                SimpleNamespace(st_dev=1, st_ino=2),
            ),
        )

    if operation_name == "text-write":

        def operation() -> object:
            return atomic.atomic_write_text(tmp_path / "output.txt", "value")
    elif operation_name == "marker-write":

        def operation() -> object:
            return atomic.atomic_write_directory(
                tmp_path / "output",
                lambda _path: None,
            )
    else:

        def operation() -> object:
            return atomic._read_atomic_target_marker(marker)

    with pytest.raises(primary_type, match="body_primary"):
        operation()


def test_compare_runtime_admission_failure_is_not_mapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    monkeypatch.setattr(
        comparison,
        "active_runtime_resource_policy",
        lambda: (_ for _ in ()).throw(ValueError("runtime_admission_failed")),
    )

    with pytest.raises(ValueError, match="runtime_admission_failed"):
        comparison.compare_datasets(left, right, resource_policy=None)


def test_compare_expected_storage_validation_still_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )

    assert (
        comparison.compare_datasets(left, right, resource_policy=policy).status
        == "INSPECTION_FAILED"
    )


def test_artifact_root_alias_is_rejected_by_unique_root_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("dataset")
    absolute_root = Path.cwd() / root
    monkeypatch.setattr(
        dataset_paths,
        "path_is_filesystem_alias",
        lambda candidate: candidate == absolute_root,
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        dataset_paths.require_artifact_root(root)


@pytest.mark.parametrize("position", ("root", "immediate", "deep"))
@pytest.mark.parametrize("alias_kind", ("symlink", "junction"))
def test_artifact_root_rejects_exact_or_ancestor_alias(
    tmp_path: Path,
    position: str,
    alias_kind: str,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    if alias_kind == "symlink":
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink_not_available: {exc}")
    else:
        if os.name != "nt" or not hasattr(Path, "is_junction"):
            pytest.skip("junction_capability_required")
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip("junction_not_available")
    root = (
        alias
        if position == "root"
        else alias / "artifact"
        if position == "immediate"
        else alias / "nested" / "artifact"
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        dataset_paths.require_artifact_root(root)


def test_artifact_root_ordinary_chain_preserves_candidate_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("relative") / "dataset"
    monkeypatch.setattr(
        dataset_paths,
        "path_is_filesystem_alias",
        lambda _candidate: False,
    )

    assert dataset_paths.require_artifact_root(root) == root


def test_dataset_file_identity_rejects_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"payload")
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        physical.dataset_file_identity(alias)


def test_dataset_file_identity_rejects_ancestor_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "part.bin").write_bytes(b"payload")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        physical.dataset_file_identity(alias / "part.bin")


def test_bounded_metadata_reader_rejects_ancestor_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text('{"schema_id":"test"}', encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="^dataset_metadata_file_invalid$"):
        manifests.read_bounded_qrp_json(alias / "manifest.json")


def test_write_dataset_requires_artifact_root_before_writer_or_data_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    monkeypatch.setattr(dataset_writing, "validate_dataset_spec", lambda _spec: None)
    monkeypatch.setattr(
        dataset_writing,
        "validate_dataset_artifact_root",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        dataset_writing,
        "require_artifact_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid_dataset_part_files")),
        raising=False,
    )
    monkeypatch.setattr(
        dataset_writing,
        "_write_dataset_under_policy",
        lambda *_args, **_kwargs: pytest.fail("writer_before_artifact_root_gate"),
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        dataset_writing.write_dataset(
            Path("dataset"),
            (_ for _ in ()),
            spec=object(),  # type: ignore[arg-type]
            resource_policy=policy,
        )


def test_compare_rechecks_artifact_roots_before_each_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = Path("left")
    right = Path("right")
    validations = {left: 0, right: 0}
    probes = {left: 0, right: 0}
    original_exists = Path.exists
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    def require_root(path: Path) -> Path:
        candidate = Path(path)
        validations[candidate] += 1
        return candidate

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate not in probes:
            return original_exists(path)
        probes[candidate] += 1
        assert validations[candidate] == probes[candidate]
        return True

    monkeypatch.setattr(
        comparison,
        "require_artifact_root",
        require_root,
        raising=False,
    )
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )

    result = comparison.compare_datasets(left, right, resource_policy=policy)

    assert result.status == "INSPECTION_FAILED"
    assert validations == {left: 2, right: 2}
    assert probes == {left: 2, right: 2}


def test_compare_maps_aliased_root_before_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = Path("left")
    right = Path("right")
    original_exists = Path.exists

    def reject_alias(path: Path) -> Path:
        if Path(path) == left:
            raise ValueError("invalid_dataset_part_files")
        return Path(path)

    monkeypatch.setattr(
        comparison,
        "require_artifact_root",
        reject_alias,
        raising=False,
    )

    def exists(path: Path) -> bool:
        if Path(path) in (left, right):
            raise AssertionError("dataset_exists_before_artifact_root_alias_gate")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)

    result = comparison.compare_datasets(left, right, resource_policy=None)

    assert result.status == "INSPECTION_FAILED"
    assert result.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("alias_side", ("left", "right"))
def test_compare_maps_ancestor_alias_before_public_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
    alias_side: str,
) -> None:
    left = Path("left-parent") / "left"
    right = Path("right-parent") / "right"
    aliased_parent = Path.cwd() / (left if alias_side == "left" else right).parent

    monkeypatch.setattr(
        dataset_paths,
        "path_is_filesystem_alias",
        lambda candidate: candidate == aliased_parent,
    )
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate == (left if alias_side == "left" else right):
            raise AssertionError("dataset_exists_before_ancestor_alias_gate")
        if candidate in (left, right):
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)

    result = comparison.compare_datasets(left, right, resource_policy=None)

    assert result.status == "INSPECTION_FAILED"
    assert result.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("alias_side", ("left", "right"))
def test_compare_internal_recheck_blocks_new_ancestor_alias_before_exists(
    monkeypatch: pytest.MonkeyPatch,
    alias_side: str,
) -> None:
    left = Path("left-parent") / "left"
    right = Path("right-parent") / "right"
    target = left if alias_side == "left" else right
    aliased_parent = Path.cwd() / target.parent
    gate_count = 0
    probe_count = {left: 0, right: 0}
    original_require = comparison.require_artifact_root
    original_exists = Path.exists

    def require_root(path: Path) -> Path:
        nonlocal gate_count
        gate_count += 1
        return original_require(path)

    def is_alias(candidate: Path) -> bool:
        internal_gate = gate_count >= (3 if alias_side == "left" else 4)
        return internal_gate and candidate == aliased_parent

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate not in probe_count:
            return original_exists(path)
        probe_count[candidate] += 1
        if candidate == target:
            assert probe_count[candidate] == 1
        return True

    monkeypatch.setattr(comparison, "require_artifact_root", require_root)
    monkeypatch.setattr(dataset_paths, "path_is_filesystem_alias", is_alias)
    monkeypatch.setattr(Path, "exists", exists)

    result = comparison.compare_datasets(
        left,
        right,
        resource_policy=capture_test_policy(
            available_memory_bytes=16 * 1024 * 1024,
            explicit_cap_bytes=1 * 1024 * 1024,
        ),
    )

    assert result.status == "INSPECTION_FAILED"
    assert probe_count[target] == 1


def test_shared_scratch_removes_only_exact_owned_paths(tmp_path: Path) -> None:
    owned_file = tmp_path / "owned-file"
    owned_file.write_bytes(b"x")
    owned_directory = tmp_path / "owned-directory"
    owned_directory.mkdir()
    (owned_directory / "part").write_bytes(b"x")

    scratch_owner.remove_owned_scratch(owned_file, owned_file.lstat())
    scratch_owner.remove_owned_scratch(owned_directory, owned_directory.lstat())

    assert not owned_file.exists()
    assert not owned_directory.exists()


def test_shared_scratch_rejects_mismatched_identity(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        scratch_owner.remove_owned_scratch(owned, other.lstat())

    assert owned.is_dir()
    assert other.is_dir()
