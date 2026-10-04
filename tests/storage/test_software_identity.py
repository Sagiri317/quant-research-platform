from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.storage.software_identity import (
    NUMERICAL_MODEL_DISTRIBUTIONS,
    production_software_identity,
)


def _identity(root: Path, versions: dict[str, str] | None = None):  # noqa: ANN202
    return production_software_identity(
        package_root=root,
        dependency_versions=versions
        or {distribution: "1.0" for distribution in NUMERICAL_MODEL_DISTRIBUTIONS},
    )


def test_package_identity_is_order_independent_and_ignores_bytecode(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package"
    (package_root / "__pycache__").mkdir(parents=True)
    (package_root / "b.py").write_bytes(b"b")
    (package_root / "a.py").write_bytes(b"a")
    ignored = package_root / "__pycache__" / "ignored.pyc"
    ignored.write_bytes(b"first")

    first = _identity(package_root)
    assert set(first.payload) == {"package_content_hash", "dependency_versions"}
    assert len(first.payload["package_content_hash"]) == 64
    assert set(first.payload["dependency_versions"]) == set(
        NUMERICAL_MODEL_DISTRIBUTIONS
    )

    ignored.write_bytes(b"second")
    second = _identity(
        package_root,
        {
            distribution: "1.0"
            for distribution in reversed(NUMERICAL_MODEL_DISTRIBUTIONS)
        },
    )
    assert second.identity_hash == first.identity_hash
    assert second.payload == first.payload


def test_package_and_dependency_changes_invalidate_only_production_identity(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "module.py").write_bytes(b"before")
    outside_docs = tmp_path / "docs"
    outside_docs.mkdir()
    versions = {distribution: "1.0" for distribution in NUMERICAL_MODEL_DISTRIBUTIONS}

    first = _identity(package_root, versions)
    (outside_docs / "README.md").write_text("documentation", encoding="utf-8")
    assert _identity(package_root, versions).identity_hash == first.identity_hash

    (package_root / "module.py").write_bytes(b"after")
    changed_package = _identity(package_root, versions)
    assert changed_package.identity_hash != first.identity_hash

    (package_root / "module.py").write_bytes(b"before")
    changed_dependency = _identity(
        package_root,
        {
            **versions,
            "numpy": "2.0",
        },
    )
    assert changed_dependency.identity_hash != first.identity_hash


def test_identity_rejects_incomplete_or_extra_dependency_versions(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "module.py").write_bytes(b"module")
    complete = {distribution: "1.0" for distribution in NUMERICAL_MODEL_DISTRIBUTIONS}

    with pytest.raises(ValueError, match="software_dependency_versions_invalid"):
        production_software_identity(
            package_root=package_root,
            dependency_versions={
                key: value for key, value in complete.items() if key != "numpy"
            },
        )
    with pytest.raises(ValueError, match="software_dependency_versions_invalid"):
        production_software_identity(
            package_root=package_root,
            dependency_versions={**complete, "other": "1.0"},
        )


def test_identity_rejects_empty_package_and_inconsistent_typed_identity(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    complete = {distribution: "1.0" for distribution in NUMERICAL_MODEL_DISTRIBUTIONS}
    with pytest.raises(ValueError, match="production_package_empty"):
        production_software_identity(
            package_root=empty,
            dependency_versions=complete,
        )

    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "module.py").write_bytes(b"module")
    identity = production_software_identity(
        package_root=package_root,
        dependency_versions=complete,
    )
    detached = identity.payload
    detached["dependency_versions"]["numpy"] = "2.0"
    assert identity.payload["dependency_versions"]["numpy"] == "1.0"
    with pytest.raises(ValueError, match="software_identity_hash_mismatch"):
        type(identity)(identity.payload, "0" * 64)


@pytest.mark.parametrize("fixed_directory_identity", (False, True))
def test_identity_rejects_file_or_root_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_directory_identity: bool,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    module_path = package_root / "module.py"
    module_path.write_bytes(b"module")
    complete = {distribution: "1.0" for distribution in NUMERICAL_MODEL_DISTRIBUTIONS}
    from quant_research_platform.storage import software_identity

    real_open = software_identity.open_filesystem_file
    if fixed_directory_identity:
        stable = software_identity._directory_identity(package_root)
        monkeypatch.setattr(
            software_identity, "_directory_identity", lambda _path: stable
        )

    def changing_open(path: Path, *args: object, **kwargs: object):  # noqa: ANN202
        if path == module_path:
            with real_open(module_path, "wb") as handle:
                handle.write(b"changed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(software_identity, "open_filesystem_file", changing_open)
    with pytest.raises(ValueError, match="production_package_identity_changed"):
        production_software_identity(
            package_root=package_root,
            dependency_versions=complete,
        )

    monkeypatch.setattr(software_identity, "open_filesystem_file", real_open)

    def changing_root_open(path: Path, *args: object, **kwargs: object):  # noqa: ANN202
        if path == module_path:
            with real_open(package_root / "new.py", "wb") as handle:
                handle.write(b"new")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(software_identity, "open_filesystem_file", changing_root_open)
    with pytest.raises(ValueError, match="production_package_identity_changed"):
        production_software_identity(
            package_root=package_root,
            dependency_versions=complete,
        )


def test_package_inventory_and_read_support_native_long_paths(tmp_path: Path) -> None:
    from quant_research_platform.storage.identity import (
        native_filesystem_path,
        open_filesystem_file,
    )

    short = tmp_path / "short"
    long = tmp_path / ("long" * 35) / ("nested" * 20)
    for root in (short, long):
        Path(native_filesystem_path(root)).mkdir(parents=True)
        with open_filesystem_file(root / "module.py", "wb") as handle:
            handle.write(b"synthetic = 1\n")
    assert len(str(long / "module.py")) > 260
    assert _identity(short).identity_hash == _identity(long).identity_hash


@pytest.mark.parametrize(
    ("boundary", "failure_type"),
    [
        (boundary, failure_type)
        for boundary in (
            "directory_stat",
            "file_stat",
            "inventory",
            "open",
            "read",
            "close",
        )
        for failure_type in (
            (OSError, PermissionError)
            if boundary in {"read", "close"}
            else (FileNotFoundError, NotADirectoryError, OSError, PermissionError)
        )
    ],
)
def test_software_identity_preserves_system_io(
    tmp_path, monkeypatch, boundary, failure_type
):
    import io

    from quant_research_platform.storage import software_identity as owner

    package = tmp_path / "package"
    package.mkdir()
    member = package / "module.py"
    member.write_bytes(b"synthetic = 1\n")
    expected = _identity(package)
    primary = failure_type("synthetic_package_failure")
    native_stat = owner.filesystem_stat
    native_directory_identity = owner._directory_identity
    native_scan = owner.filesystem_scandir
    native_open = owner.open_filesystem_file
    handles = []

    def stat(path, *args, **kwargs):
        target = package if boundary == "directory_stat" else member
        if not isinstance(path, int) and Path(path) == target:
            raise primary
        return native_stat(path, *args, **kwargs)

    def scan(path):
        if not isinstance(path, int) and Path(path) == package:
            raise primary
        return native_scan(path)

    class ObservedFile(io.BufferedReader):
        def read(self, *args):
            if boundary == "read":
                raise primary
            return super().read(*args)

        def close(self):
            was_open = not self.closed
            super().close()
            if was_open and boundary == "close":
                raise primary

    def open_member(path):
        if Path(path) != member:
            return native_open(path)
        if boundary == "open":
            raise primary
        handle = ObservedFile(io.FileIO(path, "r"))
        handles.append(handle)
        return handle

    with monkeypatch.context() as patch:
        if boundary == "directory_stat":

            def directory_identity(path):
                with monkeypatch.context() as scoped:
                    scoped.setattr(owner, "filesystem_stat", stat)
                    return native_directory_identity(path)

            patch.setattr(owner, "_directory_identity", directory_identity)
        elif boundary == "file_stat":
            patch.setattr(owner, "filesystem_stat", stat)
        elif boundary == "inventory":
            patch.setattr(owner, "filesystem_scandir", scan)
        else:
            patch.setattr(owner, "open_filesystem_file", open_member)
        missing = failure_type in (FileNotFoundError, NotADirectoryError)
        with pytest.raises(ValueError if missing else failure_type) as caught:
            _identity(package)
        if missing:
            assert str(caught.value) == (
                "production_package_root_invalid"
                if boundary == "directory_stat"
                else "production_package_enumeration_failed"
                if boundary == "inventory"
                else "production_package_file_unreadable"
            )
            assert caught.value.__cause__ is primary
        else:
            assert caught.value is primary
    assert all(handle.closed for handle in handles)
    assert member.read_bytes() == b"synthetic = 1\n"
    assert _identity(package) == expected


@pytest.mark.parametrize("boundary", ("file", "directory"))
@pytest.mark.parametrize("primary_type", (None, OSError, KeyboardInterrupt))
def test_software_identity_close_preserves_active_primary(
    tmp_path, monkeypatch, boundary, primary_type
):
    from quant_research_platform.storage import software_identity as owner

    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_bytes(b"synthetic = 1\n")
    primary = None if primary_type is None else primary_type("synthetic_primary")
    secondary = OSError("synthetic_close")
    closed = []
    native_scan = owner.filesystem_scandir

    native_open = owner.open_filesystem_file

    class Handle:
        def __init__(self):
            self.handle = native_open(package / "module.py")

        def fileno(self):
            return self.handle.fileno()

        def read(self, _size):
            if primary is not None:
                raise primary
            return self.handle.read(_size)

        def __iter__(self):
            if primary is not None:
                raise primary
            with native_scan(package) as entries:
                yield from entries

        def close(self):
            self.handle.close()
            closed.append(True)
            raise secondary

    monkeypatch.setattr(
        owner,
        "open_filesystem_file" if boundary == "file" else "filesystem_scandir",
        lambda _path: Handle(),
    )
    with pytest.raises(type(primary or secondary)) as caught:
        _identity(package)
    assert caught.value is (primary or secondary)
    assert closed == [True]
    if primary is not None:
        assert "scratch handle close also failed: OSError" in primary.__notes__


def test_package_hash_rejects_open_fd_from_swapped_ancestor(tmp_path, monkeypatch):
    from quant_research_platform.storage import software_identity as owner

    package, foreign, saved = (
        tmp_path / name for name in ("package", "foreign", "saved")
    )
    package.mkdir()
    foreign.mkdir()
    (package / "module.py").write_bytes(b"baseline")
    (foreign / "module.py").write_bytes(b"foreign!")
    baseline = _identity(package)
    native_open = owner.open_filesystem_file
    handles = []

    def swapped(path):
        package.rename(saved)
        foreign.rename(package)
        try:
            handle = native_open(path)
            handles.append(handle)
        finally:
            package.rename(foreign)
            saved.rename(package)
        return handle

    with monkeypatch.context() as scoped:
        scoped.setattr(owner, "open_filesystem_file", swapped)
        with pytest.raises(ValueError, match="production_package_identity_changed"):
            _identity(package)
    assert handles and all(handle.closed for handle in handles)
    assert _identity(package).identity_hash == baseline.identity_hash


def test_package_inventory_rejects_child_junction_on_windows(tmp_path):
    import os
    import subprocess

    if os.name != "nt":
        pytest.skip("requires a real Windows junction")
    package, external = tmp_path / "package", tmp_path / "external"
    package.mkdir()
    external.mkdir()
    (external / "module.py").write_bytes(b"synthetic")
    junction = package / "child"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        check=True,
        capture_output=True,
    )
    try:
        with pytest.raises(ValueError, match="production_package_alias"):
            _identity(package)
    finally:
        junction.rmdir()
    assert (external / "module.py").read_bytes() == b"synthetic"
