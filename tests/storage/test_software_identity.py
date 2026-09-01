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


def test_identity_rejects_file_or_root_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    module_path = package_root / "module.py"
    module_path.write_bytes(b"module")
    complete = {distribution: "1.0" for distribution in NUMERICAL_MODEL_DISTRIBUTIONS}
    real_open = Path.open

    def changing_open(path: Path, *args: object, **kwargs: object):  # noqa: ANN202
        if path == module_path:
            with real_open(module_path, "wb") as handle:
                handle.write(b"changed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changing_open)
    with pytest.raises(ValueError, match="production_package_identity_changed"):
        production_software_identity(
            package_root=package_root,
            dependency_versions=complete,
        )

    monkeypatch.setattr(Path, "open", real_open)

    def changing_root_open(path: Path, *args: object, **kwargs: object):  # noqa: ANN202
        if path == module_path:
            with real_open(package_root / "new.py", "wb") as handle:
                handle.write(b"new")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changing_root_open)
    with pytest.raises(ValueError, match="production_package_identity_changed"):
        production_software_identity(
            package_root=package_root,
            dependency_versions=complete,
        )
