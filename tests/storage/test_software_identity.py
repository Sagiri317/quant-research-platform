from __future__ import annotations

from pathlib import Path

from quant_research_platform.storage.software_identity import (
    production_software_identity,
)


def _identity(root: Path, versions: dict[str, str] | None = None):  # noqa: ANN202
    return production_software_identity(
        package_root=root,
        dependency_versions=versions or {"numpy": "2.3.5", "scipy": "1.18.0"},
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
    assert [item["path"] for item in first.payload["package_files"]] == [
        "a.py",
        "b.py",
    ]

    ignored.write_bytes(b"second")
    second = _identity(package_root, {"scipy": "1.18.0", "numpy": "2.3.5"})
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
    versions = {"numpy": "2.3.5", "scipy": "1.18.0"}

    first = _identity(package_root, versions)
    (outside_docs / "README.md").write_text("documentation", encoding="utf-8")
    assert _identity(package_root, versions).identity_hash == first.identity_hash

    (package_root / "module.py").write_bytes(b"after")
    changed_package = _identity(package_root, versions)
    assert changed_package.identity_hash != first.identity_hash

    (package_root / "module.py").write_bytes(b"before")
    changed_dependency = _identity(
        package_root,
        {"numpy": "2.3.6", "scipy": "1.18.0"},
    )
    assert changed_dependency.identity_hash != first.identity_hash
