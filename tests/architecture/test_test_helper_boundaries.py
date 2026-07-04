from __future__ import annotations

import ast
from pathlib import Path


def test_correctness_package_helper_was_removed() -> None:
    assert not Path("tests/correctness/_helpers.py").exists()


def test_research_tests_do_not_import_correctness_helpers() -> None:
    findings = _imports_matching(
        Path("tests/research"),
        {"tests.correctness._helpers"},
    )

    assert findings == []


def test_tests_do_not_cross_import_owner_local_helpers() -> None:
    findings: list[str] = []
    for path in Path("tests").rglob("test_*.py"):
        owner = _test_owner(path)
        for imported in _imported_modules(path):
            parts = imported.split(".")
            if len(parts) >= 3 and parts[0] == "tests" and parts[2] == "_helpers":
                imported_owner = parts[1]
                if imported_owner != owner:
                    findings.append(f"{path.as_posix()}:{imported}")

    assert findings == []


def test_shared_helpers_have_clear_owner_file_names() -> None:
    unclear = {
        "common.py",
        "utils.py",
        "new_helpers.py",
        "_helpers.py",
        "correctness_data.py",
    }

    assert sorted(path.name for path in Path("tests/helpers").glob("*.py")) != []
    unclear_helpers = sorted(
        path.name for path in Path("tests/helpers").glob("*.py") if path.name in unclear
    )
    assert unclear_helpers == []


def test_price_volume_reference_helper_has_only_price_volume_owner_symbols() -> None:
    denied_names = {
        "assert_forced_tiny_processed_chunks_match_default",
        "write_minimal_interim_inputs",
        "write_correctness_processed_inputs",
        "research_rows_for_dates",
        "correctness_processed_catalog",
        "correctness_research_dataset_catalog",
    }
    denied_prefix_suffix = (
        ("correctness_", ""),
        ("minimal_", "_frame"),
    )
    source = Path("tests/helpers/price_volume_reference.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    exported = defined | assigned
    denied = sorted(
        name
        for name in exported
        if name in denied_names
        or any(
            name.startswith(prefix) and (not suffix or name.endswith(suffix))
            for prefix, suffix in denied_prefix_suffix
        )
    )

    assert denied == []


def test_keyed_frame_diff_does_not_import_price_volume_reference() -> None:
    imported = _imported_modules(Path("tests/helpers/keyed_frame_diff.py"))

    assert "tests.helpers.price_volume_reference" not in imported


def test_all_tests_python_files_do_not_cross_import_owner_local_helpers() -> None:
    findings: list[str] = []
    for path in Path("tests").rglob("*.py"):
        owner = _test_owner(path)
        for imported in _imported_modules(path):
            parts = imported.split(".")
            if len(parts) >= 3 and parts[0] == "tests" and parts[2] == "_helpers":
                imported_owner = parts[1]
                if imported_owner != owner:
                    findings.append(f"{path.as_posix()}:{imported}")

    assert findings == []


def test_retired_terms_no_longer_lists_correctness_helper_as_shared_root() -> None:
    source = Path("tests/architecture/test_retired_terms.py").read_text(
        encoding="utf-8"
    )

    assert "tests/correctness/_helpers.py" not in source


def _imports_matching(root: Path, denied: set[str]) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("test_*.py"):
        for imported in _imported_modules(path):
            if imported in denied:
                findings.append(f"{path.as_posix()}:{imported}")
    return findings


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and path.parent.name:
                module = f"tests.{path.parent.name}.{module}".rstrip(".")
            imports.add(module)
    return imports


def _test_owner(path: Path) -> str:
    try:
        return path.relative_to("tests").parts[0]
    except ValueError:
        return ""
