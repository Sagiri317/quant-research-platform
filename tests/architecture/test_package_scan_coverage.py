from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")


def iter_source_files() -> tuple[Path, ...]:
    return tuple(sorted(PROJECT_SRC.rglob("*.py")))


def test_all_source_files_are_discoverable_and_parseable() -> None:
    files = iter_source_files()
    parse_failed: list[str] = []
    assert files, "source_scan_found_no_python_files"
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            parse_failed.append(f"{path}:{exc.lineno}:{exc.msg}")

    assert not parse_failed, "source_scan_parse_failed: " + "; ".join(parse_failed)


def test_top_level_packages_are_accounted_for() -> None:
    packages = {p.name for p in PROJECT_SRC.iterdir() if p.is_dir()}
    expected_minimum = {
        "agents",
        "catalog",
        "compute",
        "contracts",
        "interfaces",
        "markets",
        "reports",
        "research",
        "runtime",
        "sources",
        "storage",
        "workflows",
    }
    missing = expected_minimum - packages

    assert not missing, f"expected_top_level_packages_missing: {sorted(missing)}"
