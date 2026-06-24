from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")


def test_catalog_does_not_import_concrete_sources_markets_or_materializers() -> None:
    hits = [
        f"{path.as_posix()}:{lineno}:{module}"
        for path, lineno, module in _iter_imports(PROJECT_SRC / "catalog")
        if module.startswith(
            (
                "quant_research_platform.sources",
                "quant_research_platform.markets.implementations",
            )
        )
        or "materializer" in module
    ]

    assert hits == []


def test_profiles_research_is_the_composition_root() -> None:
    profile = PROJECT_SRC / "profiles" / "research.py"
    assert profile.is_file()

    source = profile.read_text(encoding="utf-8")
    assert "ResearchProfile" in source
    assert "def research_profile(" in source
    assert "def research_profile_id(" in source
    assert "quant_research_platform.markets.implementations" in source
    assert "materializer" in source
    assert not (PROJECT_SRC / "catalog" / "profiles.py").exists()


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    paths = (root,) if root.is_file() else tuple(sorted(root.rglob("*.py")))
    imports: list[tuple[Path, int, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)
