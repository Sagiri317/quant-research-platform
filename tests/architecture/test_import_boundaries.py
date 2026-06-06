from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")

FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "quant_research_platform.compute",
    "quant_research_platform.runtime",
    "quant_research_platform.storage",
    "quant_research_platform.markets",
    "quant_research_platform.sources",
    "quant_research_platform.interfaces",
    "quant_research_platform.agents",
    "quant_research_platform.workflows",
    "quant_research_platform.reports",
)

# Batch 2: catalog/defaults.py is the platform composition root.
DEFAULTS_A_SHARE_CATALOG_FACTORIES = (
    "src/quant_research_platform/catalog/defaults.py",
    "quant_research_platform.markets.implementations.a_share.catalog",
)
# Batch 2: catalog/defaults.py may use concrete market ids.
DEFAULTS_A_SHARE_IDS = (
    "src/quant_research_platform/catalog/defaults.py",
    "quant_research_platform.markets.implementations.a_share.ids",
)
# Batch 2: catalog/defaults.py may compose default source catalogs.
DEFAULTS_CSMAR_CATALOG = (
    "src/quant_research_platform/catalog/defaults.py",
    "quant_research_platform.sources.csmar.catalog",
)
DEFAULTS_A_SHARE_CSMAR_BINDING = (
    "src/quant_research_platform/catalog/defaults.py",
    "quant_research_platform.markets.implementations.a_share.source_bindings",
)
# Batch 3: catalog/defaults.py may compose report owner definitions.
DEFAULTS_REPORT_CATALOG = (
    "src/quant_research_platform/catalog/defaults.py",
    "quant_research_platform.reports.catalog",
)
KNOWN_IMPORT_BOUNDARY_ALLOWLIST = {
    DEFAULTS_A_SHARE_CATALOG_FACTORIES,
    DEFAULTS_A_SHARE_IDS,
    DEFAULTS_CSMAR_CATALOG,
    DEFAULTS_A_SHARE_CSMAR_BINDING,
    DEFAULTS_REPORT_CATALOG,
}


def test_contracts_do_not_import_upper_layers() -> None:
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(PROJECT_SRC / "contracts")
        if module.startswith(FORBIDDEN_CONTRACT_IMPORT_PREFIXES)
    ]

    assert not hits, "contracts_import_boundary_violation: " + "; ".join(hits)


def test_known_cross_layer_imports_are_explicitly_allowlisted() -> None:
    found = {
        (path.as_posix(), module)
        for path, _lineno, module in _iter_imports(PROJECT_SRC)
        if _is_selected_boundary(path, module)
    }
    unclassified = sorted(found - KNOWN_IMPORT_BOUNDARY_ALLOWLIST)
    stale = sorted(KNOWN_IMPORT_BOUNDARY_ALLOWLIST - found)

    assert not unclassified, "unclassified_cross_layer_import: " + repr(unclassified)
    assert not stale, "stale_cross_layer_import_allowlist_entry: " + repr(stale)


def test_catalog_defaults_is_the_only_interface_composition_root() -> None:
    defaults_path = PROJECT_SRC / "catalog" / "defaults.py"
    services_path = PROJECT_SRC / "interfaces" / "services.py"

    assert defaults_path.exists(), "catalog_defaults_missing"
    assert {
        module
        for _path, _lineno, module in _iter_imports(defaults_path)
    }.issuperset(
        {
            "quant_research_platform.markets.implementations.a_share.catalog",
            "quant_research_platform.markets.implementations.a_share.ids",
            "quant_research_platform.markets.implementations.a_share.source_bindings",
            "quant_research_platform.sources.csmar.catalog",
        }
    )
    assert "default_catalogs" in defaults_path.read_text(encoding="utf-8")
    interface_market_imports = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(PROJECT_SRC / "interfaces")
        if module.startswith("quant_research_platform.markets.implementations")
    ]

    assert services_path.exists()
    assert not interface_market_imports, (
        "interfaces_market_implementation_import: "
        + "; ".join(interface_market_imports)
    )


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    imports: list[tuple[Path, int, str]] = []
    paths = (root,) if root.is_file() else tuple(sorted(root.rglob("*.py")))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)


def _is_selected_boundary(path: Path, module: str) -> bool:
    path_text = path.as_posix()
    if path_text.startswith("src/quant_research_platform/contracts/"):
        return module.startswith(FORBIDDEN_CONTRACT_IMPORT_PREFIXES)
    if path_text.startswith("src/quant_research_platform/catalog/"):
        return module.startswith(
            (
                "quant_research_platform.markets.implementations",
                "quant_research_platform.reports",
                "quant_research_platform.sources",
            )
        )
    if path_text.startswith("src/quant_research_platform/interfaces/"):
        return module.startswith("quant_research_platform.markets.implementations")
    if path_text.startswith("src/quant_research_platform/reports/"):
        return module.startswith("quant_research_platform.markets.implementations")
    if path_text.startswith("src/quant_research_platform/sources/"):
        return module.startswith("quant_research_platform.markets.implementations")
    if path_text.startswith("src/quant_research_platform/agents/"):
        return module.startswith("quant_research_platform.markets.implementations")
    return False
