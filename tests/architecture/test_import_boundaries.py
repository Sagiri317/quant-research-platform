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
FORBIDDEN_REPORT_IMPORT_PREFIXES = (
    "quant_research_platform.compute",
    "quant_research_platform.research",
    "quant_research_platform.workflows",
    "quant_research_platform.interfaces",
)

PROFILE_A_SHARE_ROOT = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.markets.implementations.a_share",
)
PROFILE_A_SHARE_FACTORS = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.markets.implementations.a_share.factors",
)
PROFILE_A_SHARE_LABELS = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.markets.implementations.a_share.labels",
)
PROFILE_A_SHARE_PROCESSED = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.markets.implementations.a_share.processed",
)
PROFILE_CSMAR_CATALOG = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.sources.csmar.catalog",
)
PROFILE_CSMAR_INTERIM = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.sources.csmar.interim",
)
PROFILE_REPORT_CATALOG = (
    "src/quant_research_platform/catalog/profiles.py",
    "quant_research_platform.reports.catalog",
)
KNOWN_IMPORT_BOUNDARY_ALLOWLIST_REASONS = {
    PROFILE_A_SHARE_ROOT: "profiles.py composes A-share research profile owners.",
    PROFILE_A_SHARE_FACTORS: (
        "profiles.py composes the active A-share factor catalog."
    ),
    PROFILE_A_SHARE_LABELS: (
        "profiles.py composes the active A-share label catalog."
    ),
    PROFILE_A_SHARE_PROCESSED: (
        "profiles.py composes the active A-share processed catalog."
    ),
    PROFILE_CSMAR_CATALOG: "profiles.py composes the CSMAR source catalog.",
    PROFILE_CSMAR_INTERIM: (
        "profiles.py composes the source-owned CSMAR interim catalog."
    ),
    PROFILE_REPORT_CATALOG: (
        "profiles.py composes report owner definitions into the research profile."
    ),
}
KNOWN_IMPORT_BOUNDARY_ALLOWLIST = set(KNOWN_IMPORT_BOUNDARY_ALLOWLIST_REASONS)


def test_contracts_do_not_import_upper_layers() -> None:
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(PROJECT_SRC / "contracts")
        if module.startswith(FORBIDDEN_CONTRACT_IMPORT_PREFIXES)
    ]

    assert not hits, "contracts_import_boundary_violation: " + "; ".join(hits)


def test_reports_do_not_import_compute_research_workflows_or_interfaces() -> None:
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(PROJECT_SRC / "reports")
        if module.startswith(FORBIDDEN_REPORT_IMPORT_PREFIXES)
    ]

    assert not hits, "reports_import_boundary_violation: " + "; ".join(hits)


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


def test_known_cross_layer_import_allowlist_entries_have_reasons() -> None:
    missing_reason = sorted(
        entry
        for entry, reason in KNOWN_IMPORT_BOUNDARY_ALLOWLIST_REASONS.items()
        if not reason.strip()
    )

    assert not missing_reason, "missing_cross_layer_import_reason: " + repr(
        missing_reason
    )


def test_catalog_profiles_is_the_only_interface_composition_root() -> None:
    profiles_path = PROJECT_SRC / "catalog" / "profiles.py"
    services_path = PROJECT_SRC / "interfaces" / "services.py"

    assert profiles_path.exists(), "catalog_profiles_missing"
    assert {
        module
        for _path, _lineno, module in _iter_imports(profiles_path)
    }.issuperset(
        {
            "quant_research_platform.markets.implementations.a_share.processed",
            "quant_research_platform.sources.csmar.catalog",
            "quant_research_platform.sources.csmar.interim",
        }
    )
    profiles_source = profiles_path.read_text(encoding="utf-8")
    assert "research_profile" in profiles_source
    assert "local_a_share_csmar_catalogs" not in profiles_source
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
