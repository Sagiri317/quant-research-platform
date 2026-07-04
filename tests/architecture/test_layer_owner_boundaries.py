from __future__ import annotations

import importlib
from pathlib import Path

from tests.helpers.import_graph import (
    ImportEdge,
    import_edges_for_roots,
    import_edges_under,
)
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

PROJECT_SRC = Path("src/quant_research_platform")
PACKAGE_ROOT = Path("src/quant_research_platform")
PACKAGE_PREFIX = "quant_research_platform"

def test_gitignore_excludes_generated_research_data() -> None:
    lines = {
        line.strip()
        for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert "data/research/" in lines


def test_active_source_modules_import_without_side_effect_failures() -> None:
    modules = tuple(_active_module_names())
    failures: list[str] = []

    assert (
        "quant_research_platform.markets.implementations.a_share.validation"
        in modules
    )
    assert (
        "quant_research_platform.markets.implementations.a_share.validation.research_panel"
        not in modules
    )

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - failure path reports all modules.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []


def _active_module_names() -> tuple[str, ...]:
    names: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.name == "__init__.py":
            module_parts = relative.parent.parts
        else:
            module_parts = (*relative.parent.parts, relative.stem)
        module_name = ".".join((PACKAGE_PREFIX, *module_parts))
        names.append(module_name)
    return tuple(dict.fromkeys(names))


def test_owner_layer_import_boundaries_are_enforced_from_governance_contract(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    contract = load_maintenance_script_module("governance_contract")
    hits: list[str] = []

    for rule in contract.layer_boundary_import_rules():
        root = PROJECT_SRC / rule["root"]
        for edge in import_edges_under(import_edges, root):
            if any(
                edge.module.startswith(prefix)
                for prefix in rule["forbidden_prefixes"]
            ):
                hits.append(f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}")

    assert hits == []


def test_storage_layout_does_not_own_run_output_or_evidence_roots() -> None:
    source = (PROJECT_SRC / "storage" / "layout.py").read_text(encoding="utf-8")

    assert "run_output" not in source
    assert "evidence_root" not in source
    assert "report_root" not in source


def test_catalog_does_not_import_concrete_sources_markets_or_materializers() -> None:
    hits = [
        f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}"
        for edge in import_edges_for_roots((PROJECT_SRC / "catalog",))
        if edge.module.startswith(
            (
                "quant_research_platform.sources",
                "quant_research_platform.markets.implementations",
            )
        )
        or "materializer" in edge.module
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


def test_policy_does_not_import_business_layers() -> None:
    policy_root = PROJECT_SRC / "policy"
    hits = [
        f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}"
        for edge in import_edges_for_roots((policy_root,))
        if edge.module.startswith("quant_research_platform.")
        and not edge.module.startswith("quant_research_platform.policy.")
    ]

    assert hits == []


def test_contracts_do_not_own_share_safety_or_identifier_regexes() -> None:
    hits: list[str] = []
    blocked_terms = (
        "local_only_path",
        "provider",
        "source marker",
        "stock_code",
        "stock-code",
        "share_safety",
    )
    for path in sorted((PROJECT_SRC / "contracts").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for term in blocked_terms:
            if term in lowered:
                hits.append(f"{path.as_posix()}:{term}")

    assert hits == []


def test_status_aggregation_rejects_top_level_pass_with_not_run_child() -> None:
    from quant_research_platform.policy.check_status import aggregate_check_status

    assert aggregate_check_status(["PASS", "NOT_RUN"]) == "FAIL"
