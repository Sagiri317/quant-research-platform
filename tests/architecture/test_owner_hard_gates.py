from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers.import_graph import _import_edges_from_trees, import_edges_under

SOURCE_ROOT = Path("src/quant_research_platform")
WORKFLOW_ACTIONS = (
    "data_materialize",
    "dataset_build",
    "research_run",
    "report_build",
)


def test_public_actions_have_one_owner_in_their_responsibility_layer() -> None:
    actions = {
        **dict.fromkeys(WORKFLOW_ACTIONS, "workflows"),
        "status": "interfaces",
        "inspect": "interfaces",
    }
    owners = {action: [] for action in actions}
    for layer in ("workflows", "interfaces"):
        for path in (SOURCE_ROOT / layer).rglob("*.py"):
            if layer == "interfaces" and path.parent.name in {"api", "cli"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in owners
                ):
                    owners[node.name].append(layer)
    assert owners == {action: [layer] for action, layer in actions.items()}
    references = _canonical_references(
        ast.parse(
            (SOURCE_ROOT / "interfaces/cli/commands.py").read_text(encoding="utf-8")
        )
    )
    for action, layer in actions.items():
        assert any(
            reference.startswith(f"quant_research_platform.{layer}.")
            and reference.endswith("." + action)
            for reference in references
        )


def test_query_owner_has_no_workflow_or_source_traversal(import_edges):
    from tests.helpers.import_graph import import_edges_under

    for edge in import_edges_under(import_edges, SOURCE_ROOT / "interfaces/queries.py"):
        assert not edge.module.startswith(
            ("quant_research_platform.workflows", "quant_research_platform.sources")
        )
        assert not edge.module.rsplit(".", 1)[-1].startswith("write_")


def test_public_action_reference_check_accepts_import_aliases():
    expected = {
        "quant_research_platform.workflows.research.data_materialize",
        "quant_research_platform.interfaces.queries.status",
    }
    for source in (
        "from quant_research_platform.workflows import research\n"
        "from quant_research_platform.interfaces import queries\n"
        "research.data_materialize()\nqueries.status()",
        "from quant_research_platform.workflows.research "
        "import data_materialize as build\n"
        "from quant_research_platform.interfaces import queries as read\n"
        "build()\nread.status()",
    ):
        assert expected.issubset(_canonical_references(ast.parse(source)))


def test_research_and_profile_owners_have_no_provider_traversal(import_edges) -> None:
    from tests.helpers.import_graph import import_edges_under, is_ids_contract_import

    for edge in import_edges_under(import_edges, SOURCE_ROOT / "research"):
        if edge.module.startswith("quant_research_platform.markets.implementations"):
            assert is_ids_contract_import(edge.module)
        assert not edge.module.startswith("quant_research_platform.sources")
    for edge in import_edges_under(import_edges, SOURCE_ROOT / "profiles/research.py"):
        assert edge.module.rsplit(".", 1)[-1] != "write_dataset"
        assert not edge.module.startswith("pandas")


def test_profile_is_composition_only_and_research_has_one_resolver(
    import_edges,
) -> None:
    profile_path = SOURCE_ROOT / "profiles" / "research.py"
    profile_tree = ast.parse(
        profile_path.read_text(encoding="utf-8"),
        filename=profile_path.as_posix(),
    )
    forbidden_prefixes = (
        "quant_research_platform.storage",
        "quant_research_platform.runtime",
        "quant_research_platform.workflows",
    )
    imports = (edge.module for edge in import_edges_under(import_edges, profile_path))
    assert all(
        not module.startswith(forbidden)
        for module in imports
        for forbidden in forbidden_prefixes
    )
    assert {
        "StorageLayout",
        "InterimMaterializer",
        "ProcessedMaterializer",
    }.isdisjoint(_defined_symbols(profile_tree))

    resolver_defs = []
    for path in (SOURCE_ROOT / "workflows").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name in {
                "resolve_target_graph",
                "plan_targets",
            }:
                resolver_defs.append(node.name)
    assert sorted(resolver_defs) == ["plan_targets", "resolve_target_graph"]


def test_a_share_pit_owner_and_lifecycle_inputs_have_one_direction(
    import_edges,
) -> None:
    factor_path = (
        SOURCE_ROOT / "research" / "factors" / "fundamental" / "calculations.py"
    )
    factor_modules = (
        edge.module for edge in import_edges_under(import_edges, factor_path)
    )
    assert all(
        not module.startswith("quant_research_platform.markets")
        for module in factor_modules
    )

    producer_path = (
        SOURCE_ROOT
        / "markets"
        / "implementations"
        / "a_share"
        / "processed"
        / "producer.py"
    )
    producer_tree = ast.parse(producer_path.read_text(encoding="utf-8"))
    assert (
        "quant_research_platform.markets.implementations.a_share.fundamentals.construct_ttm_events"
        in _canonical_references(producer_tree)
    )

    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )
    from quant_research_platform.sources.choice.interim import first_disclosure_spec

    panel_sources = (
        catalog.processed_catalog().get("market_panel_1d").source_input_data_ids
    )
    # Industry announcements do not replace stock_master as listing truth.
    assert {"listing_status_changes", "stock_info_annual"}.isdisjoint(panel_sources)
    assert {"stock_master", "stock_info_changes"}.issubset(panel_sources)
    assert first_disclosure_spec().primary_key == (
        "asset_id",
        "report_period",
        "first_disclosure_date",
    )


def test_workflow_production_statuses_have_no_readiness_aliases() -> None:
    from quant_research_platform.contracts.targets import RunStatus, TargetStatus

    assert {item.value for item in RunStatus} == {"COMPLETE", "BLOCKED", "FAILED"}
    assert {item.value for item in TargetStatus} == {
        "REUSED",
        "BUILT",
        "BLOCKED",
        "FAILED",
    }
    for enum in (RunStatus, TargetStatus):
        assert set(enum.__members__) == {item.value for item in enum}
        for value in ("READY", "STALE", "undeclared"):
            with pytest.raises(ValueError):
                enum(value)


def test_pit_reference_check_accepts_owner_aliases_and_rejects_wrong_owner():
    owner = "quant_research_platform.markets.implementations.a_share.fundamentals"
    for source in (
        f"from {owner} import construct_ttm_events as construct\nconstruct()",
        "from quant_research_platform.markets.implementations.a_share "
        "import fundamentals as facts\nfacts.construct_ttm_events()",
        f"import {owner} as facts\nfacts.construct_ttm_events()",
    ):
        assert owner + ".construct_ttm_events" in _canonical_references(
            ast.parse(source)
        )
    source = "from wrong_owner import construct_ttm_events\nconstruct_ttm_events()"
    assert owner + ".construct_ttm_events" not in _canonical_references(
        ast.parse(source)
    )


def test_configured_model_algorithms_use_the_single_capability_owner() -> None:
    import json

    from quant_research_platform.contracts.models import _MODEL_ALGORITHM_FACTS

    root = Path(__file__).resolve().parents[2] / "configs" / "research"
    configured = {
        model["model_algorithm"]
        for path in root.rglob("*.json")
        for model in json.loads(path.read_text(encoding="utf-8"))["models"]
    }
    assert configured == set(_MODEL_ALGORITHM_FACTS)


def _defined_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.arg):
            symbols.add(node.arg)
        elif isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            symbols.add(node.arg)
    return symbols


def _canonical_references(tree):
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
    references = set()
    for node in ast.walk(tree):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in bindings:
            references.add(".".join((bindings[node.id], *reversed(parts))))
    return references


@pytest.mark.parametrize(
    "source",
    (
        "import quant_research_platform.storage.datasets",
        "from ..storage import datasets",
        "from quant_research_platform import storage",
    ),
)
def test_profile_gate_rejects_equivalent_storage_imports(source):
    path = SOURCE_ROOT / "profiles/research.py"
    edges = _import_edges_from_trees({path: ast.parse(source)})
    with pytest.raises(AssertionError):
        test_profile_is_composition_only_and_research_has_one_resolver(edges)
    legal = _import_edges_from_trees(
        {path: ast.parse("from ..contracts import targets")}
    )
    test_profile_is_composition_only_and_research_has_one_resolver(legal)
