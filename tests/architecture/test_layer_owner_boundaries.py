from __future__ import annotations

import ast
import importlib
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import pytest

from tests.helpers.import_graph import (
    ImportEdge,
    _import_edges_from_trees,
    import_edges_under,
    is_ids_contract_import,
)

SOURCE_ROOT = Path("src/quant_research_platform")
PACKAGE_PREFIX = "quant_research_platform"
LAYER_BOUNDARIES = {
    "contracts": (
        "quant_research_platform.artifacts",
        "quant_research_platform.catalog",
        "quant_research_platform.profiles",
        "quant_research_platform.research",
        "quant_research_platform.compute",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets",
        "quant_research_platform.reports",
        "quant_research_platform.runtime",
        "quant_research_platform.sources",
        "quant_research_platform.storage",
        "quant_research_platform.workflows",
    ),
    "sources": (
        "quant_research_platform.artifacts",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets",
        "quant_research_platform.reports",
        "quant_research_platform.research",
        "quant_research_platform.workflows",
    ),
    "policy": (
        "quant_research_platform.artifacts",
        "quant_research_platform.catalog",
        "quant_research_platform.compute",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets",
        "quant_research_platform.reports",
        "quant_research_platform.research",
        "quant_research_platform.runtime",
        "quant_research_platform.sources",
        "quant_research_platform.storage",
        "quant_research_platform.workflows",
    ),
    "catalog": (
        "quant_research_platform.markets.implementations",
        "quant_research_platform.reports",
        "quant_research_platform.sources",
    ),
    "runtime": ("quant_research_platform.reports",),
    # Direct operation owners may compose science, native readers and writers.
    "research": ("quant_research_platform.reports", "quant_research_platform.sources"),
    "reports": ("quant_research_platform.research", "quant_research_platform.sources"),
    "compute": (
        "quant_research_platform.reports",
        "quant_research_platform.artifacts",
        "quant_research_platform.workflows",
        "quant_research_platform.interfaces",
    ),
}


def test_active_source_modules_import_without_side_effect_failures() -> None:
    failures: list[str] = []
    for module_name in _active_module_names():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - reports every failed import.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []


def test_owner_layer_import_boundaries_are_enforced(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    violations: list[str] = []
    for root, forbidden_prefixes in LAYER_BOUNDARIES.items():
        for edge in import_edges_under(import_edges, SOURCE_ROOT / root):
            if any(edge.module.startswith(prefix) for prefix in forbidden_prefixes):
                violations.append(f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}")

    assert violations == []


@pytest.mark.parametrize(
    ("path", "source", "module"),
    (
        (
            SOURCE_ROOT / "contracts/example.py",
            "from quant_research_platform.storage import datasets\n",
            "quant_research_platform.storage.datasets",
        ),
        (
            SOURCE_ROOT / "runtime/example.py",
            "from quant_research_platform.reports import verification_summary\n",
            "quant_research_platform.reports.verification_summary",
        ),
        (
            SOURCE_ROOT / "catalog/example.py",
            "from quant_research_platform import reports\n",
            "quant_research_platform.reports",
        ),
        (
            SOURCE_ROOT / "research/example.py",
            "from ..sources import tardis\n",
            "quant_research_platform.sources.tardis",
        ),
        (
            SOURCE_ROOT / "reports/example.py",
            "from quant_research_platform.research import models\n",
            "quant_research_platform.research.models",
        ),
    ),
)
def test_owner_layer_boundaries_reject_real_dependency_directions(
    path: Path,
    source: str,
    module: str,
) -> None:
    edge = _import_edges_from_trees({path: ast.parse(source)})[0]
    root = path.relative_to(SOURCE_ROOT).parts[0]

    assert edge.module == module
    assert any(module.startswith(prefix) for prefix in LAYER_BOUNDARIES.get(root, ()))


def _active_module_names() -> tuple[str, ...]:
    names: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT)
        module_parts = (
            relative.parent.parts
            if relative.name == "__init__.py"
            else (*relative.parent.parts, relative.stem)
        )
        names.append(".".join((PACKAGE_PREFIX, *module_parts)))
    return tuple(dict.fromkeys(names))


@pytest.mark.parametrize("filename", ("__init__.py", "example.py"))
@pytest.mark.parametrize(
    "source,binding,forbidden",
    (
        ("from ..storage import datasets", "datasets", True),
        ("from . import time", "time", False),
    ),
)
def test_relative_import_edges_match_python_package_resolution(
    filename, source, binding, forbidden
):
    package = importlib.import_module("quant_research_platform.contracts")
    namespace = {"__package__": package.__package__}
    exec(source, namespace)
    actual_module = namespace[binding].__name__
    path = SOURCE_ROOT / "contracts" / filename
    (edge,) = _import_edges_from_trees({path: ast.parse(source)})
    assert edge.module == actual_module
    assert (
        any(edge.module.startswith(prefix) for prefix in LAYER_BOUNDARIES["contracts"])
        is forbidden
    )


@pytest.mark.parametrize("owner", ("artifacts", "catalog", "profiles", "research"))
def test_contract_direction_gate_rejects_each_high_level_owner(owner):
    edges = _import_edges_from_trees(
        {
            SOURCE_ROOT / "contracts" / "probe.py": ast.parse(
                f"from quant_research_platform.{owner} import value"
            )
        }
    )
    with pytest.raises(AssertionError):
        test_owner_layer_import_boundaries_are_enforced(tuple(edges))


@pytest.mark.parametrize(
    "module",
    (
        "quant_research_platform.markets.implementations.crypto.ids.VALUE",
        "quant_research_platform.markets.implementations.a_share.ids.VALUE",
        "quant_research_platform.sources.binance.ids.VALUE",
        "quant_research_platform.sources.choice.ids.VALUE",
        "quant_research_platform.sources.csmar.ids.VALUE",
    ),
)
def test_ids_contract_symbols_are_legal_but_owner_implementation_is_not(module):
    assert is_ids_contract_import(module)
    assert not is_ids_contract_import(module.replace(".ids.VALUE", ".replay.VALUE"))
    assert not is_ids_contract_import(module + ".member")


def _module_graph_order(modules, edges):
    graph = {module: set() for module in modules}
    for edge in edges:
        source = (
            ".".join(edge.path.with_suffix("").parts)
            .removeprefix("src.")
            .removesuffix(".__init__")
        )
        target = edge.module
        while target and target not in graph:
            target = target.rpartition(".")[0]
        if source in graph and target and source != target:
            graph[source].add(target)
    return tuple(TopologicalSorter(graph).static_order())


def test_active_modules_have_no_import_cycles(import_edges):
    _module_graph_order(_active_module_names(), import_edges)


@pytest.mark.parametrize("relative", (False, True))
def test_module_graph_rejects_nontrivial_cycles(relative):
    root = Path("src/quant_research_platform")
    trees = {
        root / "a.py": ast.parse(
            "from .b import value" if relative else "import quant_research_platform.b"
        ),
        root / "b.py": ast.parse(
            "from .a import value" if relative else "import quant_research_platform.a"
        ),
    }
    with pytest.raises(CycleError):
        _module_graph_order(
            ("quant_research_platform.a", "quant_research_platform.b"),
            _import_edges_from_trees(trees),
        )
