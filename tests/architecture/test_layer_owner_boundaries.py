from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from tests.helpers.import_graph import (
    ImportEdge,
    _import_edges_from_trees,
    import_edges_under,
)
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

SOURCE_ROOT = Path("src/quant_research_platform")
PACKAGE_PREFIX = "quant_research_platform"


def test_active_source_modules_import_without_side_effect_failures() -> None:
    failures: list[str] = []
    for module_name in _active_module_names():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - reports every failed import.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []


def test_owner_layer_import_boundaries_are_enforced_from_governance_contract(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    contract = load_maintenance_script_module("governance_contract")
    violations: list[str] = []
    for rule in contract.layer_boundary_import_rules():
        for edge in import_edges_under(import_edges, SOURCE_ROOT / rule["root"]):
            if any(
                edge.module.startswith(prefix)
                for prefix in rule["forbidden_prefixes"]
            ):
                violations.append(
                    f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}"
                )

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
            "from ..markets.implementations import a_share\n",
            "quant_research_platform.markets.implementations.a_share",
        ),
        (
            SOURCE_ROOT / "reports/example.py",
            "from quant_research_platform.storage import layout\n",
            "quant_research_platform.storage.layout",
        ),
    ),
)
def test_owner_layer_boundaries_reject_real_dependency_directions(
    path: Path,
    source: str,
    module: str,
) -> None:
    edge = _import_edges_from_trees({path: ast.parse(source)})[0]
    rules = {
        rule["root"]: rule["forbidden_prefixes"]
        for rule in load_maintenance_script_module(
            "governance_contract"
        ).layer_boundary_import_rules()
    }
    root = path.relative_to(SOURCE_ROOT).parts[0]

    assert edge.module == module
    assert any(module.startswith(prefix) for prefix in rules.get(root, ()))


def test_status_aggregation_rejects_pass_with_not_run_child() -> None:
    from quant_research_platform.policy.check_status import aggregate_check_status

    assert aggregate_check_status(["PASS", "NOT_RUN"]) == "FAIL"


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
