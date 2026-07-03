from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.helpers.import_graph import python_files_under

PROJECT_SRC = Path("src/quant_research_platform")
CATALOG_INIT = PROJECT_SRC / "catalog" / "__init__.py"
RESEARCH_PROFILE = PROJECT_SRC / "profiles" / "research.py"
SCAN_ROOTS = (Path("src"), Path("tests"))


def test_profiles_root_init_does_not_export_retired_entrypoints() -> None:
    source = CATALOG_INIT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CATALOG_INIT))
    all_names = _module_all_names(tree)

    assert "quant_research_platform.catalog.defaults" not in source
    assert "catalog.defaults" not in source
    assert "_DEFAULT_EXPORTS" not in source
    assert "__getattr__" not in {node.name for node in tree.body if _is_def(node)}
    assert "default_catalogs" not in all_names
    assert not [name for name in all_names if name.startswith("default_")]


def test_profiles_root_retired_entrypoints_not_imported_from_root() -> None:
    root_import_hits: list[str] = []
    for path in python_files_under(SCAN_ROOTS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "quant_research_platform.catalog":
                continue
            retired_names = [
                alias.name
                for alias in node.names
                if alias.name == "default_catalogs" or alias.name.startswith("default_")
            ]
            if retired_names:
                root_import_hits.append(f"{path}:{node.lineno}")

    assert not root_import_hits, (
        "profiles_composition_root_must_use_explicit_module: "
        + "; ".join(root_import_hits)
    )


def test_profiles_composition_root_remains_platform_composition_root() -> None:
    source = RESEARCH_PROFILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RESEARCH_PROFILE))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    all_names = _module_all_names(tree)

    assert (
        "quant_research_platform.markets.implementations.a_share.processed" in modules
    )
    assert "quant_research_platform.sources.csmar.catalog" in modules
    assert "quant_research_platform.sources.csmar.interim" in modules
    assert "quant_research_platform.artifacts.definitions" in modules
    assert "csmar_interim_catalog(" in source
    assert "source_bindings" not in source
    assert "provider_binding" not in source
    assert "research_profile" in all_names
    assert "ResearchProfile" in all_names
    assert "local_a_share_csmar_catalogs" not in all_names
    assert "CatalogBundle" not in all_names
    assert not [name for name in all_names if name.startswith("default_")]


def test_profiles_composition_root_does_not_call_retired_bootstrap_entrypoint() -> None:
    source = RESEARCH_PROFILE.read_text(encoding="utf-8")

    assert ".bootstrap_default(" not in source


def test_profiles_composition_root_delegates_concrete_definitions_to_owners() -> None:
    source = RESEARCH_PROFILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RESEARCH_PROFILE))
    forbidden_names = {
        "DataProductDefinition",
        "MarketDefinition",
        "ProviderBinding",
        "UniverseDefinition",
    }
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    instantiated_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    violations = sorted(forbidden_names & (imported_names | instantiated_names))

    assert not violations, (
        "profiles_must_delegate_concrete_definitions_to_owners: "
        + ", ".join(violations)
    )
    assert "ProviderBinding" not in instantiated_names


def test_a_share_research_profiles_do_not_import_csmar() -> None:
    path = PROJECT_SRC / "markets/implementations/a_share/research_profiles.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    imports = _imported_modules(tree)
    assert not [
        module
        for module in imports
        if module.startswith("quant_research_platform.sources.csmar")
    ]


def test_profiles_root_is_only_prod_module_composing_csmar_and_a_share() -> None:
    composers: list[str] = []
    for path in PROJECT_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imports = _imported_modules(tree)
        has_csmar = any(
            module.startswith("quant_research_platform.sources.csmar")
            for module in imports
        )
        has_a_share = any(
            module.startswith("quant_research_platform.markets.implementations.a_share")
            for module in imports
        )
        if has_csmar and has_a_share:
            composers.append(path.as_posix())

    assert composers == [
        "src/quant_research_platform/profiles/research.py",
        (
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/catalog.py"
        ),
        (
            "src/quant_research_platform/markets/implementations/a_share/"
            "verification/aggregates.py"
        ),
    ]


def test_a_share_scenario_specs_are_not_owned_by_profile_root() -> None:
    profile_source = RESEARCH_PROFILE.read_text(encoding="utf-8")
    scenario_source = (
        PROJECT_SRC / "markets/implementations/a_share/research_scenarios.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "EvaluationSpec(",
        "ModelSpec(",
        "PredictionSpec(",
        "build_prediction_catalog",
        "build_portfolio_catalog",
    )

    assert [token for token in forbidden[:3] if token in profile_source] == []
    assert all(token in scenario_source for token in forbidden)


def test_a_share_owner_does_not_define_registry_factories() -> None:
    forbidden = {
        "_".join(parts)
        for parts in (
            ("a", "share", "data", "product", "registry"),
            ("a", "share", "market", "registry"),
            ("a", "share", "universe", "catalog"),
        )
    }
    hits: list[str] = []
    for path in (PROJECT_SRC / "markets" / "implementations" / "a_share").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in forbidden:
                hits.append(f"{path}:{node.lineno}:{node.name}")

    assert not hits, "a_share_registry_factory_forbidden: " + "; ".join(hits)


def test_research_profile_does_not_own_a_share_scenario_parameters() -> None:
    source = RESEARCH_PROFILE.read_text(encoding="utf-8")

    forbidden_patterns = (
        r"_PROFILE_",
        r"DEFAULT_",
        r"20[0-9][0-9]-",
        r"\btrain_end\s*=",
        r"\btest_start\s*=",
        r"\bfrequency\s*=",
        r"\bexecution_lag_steps\s*=",
    )
    hits: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for pattern in forbidden_patterns:
            if re.search(pattern, line):
                hits.append(f"{RESEARCH_PROFILE}:{line_number}: {line.strip()}")

    assert hits == []
    assert "predictions=PredictionCatalog()" in source
    assert "portfolios=PortfolioCatalog()" in source


def test_a_share_scenarios_own_research_scenario_catalog_builders() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios,
    )

    for name in (
        "build_signal_specs",
        "build_model_specs",
        "build_prediction_specs",
        "build_evaluation_specs",
        "build_portfolio_specs",
        "build_model_catalog",
        "build_prediction_catalog",
        "build_evaluation_catalog",
        "build_portfolio_catalog",
        "default_evaluation_run_id",
        "default_prediction_smoke_id",
        "default_prediction_oos_period",
    ):
        assert hasattr(research_scenarios, name)


def test_a_share_scenario_builders_require_smoke_windows_and_frequency() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios,
    )
    from tests.helpers import frequency_ids

    try:
        research_scenarios.build_prediction_specs()
    except TypeError as exc:
        assert "train_end" in str(exc)
    else:  # pragma: no cover - failure branch for clear assertion message
        raise AssertionError("prediction specs must require explicit smoke dates")

    portfolio_specs = research_scenarios.build_portfolio_specs()
    assert portfolio_specs[0].rebalance_frequency == frequency_ids.DAILY_FREQUENCY_SLUG

    source = (
        PROJECT_SRC
        / "markets"
        / "implementations"
        / "a_share"
        / "research_scenarios.py"
    ).read_text(encoding="utf-8")
    for token in (
        "DEFAULT_TRAIN_END",
        "DEFAULT_TEST_START",
        "DEFAULT_EVALUATION_PERIOD_END",
        'train_end: str = "2014-06-30"',
        'test_start: str = "2014-07-01"',
        'frequency: str = "daily"',
    ):
        assert token not in source


def _module_all_names(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    names.append(item.value)
    return tuple(names)


def _is_def(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules
