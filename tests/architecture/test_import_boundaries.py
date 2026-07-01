from __future__ import annotations

import ast
import inspect
from pathlib import Path

from tests.helpers.import_graph import ImportEdge, import_edges_under

PROJECT_SRC = Path("src/quant_research_platform")
OWNER_LOCAL_INTERNAL_API_ROOTS = (
    PROJECT_SRC / "artifacts",
    PROJECT_SRC / "markets" / "implementations" / "a_share" / "processed",
    PROJECT_SRC / "research" / "models",
    PROJECT_SRC / "research" / "datasets",
    PROJECT_SRC / "research" / "evaluations",
    PROJECT_SRC / "research" / "predictions",
    PROJECT_SRC / "research" / "factors" / "price_volume",
    PROJECT_SRC / "compute",
    PROJECT_SRC / "storage",
)
RETIRED_OWNER_LOCAL_PRIVATE_SYMBOLS = {
    "_ensure_processed_dependencies",
    "_planned_processed_dependency_windows",
    "_merge_time_windows",
    "_available_trading_sessions",
    "_require_target_sessions",
    "_require_dataset_coverage",
    "_planned_session_windows",
    "_filter_output_sessions",
    "_processed_panel_key_coverage",
    "_estimate_panel_bytes_per_session",
    "_estimate_factor_bytes_per_session",
    "_estimate_label_bytes_per_session",
    "_select_factor_definitions_for_spec",
    "_select_label_definitions_for_spec",
    "_linear_regression_estimator",
    "_SingleFactorBaseline",
    "_RankEnsemblePredictor",
    "_ICWeightedRankEnsembleFactory",
    "_ICWeightedRankEnsemble",
    "_iter_research_dataset_chunks",
    "_evaluation_diagnostics",
    "_prediction_callable",
    "_prediction_model_input_frame",
    "_required_windows",
    "_lookback_window",
    "_windows_from_requested_factors",
    "_simulate_weighted_long_short_returns_and_cost_from_validated_panel",
}
OWNER_LOCAL_INTERNAL_PACKAGE_EXPORTS = {
    PROJECT_SRC
    / "markets"
    / "implementations"
    / "a_share"
    / "processed"
    / "__init__.py": {
        "ensure_processed_dependencies",
        "planned_processed_dependency_windows",
        "merge_time_windows",
        "available_trading_sessions",
        "require_target_sessions",
        "require_dataset_coverage",
        "planned_session_windows",
        "filter_output_sessions",
        "processed_panel_key_coverage",
        "estimate_panel_bytes_per_session",
        "estimate_factor_bytes_per_session",
        "estimate_label_bytes_per_session",
        "select_factor_definitions_for_spec",
        "select_label_definitions_for_spec",
    },
    PROJECT_SRC / "research" / "models" / "__init__.py": {
        "linear_regression_estimator",
        "SingleFactorBaseline",
        "RankEnsemblePredictor",
        "ICWeightedRankEnsembleFactory",
        "ICWeightedRankEnsemble",
    },
    PROJECT_SRC / "research" / "datasets" / "__init__.py": {
        "iter_research_dataset_chunks",
    },
    PROJECT_SRC / "research" / "predictions" / "__init__.py": {
        "prediction_callable",
        "prediction_model_input_frame",
    },
    PROJECT_SRC / "compute" / "__init__.py": {
        "simulate_weighted_long_short_returns_and_cost_from_validated_panel",
    },
}

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
    "quant_research_platform.storage",
    "quant_research_platform.workflows",
    "quant_research_platform.interfaces",
)
FORBIDDEN_RESEARCH_IMPORT_PREFIXES = (
    "quant_research_platform.markets.implementations",
    "quant_research_platform.sources",
)
FORBIDDEN_RUNTIME_IMPORT_PREFIXES = ("quant_research_platform.reports",)

PROFILE_A_SHARE_ROOT = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.markets.implementations.a_share",
)
PROFILE_A_SHARE_FACTORS = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.markets.implementations.a_share.factors",
)
PROFILE_A_SHARE_LABELS = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.markets.implementations.a_share.labels",
)
PROFILE_A_SHARE_PROCESSED = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.markets.implementations.a_share.processed",
)
PROFILE_CSMAR_CATALOG = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.sources.csmar.catalog",
)
PROFILE_CSMAR_INTERIM = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.sources.csmar.interim",
)
PROFILE_REPORT_CATALOG = (
    "src/quant_research_platform/profiles/research.py",
    "quant_research_platform.artifacts.definitions",
)
KNOWN_IMPORT_BOUNDARY_ALLOWLIST_REASONS = {
    PROFILE_A_SHARE_ROOT: "profiles.py composes A-share research profile owners.",
    PROFILE_A_SHARE_FACTORS: (
        "profiles.py composes the active A-share factor catalog."
    ),
    PROFILE_A_SHARE_LABELS: ("profiles.py composes the active A-share label catalog."),
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


def test_contracts_do_not_import_upper_layers(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    hits = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, PROJECT_SRC / "contracts")
        if edge.module.startswith(FORBIDDEN_CONTRACT_IMPORT_PREFIXES)
    ]

    assert not hits, "contracts_import_boundary_violation: " + "; ".join(hits)


def test_reports_do_not_import_compute_research_storage_workflows_or_interfaces(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    hits = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, PROJECT_SRC / "reports")
        if edge.module.startswith(FORBIDDEN_REPORT_IMPORT_PREFIXES)
    ]

    assert not hits, "reports_import_boundary_violation: " + "; ".join(hits)


def test_research_only_imports_generic_market_session_windows(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    hits = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, PROJECT_SRC / "research")
        if edge.module.startswith(FORBIDDEN_RESEARCH_IMPORT_PREFIXES)
    ]

    assert not hits, "research_import_boundary_violation: " + "; ".join(hits)


def test_runtime_does_not_import_reports(import_edges: tuple[ImportEdge, ...]) -> None:
    hits = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, PROJECT_SRC / "runtime")
        if edge.module.startswith(FORBIDDEN_RUNTIME_IMPORT_PREFIXES)
    ]

    assert not hits, "runtime_report_import_boundary_violation: " + "; ".join(hits)


def test_runtime_tests_do_not_import_reports(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    hits = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, Path("tests/runtime"))
        if edge.module.startswith(FORBIDDEN_RUNTIME_IMPORT_PREFIXES)
    ]

    assert not hits, "runtime_test_report_import_boundary_violation: " + "; ".join(hits)


def test_runtime_recorders_do_not_expose_artifact_writers() -> None:
    from quant_research_platform.runtime.execution_metrics import (
        ExecutionMetricsRecorder,
    )
    from quant_research_platform.runtime.profiling import StageProfiler

    assert not hasattr(StageProfiler, "write_artifacts")
    assert not hasattr(ExecutionMetricsRecorder, "write_json")
    assert not hasattr(ExecutionMetricsRecorder, "write_csv")
    source = inspect.getsource(
        __import__(
            "quant_research_platform.runtime.execution_metrics",
            fromlist=["unused"],
        )
    )
    assert "quant_research_platform.storage.atomic" not in source


def test_session_window_owner_is_unique(import_edges: tuple[ImportEdge, ...]) -> None:
    owner = PROJECT_SRC / "markets" / "session_windows.py"
    retired = PROJECT_SRC / "markets" / "calendars.py"

    assert owner.is_file()
    assert not retired.exists()
    imports = [
        f"{path}:{lineno}"
        for path, lineno, module in (
            (edge.path, edge.lineno, edge.module)
            for edge in import_edges_under(import_edges, PROJECT_SRC)
        )
        if module == "quant_research_platform.markets.calendars"
    ]
    assert not imports, "retired_markets_calendars_import: " + "; ".join(imports)


def test_price_volume_root_is_public_reexport_surface() -> None:
    retired_module = PROJECT_SRC / "research" / "factors" / "price_volume.py"
    package_root = PROJECT_SRC / "research" / "factors" / "price_volume"
    init_path = package_root / "__init__.py"

    assert not retired_module.exists()
    assert package_root.is_dir()
    assert init_path.is_file()

    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    local_definitions = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    assert not [
        node.name for node in tree.body if isinstance(node, local_definitions)
    ], "price_volume_root_defines_behavior"

    exports = _literal_all_exports(tree)
    assert exports, "price_volume_root_missing_all"
    private_exports = [name for name in exports if name.startswith("_")]
    assert not private_exports, "price_volume_root_private_exports: " + repr(
        private_exports
    )


def test_owner_local_internal_api_files_do_not_import_private_symbols() -> None:
    hits: list[str] = []
    for root in OWNER_LOCAL_INTERNAL_API_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        hits.append(f"{path}:{node.lineno}: {alias.name}")

    assert not hits, "owner_local_private_import: " + "; ".join(hits)


def test_retired_owner_local_private_symbols_are_not_cross_file_attributes() -> None:
    hits: list[str] = []
    for root in OWNER_LOCAL_INTERNAL_API_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in RETIRED_OWNER_LOCAL_PRIVATE_SYMBOLS
                ):
                    hits.append(f"{path}:{node.lineno}: {node.attr}")

    assert not hits, "retired_owner_local_private_attribute: " + "; ".join(hits)


def test_owner_local_internal_api_files_do_not_export_private_names() -> None:
    hits: list[str] = []
    for root in OWNER_LOCAL_INTERNAL_API_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name in _literal_all_exports(tree):
                if name.startswith("_"):
                    hits.append(f"{path}: __all__ exports {name}")

    assert not hits, "owner_local_private_all_export: " + "; ".join(hits)


def test_owner_local_internal_api_is_not_package_reexported() -> None:
    hits: list[str] = []
    for path, internal_names in OWNER_LOCAL_INTERNAL_PACKAGE_EXPORTS.items():
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exports = set(_literal_all_exports(tree))
        leaked = sorted(exports.intersection(internal_names))
        if leaked:
            hits.append(f"{path}: {', '.join(leaked)}")

    assert not hits, "owner_local_internal_api_reexported: " + "; ".join(hits)


def test_known_cross_layer_imports_are_explicitly_allowlisted(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    found = {
        (edge.path.as_posix(), edge.module)
        for edge in import_edges_under(import_edges, PROJECT_SRC)
        if _is_selected_boundary(edge.path, edge.module)
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


def test_profiles_research_is_the_only_interface_composition_root(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    profiles_path = PROJECT_SRC / "profiles" / "research.py"
    services_path = PROJECT_SRC / "interfaces" / "services.py"

    assert profiles_path.exists(), "catalog_profiles_missing"
    assert {
        edge.module for edge in import_edges_under(import_edges, profiles_path)
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
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, PROJECT_SRC / "interfaces")
        if edge.module.startswith("quant_research_platform.markets.implementations")
    ]

    assert services_path.exists()
    assert not interface_market_imports, (
        "interfaces_market_implementation_import: "
        + "; ".join(interface_market_imports)
    )


def _literal_all_exports(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        return tuple(value)
    return ()


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
    if path_text.startswith("src/quant_research_platform/profiles/"):
        return module.startswith(
            (
                "quant_research_platform.artifacts.definitions",
                "quant_research_platform.markets.implementations",
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
