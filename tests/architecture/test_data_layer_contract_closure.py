from __future__ import annotations

import ast
import re
from pathlib import Path

ACTIVE_SCAN_ROOTS = (
    Path("src"),
    Path("tests/architecture"),
    Path("tests/correctness"),
    Path("tests/research"),
    Path("tests/markets"),
    Path("tests/storage"),
)
PROJECT_SRC = Path("src/quant_research_platform")
NEGATIVE_GATE_FILES = {
    Path("tests/architecture/test_clean_rewrite_rg_gate.py"),
    Path("tests/architecture/test_data_layer_boundaries.py"),
}
# Only local/verification evidence fixture residue may name the source here.
LOCAL_PRIVATE_EVIDENCE_FIXTURE_SOURCE_LITERAL = (
    'LOCAL_PRIVATE_EVIDENCE_INTERIM_SOURCE_ID = "csmar"'
)


def test_old_catalog_entrypoints_are_not_active_imports() -> None:
    forbidden = {
        "quant_research_platform.catalog.defaults",
        "quant_research_platform.markets.implementations.a_share.catalog",
    }
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(ACTIVE_SCAN_ROOTS)
        if module in forbidden
    ]

    assert hits == []


def test_active_source_has_no_old_catalog_or_defaults_entrypoints() -> None:
    forbidden = (
        "markets.implementations.a_share.catalog",
        "from ..catalog import",
        "from .catalog import",
        "catalog.defaults",
        "default_catalogs",
        "default_factor_catalog",
        "default_label_catalog",
        "default_processed_data_catalog",
        "default_research_dataset_catalog",
        "default_evaluation_catalog",
        "default_model_catalog",
        "default_prediction_catalog",
        "PlatformCatalogs",
        "validation/research_panel.py",
        "validation\\research_panel.py",
    )
    hits: list[str] = []
    for path in _python_files((Path("src"),)):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                hits.append(f"{path}:{token}")

    assert hits == []


def test_retired_public_profile_names_are_absent() -> None:
    profiles_path = PROJECT_SRC / "catalog/profiles.py"
    defaults_path = PROJECT_SRC / "catalog/defaults.py"
    assert profiles_path.exists()
    assert not defaults_path.exists()

    tree = ast.parse(
        profiles_path.read_text(encoding="utf-8"),
        filename=str(profiles_path),
    )
    public_names = _module_all_names(tree)
    assert "ResearchProfile" in public_names
    assert "research_profile" in public_names
    assert "CatalogBundle" not in public_names
    assert "local_a_share_csmar_catalogs" not in public_names
    assert "PlatformCatalogs" not in public_names
    assert "default_catalogs" not in public_names
    assert not [name for name in public_names if name.startswith("default_")]


def test_source_market_storage_research_import_boundaries_are_closed() -> None:
    checks = (
        (
            PROJECT_SRC / "markets/implementations/a_share",
            "a_share_imports_csmar",
            ("quant_research_platform.sources.csmar",),
        ),
        (
            PROJECT_SRC / "sources/csmar",
            "csmar_imports_a_share",
            ("quant_research_platform.markets.implementations.a_share",),
        ),
        (
            PROJECT_SRC / "storage",
            "storage_imports_implementation",
            (
                "quant_research_platform.markets.implementations.a_share",
                "quant_research_platform.sources.csmar",
            ),
        ),
        (
            PROJECT_SRC / "research/datasets",
            "research_dataset_imports_implementation",
            (
                "quant_research_platform.markets.implementations.a_share",
                "quant_research_platform.sources.csmar",
            ),
        ),
        (
            PROJECT_SRC / "research/factors",
            "generic_factor_imports_a_share",
            ("quant_research_platform.markets.implementations.a_share",),
        ),
        (
            PROJECT_SRC / "research/labels",
            "generic_label_imports_a_share",
            ("quant_research_platform.markets.implementations.a_share",),
        ),
    )
    for root, error_name, prefixes in checks:
        hits = [
            f"{path}:{lineno} imports {module}"
            for path, lineno, module in _iter_imports((root,))
            if module.startswith(prefixes)
        ]
        assert hits == [], f"{error_name}: " + "; ".join(hits)


def test_interfaces_workflows_agents_use_profile_not_direct_implementations() -> None:
    roots = (
        PROJECT_SRC / "interfaces",
        PROJECT_SRC / "workflows",
        PROJECT_SRC / "agents",
    )
    forbidden_prefixes = (
        "quant_research_platform.markets.implementations.a_share",
        "quant_research_platform.sources.csmar",
    )
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(roots)
        if module.startswith(forbidden_prefixes)
    ]
    profile_users = [
        path.as_posix()
        for path, _lineno, module in _iter_imports(roots)
        if module == "quant_research_platform.catalog.profiles"
    ]

    assert hits == []
    assert profile_users


def test_deep_layers_do_not_build_default_research_profile() -> None:
    hits: list[str] = []
    forbidden = ("research_profile(", "local_a_share_csmar_catalogs(")
    roots = (
        PROJECT_SRC / "interfaces/services.py",
        PROJECT_SRC / "agents",
        PROJECT_SRC / "workflows",
    )
    for path in _python_files(roots):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                hits.append(f"{path}:{token}")

    assert hits == []


def test_shared_services_receive_research_profile_by_parameter() -> None:
    services_path = PROJECT_SRC / "interfaces/services.py"
    source = services_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(services_path))
    expected_profile_parameter = {
        "list_factors",
        "list_labels",
        "describe_research_dataset_requirements",
    }
    function_parameters = {
        node.name: {arg.arg for arg in node.args.args}
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in expected_profile_parameter
    }
    forbidden_import_names = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "quant_research_platform.catalog.profiles"
        for alias in node.names
        if alias.name == "research_profile"
    ]

    assert expected_profile_parameter == set(function_parameters)
    assert all("profile" in names for names in function_parameters.values())
    assert forbidden_import_names == []


def test_evaluation_model_prediction_do_not_import_a_share_annotation_rules() -> None:
    roots = (
        PROJECT_SRC / "research/evaluations",
        PROJECT_SRC / "research/models",
        PROJECT_SRC / "research/predictions",
    )
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(roots)
        if module.startswith(
            "quant_research_platform.markets.implementations.a_share.processed.annotations"
        )
    ]

    assert hits == []


def test_materializer_has_no_old_family_specific_or_source_imports() -> None:
    path = PROJECT_SRC / "markets/implementations/a_share/processed/materializer.py"
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "CSMAR_SOURCE_ID",
        "calculate_price_volume_factors_for_panel",
        "calculate_forward_return_labels_for_panel",
        "a_share_forward_return_label_definitions",
        "a_share_price_volume_factor_definitions",
        "markets.implementations.a_share.catalog",
    )

    assert [token for token in forbidden if token in source] == []


def test_panel_layer_has_no_annotation_rule_terms() -> None:
    path = PROJECT_SRC / "markets/implementations/a_share/processed/panels.py"
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "_add_one_word_limit_masks",
        "_shifted_bool_mask",
        "_bool_like_column",
        "one_word_limit",
        "can_enter",
        "can_exit",
        "eligible_base",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "next_day_is_trading",
        "next_day_limit_status",
    )

    assert [token for token in forbidden if token in source] == []


def test_deleted_panel_helper_names_only_appear_in_negative_gates() -> None:
    allowed_gate_files = {
        Path(__file__),
        Path("tests/markets/a_share/test_annotation_ownership.py"),
        PROJECT_SRC / "markets/implementations/a_share/processed/annotations.py",
    }
    forbidden = (
        "_add_one_word_limit_masks",
        "_shifted_bool_mask",
        "_bool_like_column",
    )
    hits: list[str] = []
    for path in _python_files(ACTIVE_SCAN_ROOTS):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source and path not in allowed_gate_files:
                hits.append(f"{path}:{token}")

    assert hits == []


def test_a_share_csmar_residue_is_limited_to_verification_evidence_fixture() -> None:
    a_share_root = PROJECT_SRC / "markets/implementations/a_share"
    hits: list[str] = []
    csmar_hits: list[tuple[Path, str]] = []
    for path in sorted(a_share_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "CSMAR_SOURCE_ID" in source:
            hits.append(f"{path}:CSMAR_SOURCE_ID")
        if "from quant_research_platform.sources.csmar" in source:
            hits.append(f"{path}:sources.csmar import")
        if re.search(r"(?i)\bcsmar\b", source):
            if path == a_share_root / "verification/private_aggregate.py":
                for line in source.splitlines():
                    if re.search(r"(?i)\bcsmar\b", line):
                        csmar_hits.append(
                            (path.relative_to(a_share_root), line.strip())
                        )
            else:
                csmar_hits.append((path.relative_to(a_share_root), "csmar"))

    assert hits == []
    assert csmar_hits == [
        (
            Path("verification/private_aggregate.py"),
            LOCAL_PRIVATE_EVIDENCE_FIXTURE_SOURCE_LITERAL,
        )
    ]


def test_active_code_has_no_retired_data_layer_terms() -> None:
    forbidden_patterns = (
        r"\basset_batches\b",
        r"\bdata_product_id\b",
        r"\bChunkAxis\.YEAR\b",
        r"\bChunkAxis\.ENTITY\b",
        r"\b_PANDAS_MOMENT_FALLBACK_VARIANCE\b",
        r"pandas fallback",
    )
    hits: list[str] = []
    for path in _python_files(ACTIVE_SCAN_ROOTS):
        if path == Path(__file__) or path in NEGATIVE_GATE_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if re.search(pattern, source):
                hits.append(f"{path}:{pattern}")

    assert hits == []


def _iter_imports(roots: tuple[Path, ...]) -> tuple[tuple[Path, int, str], ...]:
    imports: list[tuple[Path, int, str]] = []
    for path in _python_files(roots):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)


def _python_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            raise AssertionError(f"python_scan_root_missing: {root}")
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    return tuple(
        sorted(
            path
            for path in files
            if path.resolve() != Path(__file__).resolve()
        )
    )


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
            names.extend(
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return tuple(names)
