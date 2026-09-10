from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")
WORKFLOW_ACTIONS = (
    "data_materialize",
    "dataset_build",
    "research_run",
    "report_build",
    "status",
    "inspect",
)


def test_public_workflow_actions_have_one_owner_and_direct_callers() -> None:
    workflow_root = SOURCE_ROOT / "workflows"
    owners = {action: [] for action in WORKFLOW_ACTIONS}
    for path in workflow_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in owners:
                    owners[node.name].append(path)

    assert owners == {
        action: [workflow_root / "research.py"] for action in WORKFLOW_ACTIONS
    }

    cli_path = SOURCE_ROOT / "interfaces" / "cli" / "commands.py"
    assert _imports_research_workflow(cli_path)
    assert _direct_workflow_calls(cli_path) == set(WORKFLOW_ACTIONS)


def test_research_and_profile_owners_have_no_provider_traversal() -> None:
    research_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (SOURCE_ROOT / "research").rglob("*.py")
    )
    assert "markets.implementations" not in research_sources
    assert "quant_research_platform.sources" not in research_sources

    profile = (SOURCE_ROOT / "profiles" / "research.py").read_text(encoding="utf-8")
    assert "write_dataset" not in profile
    assert "pandas" not in profile
    assert "memory_budget_bytes" not in profile


def test_profile_is_composition_only_and_research_has_one_resolver() -> None:
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
    imports = _imported_modules(profile_tree)
    assert all(
        not module.startswith(forbidden)
        for module in imports
        for forbidden in forbidden_prefixes
    )
    assert {
        "StorageLayout",
        "RuntimeResourcePolicy",
        "InterimMaterializer",
        "ProcessedMaterializer",
    }.isdisjoint(_defined_symbols(profile_tree))

    resolver_owner = SOURCE_ROOT / "workflows" / "research.py"
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
                resolver_defs.append((node.name, path))
    assert resolver_defs == [
        ("resolve_target_graph", resolver_owner),
        ("plan_targets", resolver_owner),
    ]


def test_a_share_pit_owner_and_lifecycle_inputs_have_one_direction() -> None:
    factor_path = (
        SOURCE_ROOT / "research" / "factors" / "fundamental" / "calculations.py"
    )
    factor_modules = _imported_modules(
        ast.parse(factor_path.read_text(encoding="utf-8"))
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
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module
        == "quant_research_platform.markets.implementations.a_share.fundamentals"
        and any(alias.name == "construct_ttm_events" for alias in node.names)
        for node in ast.walk(producer_tree)
    )

    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )
    from quant_research_platform.sources.choice.interim import first_disclosure_spec

    assert {
        "listing_status_changes",
        "stock_info_changes",
        "stock_info_annual",
    }.isdisjoint(
        catalog.processed_catalog().get("market_panel_1d").source_input_data_ids
    )
    assert first_disclosure_spec().primary_key == (
        "asset_id",
        "report_period",
        "first_disclosure_date",
    )


def test_workflow_production_statuses_have_no_readiness_aliases() -> None:
    literals: set[str] = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        literals.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and type(node.value) is str
        )
    assert {"READY", "STALE"}.isdisjoint(literals)


def test_model_algorithm_surface_matches_a_share_and_crypto_protocols() -> None:
    from quant_research_platform.contracts.models import _MODEL_ALGORITHM_FACTS
    from quant_research_platform.research.protocols.a_share import MODEL_FAMILY_ORDER
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_MODEL_ALGORITHMS,
    )

    expected = {
        *MODEL_FAMILY_ORDER,
        *HFT_MODEL_ALGORITHMS,
    }
    assert set(_MODEL_ALGORITHM_FACTS) == expected


def test_tracked_surfaces_have_no_retired_naming() -> None:
    violations: list[str] = []
    repository_root = Path(__file__).resolve().parents[2]
    for relative_path in _tracked_surface_paths():
        if _contains_retired_token(relative_path.as_posix()):
            violations.append(f"{relative_path}:path")
        path = repository_root / relative_path
        if path.suffix == ".parquet":
            continue
        text = path.read_text(encoding="utf-8")
        if _contains_retired_token(text):
            violations.append(f"{relative_path}:text")
        if path.suffix == ".py":
            tree = ast.parse(text, filename=relative_path.as_posix())
            for symbol in _defined_symbols(tree):
                if _is_retired_symbol(symbol):
                    violations.append(f"{relative_path}:{symbol}")

    assert violations == []


def test_retired_contract_separator_variants_are_detected() -> None:
    for separator in ("_", "-", " "):
        assert _contains_retired_token("current" + separator + "contract")


def test_csmar_source_surface_has_one_root_without_retired_sibling_api() -> None:
    retired_surface_tokens = (
        "csmar" + "_" + "aux",
        "source_root" + "_" + "resolver",
        "resolve" + "_" + "source_root",
    )
    repository_root = Path(__file__).resolve().parents[2]
    paths = set(_tracked_surface_paths())
    violations = []
    for relative_path in sorted(paths):
        text = (
            ""
            if relative_path.suffix == ".parquet"
            else (repository_root / relative_path).read_text(encoding="utf-8")
        )
        for token in retired_surface_tokens:
            if (
                token.casefold() in text.casefold()
                or token.casefold() in relative_path.as_posix().casefold()
            ):
                violations.append(f"{relative_path}:{token}")
    assert violations == []


_RETIRED_TEXT = re.compile(
    r"(?i)(?<![a-z0-9])(?:phase\s*[_-]?\s*[12]|"
    "current" + r"[_ -]" + "contract|"
    "unified" + r"[_-]" + "research|"
    "leg" + "acy|"
    "com" + r"pat(?:ibility)?|"
    r"_" + "v2)(?![a-z0-9])"
)


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


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _is_retired_symbol(symbol: str) -> bool:
    return bool(_RETIRED_TEXT.fullmatch(symbol))


def _contains_retired_token(value: str) -> bool:
    return _RETIRED_TEXT.search(value) is not None


def _tracked_surface_paths() -> tuple[Path, ...]:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    tracked = (Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw)
    selected = tuple(
        path
        for path in tracked
        if path.name == "README.md"
        or path.parts[:1] in (("docs",), ("src",), ("tests",))
    )
    # The scan is intentionally limited to live tracked files during a dirty deletion.
    return tuple(path for path in selected if (repository_root / path).exists())


def _imports_research_workflow(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "quant_research_platform.workflows"
        and any(
            alias.name == "research" and alias.asname is None for alias in node.names
        )
        for node in ast.walk(tree)
    )


def _direct_workflow_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "research"
        and node.func.attr in WORKFLOW_ACTIONS
    }
