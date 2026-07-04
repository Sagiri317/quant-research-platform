from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.markets.implementations.a_share import ids as market_ids
from quant_research_platform.research.factors.price_volume import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from tests.helpers.governance_fixtures import rule_form, text_rule_example
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

ROOTS_TO_SCAN = (
    Path("src"),
    Path(".agents/skills/quant-research-platform-maintenance/scripts"),
)
ACTIVE_DOC_ROOTS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path(".agents/briefs"),
    Path(".agents/skills/quant-research-platform-maintenance/SKILL.md"),
    Path(".agents/skills/quant-research-platform-maintenance/references"),
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/README.md"
    ),
    Path("docs/architecture"),
    Path("docs/code-map"),
    Path("docs/api"),
    Path("docs/data"),
    Path("data/README.md"),
)
SHARED_TEST_HELPER_ROOTS = (
    Path("tests/helpers"),
)

def test_workflow_run_policy_replaces_retired_action_contract() -> None:
    from quant_research_platform.contracts.workflows import WorkflowRunPolicy

    assert WorkflowRunPolicy.VALIDATE_ONLY.value == "validate_only"
    assert WorkflowRunPolicy.WRITE_ARTIFACTS.value == "write_artifacts"
    assert not Path("src/quant_research_platform/contracts/execution.py").exists()
    assert not Path("src/quant_research_platform/contracts/workflow_run.py").exists()


def test_runtime_workflow_outputs_do_not_use_action_wording(
    repo_python_files: tuple[Path, ...],
) -> None:
    action_word = "exec" + "ute"
    dry_run = "dry" + "-run"
    forbidden = re.compile(
        r"\b(?:"
        + re.escape(action_word)
        + r"[A-Za-z0-9_]*|[A-Za-z0-9_]*dry_run[A-Za-z0-9_]*|"
        r"[A-Za-z0-9_]*" + re.escape(dry_run) + r"[A-Za-z0-9_-]*)\b",
        re.IGNORECASE,
    )
    hits: list[str] = []
    roots = (
        "src/quant_research_platform/workflows/",
        "src/quant_research_platform/contracts/",
        "tests/workflows/",
    )
    for path in repo_python_files:
        if not path.as_posix().startswith(roots):
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if forbidden.search(line) or "WorkflowRun" + "Mode" in line:
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_run_output_architecture_retired_paths_are_absent() -> None:
    contract = load_maintenance_script_module("governance_contract")

    existing = [
        path
        for path in contract.retired_run_output_architecture_paths()
        if Path(path).exists()
    ]

    assert existing == []


def test_retired_run_output_names_are_absent() -> None:
    contract = load_maintenance_script_module("governance_contract")
    patterns = contract.retired_run_output_architecture_name_patterns()
    hits: list[str] = []

    for path in _scan_files((*ROOTS_TO_SCAN, *ACTIVE_DOC_ROOTS)):
        if contract.is_run_output_architecture_negative_test(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits.append(f"{path.as_posix()}:{name}")

    assert hits == []


def _scan_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower()
                in {".md", ".py", ".json", ".toml", ".yml", ".yaml"}
            )
    return tuple(sorted(set(files)))


def test_retired_surface_findings_are_absent(
    live_source_tree_scan: dict[str, dict[str, object]],
) -> None:
    scan_groups = live_source_tree_scan

    assert scan_groups["retired_text"]["findings"] == []
    assert scan_groups["python_symbols"]["findings"] == []
    assert scan_groups["synthesized_strings"]["findings"] == []


def test_shared_test_helpers_do_not_handwrite_active_derived_names() -> None:
    market_bars_id = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
    market_panel_id = market_ids.market_panel_data_id(DataFrequency.DAILY)
    momentum_column = price_volume_window_column("mom", 20)
    reversal_column = price_volume_window_column("rev", 1)
    forward_return_column = forward_return_label_column(1)
    forbidden = {
        market_bars_id: _literal_pattern(market_bars_id),
        market_panel_id: _literal_pattern(market_panel_id),
        momentum_column: _literal_pattern(momentum_column),
        reversal_column: _literal_pattern(reversal_column),
        "forward_return_horizon": re.compile(
            rf"\b{re.escape(forward_return_column[:-2])}[0-9]+d\b"
        ),
    }
    hits: list[str] = []
    for path in _iter_shared_test_helper_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in forbidden.items():
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}: {line.strip()}")

    assert hits == []


def test_shared_test_helpers_do_not_import_contract_market_bar_id_builder() -> None:
    forbidden_modules = {"quant_research_platform.contracts.data"}
    forbidden_names = {"market_bars_data_id"}
    hits: list[str] = []
    for path in _iter_shared_test_helper_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        hits.append(f"{path}:{node.lineno}: {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                imported_names = {alias.name for alias in node.names}
                if "*" in imported_names or imported_names & forbidden_names:
                    hits.append(
                        f"{path}:{node.lineno}: {node.module}: "
                        f"{', '.join(sorted(imported_names))}"
                    )

    assert hits == []


def _iter_shared_test_helper_files() -> list[Path]:
    files: list[Path] = []
    for root in SHARED_TEST_HELPER_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return files


def _literal_pattern(value: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(value) + r"\b")


def test_retired_owner_local_function_names_are_absent() -> None:
    contract = load_maintenance_script_module("governance_contract")
    patterns = contract.python_symbol_forbidden_patterns()
    modules = (
        "quant_research_platform.markets.implementations.a_share.definition",
        "quant_research_platform.markets.implementations.a_share.fields",
        "quant_research_platform.markets.implementations.a_share.processed.catalog",
        "quant_research_platform.markets.implementations.a_share.factors.definitions",
        "quant_research_platform.markets.implementations.a_share.labels.definitions",
        "quant_research_platform.markets.implementations.a_share.research_profiles",
        "quant_research_platform.markets.implementations.a_share.research_scenarios",
    )

    for module_name in modules:
        module = __import__(module_name, fromlist=["*"])
        hits = [
            (name, rule)
            for name in vars(module)
            for rule, pattern in patterns.items()
            if pattern.search(name)
        ]
        assert hits == []


def test_retired_workflow_run_module_is_absent() -> None:
    contract = load_maintenance_script_module("governance_contract")
    retired_module = "quant_research_platform." + text_rule_example(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "retired_workflow_module",
    )

    assert importlib.util.find_spec(retired_module) is None
    assert not (Path("src") / Path(retired_module.replace(".", "/") + ".py")).exists()


def test_retired_public_constant_aliases_are_absent() -> None:
    import quant_research_platform.contracts.models as contracts_models
    from quant_research_platform.markets.implementations.a_share import (
        ids as market_ids,
    )
    from quant_research_platform.sources.csmar import ids as source_ids

    contract = load_maintenance_script_module("governance_contract")
    cases = (
        (
            contracts_models,
            "retired_model_algorithm_collection",
        ),
        (source_ids, "retired_csmar_source_alias"),
        (market_ids, "retired_a_share_market_alias"),
    )

    for module, rule_name in cases:
        assert not hasattr(
            module,
            rule_form(contract.RETIRED_PUBLIC_SURFACE_RULES, rule_name),
        )


def test_retired_constant_fixtures_are_derived_from_governance_contract() -> None:
    contract = load_maintenance_script_module("governance_contract")

    assert rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "retired_model_algorithm_collection",
    )
    assert rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "retired_csmar_source_alias",
    )



def test_retired_pr_body_renderer_is_absent() -> None:
    contract = load_maintenance_script_module("governance_contract")
    renderer = text_rule_example(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_render_pr_body",
    )
    body = text_rule_example(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_pr_body",
    )
    assert (
        not Path(".agents/skills/quant-research-platform-maintenance/scripts")
        .joinpath(renderer)
        .exists()
    )
    assert (
        not Path("tests/agents/handoff")
        .joinpath(f"test_{Path(body).stem}_consistency.py")
        .exists()
    )


def test_handoff_source_tree_scan_rejects_retired_pr_body_renderer() -> None:
    contract = load_maintenance_script_module("governance_contract")

    patterns = contract.source_tree_forbidden_patterns()

    assert "forbidden_pr_body" in patterns
    assert "forbidden_render_pr_body" in patterns


def test_share_safety_does_not_export_retired_local_output_path_inventory() -> None:
    import quant_research_platform.policy.share_safety as share_safety

    assert "LEGACY_LOCAL_OUTPUT_PATH_NAMES" not in share_safety.__all__
    assert not hasattr(share_safety, "LEGACY_LOCAL_OUTPUT_PATH_NAMES")


def test_active_source_has_no_retired_local_output_inventory_name() -> None:
    source = Path("src/quant_research_platform/policy/share_safety.py").read_text(
        encoding="utf-8"
    )

    assert "LEGACY_LOCAL_OUTPUT_PATH_NAMES" not in source


def test_retired_local_only_report_paths_remain_share_unsafe() -> None:
    contract = load_maintenance_script_module("governance_contract")
    from quant_research_platform.policy.share_safety import (
        is_local_only_path,
        unsafe_for_share_reason,
    )

    private_real = rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_private_real",
    )
    private_data = rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_private_data",
    )
    for path in (
        Path("reports") / private_real / "a_share" / "run" / "summary.json",
        Path("reports") / private_data / "a_share" / "run" / "summary.json",
        Path("reports/private_research/a_share/run/summary.json"),
    ):
        assert is_local_only_path(path)
        assert unsafe_for_share_reason(path.as_posix()) == "local_only_path"
