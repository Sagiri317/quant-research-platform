from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tests.helpers.governance_fixtures import rule_form, text_rule_example


def test_retired_owner_local_function_names_are_absent() -> None:
    contract = _load_governance_contract()
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
    contract = _load_governance_contract()
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

    contract = _load_governance_contract()
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
    contract = _load_governance_contract()

    assert rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "retired_model_algorithm_collection",
    )
    assert rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "retired_csmar_source_alias",
    )


def test_retired_handoff_evidence_names_are_absent(
    live_source_tree_scan: dict[str, dict[str, object]],
) -> None:
    scan_groups = live_source_tree_scan

    assert scan_groups["retired_text"]["findings"] == []
    assert scan_groups["python_symbols"]["findings"] == []
    assert scan_groups["synthesized_strings"]["findings"] == []


def test_retired_pr_body_renderer_is_absent() -> None:
    contract = _load_governance_contract()
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
    contract = _load_governance_contract()

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
    contract = _load_governance_contract()
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


def _load_governance_contract():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "governance_contract.py"
    )
    spec = importlib.util.spec_from_file_location("governance_contract", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
