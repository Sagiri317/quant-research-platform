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


def test_retired_handoff_evidence_names_are_absent() -> None:
    scan_groups = _load_governance_contract().source_tree_scan_groups(Path("."))

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


def test_retired_handoff_evidence_input_fields_fail_fast(
    tmp_path: Path,
) -> None:
    module = _load_handoff_builder_integration_tests()
    contract = _load_governance_contract()
    body = text_rule_example(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_pr_body",
    )
    forbidden_keys = ("check", f"{Path(body).stem}_path")

    for forbidden_key in forbidden_keys:
        case_root = tmp_path / forbidden_key
        case_root.mkdir()
        result, out_dir = module._run_builder_with_handoff_evidence(
            case_root,
            {
                "custom-note": {
                    "schema_id": "qrp-custom-note",
                    "status": "PASS",
                    forbidden_key: "old input",
                }
            },
        )

        assert result.returncode != 0
        assert "handoff_evidence_forbidden_field" in result.stderr
        assert not (out_dir / "synthetic-handoff-negative.zip").exists()
        assert not (out_dir / "synthetic-handoff-negative.json").exists()


def _load_handoff_builder():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "handoff"
        / "build_handoff_package.py"
    )
    spec = importlib.util.spec_from_file_location("build_handoff_package", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_handoff_builder_integration_tests():
    script = Path("tests/agents/handoff/test_build_handoff_package_integration.py")
    spec = importlib.util.spec_from_file_location(
        "test_build_handoff_package_integration",
        script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
