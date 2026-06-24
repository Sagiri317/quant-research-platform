from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.helpers.governance_fixtures import rule_form


def test_retired_owner_local_function_names_are_absent() -> None:
    modules = {
        "quant_research_platform.markets.implementations.a_share.definition": (
            "a_share_market_definition",
        ),
        "quant_research_platform.markets.implementations.a_share.fields": (
            "a_share_field_catalog",
        ),
        "quant_research_platform.markets.implementations.a_share.processed.catalog": (
            "a_share_processed_catalog",
        ),
        "quant_research_platform.markets.implementations.a_share.factors.definitions": (
            "a_share_factor_definitions",
            "a_share_factor_catalog",
        ),
        "quant_research_platform.markets.implementations.a_share.labels.definitions": (
            "a_share_label_definitions",
            "a_share_label_catalog",
        ),
        "quant_research_platform.markets.implementations.a_share.research_profiles": (
            "build_a_share_research_dataset_catalog",
            "build_a_share_research_dataset_specs",
        ),
        "quant_research_platform.markets.implementations.a_share.research_scenarios": (
            "build_a_share_model_catalog",
            "build_a_share_model_specs",
            "build_a_share_prediction_catalog",
            "build_a_share_prediction_specs",
            "build_a_share_portfolio_catalog",
            "build_a_share_portfolio_specs",
            "default_a_share_prediction_smoke_id",
            "default_a_share_prediction_oos_period",
        ),
    }

    for module_name, retired_names in modules.items():
        module = __import__(module_name, fromlist=["*"])
        assert [name for name in retired_names if hasattr(module, name)] == []


def test_retired_workflow_run_module_is_absent() -> None:
    assert (
        importlib.util.find_spec("quant_research_platform.contracts.workflow_run")
        is None
    )
    assert not Path("src/quant_research_platform/contracts/workflow_run.py").exists()


def test_retired_public_constant_aliases_are_absent() -> None:
    import quant_research_platform.contracts.models as contracts_models
    from quant_research_platform.markets.implementations.a_share import (
        ids as market_ids,
    )
    from quant_research_platform.sources.csmar import ids as source_ids

    assert not hasattr(contracts_models, "MODEL_ALGORITHMS")
    assert not hasattr(source_ids, "CSMAR_SOURCE_ID")
    assert not hasattr(market_ids, "A_SHARE_MARKET_ID")


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
    scan_groups = _load_handoff_builder().source_tree_scan_groups(Path("."))

    assert scan_groups["retired_text"]["findings"] == []
    assert scan_groups["python_symbols"]["findings"] == []
    assert scan_groups["synthesized_strings"]["findings"] == []


def test_retired_pr_body_renderer_is_absent() -> None:
    assert not Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/render_pr_body_from_validation_evidence.py"
    ).exists()
    assert not Path("tests/agents/handoff/test_pr_body_consistency.py").exists()


def test_handoff_source_tree_scan_rejects_retired_pr_body_renderer() -> None:
    module = _load_handoff_builder()

    patterns = module.source_tree_forbidden_patterns()

    assert "forbidden_pr_body" in patterns
    assert "forbidden_render_pr_body" in patterns


@pytest.mark.parametrize("forbidden_key", ("check", "pr_body_path"))
def test_retired_handoff_evidence_input_fields_fail_fast(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    module = _load_handoff_builder_integration_tests()

    result, out_dir = module._run_builder_with_handoff_evidence(
        tmp_path,
        {
            "custom-note": {
                "schema_version": "custom-note/0.1",
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
