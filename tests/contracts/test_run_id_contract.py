from __future__ import annotations

import pytest

import quant_research_platform.contracts.evaluations as evaluations_contracts
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.run_outputs import OutputScope, RunOutputSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.workflows import (
    WorkflowRunPolicy,
    WorkflowRunSummary,
)


def test_run_id_accepts_public_safe_lowercase_identifiers() -> None:
    report = RunOutputSpec(
        output_id="stage_profile",
        output_definition_id="stage_profile_local_diagnostic",
        run_id="run-01.alpha_beta",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
        scope=OutputScope.LOCAL_ONLY,
    )

    assert report.run_id == "run-01.alpha_beta"
    assert (
        RunOutputSpec(
            output_id="date_profile",
            output_definition_id="stage_profile_local_diagnostic",
            run_id="run_202606",
            market_id=MarketId("a_share"),
            run_type="diagnostics",
            scope=OutputScope.LOCAL_ONLY,
        ).run_id
        == "run_202606"
    )
    assert (
        RunOutputSpec(
            output_id="date_parts_profile",
            output_definition_id="stage_profile_local_diagnostic",
            run_id="run_2026_06",
            market_id=MarketId("a_share"),
            run_type="diagnostics",
            scope=OutputScope.LOCAL_ONLY,
        ).run_id
        == "run_2026_06"
    )


def test_signal_spec_is_not_reexported_from_evaluation_contracts() -> None:
    assert not hasattr(evaluations_contracts, "SignalSpec")


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "Run-01",
        "run 01",
        "run/01",
        r"run\01",
        "csmar-run",
        "provider_run",
        "source_run",
        "data/raw/run",
        "data/interim/run",
        "data/processed/run",
        "data/research/run",
        r"data\research\run",
        "run_outputs/local/run",
        "600000.sh",
        "000001.sz",
        "000001",
        "688001",
        "run_000001",
        "run-688001",
        "run_600000_alpha",
        "300001",
        "a" * 129,
    ],
)
def test_run_output_spec_rejects_invalid_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        RunOutputSpec(
            output_id="stage_profile",
            output_definition_id="stage_profile_local_diagnostic",
            run_id=run_id,
            market_id=MarketId("a_share"),
            run_type="diagnostics",
            scope=OutputScope.LOCAL_ONLY,
        )


def test_evaluation_prediction_and_workflow_summaries_validate_run_ids() -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        EvaluationSpec(
            run_id="run 01",
            market_id=MarketId("a_share"),
            research_dataset_id="dataset",
            signals=(
                SignalSpec(
                    signal_id="factor",
                    source_kind="factor",
                    source_column="factor",
                ),
            ),
            label_columns=("label",),
        )

    with pytest.raises(ValueError, match="invalid_run_id"):
        PredictionSpec(
            prediction_id="data/raw/run",
            model_id="model",
            research_dataset_id="dataset",
            train_end="2014-01-02",
            test_start="2014-01-03",
        )

    with pytest.raises(ValueError, match="invalid_run_id"):
        WorkflowRunSummary(
            workflow_id="workflow",
            workflow_version="0.1",
            run_policy=WorkflowRunPolicy.VALIDATE_ONLY,
            run_id="000001.sz",
            status="failed",
            market_id="a_share",
            run_type="workflow",
            input_spec_ids=(),
            output_spec_ids=(),
            node_count=0,
            completed_nodes=(),
            scope=True,
            artifact_writer_policy="none",
            failure_code="invalid_run_id",
        )
