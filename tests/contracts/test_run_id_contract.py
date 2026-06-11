from __future__ import annotations

import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.contracts.workflows import WorkflowRunSummary


def test_run_id_accepts_public_safe_lowercase_identifiers() -> None:
    report = ReportSpec(
        report_id="stage_profile",
        report_definition_id="stage_profile_private_diagnostic",
        run_id="run-01.alpha_beta",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
    )

    assert report.run_id == "run-01.alpha_beta"


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "Run-01",
        "run 01",
        "run/01",
        r"run\01",
        "csmar-run",
        "data/raw/run",
        "data/interim/run",
        "data/processed/run",
        "reports/private_real/run",
        "600000.sh",
        "000001.sz",
        "a" * 129,
    ],
)
def test_report_spec_rejects_invalid_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        ReportSpec(
            report_id="stage_profile",
            report_definition_id="stage_profile_private_diagnostic",
            run_id=run_id,
            market_id=MarketId("a_share"),
            run_type="diagnostics",
        )


def test_evaluation_prediction_and_workflow_summaries_validate_run_ids() -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        EvaluationSpec(
            run_id="run 01",
            market_id=MarketId("a_share"),
            research_dataset_id="dataset",
            feature_columns=("factor",),
            label_columns=("label",),
        )

    with pytest.raises(ValueError, match="invalid_run_id"):
        PredictionSpec(
            prediction_id="data/raw/run",
            model_id="model",
            research_dataset_id="dataset",
            prediction_column="prediction",
            primary_key=("market_id", "asset_id", "timestamp"),
            train_end="2014-01-02",
            test_start="2014-01-03",
        )

    with pytest.raises(ValueError, match="invalid_run_id"):
        WorkflowRunSummary(
            workflow_id="workflow",
            workflow_version="0.1",
            execution_mode=ExecutionMode.DRY_RUN,
            run_id="000001.sz",
            status="failed",
            market_id="a_share",
            data_product_id="a_share_daily_bar",
            run_type="workflow",
            input_spec_ids=(),
            output_spec_ids=(),
            node_count=0,
            completed_nodes=(),
            private_boundary_required=True,
            artifact_writer_policy="none",
            failure_code="invalid_run_id",
        )
