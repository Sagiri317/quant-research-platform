from __future__ import annotations

import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec


def test_run_id_accepts_share_safe_lowercase_identifiers() -> None:
    report = OutputSpec(
        output_id="prediction_daily",
        output_definition_id="prediction_outputs",
        run_id="run-01.alpha_beta",
        market_id=MarketId("a_share"),
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
    )

    assert report.run_id == "run-01.alpha_beta"
    assert (
        OutputSpec(
            output_id="prediction_monthly",
            output_definition_id="prediction_outputs",
            run_id="run_202606",
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        ).run_id
        == "run_202606"
    )
    assert (
        OutputSpec(
            output_id="prediction_date_parts",
            output_definition_id="prediction_outputs",
            run_id="run_2026_06",
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        ).run_id
        == "run_2026_06"
    )


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "Run-01",
        "run 01",
        "run/01",
        r"run\01",
        "data/raw/run",
        "data/interim/run",
        "data/processed/run",
        "data/research/run",
        r"data\research\run",
        "outputs/prediction_daily/run",
        "a" * 129,
    ],
)
def test_output_spec_rejects_invalid_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        OutputSpec(
            output_id="prediction_daily",
            output_definition_id="prediction_outputs",
            run_id=run_id,
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )


@pytest.mark.parametrize(
    "run_id",
    (
        "csmar-run",
        "provider_run",
        "source_run",
        "600000.sh",
        "000001.sz",
        "000001",
        "688001",
        "run_000001",
        "run-688001",
        "run_600000_alpha",
        "run_600000_sh",
        "300001",
    ),
)
def test_output_spec_preserves_typed_content_identifiers(run_id: str) -> None:
    report = OutputSpec(
        output_id="prediction_daily",
        output_definition_id="prediction_outputs",
        run_id=run_id,
        market_id=MarketId("a_share"),
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
    )

    assert report.run_id == run_id


def test_evaluation_and_prediction_specs_validate_run_ids() -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        EvaluationSpec(
            evaluation_id="run 01",
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
