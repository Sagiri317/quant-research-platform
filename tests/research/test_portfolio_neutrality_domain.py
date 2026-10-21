"""Research-pool neutrality uses context availability before forecast coverage."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.portfolios import production
from tests.helpers.portfolio_inputs import decision_inputs


def _inputs(missing):
    args = decision_inputs(benchmark=(1 / 3, 1 / 3, 1 / 3))
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        top_quantile=0.5,
        benchmark_id=None,
        risk_style_columns=(),
        neutrality="industry_size",
        industry_neutrality_limit=0.0,
        size_neutrality_limit=0.0,
    )
    args["enhanced_decision"] = None
    args["panel"] = args["panel"].assign(
        industry_standard_id="fixture", industry_level1_code=["A", "B", "C"]
    )
    args["dataset"] = args["dataset"].assign(
        float_market_cap=[1000.0, 2000.0, 3000.0],
        total_market_cap=[1000.0, 2000.0, 3000.0],
    )
    if missing == "industry":
        args["panel"].loc[args["panel"].asset_id.eq("c"), "industry_level1_code"] = (
            pd.NA
        )
    else:
        args["dataset"].loc[args["dataset"].asset_id.eq("c"), missing] = np.nan
    args["alpha"].loc["c"] = np.nan
    return args


@pytest.mark.parametrize(
    "missing", ["industry", "float_market_cap", "total_market_cap"]
)
def test_unavailable_context_exits_without_blocking_known_pool(missing):
    args = _inputs(missing)
    source = args["dataset"].copy(deep=True)
    predictions = args["alpha"].copy(deep=True)
    account, _, _, _, ledger = production._build_session(**args)
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] is None
    assert ledger.attrs["decision_availability"]["missing_prediction_count"] == 0
    # Exact industry matching fixes the known-pool weights at 1/3 and 2/3.
    assert ledger.set_index("asset_id").post_trade_notional.to_dict() == pytest.approx(
        {"a": 100 / 3, "b": 200 / 3, "c": 0.0}, abs=1e-8
    )
    assert account.nav_after == pytest.approx(100.0, abs=1e-10)
    facts = ledger.attrs["neutrality_exposures"]
    assert facts["neutrality_reference_eligible_count"] == 3
    assert facts["neutrality_reference_calculable_count"] == 2
    assert facts["neutrality_reference_excluded_known_float_cap_weight"] == (
        0.0 if missing == "float_market_cap" else 0.5
    )
    assert facts["benchmark_exposure:industry:fixture:A"] == pytest.approx(1 / 3)
    assert facts["benchmark_exposure:industry:fixture:B"] == pytest.approx(2 / 3)
    pd.testing.assert_frame_equal(args["dataset"], source, check_exact=True)
    pd.testing.assert_series_equal(args["alpha"], predictions, check_exact=True)


def test_missing_forecast_inside_calculable_pool_still_skips_and_exits():
    args = _inputs("industry")
    args["alpha"].loc["a"] = np.nan
    _, _, _, _, ledger = production._build_session(**args)
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] == (
        "partial_predictions_missing_neutrality"
    )
    assert ledger.attrs["decision_availability"]["missing_prediction_count"] == 1
    amounts = ledger.set_index("asset_id").post_trade_notional
    assert amounts.to_dict() == pytest.approx({"a": 100 / 3, "b": 100 / 3, "c": 0.0})


@pytest.mark.parametrize("can_exit", [False, True])
def test_context_loss_requests_independent_exit_off_schedule(can_exit):
    args = _inputs("industry")
    args["candidate"] = replace(
        args["candidate"], rebalance_schedule="weekly-last-trading-day"
    )
    args["execution"].loc[args["execution"].asset_id.eq("c"), "can_exit"] = can_exit
    _, _, _, delta, ledger = production._build_session(**args)
    assert delta["c"] == pytest.approx(-1 / 3)
    assert ledger.set_index("asset_id").loc[
        "c", "post_trade_notional"
    ] == pytest.approx(0.0 if can_exit else 100 / 3)
    facts = ledger.attrs["neutrality_exposures"]
    assert facts["realized_exposure_status"] == ("DEFINED" if can_exit else "UNDEFINED")
    if not can_exit:
        assert facts["realized_exposure_reason"] == "outside_benchmark_domain"


def test_non_neutral_account_does_not_depend_on_context_pool():
    args = _inputs("industry")
    args["candidate"] = replace(args["candidate"], neutrality="none")
    reference = deepcopy(args)
    reference["panel"]["industry_level1_code"] = "known"
    reference["dataset"]["total_market_cap"] = np.nan
    actual = production._build_session(**args)
    expected = production._build_session(**reference)
    pd.testing.assert_frame_equal(actual[4], expected[4], check_exact=True)
    assert actual[4].attrs == expected[4].attrs


def test_public_neutral_pool_persists_reference_and_blocked_exit(tmp_path):
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from tests.research.test_portfolio_missing_predictions import (
        _public_prediction_config,
        _replace_public_input,
    )

    config = _public_prediction_config(tmp_path)
    config["candidate"].update(
        neutrality="industry_size",
        size_cap_column="float_market_cap",
        industry_neutrality_limit=0.0,
        size_neutrality_limit=0.0,
    )
    rows, metadata = read_predictions(config["predictions"][0]["path"])
    config["predictions"][0]["path"] = str(
        write_prediction_outputs(
            output_path=tmp_path / "known-predictions",
            output_chunks=[rows.loc[rows.asset_id.ne("b5")]],
            metadata={**metadata, "prediction_rows": None},
        )
    )

    def panel(frame):
        frame["industry_level1_code"] = frame.asset_id
        frame.loc[
            frame.asset_id.eq("b5")
            | (frame.asset_id.eq("a5") & frame.timestamp.ge("2023-01-03")),
            "industry_level1_code",
        ] = pd.NA
        return frame

    _replace_public_input(tmp_path, config, "panel", panel)
    _replace_public_input(
        tmp_path,
        config,
        "dataset",
        lambda frame: frame.assign(
            float_market_cap=1000.0 * (frame.asset_id.str[1:].astype(int) + 1)
        ),
    )
    _replace_public_input(
        tmp_path,
        config,
        "execution",
        lambda frame: frame.assign(can_exit=frame.asset_id.ne("a5")),
    )
    _replace_public_input(
        tmp_path, config, "fees", lambda frame: frame.assign(rate=0.0)
    )
    artifact = read_portfolio_ledger_artifact(production.backtest(config))
    assert artifact.metadata["neutrality_reference_domain"] == (
        "context_available_research_pool"
    )
    batches = list(artifact.iter_chunks())
    assert len(batches) == 3
    first = batches[0]
    facts = {
        row.name: row.number if row.number is not None else row.text
        for row in first.diagnostics
        if row.asset_id == "__cash__"
    }
    assert facts["neutrality_reference_eligible_count"] == 12
    assert facts["neutrality_reference_calculable_count"] == 11
    assert facts["neutrality_reference_excluded_known_float_cap_weight"] == (
        pytest.approx(1 / 7)
    )
    assert facts["missing_prediction_count"] == 0
    assert facts["benchmark_exposure:industry:fixture:a5"] == pytest.approx(1 / 6)
    assert {row.asset_id for row in first.trades} == {
        *(f"a{i}" for i in range(6)),
        *(f"b{i}" for i in range(5)),
    }
    assert all(row.asset_id != "b5" for batch in batches for row in batch.trades)
    for batch in batches[1:]:
        assert any(
            row.name == "realized_exposure_reason"
            and row.text == "outside_benchmark_domain"
            for row in batch.diagnostics
        )
        held = {row.phase: row for row in batch.valuations if row.asset_id == "a5"}
        assert held["after"].quantity == held["before"].quantity > 0
        assert any(
            row.asset_id == "a5"
            and row.name == "decision_delta_weight"
            and row.number < 0
            for row in batch.diagnostics
        )
