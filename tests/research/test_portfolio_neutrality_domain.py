"""Unknown industry, unavailable size and missing forecasts have distinct domains."""

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


@pytest.mark.parametrize("missing", ["float_market_cap", "total_market_cap"])
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


def test_fixed_missing_forecasts_only_skip_when_constraints_are_infeasible():
    args = _inputs("industry")
    args["alpha"].loc["a"] = np.nan
    _, _, _, _, ledger = production._build_session(**args)
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] == (
        "target_neutrality_infeasible"
    )
    assert ledger.attrs["decision_availability"]["missing_prediction_count"] == 2
    amounts = ledger.set_index("asset_id").post_trade_notional
    assert amounts.to_dict() == pytest.approx(
        {"a": 100 / 3, "b": 100 / 3, "c": 100 / 3}
    )


def test_partial_forecast_rebalances_known_names_with_nonzero_fixed_weight():
    args = _inputs("industry")
    args["candidate"] = replace(args["candidate"], neutrality="industry")
    _, _, _, _, ledger = production._build_session(**args)
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] is None
    assert ledger.attrs["decision_availability"]["decision_status"] == "PARTIAL"
    # c is fixed at 1/3; the remaining 2/3 follows A:B = 1:2.
    assert ledger.set_index("asset_id").post_trade_notional.to_dict() == pytest.approx(
        {"a": 200 / 9, "b": 400 / 9, "c": 100 / 3}, abs=1e-8
    )


@pytest.mark.parametrize("unknown_score,expected", [(2.0, 1.0), (-1.0, 1 / 7)])
def test_unknown_industry_can_buy_or_sell_by_normal_projection(unknown_score, expected):
    args = _inputs("industry")
    args["candidate"] = replace(args["candidate"], neutrality="industry")
    args["alpha"].loc["c"] = unknown_score
    _, _, _, delta, ledger = production._build_session(**args)
    assert delta["c"] == pytest.approx(expected - 1 / 3, abs=1e-8)
    assert ledger.set_index("asset_id").loc[
        "c", "post_trade_notional"
    ] == pytest.approx(100 * expected, abs=1e-8)
    facts = ledger.attrs["neutrality_exposures"]
    assert facts["benchmark_unknown_industry_weight"] == 0.5
    assert facts["target_unknown_industry_weight"] == pytest.approx(expected, abs=1e-8)
    if unknown_score < 0:
        # The closest point to (1,0,0) on w_b=2*w_a, sum(w)=1 is (2,4,1)/7.
        assert facts["target_exposure:industry:fixture:A"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("can_exit", [False, True])
def test_industry_loss_alone_requests_no_exit_off_schedule(can_exit):
    args = _inputs("industry")
    args["candidate"] = replace(
        args["candidate"], rebalance_schedule="weekly-last-trading-day"
    )
    args["execution"].loc[args["execution"].asset_id.eq("c"), "can_exit"] = can_exit
    args["alpha"].loc["c"] = 2.0
    _, _, _, delta, ledger = production._build_session(**args)
    assert delta["c"] == 0.0
    assert ledger.set_index("asset_id").loc[
        "c", "post_trade_notional"
    ] == pytest.approx(100 / 3)
    facts = ledger.attrs["neutrality_exposures"]
    assert facts["realized_exposure_status"] == "DEFINED"
    assert facts["realized_unknown_industry_weight"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("neutrality", ["industry", "industry_size"])
def test_all_industries_unknown_keeps_trading_and_available_size(neutrality):
    args = _inputs("industry")
    args["candidate"] = replace(args["candidate"], neutrality=neutrality)
    args["panel"]["industry_level1_code"] = pd.NA
    args["alpha"].loc["c"] = 0.0
    _, _, _, _, ledger = production._build_session(**args)
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] is None
    facts = ledger.attrs["neutrality_exposures"]
    for kind in ("benchmark", "target", "realized"):
        assert facts[f"{kind}_unknown_industry_weight"] == pytest.approx(1.0)
        assert facts[f"{kind}_industry_exposure_status"] == "UNDEFINED"
        assert facts[f"{kind}_industry_exposure_reason"] == "no_known_industry_weight"
        assert not any(name.startswith(f"{kind}_exposure:industry:") for name in facts)
    if neutrality == "industry_size":
        assert facts["target_exposure:size"] == pytest.approx(
            facts["benchmark_exposure:size"], abs=1e-8
        )


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


def test_public_neutral_pool_persists_known_ratios_and_unknown_holdings(tmp_path):
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
        size_neutrality_limit=0.2,
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
        "capitalization_available_research_pool"
    )
    assert (
        artifact.metadata["industry_neutrality_domain"] == "known_industry_subportfolio"
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
    assert facts["neutrality_reference_calculable_count"] == 12
    assert facts["neutrality_reference_excluded_known_float_cap_weight"] == (0.0)
    assert facts["missing_prediction_count"] == 1
    assert facts["benchmark_unknown_industry_weight"] == pytest.approx(1 / 7)
    assert facts["target_unknown_industry_weight"] == 0.0
    assert facts["benchmark_exposure:industry:fixture:a5"] == pytest.approx(1 / 6)
    assert {row.asset_id for row in first.trades} == {
        *(f"a{i}" for i in range(6)),
        *(f"b{i}" for i in range(5)),
    }
    assert all(row.asset_id != "b5" for batch in batches for row in batch.trades)
    for batch in batches[1:]:
        assert any(
            row.name == "realized_exposure_status" and row.text == "DEFINED"
            for row in batch.diagnostics
        )
        held = {row.phase: row for row in batch.valuations if row.asset_id == "a5"}
        assert held["after"].quantity == held["before"].quantity > 0
        assert any(
            row.asset_id == "a5"
            and row.name == "decision_delta_weight"
            and row.number == 0
            for row in batch.diagnostics
        )
        assert any(
            row.name == "realized_unknown_industry_weight" and row.number > 0
            for row in batch.diagnostics
        )

    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
        portfolio_statistics,
    )
    from quant_research_platform.research.portfolios.risk import HoldingFactorInputs

    stats = read_statistics_artifact(
        portfolio_statistics(
            {
                "ledgers": [str(artifact.root)],
                "window": config["output_window"],
                "output": str(tmp_path / "performance"),
                "analyses": ["performance"],
            }
        )
    )
    candidate = production.PortfolioCandidateSpec(
        **{
            **artifact.metadata["candidate"],
            "risk_style_columns": tuple(
                artifact.metadata["candidate"]["risk_style_columns"]
            ),
        }
    )
    assert stats.metadata["industry_neutrality_domains"] == {
        candidate.candidate_id: "known_industry_subportfolio"
    }
    dates = pd.bdate_range("2023-01-02", periods=5)
    tables = _portfolio_statistics_tables(
        ledgers={candidate.candidate_id: artifact},
        candidates=(candidate,),
        config={"analyses": ["attribution"], "sample_role": "validation"},
        time_window=TimeWindow(**config["output_window"]),
        benchmark_opens=pd.DataFrame(),
        calendar=dates,
        unified_trading_sessions=lambda x: x,
        holding_inputs={
            (): [
                HoldingFactorInputs(
                    *dates[i : i + 3],
                    None,
                    pd.Series(dtype=float),
                    pd.Series(dtype=float),
                    pd.DataFrame(),
                    reason="attribution_pit_history_missing",
                )
                for i in range(3)
            ]
        },
    )
    published = read_statistics_artifact(
        write_statistics_artifact(
            tmp_path / "exposures",
            metadata={
                **stats.metadata,
                "analyses": ["attribution"],
                "sample_role": "validation",
            },
            tables=tables,
            table_group="portfolio",
        )
    )
    with published.table("portfolio_exposures.parquet") as exposures:
        for i, batch in enumerate(batches):
            expected = {
                row.name: row.number
                for row in batch.diagnostics
                if row.asset_id == "__cash__"
                and row.number is not None
                and (
                    row.name.startswith(
                        (
                            "target_exposure:",
                            "realized_exposure:",
                            "benchmark_exposure:",
                        )
                    )
                    or row.name.endswith("_unknown_industry_weight")
                )
            }
            actual = (
                exposures.loc[exposures.entry_at.eq(dates[i + 1])]
                .set_index("factor_id")
                .value.to_dict()
            )
            assert actual == expected
