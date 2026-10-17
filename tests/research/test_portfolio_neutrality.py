"""Hand-check full-domain projection and separate target/execution facts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.production import _validate_windows


@pytest.mark.parametrize(
    "neutrality,expected",
    [
        ("industry_size", [0.0, 0.5, 1 / 3, 1 / 6]),
        ("industry", [0.0, 0.5, 0.25, 0.25]),
        ("size", [1 / 6, 2 / 3, 1 / 6, 0.0]),
    ],
)
def test_projection_handcheck_includes_unpredicted_benchmark_assets(
    neutrality, expected
):
    assets = pd.Index(["a", "b", "c", "d"])
    caps = pd.Series([1.0, 2.0, 1.0, 2.0], index=assets)
    weights = build_simple_target_weights(
        pd.Series([np.nan, 1.0, np.nan, np.nan], index=assets),
        eligible=pd.Series(True, index=assets),
        float_market_cap=caps,
        total_market_cap=caps,
        industry=pd.Series(["A", "A", "B", "B"], index=assets)
        if neutrality != "size"
        else None,
        neutrality=neutrality,
        industry_neutrality_limit=0.0,
        size_neutrality_limit=0.0,
    )
    assert weights.tolist() == pytest.approx(expected, abs=1e-8)
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert weights.ge(0).all()
    if neutrality != "size":
        assert weights.iloc[:2].sum() == pytest.approx(0.5)
    if neutrality != "industry":
        assert float(weights @ np.log(caps)) == pytest.approx(2 / 3 * np.log(2))


def test_projection_failure_does_not_silently_relax(monkeypatch):
    import quant_research_platform.research.portfolios.optimizer as optimizer

    monkeypatch.setattr(
        optimizer, "minimize", lambda *a, **kw: SimpleNamespace(success=False)
    )
    with pytest.raises(PortfolioTargetBlocked, match="target_neutrality_solver_failed"):
        build_simple_target_weights(
            pd.Series({"a": 1.0, "b": 0.0}),
            eligible=pd.Series({"a": True, "b": True}),
            float_market_cap=pd.Series({"a": 1.0, "b": 2.0}),
            total_market_cap=pd.Series({"a": 1.0, "b": 2.0}),
            neutrality="size",
        )


def test_mixed_precision_windows_keep_whole_date_and_nanosecond_bounds():
    _validate_windows(
        TimeWindow("2023-01-02T12:00:00.000000001", "2023-01-02T23:59:59.999999999"),
        TimeWindow("2023-01-01", "2023-01-02"),
        TimeWindow("2023-01-02", "2023-01-02"),
    )
    with pytest.raises(ValueError, match="portfolio_windows_invalid"):
        _validate_windows(
            TimeWindow("2023-01-02", "2023-01-02"),
            TimeWindow("2023-01-01T00:00:00", "2023-01-02T23:59:59.999999998"),
            TimeWindow("2023-01-01", "2023-01-02"),
        )


def test_neutral_target_and_blocked_realized_exposure_are_distinct(
    tmp_path, monkeypatch
):
    from tests.research.test_portfolio_production import _build_producer, _inputs

    frames, fees, window = _inputs()
    frames["panel"]["industry_standard_id"] = "fixture"
    frames["panel"]["industry_level1_code"] = frames["panel"].asset_id
    frames["execution"].loc[frames["execution"].asset_id.eq("a"), "can_enter"] = False
    candidate = PortfolioCandidateSpec(
        "neutral",
        "global",
        "quantile_equal_weight",
        "ridge",
        "daily",
        0.5,
        None,
        neutrality="industry_size",
        industry_neutrality_limit=0.0,
        size_neutrality_limit=0.0,
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        typed_batches=True,
    )
    with closing_iterator(batches):
        first = next(iter(batches))
    facts = {
        row.name: row.number if row.number is not None else row.text
        for row in first.diagnostics
        if row.asset_id == "__cash__"
    }
    assert facts["target_exposure:industry:fixture:a"] == pytest.approx(1 / 3)
    assert facts["benchmark_exposure:industry:fixture:a"] == pytest.approx(1 / 3)
    assert facts["realized_exposure:industry:fixture:a"] == 0.0
    assert facts["realized_exposure_status"] == "DEFINED"
    assert any(row.name == "raw_target_weight" for row in first.diagnostics)
    assert replace(candidate, neutrality="none").neutrality == "none"


def test_nonrebalance_blocked_exit_keeps_realized_exposure_unknown(
    tmp_path, monkeypatch
):
    from tests.research.test_portfolio_production import _build_producer, _inputs

    frames, fees, window = _inputs()
    for name in ("panel", "dataset", "execution"):
        last = frames[name].loc[frames[name].timestamp.eq("2023-01-10")]
        frames[name] = pd.concat(
            (frames[name], last.assign(timestamp=pd.Timestamp("2023-01-11"))),
            ignore_index=True,
        )
    frames["calendar"] = pd.concat(
        (
            frames["calendar"],
            frames["calendar"]
            .iloc[-1:]
            .assign(calendar_date=pd.Timestamp("2023-01-11")),
        ),
        ignore_index=True,
    )
    frames["panel"]["industry_standard_id"] = "fixture"
    frames["panel"]["industry_level1_code"] = frames["panel"].asset_id
    frames["dataset"].loc[
        frames["dataset"].asset_id.eq("b")
        & frames["dataset"].timestamp.ge("2023-01-09"),
        "eligible",
    ] = False
    # Loss of research eligibility requests an independent exit; a real execution
    # restriction is needed to leave the old position outside the benchmark.
    frames["execution"].loc[
        frames["execution"].asset_id.eq("b")
        & frames["execution"].timestamp.ge("2023-01-09"),
        "can_exit",
    ] = False
    candidate = PortfolioCandidateSpec(
        "neutral",
        "global",
        "quantile_equal_weight",
        "ridge",
        "weekly-last-trading-day",
        0.5,
        None,
        neutrality="industry",
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=TimeWindow("2023-01-03", "2023-01-11"),
        input_window=TimeWindow("2022-12-01", "2023-01-11"),
        formal_dataset_window=TimeWindow("2022-12-01", "2023-01-11"),
        candidate=candidate,
        typed_batches=True,
    )
    matched = []
    initial_cash = []
    for batch in batches:
        facts = {
            row.name: row.number if row.number is not None else row.text
            for row in batch.diagnostics
            if row.asset_id == "__cash__"
        }
        if pd.Timestamp(facts["decision_timestamp"]) < pd.Timestamp("2023-01-06"):
            initial_cash.append(batch)
            assert not batch.trades
            assert all(row.asset_id == "__cash__" for row in batch.valuations)
            for kind in ("target", "realized"):
                assert facts[f"{kind}_exposure_status"] == "UNDEFINED"
                assert facts[f"{kind}_exposure_reason"] == "nonpositive_stock_value"
            continue
        if pd.Timestamp(facts["decision_timestamp"]) >= pd.Timestamp("2023-01-09"):
            matched.append(batch)
            assert facts["target_exposure_status"] == "DEFINED"
            assert facts["target_exposure_excluded_weight"] == 0.0
            assert facts["target_exposure:industry:fixture:a"] == pytest.approx(1.0)
            assert facts["realized_exposure_status"] == "UNDEFINED"
            assert facts["realized_exposure_reason"] == "outside_benchmark_domain"
            assert facts["realized_exposure_excluded_weight"] > 0.0
            assert not any(key.startswith("realized_exposure:") for key in facts)
            held = {row.phase: row for row in batch.valuations if row.asset_id == "b"}
            assert held["before"].quantity > 0
            assert held["after"].quantity == held["before"].quantity
            assert held["after"].value > 0
            assert not any(row.asset_id == "b" for row in batch.trades)
            target = {
                row.name: row.number for row in batch.diagnostics if row.asset_id == "b"
            }
            assert target["target_weight"] == 0.0
            assert target["decision_delta_weight"] < 0.0
    assert initial_cash
    assert matched


def test_neutral_projection_satisfies_kkt_on_active_and_boundary_assets():
    from quant_research_platform.research.portfolios.optimizer import (
        _project_neutral_target,
    )

    context = pd.DataFrame(
        {
            "benchmark_weight": [0.2, 0.2, 0.2, 0.2, 0.2],
            "size": [-2.0, -1.0, 0.0, 1.0, 2.0],
        },
        index=list("abcde"),
    )
    target = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=context.index)
    weights = _project_neutral_target(
        target, context, industry_limit=0.0, size_limit=0.0
    ).to_numpy()
    matrix = np.vstack((np.ones(5), context["size"]))
    gradient = weights - target.to_numpy()
    active = weights > 1e-9
    multiplier = np.linalg.lstsq(matrix[:, active].T, -gradient[active], rcond=None)[0]
    reduced_gradient = gradient + matrix.T @ multiplier
    assert matrix @ weights == pytest.approx(
        matrix @ context.benchmark_weight, abs=1e-8
    )
    assert reduced_gradient[active] == pytest.approx(np.zeros(active.sum()), abs=1e-7)
    assert (reduced_gradient[~active] >= -1e-7).all()
    assert (weights >= 0.0).all()


@pytest.mark.parametrize("missing", ["absent", "null", "zero", "negative", "industry"])
def test_missing_neutral_context_carries_account_and_skips_new_target(
    missing,
):
    from quant_research_platform.research.portfolios.production import _build_session
    from tests.research.test_portfolio_decision_execution import _decision_inputs

    inputs = _decision_inputs()
    inputs["candidate"] = replace(
        inputs["candidate"],
        construction="quantile_equal_weight",
        top_quantile=0.5,
        benchmark_id=None,
        risk_style_columns=(),
        rebalance_schedule="weekly-last-trading-day",
        neutrality="industry" if missing == "industry" else "size",
    )
    inputs["enhanced_decision"] = None
    if missing == "absent":
        inputs["dataset"] = inputs["dataset"].drop(columns="float_market_cap")
    elif missing != "industry":
        inputs["dataset"]["float_market_cap"] = {
            "null": np.nan,
            "zero": 0.0,
            "negative": -1.0,
        }[missing]
    account, _, _, _, ledger = _build_session(**inputs)
    assert account.nav_after == 100.0
    assert account.post_trade_cash == 0.0
    assert ledger.executed_delta_notional.tolist() == [0.0, 0.0]
    assert ledger.post_trade_notional.tolist() == [50.0, 50.0]
    diagnostics = ledger.attrs["neutrality_exposures"]
    for kind in ("target", "realized"):
        assert diagnostics[f"{kind}_exposure_status"] == "UNDEFINED"
        assert (
            diagnostics[f"{kind}_exposure_reason"]
            == "target_neutrality_benchmark_empty"
        )
        assert diagnostics[f"{kind}_exposure_excluded_weight"] is None
    assert not any(key.startswith("benchmark_exposure:") for key in diagnostics)
    inputs["candidate"] = replace(inputs["candidate"], rebalance_schedule="daily")
    account, _, optimized, _, ledger = _build_session(**inputs)
    assert optimized is None
    assert account.nav_after == 100.0
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "target_neutrality_benchmark_empty"
    )
    assert (
        ledger.attrs["neutrality_exposures"]["realized_exposure_status"] == "UNDEFINED"
    )


def test_enhanced_nonrebalance_unknown_industry_preserves_held_account(
    tmp_path, monkeypatch
):
    from tests.research.test_portfolio_production import (
        RISK_STYLE_COLUMNS,
        _build_producer,
        _inputs,
    )

    frames, fees, window = _inputs()
    frames["panel"] = frames["panel"].assign(
        period_return=0.0,
        total_market_cap=1000.0,
        float_market_cap=1000.0,
        industry_standard_id="fixture",
        industry_level1_code="known",
        is_nonfinancial_industry=True,
    )
    frames["panel"].loc[
        frames["panel"].timestamp.eq("2023-01-04"), "industry_level1_code"
    ] = pd.NA
    frames["benchmark"] = frames["panel"][["asset_id", "timestamp"]].assign(
        benchmark_id="csi500", weight=0.5
    )
    candidate = PortfolioCandidateSpec(
        "enhanced",
        "global",
        "benchmark_enhanced",
        "champion",
        "weekly-last-trading-day",
        None,
        "csi500",
        RISK_STYLE_COLUMNS,
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        typed_batches=True,
    )
    batches = list(batches)
    matched = []
    for batch in batches:
        facts = {
            row.name: row.text
            for row in batch.diagnostics
            if row.asset_id == "__cash__"
        }
        if facts.get("target_exposure_reason") == "portfolio_industry_context_missing":
            matched.append(batch)
            assert facts["target_exposure_status"] == "UNDEFINED"
            assert facts["realized_exposure_status"] == "UNDEFINED"
            assert not batch.trades and not batch.cashflows
            before = {
                row.asset_id: row.quantity
                for row in batch.valuations
                if row.phase == "before"
            }
            after = {
                row.asset_id: row.quantity
                for row in batch.valuations
                if row.phase == "after"
            }
            assert before.keys() == after.keys()
            for asset in before:
                if asset == "__cash__":
                    assert before[asset] == pytest.approx(
                        after[asset], abs=1e-10, rel=0.0
                    )
                else:
                    assert before[asset] == after[asset]
    assert len(matched) == 1


@pytest.mark.parametrize("schedule", ["daily", "weekly-last-trading-day"])
def test_unheld_missing_cap_does_not_make_known_exposures_undefined(schedule):
    from quant_research_platform.research.portfolios.production import _build_session
    from tests.research.test_portfolio_decision_execution import _decision_inputs

    inputs = _decision_inputs(benchmark=(0.5, 0.5, 0.0))
    inputs["candidate"] = replace(
        inputs["candidate"],
        construction="quantile_equal_weight",
        top_quantile=0.5,
        benchmark_id=None,
        risk_style_columns=(),
        rebalance_schedule=schedule,
        neutrality="size",
    )
    inputs["enhanced_decision"] = None
    inputs["dataset"].loc[inputs["dataset"].asset_id.eq("c"), "float_market_cap"] = (
        np.nan
    )
    account, _, _, _, ledger = _build_session(**inputs)
    assert account.nav_after == 100.0
    assert ledger.attrs["neutrality_exposures"]["realized_exposure_status"] == "DEFINED"
    assert ledger.attrs["neutrality_exposures"]["target_exposure_status"] == "DEFINED"
    assert not ledger.loc[ledger.asset_id.eq("c"), "post_trade_notional"].any()


def test_absent_cap_column_is_local_to_actual_scheduled_neutral_target(
    tmp_path, monkeypatch
):
    from tests.research.test_portfolio_production import _build_producer, _inputs

    frames, fees, _ = _inputs()
    frames["dataset"] = frames["dataset"].drop(columns="float_market_cap")
    candidate = PortfolioCandidateSpec(
        "missing-cap",
        "global",
        "quantile_equal_weight",
        "ridge",
        "weekly-last-trading-day",
        0.5,
        None,
        neutrality="size",
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=TimeWindow("2023-01-03", "2023-01-06"),
        candidate=candidate,
        typed_batches=True,
        model_inputs=("total_market_cap",),
    )
    selected = list(batches)
    assert len(selected) == 2
    for batch in selected:
        assert not batch.trades
        assert all(
            row.value == 100.0 for row in batch.valuations if row.asset_id == "__cash__"
        )
        assert any(
            row.name == "realized_exposure_reason"
            and row.text == "target_neutrality_benchmark_empty"
            for row in batch.diagnostics
        )
    longer, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=TimeWindow("2023-01-03", "2023-01-10"),
        candidate=candidate,
        typed_batches=True,
        model_inputs=("total_market_cap",),
    )
    remaining = list(longer)
    assert remaining
    assert all(not batch.trades for batch in remaining)
    assert any(
        row.name == "decision_skip_reason"
        and row.text == "target_neutrality_benchmark_empty"
        for batch in remaining
        for row in batch.diagnostics
    )
