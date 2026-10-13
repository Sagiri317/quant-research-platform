import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import (
    portfolio_sessions as market_sessions,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    prepare_trade_cost,
    require_fee_coverage,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.diagnostics import (
    iter_holding_factor_inputs,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    holding_attribution,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.portfolio_inputs import (
    _interim_specs,
    _processed_specs,
)
from tests.helpers.portfolio_ledger import session_frame, write_market_evidence
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.test_portfolio_production import _inputs


def _native_inputs(tmp_path, *, fault=None, open_assets=6):
    dates = pd.bdate_range("2022-12-01", "2023-01-06")
    closed = pd.DatetimeIndex(
        []
        if fault in {"normal_open", "confirmed_suspension", "adjacent_relisting"}
        else ["2023-01-03", "2023-01-04"]
    )
    assets = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(open_assets)]
    panel = pd.DataFrame(
        [
            {
                "listing_date": dates[0],
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": day,
                "market_type": 1 if asset.startswith("a") else 4,
                "adj_open": 121.0
                if asset.startswith("a") and day >= pd.Timestamp("2023-01-05")
                else 100.0,
                "adj_close": 110.0
                if asset.startswith("a") and day == pd.Timestamp("2023-01-02")
                else 121.0
                if asset.startswith("a") and day >= pd.Timestamp("2023-01-05")
                else 100.0,
                "amount": 1000.0,
                "period_return": 0.0,
                "float_market_cap": 1000.0,
                "industry_standard_id": "fixture",
                "industry_level1_code": "one",
                "is_nonfinancial_industry": True,
                "is_a_share_scope": True,
                "is_trading": True,
                "is_st": False,
                "is_pt": False,
                "listing_age_trading_days": 300,
                "has_252_trading_sessions": True,
                "close": 100.0,
                "volume": 1000.0,
                "turnover": 0.01,
            }
            for day in dates
            for asset in assets
        ]
    )
    bars = panel[["asset_id", "timestamp", "market_type"]].rename(
        columns={"asset_id": "source_asset_id"}
    )
    bars["filling_flag"] = 0
    holiday = bars.source_asset_id.str.startswith("a") & bars.timestamp.isin(closed)
    bars.loc[holiday, "filling_flag"] = 2
    bars["adj_close"] = 999.0
    panel = panel.loc[
        ~(panel.asset_id.str.startswith("a") & panel.timestamp.isin(closed))
    ].copy()
    calendar = pd.DataFrame(
        [
            {
                "timestamp": day,
                "calendar_date": day,
                "source_market_type": market,
                "is_trading_day": day in dates and not (market == 1 and day in closed),
            }
            for day in pd.date_range(dates[0], dates[-1])
            for market in (1, 4)
        ]
    )
    stocks = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": dates[0],
            "listing_date": dates[0],
            "delisting_date": pd.NaT,
            "company_status": "active",
        }
    )
    if fault == "actual_activity_on_closed":
        bars.loc[holiday, "filling_flag"] = 0
    elif fault == "unknown_barrier":
        bars.loc[
            bars.source_asset_id.eq("a0")
            & bars.timestamp.eq(pd.Timestamp("2023-01-02")),
            "filling_flag",
        ] = pd.NA
    elif fault == "missing_calendar":
        calendar = calendar.loc[
            ~(calendar.source_market_type.eq(1) & calendar.calendar_date.eq(closed[0]))
        ]
    elif fault == "unknown_membership":
        bars.loc[
            bars.source_asset_id.eq("a0") & bars.timestamp.eq(closed[0]), "market_type"
        ] = pd.NA
    elif fault == "delisted":
        stocks.loc[stocks.source_asset_id.eq("a0"), "delisting_date"] = pd.Timestamp(
            "2023-01-02"
        )
    elif fault == "future_lifecycle":
        stocks.loc[stocks.source_asset_id.eq("a0"), ["timestamp", "listing_date"]] = (
            pd.Timestamp("2023-01-05")
        )
    elif fault == "missing_open_price":
        panel.loc[
            panel.asset_id.eq("a0") & panel.timestamp.eq(pd.Timestamp("2023-01-02")),
            "adj_open",
        ] = np.nan
    elif fault == "missing_open_row":
        panel = panel.loc[
            ~(panel.asset_id.eq("a0") & panel.timestamp.eq(pd.Timestamp("2023-01-02")))
        ]
    if fault == "closed_only_axis":
        panel = panel.loc[panel.asset_id.str.startswith("a")].copy()
        bars = bars.loc[bars.source_asset_id.str.startswith("a")].copy()
        stocks = stocks.loc[stocks.source_asset_id.str.startswith("a")].copy()
    if fault == "new_lifecycle":
        new = stocks.loc[stocks.source_asset_id.eq("a0")].assign(
            timestamp=closed[0], listing_date=closed[0]
        )
        stocks.loc[stocks.source_asset_id.eq("a0"), "delisting_date"] = closed[
            0
        ] - pd.Timedelta(days=1)
        stocks = pd.concat((stocks, new), ignore_index=True)
    if fault == "future_lifecycle_append":
        selected = stocks.source_asset_id.eq("a0")
        new = stocks.loc[selected].assign(
            timestamp=pd.Timestamp("2023-01-09"),
            listing_date=pd.Timestamp("2023-01-09"),
        )
        stocks.loc[selected, "delisting_date"] = pd.Timestamp("2023-01-08")
        stocks = pd.concat((stocks, new), ignore_index=True)
    if fault == "unknown_before_suspension":
        calendar = calendar.loc[
            ~(calendar.source_market_type.eq(1) & calendar.calendar_date.eq(closed[0]))
        ].copy()
        calendar.loc[
            calendar.source_market_type.eq(4) & calendar.calendar_date.eq(closed[0]),
            "is_trading_day",
        ] = False
        calendar.loc[
            calendar.source_market_type.eq(1) & calendar.calendar_date.eq(closed[1]),
            "is_trading_day",
        ] = True
        bars = bars.loc[
            ~(bars.source_asset_id.str.startswith("b") & bars.timestamp.eq(closed[0]))
        ].copy()
        bars.loc[
            bars.source_asset_id.str.startswith("a") & bars.timestamp.eq(closed[1]),
            "filling_flag",
        ] = 1
        suspended = panel.loc[
            panel.asset_id.str.startswith("a")
            & panel.timestamp.eq(pd.Timestamp("2023-01-02"))
        ].assign(
            timestamp=closed[1],
            adj_open=110.0,
            adj_close=110.0,
            is_trading=False,
            amount=0.0,
            volume=0.0,
        )
        panel = pd.concat(
            (panel.loc[~panel.timestamp.eq(closed[0])], suspended), ignore_index=True
        )
    if fault == "confirmed_suspension":
        suspension_dates = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
        suspended = panel.asset_id.str.startswith("a") & panel.timestamp.isin(
            suspension_dates
        )
        bars.loc[
            bars.source_asset_id.str.startswith("a")
            & bars.timestamp.isin(suspension_dates),
            "filling_flag",
        ] = 1
        panel.loc[suspended, ["adj_open", "adj_close"]] = 110.0
        panel.loc[suspended, ["amount", "volume"]] = 0.0
        panel.loc[suspended, "is_trading"] = False
    if fault == "missing_decision_industry":
        panel.loc[
            panel.asset_id.eq("a0") & panel.timestamp.eq(pd.Timestamp("2023-01-02")),
            "industry_level1_code",
        ] = pd.NA
    if fault == "adjacent_relisting":
        boundary = dates[-1]
        changed = stocks.source_asset_id.str.startswith("a")
        new = stocks.loc[changed].assign(timestamp=boundary, listing_date=boundary)
        stocks.loc[changed, "delisting_date"] = dates[-2]
        stocks = pd.concat((stocks, new), ignore_index=True)
        rows = panel.asset_id.str.startswith("a") & panel.timestamp.eq(boundary)
        panel.loc[rows, "listing_date"] = boundary
        panel.loc[panel.asset_id.str.startswith("a"), ["adj_open", "adj_close"]] = 10.0
        panel.loc[rows, ["adj_open", "adj_close"]] = 100.0
        panel.loc[rows, "has_252_trading_sessions"] = False
        panel.loc[rows, "listing_age_trading_days"] = 1
    if fault in {"partial_open_quote", "partial_close_quote"}:
        selected = panel.asset_id.eq("a0")
        panel.loc[selected, ["adj_open", "adj_close"]] = 10.0
        panel.loc[selected & panel.timestamp.eq("2023-01-02"), "adj_close"] = 11.0
        partial = selected & panel.timestamp.eq("2023-01-05")
        panel.loc[partial, ["adj_open", "adj_close"]] = np.nan
        phase = "open" if fault == "partial_open_quote" else "close"
        panel.loc[partial, f"adj_{phase}"] = 12.0
        panel.loc[
            selected & panel.timestamp.eq("2023-01-06"), ["adj_open", "adj_close"]
        ] = np.nan
    execution = panel[["market_id", "asset_id", "timestamp"]].assign(
        can_enter=True, can_exit=True
    )
    initial = execution.timestamp.eq(
        pd.Timestamp("2022-12-30")
    ) & execution.asset_id.isin(["a3", "a4", "a5"])
    if fault != "adjacent_relisting":
        execution.loc[initial, "can_enter"] = False
    if fault == "unknown_reopen_mask":
        execution["can_exit"] = execution.can_exit.astype("boolean")
        execution.loc[
            execution.asset_id.eq("a0")
            & execution.timestamp.eq(pd.Timestamp("2023-01-02")),
            "can_exit",
        ] = pd.NA
    data = panel[["market_id", "asset_id", "timestamp", "float_market_cap"]].assign(
        eligible=True, signal=1.0, label=0.0
    )
    roots = write_market_evidence(tmp_path, panel, bars=bars, stocks=stocks)
    specs = _processed_specs()
    dataset_spec = make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=("signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    for target, frame, spec in (
        ("market_panel_1d", panel, specs.get("market_panel_1d")),
        ("execution_mask", execution, specs.get("execution_mask")),
        ("trading_calendar", calendar, _interim_specs().get("trading_calendar")),
        (
            "a_share_fee_schedule",
            _inputs()[1].assign(
                included_in=lambda frame: frame.included_in.astype("string"),
                effective_date=pd.Timestamp("2022-12-01"),
            ),
            _interim_specs().get("a_share_fee_schedule"),
        ),
        ("fixture", data, dataset_spec),
    ):
        root = tmp_path / target
        write_dataset(root, frame, spec=spec)
        roots[target] = root
    inputs = {
        "dataset": roots["fixture"],
        "panel": roots["market_panel_1d"],
        "execution": roots["execution_mask"],
        "calendar": roots["trading_calendar"],
        "fees": roots["a_share_fee_schedule"],
        "market_bars": roots["market_bars_1d"],
        "stock_master": roots["stock_master"],
        "risk_context": (),
    }
    return inputs, roots, calendar, panel


def _ledger(layout, roots):
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="closed_market_portfolio",
        construction="top20_equal_weight",
        rebalance_schedule="daily",
        top_quantile=0.5,
        benchmark_id=None,
    )

    class Predictions:
        def values_for_frame(self, frame, **kwargs):
            return [
                10.0 if str(asset).startswith("a") else 1.0 for asset in frame.asset_id
            ], [True] * len(frame)

        def coverage_counts(self):
            return 0, 0

    return production.backtest_a_share(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        candidate=candidate,
        inputs=layout,
        output_window=TimeWindow("2022-12-30", "2023-01-06"),
        input_window=TimeWindow("2022-12-01", "2023-01-06"),
        formal_dataset_window=TimeWindow("2022-12-01", "2023-01-06"),
        prediction_lookup=Predictions(),
        prediction_domains=(
            (
                TimeWindow("2022-12-30", "2023-01-06"),
                ("signal",),
                "equal_weight_rank_ensemble",
            ),
        ),
        sample_role="validation",
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        fee_schedule_from_frame=fee_schedule_from_frame,
        require_fee_coverage=require_fee_coverage,
        prepare_trade_cost=prepare_trade_cost,
        market_name_for_type=lambda market: "shanghai" if market == 1 else "shenzhen",
        unified_trading_sessions=unified_trading_sessions,
    )


def test_closed_market_ledger_reopens_on_current_target_and_keeps_native_endpoints(
    tmp_path,
):
    layout, roots, calendar, panel = _native_inputs(tmp_path)
    # Synthetic identities isolate scientific payload validation from model fitting.
    chunks = list(_ledger(layout, roots))
    assert len(chunks) == 4
    frames = [session_frame(batch).set_index("asset_id") for batch in chunks]
    for frame, expected in zip(frames[:3], (0.1, 0.0, 0.1), strict=True):
        assert frame.loc[
            ["a0", "a1", "a2"], "holding_return"
        ].tolist() == pytest.approx([expected] * 3)
    for batch in chunks[1:3]:
        assert all(not trade.asset_id.startswith("a") for trade in batch.trades)
        assert all(not flow.asset_id.startswith("a") for flow in batch.cashflows)
    assert any(
        trade.asset_id.startswith("b") and trade.quantity > 0
        for trade in chunks[2].trades
    )
    assert frames[1].loc["__cash__", "post_trade_notional"] >= 0.0
    assert frames[1].loc[["a0", "a1", "a2"], "adv20"].tolist() == [1000.0] * 3
    assert all(
        trade.asset_id not in {"a3", "a4", "a5"}
        for batch in chunks
        for trade in batch.trades
    )
    assert any(
        trade.asset_id == "a0" and trade.quantity < 0 for trade in chunks[3].trades
    )
    for before, after in zip(frames[:-1], frames[1:], strict=True):
        assert before.end_notional.sum() == pytest.approx(
            after.pre_trade_notional.sum()
        )
    capital = PortfolioCapital(
        "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
    )
    metadata = {
        "window": {"start": "2022-12-30", "end": "2023-01-06"},
        "sample_role": "validation",
    }

    def produce(emit):
        for batch in chunks:
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "native-ledger", capital=capital, metadata=metadata, produce=produce
    )
    saved = read_portfolio_ledger_artifact(root)
    assert list(saved.iter_chunks()) == chunks
    from dataclasses import replace
    from hashlib import sha256

    before = {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    rejected = replace(
        chunks[0],
        trades=(
            replace(chunks[0].trades[0], notional=chunks[0].trades[0].notional * 2),
            *chunks[0].trades[1:],
        ),
    )
    capital = PortfolioCapital(
        "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
    )
    metadata = {
        "window": {"start": "2022-12-30", "end": "2023-01-06"},
        "sample_role": "validation",
    }
    with pytest.raises(ValueError, match="^portfolio_ledger_direction_invalid$"):
        write_portfolio_ledger_artifact(
            tmp_path / "rejected-ledger",
            capital=capital,
            metadata=metadata,
            produce=lambda emit: emit(rejected),
        )
    assert before == {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert list(saved.iter_chunks()) == chunks
    inputs = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        inputs=layout,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
        unified_trading_sessions=unified_trading_sessions,
        style_columns=(),
        benchmark_required=False,
        risk_config=FactorRiskConfig(minimum_cross_section=10),
    )
    with closing_iterator(inputs):
        inputs = list(inputs)
    assert len(inputs) == 4
    for expected, actual in zip(frames, inputs, strict=True):
        held = expected.post_trade_notional.drop("__cash__")
        held = held[held.gt(0)]
        assert actual.entry_open.reindex(held.index).notna().all()
        values, reasons, exposures = holding_attribution(
            actual,
            after_notionals=held,
            nav_before=float(expected.pre_trade_notional.sum()),
            nav_post=float(expected.post_trade_notional.sum()),
            benchmark_id=None,
            benchmark_return=None,
            holding_returns=expected.holding_return.drop("__cash__"),
        )
        assert "holding_return_ledger_mismatch" not in reasons.values()
        if actual.decision_at == pd.Timestamp("2023-01-03"):
            assert actual.reason == "holding_factor_fit_unidentified"
            assert reasons


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_barrier",
        "missing_calendar",
        "unknown_membership",
        "delisted",
        "missing_open_price",
        "missing_open_row",
        "unknown_reopen_mask",
        "unknown_before_suspension",
    ],
)
def test_research_valuation_retains_unquoted_rights_without_claiming_execution(
    tmp_path, fault
):
    layout, roots, _calendar, _panel = _native_inputs(tmp_path, fault=fault)
    batches = list(_ledger(layout, roots))
    assert batches
    facts = [item for batch in batches for item in batch.diagnostics]
    assert any(item.name == "research_valuation_version" for item in facts)
    if fault == "unknown_reopen_mask":
        assert any(
            item.name == "block_reason" and item.text == "execution_fact_unknown"
            for item in facts
        )
    else:
        assert any(
            item.name in {"research_entry_mark_at", "research_end_mark_at"}
            for item in facts
        )
    for batch in batches:
        frame = session_frame(batch).set_index("asset_id")
        assert np.isfinite(frame.end_notional).all()
        assert frame.loc["__cash__", "end_notional"] >= 0.0
        if any(
            item.asset_id == "a0" and item.name == "research_entry_mark_at"
            for item in batch.diagnostics
        ):
            assert all(trade.asset_id != "a0" for trade in batch.trades)


@pytest.mark.parametrize("fault", ["future_lifecycle", "new_lifecycle"])
def test_research_valuation_still_blocks_unproved_listing_life(tmp_path, fault):
    layout, roots, _calendar, _panel = _native_inputs(tmp_path, fault=fault)
    with pytest.raises(PortfolioTargetBlocked):
        list(_ledger(layout, roots))


@pytest.mark.parametrize("phase", ["open", "close"])
def test_native_partial_quote_becomes_phase_ordered_research_anchor(tmp_path, phase):
    layout, roots, calendar, _panel = _native_inputs(
        tmp_path, fault=f"partial_{phase}_quote"
    )
    with closing_iterator(
        market_sessions.iter_portfolio_sessions(
            market_bars_root=layout["market_bars"],
            stock_master_root=layout["stock_master"],
            panel_root=roots["market_panel_1d"],
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2022-12-30", "2023-01-06"),
        )
    ) as source:
        views = list(source)
    partial = next(
        view for view in views if view.timestamp == pd.Timestamp("2023-01-05")
    )
    raw = partial.panel.set_index("asset_id").loc["a0"]
    assert raw[f"adj_{phase}"] == 12.0
    assert pd.isna(raw[f"adj_{'close' if phase == 'open' else 'open'}"])
    assert partial.valuation.set_index("asset_id").loc["a0", f"adj_{phase}"] == 12.0

    marks = production._ResearchValuationMarks()
    projected = {
        view.timestamp: marks.project(view).set_index("asset_id") for view in views
    }
    current = projected[pd.Timestamp("2023-01-05")].loc["a0"]
    assert current[f"adj_{phase}"] == 12.0
    other = "close" if phase == "open" else "open"
    # Missing open uses only yesterday's 11, never the later real close of 12.
    # A real open of 12 may anchor the later missing close.
    assert current[f"adj_{other}"] == (12.0 if phase == "open" else 11.0)
    assert pd.notna(current[f"research_{other}_mark_at"])
    assert pd.isna(current[f"research_{phase}_mark_at"])
    next_day = projected[pd.Timestamp("2023-01-06")].loc["a0"]
    assert next_day.adj_open == next_day.adj_close == 12.0
    assert partial.panel.set_index("asset_id").loc["a0"].equals(raw)
    batches = list(_ledger(layout, roots))
    assert all(np.isfinite(row.value) for batch in batches for row in batch.valuations)
    if phase == "close":
        assert not any(
            trade.asset_id == "a0"
            and trade.timestamp_ns == pd.Timestamp("2023-01-05").value
            for batch in batches
            for trade in batch.trades
        )
    else:
        assert all(
            row.quantity == 0
            for row in batches[-1].valuations
            if row.asset_id == "a0" and row.phase != "before"
        )


def test_future_lifecycle_append_cannot_change_previous_ledger_prefix(tmp_path):
    original, original_roots, calendar, _panel = _native_inputs(tmp_path / "original")
    appended, appended_roots, _calendar, _panel = _native_inputs(
        tmp_path / "appended", fault="future_lifecycle_append"
    )
    assert list(_ledger(original, original_roots)) == list(
        _ledger(appended, appended_roots)
    )
    for layout in (original, appended):
        with closing_iterator(
            market_sessions.iter_portfolio_sessions(
                market_bars_root=layout["market_bars"],
                stock_master_root=layout["stock_master"],
                panel_root=layout["panel"],
                panel_columns=production._PANEL_COLUMNS,
                calendar=calendar,
                time_window=TimeWindow("2022-12-30", "2023-01-06"),
            )
        ) as source:
            assert all(not view.unproved_lifecycle_assets for view in source)


def test_closed_calendar_rejects_real_activity_even_in_future_source(tmp_path):
    layout, _roots, calendar, _panel = _native_inputs(
        tmp_path, fault="actual_activity_on_closed"
    )
    source = market_sessions.iter_portfolio_sessions(
        market_bars_root=layout["market_bars"],
        stock_master_root=layout["stock_master"],
        panel_root=layout["panel"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2022-12-01", "2022-12-30"),
    )
    with pytest.raises(ValueError, match="^market_bar_session_missing_from_calendar$"):
        next(source)


def test_identified_open_factor_domain_does_not_invent_closed_exposures(tmp_path):
    from dataclasses import replace

    layout, _roots, calendar, _panel = _native_inputs(tmp_path, open_assets=24)
    source = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        inputs=layout,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
        unified_trading_sessions=unified_trading_sessions,
        style_columns=(),
        benchmark_required=False,
        risk_config=FactorRiskConfig(minimum_cross_section=10),
    )
    with closing_iterator(source):
        all_inputs = list(source)
    prior_open = next(
        item for item in all_inputs if item.decision_at == pd.Timestamp("2023-01-02")
    )
    assert prior_open.fitted is not None and len(prior_open.fitted.exposures) == 24
    assert (
        "a0" not in prior_open.fitted.exposures.index
        and "a0" in prior_open.closed_assets
    )
    inputs = next(
        item for item in all_inputs if item.decision_at == pd.Timestamp("2023-01-03")
    )
    assert inputs.fitted is not None
    assert len(inputs.fitted.exposures) == 24
    assert "a0" in inputs.closed_assets and "a0" not in inputs.fitted.exposures.index

    def reasons_for(value, asset="a0"):
        return holding_attribution(
            value,
            after_notionals=pd.Series({asset: 1.0}),
            nav_before=1.0,
            nav_post=1.0,
            benchmark_id=None,
            benchmark_return=None,
            holding_returns=pd.Series({asset: 0.1}),
        )[1]

    assert set(reasons_for(inputs).values()) == {
        "holding_closed_market_domain_unidentified"
    }
    assert set(reasons_for(replace(inputs, closed_assets=frozenset())).values()) == {
        "holding_exposures_incomplete"
    }
    assert set(reasons_for(inputs, "ordinary_missing").values()) == {
        "holding_exposures_incomplete"
    }


def test_union_session_keeps_closed_holdings_without_any_raw_asset_rows(tmp_path):
    layout, roots, calendar, _panel = _native_inputs(tmp_path, fault="closed_only_axis")
    chunks = list(_ledger(layout, roots))
    assert len(chunks) == 4
    for batch in chunks[1:3]:
        assert not batch.trades and not batch.cashflows
    assert any(
        trade.asset_id.startswith("a") and trade.quantity < 0
        for trade in chunks[3].trades
    )
    stream = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        inputs=layout,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
        unified_trading_sessions=unified_trading_sessions,
        style_columns=(),
        benchmark_required=False,
        risk_config=FactorRiskConfig(minimum_cross_section=10),
    )
    with closing_iterator(stream):
        inputs = list(stream)
    assert len(inputs) == 4
    assert inputs[2].entry_open.notna().all()


def test_closed_endpoints_do_not_hide_missing_decision_risk_facts(tmp_path):
    layout, _roots, calendar, _panel = _native_inputs(
        tmp_path, fault="missing_decision_industry", open_assets=24
    )
    source = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        inputs=layout,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
        unified_trading_sessions=unified_trading_sessions,
        style_columns=(),
        benchmark_required=False,
        risk_config=FactorRiskConfig(minimum_cross_section=10),
    )
    with closing_iterator(source):
        inputs = next(
            item for item in source if item.decision_at == pd.Timestamp("2023-01-02")
        )
    assert inputs.fitted is not None and "a0" not in inputs.closed_assets
    reasons = holding_attribution(
        inputs,
        after_notionals=pd.Series({"a0": 1.0}),
        nav_before=1.0,
        nav_post=1.0,
        benchmark_id=None,
        benchmark_return=None,
        holding_returns=pd.Series({"a0": 0.0}),
    )[1]
    assert set(reasons.values()) == {"holding_exposures_incomplete"}


def test_confirmed_suspension_rows_remain_in_real_factor_pair_domain(tmp_path):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.types import DatasetReadMode

    layout, roots, calendar, _panel = _native_inputs(
        tmp_path, fault="confirmed_suspension", open_assets=24
    )
    dates = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
    for target, asset_column, value_column, expected in (
        ("market_bars_1d", "source_asset_id", "filling_flag", 1),
        ("market_panel_1d", "asset_id", "is_trading", False),
        ("market_panel_1d", "asset_id", "amount", 0.0),
    ):
        frame = read_dataset(
            roots[target],
            columns=(asset_column, "timestamp", value_column),
            mode=DatasetReadMode.MATERIALIZED,
        )
        suspended = frame.loc[
            frame[asset_column].eq("a0") & frame.timestamp.isin(dates)
        ]
        assert len(suspended) == 2 and suspended[value_column].eq(expected).all()
    source = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        inputs=layout,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
        unified_trading_sessions=unified_trading_sessions,
        style_columns=(),
        benchmark_required=False,
        risk_config=FactorRiskConfig(minimum_cross_section=10),
    )
    with closing_iterator(source):
        inputs = next(
            item for item in source if item.decision_at == pd.Timestamp("2023-01-02")
        )
    assert inputs.fitted is not None and len(inputs.fitted.exposures) == 30
    assert inputs.fitted.asset_returns.loc["a0"] == 0.0
    assert "a0" not in inputs.closed_assets


def test_asset_ordered_source_parts_preserve_portfolio_sessions(tmp_path):
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets.reading import (
        materialize_dataset_session,
        open_dataset_session,
    )

    layout, roots, calendar, _panel = _native_inputs(tmp_path)

    def sessions(selected_layout):
        with closing_iterator(
            market_sessions.iter_portfolio_sessions(
                market_bars_root=selected_layout["market_bars"],
                stock_master_root=selected_layout["stock_master"],
                panel_root=roots["market_panel_1d"],
                panel_columns=production._PANEL_COLUMNS,
                calendar=calendar,
                time_window=TimeWindow("2022-12-30", "2023-01-06"),
                execution_root=roots["execution_mask"],
            )
        ) as frames:
            return list(frames)

    expected = sessions(layout)
    bars = materialize_dataset_session(
        open_dataset_session(roots["market_bars_1d"], columns=None, time_window=None)
    )
    reordered = roots["market_bars_1d"].parent / ("b" * 64)
    write_dataset(
        reordered,
        (
            group
            for _, group in bars.sort_values(["source_asset_id", "timestamp"]).groupby(
                "source_asset_id", sort=False
            )
        ),
        spec=interim_catalog().get("market_bars_1d"),
    )
    actual = sessions({**layout, "market_bars": reordered})
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.timestamp == right.timestamp
        assert left.closed_assets == right.closed_assets
        for name in ("panel", "valuation", "execution"):
            pd.testing.assert_frame_equal(getattr(left, name), getattr(right, name))


def test_native_ledger_rejects_terminal_relisting_and_preserves_published_artifact(
    tmp_path,
):
    from hashlib import sha256

    layout, roots, _, _ = _native_inputs(tmp_path / "control")
    capital = PortfolioCapital(
        "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
    )
    metadata = {
        "window": {"start": "2022-12-30", "end": "2023-01-06"},
        "sample_role": "validation",
    }

    def produce(emit):
        with closing_iterator(_ledger(layout, roots)) as chunks:
            for batch in chunks:
                emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    saved = read_portfolio_ledger_artifact(root)
    good = list(saved.iter_chunks())
    assert good
    hashes = {
        p.relative_to(root): sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    layout, roots, _, _ = _native_inputs(
        tmp_path / "relisted", fault="adjacent_relisting"
    )
    with pytest.raises(
        PortfolioTargetBlocked, match="portfolio_held_lifecycle_unproven"
    ):
        write_portfolio_ledger_artifact(
            tmp_path / "rejected-ledger",
            capital=capital,
            metadata=metadata,
            produce=produce,
        )
    assert hashes == {
        p.relative_to(root): sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert list(read_portfolio_ledger_artifact(root).iter_chunks()) == good
