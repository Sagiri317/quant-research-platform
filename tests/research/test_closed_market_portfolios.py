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
from tests.helpers.portfolio_inputs import closed_market_inputs as _native_inputs
from tests.helpers.portfolio_ledger import session_frame


def _ledger(layout, roots):
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="closed_market_portfolio",
        construction="quantile_equal_weight",
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
        candidate=candidate,
        inputs=layout,
        output_window=TimeWindow("2022-12-30", "2023-01-06"),
        input_window=TimeWindow("2022-12-01", "2023-01-06"),
        formal_dataset_window=TimeWindow("2022-12-01", "2023-01-06"),
        prediction_lookup=Predictions(),
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
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
    capital = PortfolioCapital("closed_market_portfolio", 100.0, "CNY", "cash_equity")
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
    capital = PortfolioCapital("closed_market_portfolio", 100.0, "CNY", "cash_equity")
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
    assert any(item.name == "research_valuation_policy" for item in facts)
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
    assert raw[phase] == 12.0
    assert pd.isna(raw["close" if phase == "open" else "open"])
    assert partial.valuation.set_index("asset_id").loc["a0", phase] == 12.0

    marks = production._ResearchValuationMarks()
    projected = {
        view.timestamp: marks.project(view).set_index("asset_id") for view in views
    }
    current = projected[pd.Timestamp("2023-01-05")].loc["a0"]
    assert current[phase] == 12.0
    other = "close" if phase == "open" else "open"
    # Missing open uses only yesterday's 11, never the later real close of 12.
    # A real open of 12 may anchor the later missing close.
    assert current[other] == (12.0 if phase == "open" else 11.0)
    assert pd.notna(current[f"research_{other}_mark_at"])
    assert pd.isna(current[f"research_{phase}_mark_at"])
    next_day = projected[pd.Timestamp("2023-01-06")].loc["a0"]
    assert next_day.open == next_day.close == 12.0
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


def test_closed_calendar_validates_only_consumed_history(tmp_path):
    from dataclasses import fields

    original, _, calendar, _ = _native_inputs(tmp_path / "original")
    conflicted, _, _, _ = _native_inputs(
        tmp_path / "conflicted", fault="actual_activity_on_closed"
    )

    def sessions(layout, calendar, window):
        with closing_iterator(
            market_sessions.iter_portfolio_sessions(
                market_bars_root=layout["market_bars"],
                stock_master_root=layout["stock_master"],
                panel_root=layout["panel"],
                panel_columns=production._PANEL_COLUMNS,
                calendar=calendar,
                time_window=window,
            )
        ) as source:
            return list(source)

    historical = TimeWindow("2022-12-01", "2022-12-30")
    expected = sessions(original, calendar, historical)
    assert len(expected) == 22
    for layout in (original, conflicted):
        for name in ("market_bars", "stock_master", "panel"):
            unrelated = layout[name] / "data/year=2030"
            unrelated.mkdir()
            (unrelated / "unselected.parquet").write_bytes(b"unselected defect")
    duplicate_future_calendar = pd.concat(
        (calendar, calendar.loc[calendar.calendar_date.eq("2023-01-03")]),
        ignore_index=True,
    )
    for layout, supplied_calendar, reason in (
        (conflicted, calendar, "market_bar_session_missing_from_calendar"),
        (original, duplicate_future_calendar, "duplicate_trading_calendar_key"),
    ):
        actual = sessions(layout, supplied_calendar, historical)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            for field in fields(left):
                observed, control = (
                    getattr(left, field.name),
                    getattr(right, field.name),
                )
                if isinstance(observed, pd.DataFrame):
                    pd.testing.assert_frame_equal(observed, control)
                else:
                    assert observed == control, field.name
        with pytest.raises(ValueError, match=f"^{reason}$"):
            sessions(layout, supplied_calendar, TimeWindow("2022-12-30", "2023-01-04"))

    missing_endpoint = calendar.loc[calendar.calendar_date.ne("2023-01-06")]
    with pytest.raises(ValueError, match="^market_bar_session_missing_from_calendar$"):
        sessions(original, missing_endpoint, TimeWindow("2022-12-30", "2023-01-06"))


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
    capital = PortfolioCapital("closed_market_portfolio", 100.0, "CNY", "cash_equity")
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


def test_empty_research_valuation_keeps_boolean_lifecycle_masks():
    frame = pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in {
                "asset_id": "string",
                "listing_date": "datetime64[ns]",
                "timestamp": "datetime64[ns]",
                "market_type": "Int64",
                "open": "float64",
                "close": "float64",
                "is_trading": "boolean",
            }.items()
        }
    )
    view = market_sessions._PortfolioSession(
        timestamp=pd.Timestamp("2023-01-03"),
        panel=frame.copy(),
        valuation=frame.copy(),
        closed_assets=frozenset(),
        execution=pd.DataFrame(),
        unproved_lifecycle_assets=frozenset(),
        confirmed_delisted_lifecycles=frozenset(),
        unknown_market_assets=frozenset(),
        rights_events=pd.DataFrame(),
    )
    marks = production._ResearchValuationMarks()
    actual = marks.project(view)
    assert actual.empty
    assert {"asset_id", "open", "close"}.issubset(actual.columns)
    assert marks.last.empty
