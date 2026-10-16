from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_ENHANCED,
    PORTFOLIO_DECISION_REASON_MAINTENANCE,
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import (
    portfolio_sessions as market_sessions,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    prepare_trade_cost,
)
from quant_research_platform.research.portfolios import accounting, production
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskEstimate,
    FactorRiskState,
)
from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
from quant_research_platform.storage.datasets import DatasetReadMode
from quant_research_platform.storage.datasets import reading as dataset_reading
from quant_research_platform.storage.identity import filesystem_mkdir
from tests.helpers.portfolio_ledger import (
    CONTINUOUS_EXECUTION_COLUMNS,
    execute_continuous_target,
    ledger_frame,
    session_frame,
    write_test_sessions,
)


def test_enhanced_decision_requires_history_for_current_industry_domain() -> None:
    state = FactorRiskState(
        style_columns=("style",),
        config=FactorRiskConfig(
            minimum_cross_section=10,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
            factor_window=3,
            specific_window=3,
        ),
    )
    x = np.arange(20, dtype="float64")
    cross = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"a{index}" for index in range(20)],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "total_market_cap": 100 + x,
            "float_market_cap": 100 + x,
            "style": x,
            "asset_return": np.sin(x) / 100,
        }
    )
    for day in range(3):
        state.observe_return_session(
            cross.assign(timestamp=pd.Timestamp("2020-01-01") + pd.Timedelta(days=day))
        )
    previous = state.estimate()
    assert tuple(previous.factor_covariance.columns) == ("industry_CSRC2012:A", "style")
    cross.loc[cross.index >= 10, "industry_level1_code"] = "B"
    scheduled = pd.DataFrame(
        {
            column: pd.Series(dtype="datetime64[ns]")
            for column in ("timestamp", "entry_at", "available_at")
        }
    )

    def decision():
        return production._enhanced_decision(
            benchmark_weights=pd.Series(1 / len(cross), index=cross.asset_id),
            alpha=pd.Series(0.0, index=cross.asset_id),
            factor_risk=state,
            current_cross=cross,
            calibration_rows=[],
            calibration_intervals=scheduled,
            as_of=pd.Timestamp("2020-01-07"),
            risk_style_columns=("style",),
        )

    assert decision().risk is None
    for day in range(3, 6):
        state.observe_return_session(
            cross.assign(timestamp=pd.Timestamp("2020-01-01") + pd.Timedelta(days=day))
        )
    current = decision()
    assert current.risk is not None
    assert tuple(current.risk.factor_covariance.columns) == (
        "industry_CSRC2012:A",
        "industry_CSRC2012:B",
        "style",
    )
    assert current.risk.factor_returns.timestamp.min() == pd.Timestamp("2020-01-04")
    assert tuple(previous.factor_covariance.columns) == ("industry_CSRC2012:A", "style")


class _PredictionLookup:
    def __init__(self, *, missing: bool = False, extra: bool = False) -> None:
        self.missing = missing
        self.extra = extra
        self.matched: list[tuple[pd.Timestamp, str]] = []

    def values_for_frame(self, frame: pd.DataFrame, *, mark_matched: bool):
        if mark_matched:
            self.matched.extend(
                (
                    (pd.Timestamp(timestamp), str(asset_id))
                    for timestamp, asset_id in frame.loc[
                        :, ["timestamp", "asset_id"]
                    ].itertuples(index=False, name=None)
                )
            )
        values = frame["asset_id"].map({"a": 0.0, "b": 1.0}).to_numpy()
        found = bytearray([0 if self.missing else 1] * len(values))

        def result():
            return (values, found)

        return result()

    def coverage_counts(self) -> tuple[int, int]:
        return (0, 1 if self.extra else 0)


def _inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, TimeWindow]:
    dates = pd.bdate_range("2022-11-14", "2023-01-10")
    assets = ("a", "b")
    panel = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "listing_date": dates[0],
                "open": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "adj_open": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "adj_close": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "amount": 1000.0,
                "total_market_cap": 1000.0 if asset == "a" else 2000.0,
                "market_type": 1,
                "is_a_share_scope": True,
                "is_trading": True,
                "is_st": False,
                "is_pt": False,
                "listing_age_trading_days": 300,
                "has_252_trading_sessions": True,
                "close": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "volume": 10.0,
                "turnover": 0.01,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    dataset = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "eligible": True,
                "total_market_cap": 1000.0 if asset == "a" else 2000.0,
                "float_market_cap": 1000.0 if asset == "a" else 2000.0,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    execution = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "can_enter": True,
                "can_exit": True,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    fee = pd.DataFrame(
        [
            {
                "fee_code": fee_code,
                "effective_date": pd.Timestamp("2023-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": side,
                "basis": "amount",
                "rate": rate,
                "included_in_ledger": True,
                "included_in": None,
            }
            for fee_code, rate in (
                ("BROKER_COMMISSION", 0.0003),
                ("TRANSFER_FEE", 0.0001),
            )
            for side in ("buy", "sell")
        ]
        + [
            {
                "fee_code": "STAMP_DUTY",
                "effective_date": pd.Timestamp("2023-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": "sell",
                "basis": "amount",
                "rate": 0.0005,
                "included_in_ledger": True,
                "included_in": None,
            }
        ]
    )
    window = TimeWindow("2023-01-03", "2023-01-06")
    return (
        {
            "dataset": dataset,
            "panel": panel,
            "execution": execution,
            "fee": fee,
            "calendar": pd.DataFrame(
                {
                    "calendar_date": dates,
                    "source_market_type": 1,
                    "is_trading_day": True,
                }
            ),
        },
        fee,
        window,
    )


def _build_producer(
    tmp_path: Path,
    monkeypatch,
    *,
    frames: dict[str, pd.DataFrame],
    fee_frame: pd.DataFrame,
    output_window: TimeWindow,
    prediction_lookup: _PredictionLookup | None = None,
    input_window: TimeWindow | None = None,
    formal_dataset_window: TimeWindow | None = None,
    candidate: PortfolioCandidateSpec | None = None,
    prediction_window: TimeWindow | None = None,
    model_inputs: tuple[str, ...] = ("float_market_cap",),
    model_algorithm: str = "equal_weight_rank_ensemble",
    final_calibrations=None,
    frozen_calibration: production.AlphaCalibration | None = None,
    typed_batches: bool = False,
    construction: str = "quantile_equal_weight",
    identity_overrides=None,
    raw_bar_orders=None,
    initial_nav=100.0,
    risk_config=None,
):
    final_calibrations = [] if final_calibrations is None else final_calibrations
    enhanced = candidate is not None and candidate.construction == "benchmark_enhanced"
    if enhanced:
        risk_columns = list(RISK_STYLE_COLUMNS)
        frames = {
            **frames,
            "risk_context": frames["risk_context"]
            if "risk_context" in frames
            else frames["dataset"].reindex(
                columns=["asset_id", "timestamp", *risk_columns]
            ),
            "dataset": frames["dataset"].drop(columns=risk_columns, errors="ignore"),
        }
    if candidate is None:
        candidate = PortfolioCandidateSpec(
            selection_group="global",
            prediction_selector="champion",
            candidate_id="synthetic_quantile_equal_weight",
            construction=construction,
            rebalance_schedule="daily",
            top_quantile=0.5,
            benchmark_id=None,
        )
    frames = dict(frames)
    frames.setdefault(
        "corporate_actions",
        pd.DataFrame(columns=market_sessions._CORPORATE_SOURCE_COLUMNS),
    )
    frames["market_bars_1d"] = (
        frames["panel"]
        .loc[:, ["asset_id", "timestamp", "market_type"]]
        .rename(columns={"asset_id": "source_asset_id"})
        .assign(
            filling_flag=frames["panel"].is_trading.map({True: 0, False: 1}).to_numpy()
        )
    )
    identities = {
        target_id: ("a" if target_id not in {"fee", "calendar"} else "f") * 64
        for target_id in (
            "dataset",
            "panel",
            "execution",
            "fee",
            "calendar",
            "benchmark",
            "risk_context",
            "market_bars_1d",
            "stock_master",
            "corporate_actions",
        )
    }
    if "equity_events" in frames:
        identities["equity_events"] = "e" * 64
    identities.update(identity_overrides or {})
    roots = {}
    for target_id, identity in identities.items():
        root = tmp_path / target_id / identity
        filesystem_mkdir(root, parents=True, exist_ok=True)
        roots[target_id] = root
    calls: list[tuple[str, DatasetReadMode, TimeWindow | None]] = []

    def fake_read_dataset(
        dataset_root: Path,
        *,
        columns=None,
        time_window=None,
        mode=DatasetReadMode.MATERIALIZED,
    ):
        root = dataset_root
        target_id = next(
            (
                target_id
                for target_id, candidate_root in roots.items()
                if root == candidate_root
            )
        )
        calls.append((target_id, mode, time_window))
        if mode is DatasetReadMode.CHUNKED:
            assert isinstance(time_window, TimeWindow)
            selected = frames[target_id].loc[
                frames[target_id].timestamp.between(
                    pd.Timestamp(time_window.start), pd.Timestamp(time_window.end)
                ),
                list(columns),
            ]
            if target_id == "market_bars_1d" and raw_bar_orders is not None:
                raw_bar_orders.append(selected.timestamp.is_monotonic_increasing)
            else:
                selected = selected.sort_values(
                    [
                        "timestamp",
                        "asset_id" if "asset_id" in selected else "source_asset_id",
                    ],
                    kind="mergesort",
                )
            return iter((selected.copy(),))
        if target_id == "fee":
            assert columns == production._FEE_COLUMNS
            assert time_window is None
            return fee_frame.copy()
        if target_id == "calendar":
            assert columns == ("calendar_date", "source_market_type", "is_trading_day")
            assert time_window is None
            return frames["calendar"].copy()
        raise AssertionError(f"Unexpected materialized target: {target_id}")

    def input_session(dataset_root, *, columns=None, time_window=None):
        root = dataset_root
        target = next((key for key, value in roots.items() if value == root))
        if time_window is None and frames[target].empty:
            time_window = TimeWindow("2022-12-01", "2023-01-10")
        if time_window is None:
            time_window = TimeWindow(
                frames[target].timestamp.min(), frames[target].timestamp.max()
            )
        return SimpleNamespace(
            root=root,
            time_window=time_window,
            arrow_schema=SimpleNamespace(names=tuple(frames[target].columns)),
            spec={"column_roles": dict.fromkeys(frames[target].columns, "context")},
            read_paths=(root / "synthetic.parquet",),
            logical_columns=tuple(columns)
            if columns is not None
            else tuple(frames[target].columns),
            physical_columns=tuple(columns)
            if columns is not None
            else tuple(frames[target].columns),
            projected_view=lambda selected: input_session(
                root, columns=selected, time_window=time_window
            ),
            filtered_view=lambda window: input_session(
                root, columns=columns, time_window=window
            ),
        )

    def file_batches(session, *, columns, **_kwargs):
        for frame in fake_read_dataset(
            session.root,
            columns=columns,
            time_window=session.time_window,
            mode=DatasetReadMode.CHUNKED,
        ):
            yield (session.root / "synthetic.parquet", frame)

    def source_evidence(
        market_bars_root,
        stock_master_root,
        panel_root,
        calendar,
        *,
        panel_columns,
        include_asset_sessions,
    ):
        from quant_research_platform.markets.implementations.a_share.processed import (
            panels,
        )

        panel = frames["panel"]
        stocks = (
            panel.groupby("asset_id", sort=False)
            .timestamp.min()
            .rename("listing_date")
            .reset_index()
        )
        stocks["source_asset_id"] = stocks.asset_id
        stocks["timestamp"] = stocks.listing_date
        stocks["delisting_date"] = pd.NaT
        return (
            input_session(roots["market_bars_1d"]),
            input_session(panel_root),
            panel.loc[:, ["asset_id", "timestamp", "market_type"]],
            {
                asset: tuple(sorted(group.timestamp.unique()))
                for asset, group in panel.groupby("asset_id", sort=False)
            },
            panels._canonical_stock_master(frames.get("stock_master", stocks)),
            (
                tuple(sorted(frames["market_bars_1d"].timestamp.unique())),
                frozenset(frames["market_bars_1d"].source_asset_id),
            ),
        )

    def materialize_source(session):
        return pd.concat(
            (
                frame
                for _, frame in file_batches(session, columns=session.logical_columns)
            ),
            ignore_index=True,
        )

    monkeypatch.setattr(
        market_sessions, "materialize_dataset_session", materialize_source
    )
    monkeypatch.setattr(market_sessions, "_portfolio_session_evidence", source_evidence)
    monkeypatch.setattr(market_sessions, "open_dataset_session", input_session)
    monkeypatch.setattr(production, "open_dataset_session", input_session)
    monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", file_batches)
    monkeypatch.setattr(production, "read_dataset", fake_read_dataset)
    assumptions = ExecutionAssumptions(
        initial_nav=initial_nav,
        slippage_bps=0.0,
        impact_bps=0.0,
        adv_participation=1.0,
        adv_sensitivities=(1.0,),
    )

    def output():
        chunks = production.backtest_a_share(
            candidate=candidate,
            inputs={
                "dataset": roots["dataset"],
                "panel": roots["panel"],
                "execution": roots["execution"],
                "calendar": roots["calendar"],
                "fees": roots["fee"],
                "market_bars": roots["market_bars_1d"],
                "stock_master": roots["stock_master"],
                "corporate_actions": roots["corporate_actions"],
                **(
                    {"equity_events": roots["equity_events"]}
                    if "equity_events" in roots
                    else {}
                ),
                "benchmark": roots["benchmark"],
                "risk_context": (roots["risk_context"],),
            },
            output_window=output_window,
            input_window=input_window or TimeWindow("2022-12-01", "2023-01-10"),
            formal_dataset_window=formal_dataset_window
            or TimeWindow("2022-12-01", "2023-01-10"),
            prediction_window=prediction_window,
            prediction_lookup=prediction_lookup or _PredictionLookup(),
            prediction_domains=(
                (
                    prediction_window
                    or formal_dataset_window
                    or TimeWindow("2022-12-01", "2023-01-10"),
                    model_inputs,
                    model_algorithm,
                ),
            ),
            frozen_calibration=frozen_calibration,
            assumptions=assumptions,
            risk_config=risk_config,
            on_final_calibration=final_calibrations.append,
        )
        for batch in chunks:
            yield (batch if typed_batches else session_frame(batch))

    return (output(), calls)


@pytest.mark.parametrize("unknown_unrelated_amount", (False, True))
def test_simple_producer_publishes_one_bounded_ledger_and_reuses_it(
    tmp_path: Path, monkeypatch, unknown_unrelated_amount
) -> None:
    frames, fee_frame, output_window = _inputs()
    if unknown_unrelated_amount:
        frames["panel"].loc[frames["panel"].asset_id.eq("a"), "amount"] = np.nan
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        typed_batches=True,
    )
    original_batches = list(chunks)
    frames_out = [session_frame(batch) for batch in original_batches]
    assert frames_out
    assert all(
        tuple(frame.columns) == CONTINUOUS_EXECUTION_COLUMNS for frame in frames_out
    )
    assert all(frame["asset_id"].iloc[-1] == "__cash__" for frame in frames_out)
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "synthetic_portfolio",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": "2023-01-03",
                "end": "2023-01-06",
                "access": "development",
            }
        },
    }

    def produce(emit):
        for batch in original_batches:
            emit(batch)

    artifact_root = write_portfolio_ledger_artifact(
        tmp_path / "artifacts",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(artifact_root)
    persisted = list(artifact.iter_chunks())
    assert persisted == original_batches
    assert any(batch.trades for batch in persisted)
    for expected, batch in zip(frames_out, persisted, strict=True):
        actual = session_frame(batch)
        np.testing.assert_array_equal(
            actual.decision_delta_weight, expected.decision_delta_weight
        )
        assert abs(float(actual.decision_delta_weight.sum())) < 1e-14
    assert np.isclose(frames_out[0]["post_trade_notional"].sum(), 100.0 - 0.04)
    first = original_batches[0]
    assert first.trades
    damaged = replace(
        first,
        trades=tuple(
            replace(trade, notional=trade.notional * 2) for trade in first.trades
        ),
    )
    with pytest.raises(ValueError, match="^portfolio_ledger_direction_invalid$"):
        write_portfolio_ledger_artifact(
            tmp_path / "damaged_artifacts",
            capital=envelope["capital"],
            metadata=envelope["metadata"],
            produce=lambda emit: emit(damaged),
        )


def test_portfolio_stream_count_does_not_scale_with_decision_dates(
    tmp_path, monkeypatch
):
    from collections import Counter

    frames, fee_frame, output_window = _inputs()
    chunks, calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
    )
    assert list(chunks)
    streams = Counter(
        (target for target, mode, _window in calls if mode is DatasetReadMode.CHUNKED)
    )
    assert streams == {
        "panel": 1,
        "dataset": 1,
        "execution": 1,
        "market_bars_1d": 1,
        "corporate_actions": 1,
    }
    assert all(
        (
            target in {"fee", "calendar"}
            for target, mode, _window in calls
            if mode is DatasetReadMode.MATERIALIZED
        )
    )


def test_benchmark_stream_keeps_future_snapshot_across_missing_sessions(monkeypatch):
    sessions = pd.date_range("2020-01-01", periods=5)
    observed = []

    def snapshots(*args, **kwargs):
        observed.append("open")
        try:
            for timestamp in sessions[[1, 3]]:
                yield pd.DataFrame(
                    {
                        "benchmark_id": ["csi500", "csi500"],
                        "asset_id": ["a", "b"],
                        "timestamp": timestamp,
                        "weight": [0.25, 0.75],
                    }
                )
        finally:
            observed.append("closed")

    monkeypatch.setattr(production, "_iter_input_sessions", snapshots)
    weights = list(production._iter_benchmark_weights(None, "csi500", sessions))
    assert [item.empty for item in weights] == [True, False, True, False, True]
    for item in (weights[1], weights[3]):
        np.testing.assert_array_equal(item.to_numpy(), [0.25, 0.75])
    assert observed == ["open", "closed"]


def test_float_cap_roundoff_keeps_nonnegative_persisted_cash_target(tmp_path):
    assets = pd.Index([f"asset_{index:02}" for index in range(20)])
    caps = pd.Series(np.random.default_rng(4).uniform(1, 100, 20), index=assets)
    target = caps / caps.sum()
    nav = 100000000.0
    assert nav - float((target * nav).sum()) < 0
    zero = pd.Series(0.0, index=assets)
    adv = pd.Series(1000000000000.0, index=assets)
    permitted = pd.Series(True, index=assets)
    _, fee_frame, _ = _inputs()
    fee_schedule = fee_schedule_from_frame(fee_frame, content_identity="f" * 64)
    assumptions = ExecutionAssumptions(nav, 0.0, 0.0, 1.0, (1.0,))
    execution_date = pd.Timestamp("2023-01-04")
    account = execute_continuous_target(
        zero,
        target,
        nav=nav,
        average_daily_volume=adv,
        can_enter=permitted,
        can_exit=permitted,
        execution_date=execution_date.date(),
        cost_for_trade=lambda timestamp, _asset, side, amount, volume, settings: (
            prepare_trade_cost(
                timestamp, "shanghai", side, fee_schedule, settings
            ).trade(amount, volume)
        ),
        assumptions=assumptions,
        cost_source_identity="f" * 64,
    )
    frame = ledger_frame(
        account=account,
        decision_timestamp=pd.Timestamp("2023-01-03"),
        returns=zero,
        prediction_rank_score=caps,
        adv20=adv,
        enhanced_decision=None,
        benchmark_weights=None,
        optimization_result=None,
        decision_reason=PORTFOLIO_DECISION_REASON_SIMPLE,
        decision_delta=target,
    )
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "float_cap_ledger",
                "initial_capital": nav,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": "2023-01-03",
                "end": "2023-01-04",
                "access": "development",
            }
        },
    }
    root = write_test_sessions(
        tmp_path / "ledger",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        ledger_chunks=iter((frame,)),
    )
    saved = read_portfolio_ledger_artifact(root)
    (batch,) = list(saved.iter_chunks())
    cash_target = next(
        row.number
        for row in batch.diagnostics
        if row.asset_id == "__cash__" and row.name == "target_notional"
    )
    assert cash_target == 0.0
    assert account.nav_after == pytest.approx(
        nav + sum(row.amount for row in batch.cashflows if row.kind != "principal")
    )


def test_enhanced_producer_uses_benchmark_only_when_risk_is_insufficient(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    frames["panel"]["period_return"] = 0.001
    frames["panel"]["float_market_cap"] = frames["panel"]["asset_id"].map(
        {"a": 1000.0, "b": 2000.0}
    )
    frames["panel"]["industry_standard_id"] = "CSRC2012"
    frames["panel"]["industry_level1_code"] = "A01"
    frames["panel"]["is_nonfinancial_industry"] = True
    dates = pd.DatetimeIndex(frames["calendar"]["calendar_date"]).normalize()
    snapshots = dates[
        (dates >= pd.Timestamp(output_window.start))
        & (dates <= pd.Timestamp(output_window.end))
    ]
    frames["benchmark"] = pd.DataFrame(
        [
            {
                "benchmark_id": "csi500",
                "asset_id": asset,
                "timestamp": timestamp,
                "weight": 0.5,
            }
            for timestamp in snapshots
            for asset in ("a", "b")
        ]
    )
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        candidate=candidate,
    )
    result = list(chunks)
    assert result
    first = result[0]
    assert first["asset_id"].tolist() == ["a", "b", "__cash__"]
    assert first["benchmark_weight"].iloc[:2].tolist() == pytest.approx([0.5, 0.5])
    assert (
        first.loc[first["asset_id"].eq("__cash__"), "decision_reason"]
        .eq("enhanced_benchmark_only")
        .all()
    )
    assert first["pre_trade_notional"].iloc[:2].tolist() == pytest.approx([0.0, 0.0])
    assert first["target_notional"].iloc[:2].tolist() == pytest.approx([50.0, 50.0])
    assert first.loc[first["asset_id"].eq("__cash__"), "post_trade_notional"].iloc[
        0
    ] == pytest.approx(0.0)


@pytest.mark.parametrize("missing_session", [None, "2023-01-03", "2023-01-04"])
def test_enhanced_holdout_reads_research_sessions_from_prediction_start(
    tmp_path, monkeypatch, missing_session
):
    frames, fee_frame, output_window = _inputs()
    frames["panel"]["period_return"] = 0.001
    frames["panel"]["float_market_cap"] = frames["panel"].asset_id.map(
        {"a": 1000.0, "b": 2000.0}
    )
    frames["panel"]["industry_standard_id"] = "CSRC2012"
    frames["panel"]["industry_level1_code"] = "A01"
    frames["panel"]["is_nonfinancial_industry"] = True
    frames["risk_context"] = frames["dataset"].loc[:, ["asset_id", "timestamp"]].copy()
    for column in RISK_STYLE_COLUMNS:
        frames["risk_context"][column] = frames["risk_context"].asset_id.map(
            {"a": -0.2, "b": 0.2}
        )
    frames["dataset"] = (
        frames["dataset"]
        .loc[
            frames["dataset"].timestamp.between(
                pd.Timestamp(output_window.start), pd.Timestamp(output_window.end)
            )
        ]
        .copy()
    )
    if missing_session is not None:
        frames["dataset"] = (
            frames["dataset"]
            .loc[frames["dataset"].timestamp.ne(pd.Timestamp(missing_session))]
            .copy()
        )
    frames["benchmark"] = pd.DataFrame(
        [
            {
                "benchmark_id": "csi500",
                "asset_id": asset,
                "timestamp": timestamp,
                "weight": 0.5,
            }
            for timestamp in pd.bdate_range(output_window.start, output_window.end)
            for asset in ("a", "b")
        ]
    )
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        rebalance_schedule="daily",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    calibration = production.AlphaCalibration(
        slope=1.0,
        intercept=0.0,
        slope_standard_error=1.0,
        slope_t_statistic=1.0,
        slope_pvalue=0.5,
        slope_ci_low=-1.0,
        slope_ci_high=3.0,
        weighted_r2=0.0,
        coverage=1.0,
        valid_rebalance_dates=6,
        scheduled_rebalance_dates=6,
        undefined_reason=None,
    )
    risk_dates = []
    observe_return_session = production.FactorRiskState.observe_return_session

    def observe(state, frame, **kwargs):
        risk_dates.append(pd.Timestamp(frame.timestamp.iloc[0]))
        return observe_return_session(state, frame, **kwargs)

    def reject_recalibration(*args, **kwargs):
        pytest.fail("holdout must retain its frozen alpha calibration")

    monkeypatch.setattr(production.FactorRiskState, "observe_return_session", observe)
    monkeypatch.setattr(production, "calibrate_expected_alpha", reject_recalibration)
    lookup = _PredictionLookup()
    final_calibrations = []
    chunks, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        formal_dataset_window=output_window,
        prediction_window=output_window,
        candidate=candidate,
        prediction_lookup=lookup,
        frozen_calibration=calibration,
        final_calibrations=final_calibrations,
    )
    if missing_session is not None:
        with pytest.raises(
            PortfolioTargetBlocked, match="^portfolio_research_dataset_session_missing$"
        ):
            list(chunks)
        assert min(risk_dates) < pd.Timestamp(output_window.start)
        assert max(risk_dates) == pd.Timestamp(missing_session)
        return
    result = list(chunks)
    assert result
    assert min(risk_dates) == pd.Timestamp("2022-12-02")
    assert max(risk_dates) == pd.Timestamp(output_window.end)
    assert set(lookup.matched) == {
        (timestamp, asset)
        for timestamp in pd.bdate_range(output_window.start, output_window.end)
        for asset in ("a", "b")
    }
    assert all(
        frame.loc[frame.asset_id.eq("__cash__"), "alpha_slope"]
        .eq(calibration.slope)
        .all()
        for frame in result
    )
    assert final_calibrations == []


@pytest.mark.parametrize("can_sell_removed", [False, True])
def test_enhanced_membership_maintenance_keeps_explicit_benchmark_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, can_sell_removed: bool
) -> None:
    frames, fee_frame, _window = _inputs()
    output_window = TimeWindow("2023-01-03", "2023-01-06")
    frames["panel"] = frames["panel"].assign(
        open=100.0,
        close=100.0,
        period_return=0.0,
        total_market_cap=1000.0,
        float_market_cap=1000.0,
        industry_standard_id="CSRC2012",
        industry_level1_code="A01",
        is_nonfinancial_industry=True,
    )
    frames["benchmark"] = pd.DataFrame(
        [
            {
                "benchmark_id": "csi500",
                "asset_id": asset,
                "timestamp": pd.Timestamp("2023-01-03"),
                "weight": weight,
            }
            for asset, weight in (("a", 0.1), ("b", 0.9))
        ]
        + [
            {
                "benchmark_id": "csi500",
                "asset_id": "b",
                "timestamp": timestamp,
                "weight": 1.0,
            }
            for timestamp in pd.bdate_range("2023-01-04", "2023-01-06")
        ]
    )
    frames["execution"].loc[
        frames["execution"].asset_id.eq("a")
        & frames["execution"].timestamp.eq("2023-01-04"),
        "can_exit",
    ] = can_sell_removed
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="maintenance",
        construction="benchmark_enhanced",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    risk = FactorRiskEstimate(
        factor_returns=pd.DataFrame(),
        factor_covariance=pd.DataFrame(
            np.eye(len(RISK_STYLE_COLUMNS)),
            index=RISK_STYLE_COLUMNS,
            columns=RISK_STYLE_COLUMNS,
        ),
        specific_variance=pd.Series({"a": 0.01, "b": 0.01}),
        factor_half_life=126.0,
        specific_half_life=63.0,
        diagonal_shrinkage=0.2,
        annualization_factor=252.0,
    )
    original_calibrate = production.calibrate_expected_alpha
    estimates = iter([None, risk, risk, risk])
    monkeypatch.setattr(
        production.FactorRiskState, "estimate", lambda _self: next(estimates)
    )
    monkeypatch.setattr(
        production,
        "calibrate_expected_alpha",
        lambda *args, **kwargs: replace(
            original_calibrate(*args, **kwargs), undefined_reason=None, slope=0.01
        ),
    )

    def reject_optimizer(*_args, **_kwargs):
        raise AssertionError("maintenance must execute its explicit benchmark target")

    monkeypatch.setattr(production, "optimize_portfolio", reject_optimizer)
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        candidate=candidate,
    )
    sessions = list(chunks)
    maintenance = next(
        frame
        for frame in sessions
        if frame.decision_timestamp.iloc[0] == pd.Timestamp("2023-01-04")
    )
    indexed = maintenance.set_index("asset_id")
    initial = sessions[0].set_index("asset_id")
    initial_nav = initial.post_trade_notional.sum()
    initial_a = initial.loc["a", "post_trade_notional"]
    initial_b = initial.loc["b", "post_trade_notional"]
    assert initial_nav < 100.0
    assert initial.loc["a", "pre_trade_notional"] == 0.0
    assert initial.loc["b", "pre_trade_notional"] == 0.0
    assert (
        indexed.loc["__cash__", "decision_reason"]
        == PORTFOLIO_DECISION_REASON_MAINTENANCE
    )
    assert indexed.loc["a", "target_notional"] == 0.0
    assert indexed.loc["b", "target_notional"] == pytest.approx(initial_nav)
    assert indexed.loc["a", "decision_delta_weight"] == pytest.approx(
        -initial_a / initial_nav
    )
    assert indexed.loc["b", "decision_delta_weight"] == pytest.approx(
        1.0 - initial_b / initial_nav
    )
    assert indexed.loc["a", "post_trade_notional"] == pytest.approx(
        0.0 if can_sell_removed else initial_a
    )
    assert indexed.loc["__cash__", "post_trade_notional"] >= 0.0
    costs = maintenance.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum()
    assert maintenance.post_trade_notional.sum() + costs == pytest.approx(initial_nav)
    if can_sell_removed:
        assert costs > 0.0
    else:
        assert indexed.loc["b", "executed_delta_notional"] == 0.0
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "maintenance",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": output_window.start.isoformat(),
                "end": output_window.end.isoformat(),
                "access": "development",
            }
        },
    }
    root = write_test_sessions(
        tmp_path / "ledger",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        ledger_chunks=iter(sessions),
    )
    saved = read_portfolio_ledger_artifact(root)
    loaded = [session_frame(batch) for batch in saved.iter_chunks()]
    assert len(loaded) == len(sessions)
    for actual, expected in zip(loaded, sessions, strict=True):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def _enhanced_inputs(monkeypatch):
    frames, fee_frame, output_window = _inputs()
    output_window = TimeWindow("2023-01-03", "2023-01-10")
    frames["panel"]["period_return"] = 0.001
    dates = pd.DatetimeIndex(frames["calendar"]["calendar_date"]).normalize()
    date_positions = {
        timestamp: position
        for position, timestamp in enumerate(dates.drop_duplicates())
    }
    frames["panel"]["open"] = [
        100.0 * 1.001 ** date_positions[timestamp]
        if asset == "a"
        else 100.0 * 1.003 ** date_positions[timestamp]
        for timestamp, asset in zip(
            frames["panel"]["timestamp"], frames["panel"]["asset_id"], strict=True
        )
    ]
    frames["panel"]["close"] = frames["panel"]["open"].copy()
    # This no-action synthetic history has the same raw and adjusted return path.
    frames["panel"]["adj_open"] = frames["panel"]["open"].copy()
    frames["panel"]["adj_close"] = frames["panel"]["close"].copy()
    frames["panel"]["float_market_cap"] = frames["panel"]["asset_id"].map(
        {"a": 1000.0, "b": 2000.0}
    )
    frames["panel"]["industry_standard_id"] = "CSRC2012"
    frames["panel"]["industry_level1_code"] = "A01"
    frames["panel"]["is_nonfinancial_industry"] = True
    style_values = {"a": -0.2, "b": 0.2}
    for column in RISK_STYLE_COLUMNS:
        frames["dataset"][column] = frames["dataset"]["asset_id"].map(style_values)
    snapshots = dates[
        (dates >= pd.Timestamp(output_window.start))
        & (dates <= pd.Timestamp(output_window.end))
    ]
    frames["benchmark"] = pd.DataFrame(
        [
            {
                "benchmark_id": "csi500",
                "asset_id": asset,
                "timestamp": timestamp,
                "weight": 0.5,
            }
            for timestamp in snapshots
            for asset in ("a", "b")
        ]
    )
    exposure_columns = ("industry_CSRC2012:A01", *RISK_STYLE_COLUMNS)
    covariance = pd.DataFrame(
        np.eye(len(exposure_columns)) * 0.01,
        index=exposure_columns,
        columns=exposure_columns,
    )
    risk = FactorRiskEstimate(
        factor_returns=pd.DataFrame(),
        factor_covariance=covariance,
        specific_variance=pd.Series({"a": 0.01, "b": 0.01}),
        factor_half_life=126.0,
        specific_half_life=63.0,
        diagonal_shrinkage=0.2,
        annualization_factor=252.0,
    )
    frames["dataset"]["beta_252d"] = 1.0
    frames["dataset"]["residual_volatility_60d"] = 0.1
    monkeypatch.setattr(production.FactorRiskState, "estimate", lambda self: risk)
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    return frames, fee_frame, output_window, candidate


def test_enhanced_calibration_excludes_research_marks(tmp_path, monkeypatch):
    frames, fees, window, candidate = _enhanced_inputs(monkeypatch)
    gap = pd.Timestamp("2022-12-05")
    endpoint_gap = pd.Timestamp("2023-01-02")
    mask = frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.isin(
        [gap, endpoint_gap]
    )
    frames["panel"].loc[mask, ["open", "close"]] = np.nan
    actual = production._completed_calibration_rows
    checked = {"entry": False, "endpoint": False}

    def observe(pending, panel):
        if panel.timestamp.eq(endpoint_gap).any():
            assert not panel.asset_id.eq("b").any()
            checked["endpoint"] = True
        if pending[2] == pd.Timestamp("2022-12-02"):
            assert "b" not in pending[1].index.get_level_values("asset_id")
            checked["entry"] = True
        return actual(pending, panel)

    monkeypatch.setattr(production, "_completed_calibration_rows", observe)
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
    )
    assert list(chunks)
    assert all(checked.values())


def test_enhanced_initial_allocation_cannot_create_unquoted_rights(
    tmp_path, monkeypatch
):
    frames, fees, window, candidate = _enhanced_inputs(monkeypatch)
    mask = frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.eq("2023-01-04")
    frames["panel"].loc[mask, ["open", "close"]] = np.nan
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        typed_batches=True,
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
    )
    batches = list(chunks)
    first = batches[0]
    assert not any(trade.asset_id == "b" for trade in first.trades)
    assert all(
        float(row.quantity) == 0.0 for row in first.valuations if row.asset_id == "b"
    )
    assert sum(row.value for row in first.valuations if row.phase == "before") == 100.0
    assert any(trade.asset_id == "b" for batch in batches[1:] for trade in batch.trades)


def test_enhanced_producer_reaches_optimizer_and_publishes_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window, candidate = _enhanced_inputs(monkeypatch)
    from quant_research_platform.contracts.portfolios import (
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )

    observed = []
    actual_build = production._build_session

    def observe_decision(**kwargs):
        account, adv, optimization, decision_delta, snapshot = actual_build(**kwargs)
        if kwargs["decision_timestamp"] == pd.Timestamp("2023-01-06"):
            assert optimization is not None
            observed.append(
                (
                    kwargs["holdings"].div(kwargs["nav"]).copy(),
                    kwargs["nav"],
                    optimization.weights.copy(),
                    account.nav_before,
                    decision_delta.copy(),
                )
            )
        return (account, adv, optimization, decision_delta, snapshot)

    monkeypatch.setattr(production, "_build_session", observe_decision)
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        input_window=TimeWindow("2022-11-14", "2023-01-10"),
        candidate=candidate,
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
    )
    result = list(chunks)
    optimized = [
        frame
        for frame in result
        if frame["decision_reason"].eq("enhanced_optimizer").any()
    ]
    assert optimized, [
        (frame["decision_timestamp"].iloc[0], frame["decision_reason"].iloc[-1])
        for frame in result
    ]
    session = optimized[-1]
    assert session["expected_alpha"].iloc[:2].notna().all()
    assert session["asset_specific_variance"].iloc[:2].tolist() == pytest.approx(
        [0.01, 0.01]
    )
    assert (
        session.loc[session["asset_id"].eq("__cash__")].iloc[0][
            "predicted_tracking_error"
        ]
        >= 0.0
    )
    assert result[0].pre_trade_notional.sum() == pytest.approx(100.0)
    first = result[0]
    assert (
        first.loc[first.asset_id.eq("__cash__"), "post_trade_notional"].iloc[0] >= 0.0
    )
    assert first.loc[~first.asset_id.eq("__cash__"), "pre_trade_notional"].eq(0.0).all()
    assert (
        first.loc[~first.asset_id.eq("__cash__"), "executed_delta_notional"].sum() > 0.0
    )
    initial_cost = first.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum()
    assert initial_cost > 0.0
    assert first.post_trade_notional.sum() + initial_cost == pytest.approx(100.0)
    for row_frame in result:
        if row_frame is not first and row_frame.decision_timestamp.iloc[
            0
        ] != pd.Timestamp("2023-01-06"):
            assert np.allclose(
                row_frame.executed_delta_notional, 0.0, rtol=0.0, atol=1e-12
            )
            assert np.allclose(
                row_frame.decision_delta_weight, 0.0, rtol=0.0, atol=1e-12
            )
            assert (
                row_frame.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)]
                .eq(0.0)
                .all()
                .all()
            )
    assert optimized[-1].loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum() > 0.0
    changed = {name: value.copy(deep=True) for name, value in frames.items()}
    future = changed["panel"].timestamp.eq(pd.Timestamp("2023-01-09")) & changed[
        "panel"
    ].asset_id.eq("a")
    changed["panel"].loc[future, "open"] *= 1.5
    second, _ = _build_producer(
        tmp_path / "future",
        monkeypatch,
        frames=changed,
        fee_frame=fee_frame,
        output_window=output_window,
        input_window=TimeWindow("2022-11-14", "2023-01-10"),
        candidate=candidate,
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
    )
    list(second)
    assert len(observed) == 2
    pd.testing.assert_series_equal(observed[0][0], observed[1][0], check_exact=True)
    assert observed[0][1] == observed[1][1]
    pd.testing.assert_series_equal(observed[0][2], observed[1][2], check_exact=True)
    assert observed[0][3] != observed[1][3]
    pd.testing.assert_series_equal(observed[0][4], observed[1][4], check_exact=True)
    with monkeypatch.context() as patch:
        patch.setattr(production, "_build_session", actual_build)
        initial, _ = _build_producer(
            tmp_path / "initial",
            patch,
            frames=frames,
            fee_frame=fee_frame,
            output_window=TimeWindow("2023-01-06", "2023-01-10"),
            input_window=TimeWindow("2022-11-14", "2023-01-10"),
            candidate=candidate,
            prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
        )
        first_active = list(initial)[0]
        assert first_active.pre_trade_notional.sum() == pytest.approx(100.0)
        assert first_active.loc[
            first_active.asset_id.ne("__cash__"), "pre_trade_notional"
        ].tolist() == pytest.approx([0.0, 0.0])
        assert (
            first_active.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum() > 0.0
        )
        original_calibrate = production.calibrate_expected_alpha
        calibrations = {}

        def bounded_calibrate(frame, **kwargs):
            assert (
                kwargs["scheduled_intervals"]
                .timestamp.ge(pd.Timestamp("2022-12-01"))
                .all()
            )
            calibrated = original_calibrate(frame, **kwargs)
            calibrations[pd.Timestamp(kwargs["as_of"])] = calibrated
            return calibrated

        patch.setattr(production, "calibrate_expected_alpha", bounded_calibrate)
        lookup = _PredictionLookup()
        short, _ = _build_producer(
            tmp_path / "short",
            patch,
            frames=frames,
            fee_frame=fee_frame,
            output_window=output_window,
            input_window=TimeWindow("2022-11-14", "2023-01-10"),
            candidate=candidate,
            prediction_window=TimeWindow("2022-12-01", "2023-01-10"),
            prediction_lookup=lookup,
        )
        list(short)
        decision_calibration = calibrations[pd.Timestamp("2023-01-06")]
        final_calibration = calibrations[pd.Timestamp(output_window.end)]
        assert decision_calibration.scheduled_rebalance_dates == 4
        assert decision_calibration.coverage == 1.0
        assert decision_calibration.benchmark_only
        assert final_calibration.scheduled_rebalance_dates == 5
        assert final_calibration.coverage == 1.0
        assert final_calibration.benchmark_only
        assert min((timestamp for timestamp, _asset in lookup.matched)) >= pd.Timestamp(
            "2022-12-01"
        )
    completed = []
    primary = KeyboardInterrupt("calibration_state_interrupted")
    actual_complete = production._completed_calibration_rows

    def observed_completion(*args, **kwargs):
        rows = actual_complete(*args, **kwargs)
        completed.append(len(rows))
        if len(completed) == 2:
            rows = None
            raise primary
        return rows

    with monkeypatch.context() as patch:
        patch.setattr(production, "_build_session", actual_build)
        patch.setattr(production, "_completed_calibration_rows", observed_completion)
        failing, _ = _build_producer(
            tmp_path / "interrupt",
            patch,
            frames=frames,
            fee_frame=fee_frame,
            output_window=output_window,
            input_window=TimeWindow("2022-11-14", "2023-01-10"),
            candidate=candidate,
            prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
        )
        with pytest.raises(KeyboardInterrupt) as caught:
            list(failing)
        assert completed and caught.value is primary


@pytest.mark.parametrize("missing_prediction", [False, True])
@pytest.mark.parametrize("removed_member", [False, True])
def test_optimizer_assets_keep_residual_cash_as_explicit_current_weight(
    removed_member: bool, missing_prediction: bool
) -> None:
    assets = pd.Index(
        ("a", "b", "old") if removed_member else ("a", "b"), dtype="string"
    )
    exposures = pd.DataFrame(
        {
            "asset_id": assets,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A01",
            **{
                column: np.linspace(0.1, -0.1, len(assets))
                for column in RISK_STYLE_COLUMNS
            },
        }
    )
    result = production._optimizer_assets(
        benchmark_weights=pd.Series(0.5, index=assets[:2]),
        current_weights=pd.Series(
            [0.4, 0.4, 0.1] if removed_member else [0.45, 0.45], index=assets
        ),
        cash=10.0,
        nav=100.0,
        expected_alpha=pd.Series(
            [np.nan if missing_prediction else 0.01, *[0.01] * (len(assets) - 1)],
            index=assets,
        ),
        exposures=exposures,
        specific_variance=pd.Series(0.01, index=assets),
        adv=pd.Series(1000.0, index=assets),
        eligible=pd.Series(True, index=assets),
        panel=pd.DataFrame({"asset_id": assets, "is_nonfinancial_industry": True}),
    )
    assert result["current_weight"].sum() == pytest.approx(1.0)
    cash_row = result.loc[result["asset_id"].eq("__cash__")]
    assert len(cash_row) == 1
    assert cash_row["current_weight"].iloc[0] == pytest.approx(0.1)
    assert cash_row["is_cash"].iloc[0]
    if removed_member:
        old = result.set_index("asset_id").loc["old"]
        assert old.current_weight == pytest.approx(0.1)
        assert old.benchmark_weight == 0.0
        optimized = production.optimize_portfolio(
            result,
            factor_covariance=pd.DataFrame(
                np.eye(len(RISK_STYLE_COLUMNS)) * 0.01,
                index=RISK_STYLE_COLUMNS,
                columns=RISK_STYLE_COLUMNS,
            ),
            style_columns=RISK_STYLE_COLUMNS,
            expected_transaction_cost=lambda _weights: 0.0,
            config=production.OptimizerConfig(),
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
            eligibility_column="eligible",
            financial_column="is_financial",
            cash_column="is_cash",
        )
        assert optimized.weights.loc["old"] == 0.0
        if missing_prediction:
            assert optimized.weights.loc["a"] == 0.5
            assert pd.isna(
                result.loc[result.asset_id.eq("a"), "expected_alpha"].iloc[0]
            )
        assert optimized.weights.sum() == pytest.approx(1.0)
        assert optimized.target_turnover <= 0.2 + 1e-08


@pytest.mark.parametrize("retain_cash", [False, True])
@pytest.mark.parametrize(
    "optimizer_error",
    (None, "optimizer_trade_adv_unavailable", "optimizer_failed_closed"),
)
def test_optimizer_explicit_cash_maps_to_stock_execution_and_persisted_cash(
    tmp_path: Path, retain_cash: bool, optimizer_error, monkeypatch
) -> None:
    frames, fee_frame, _window = _inputs()
    panel = frames["panel"].assign(
        open=100.0,
        close=100.0,
        period_return=0.0,
        total_market_cap=1000.0,
        float_market_cap=1000.0,
        industry_standard_id="CSRC2012",
        industry_level1_code="A01",
        is_nonfinancial_industry=True,
    )

    def at_date(frame, day):
        return frame.loc[frame.timestamp.eq(pd.Timestamp(day))].copy()

    alpha = pd.Series({"a": -0.1 if retain_cash else 0.1, "b": -0.2})
    risk = FactorRiskEstimate(
        pd.DataFrame(),
        pd.DataFrame(
            np.zeros((2, 2)),
            index=("industry_CSRC2012:A01", "style"),
            columns=("industry_CSRC2012:A01", "style"),
        ),
        pd.Series({"a": 0.01, "b": 0.01}),
        126.0,
        63.0,
        0.2,
        252.0,
    )
    enhanced = production._EnhancedDecision(
        pd.Series({"a": 0.5, "b": 0.5}),
        alpha,
        pd.DataFrame(
            {"asset_id": ["a", "b"], "industry_CSRC2012:A01": 1.0, "style": [0.1, -0.1]}
        ),
        risk,
        production.AlphaCalibration(
            1.0, 0.0, 1.0, 1.0, 0.5, -1.0, 1.0, 0.0, 1.0, 6, 6, None
        ),
    )
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        candidate_id="cash_mapping",
        construction="benchmark_enhanced",
        prediction_selector="champion",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=("style",),
    )
    assumptions = ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,))

    def build():
        return production._build_session(
            candidate=candidate,
            decision_timestamp=pd.Timestamp("2023-01-06"),
            next_timestamp=pd.Timestamp("2023-01-09"),
            panel=production._portfolio_session_frame(
                at_date(panel, "2023-01-06"), production._ENHANCED_PANEL_COLUMNS
            ),
            next_panel=at_date(panel, "2023-01-09"),
            endpoint_panel=at_date(panel, "2023-01-10"),
            closed_assets=frozenset(),
            dataset=production._portfolio_session_frame(
                at_date(frames["dataset"], "2023-01-06"), production._DATASET_COLUMNS
            ),
            execution=production._portfolio_session_frame(
                at_date(frames["execution"], "2023-01-06"),
                market_sessions._EXECUTION_COLUMNS,
            ).assign(execution_at=pd.Timestamp("2023-01-09")),
            holdings=pd.Series({"a": 45.0, "b": 45.0}),
            recovery_cash=0.0,
            decision_receivable=0.0,
            receivable=0.0,
            locked_quantities={},
            held_units=pd.Series({"a": 45.0, "b": 45.0})
            / at_date(panel, "2023-01-06").set_index("asset_id").close,
            cash=10.0,
            nav=100.0,
            amount_history={
                (asset, panel.listing_date.iloc[0]): deque([1000.0] * 20, maxlen=20)
                for asset in ("a", "b")
            },
            alpha=alpha,
            enhanced_decision=enhanced,
            target_override=None,
            decision_reason=PORTFOLIO_DECISION_REASON_ENHANCED,
            assumptions=assumptions,
            fee_schedule=fee_schedule_from_frame(fee_frame, content_identity="f" * 64),
            fee_identity="f" * 64,
            prepare_trade_cost=prepare_trade_cost,
            market_name_for_type=lambda _market: "shanghai",
        )

    if optimizer_error is not None:

        def fail_optimizer(*args, **kwargs):
            raise ValueError(optimizer_error)

        monkeypatch.setattr(production, "optimize_portfolio", fail_optimizer)
        if optimizer_error == "optimizer_trade_adv_unavailable":
            account, _adv, optimized, delta, snapshot = build()
            assert optimized is None
            assert delta.eq(0.0).all()
            assert account.asset_ledger.executed_delta_notional.eq(0.0).all()
            assert account.asset_ledger.post_trade_notional.tolist() == [45.0, 45.0]
            assert account.post_trade_cash == 10.0
            assert snapshot.attrs["decision_availability"] == {
                "decision_status": "SKIPPED",
                "decision_skip_reason": optimizer_error,
                "missing_prediction_count": 0.0,
            }
        else:
            with pytest.raises(ValueError, match=f"^{optimizer_error}$"):
                build()
        return
    account, adv, optimized, delta, _snapshot = build()
    assert optimized is not None and "__cash__" in optimized.weights
    assert "__cash__" not in set(account.asset_ledger.asset_id)
    if retain_cash:
        assert optimized.weights.loc["__cash__"] > 0.0
    ledger = ledger_frame(
        account=account,
        decision_timestamp=pd.Timestamp("2023-01-06"),
        returns=pd.Series(0.0, index=account.asset_ledger.asset_id),
        prediction_rank_score=alpha,
        adv20=adv,
        enhanced_decision=enhanced,
        benchmark_weights=enhanced.benchmark_weights,
        optimization_result=optimized,
        decision_reason=PORTFOLIO_DECISION_REASON_ENHANCED,
        decision_delta=delta,
    )
    cash_row = ledger.loc[ledger.asset_id.eq("__cash__")].iloc[0]
    assert cash_row.target_notional == pytest.approx(
        optimized.weights.loc["__cash__"] * 100.0
    )
    assert cash_row.post_trade_notional >= 0.0
    assert ledger.post_trade_notional.sum() + ledger.loc[
        :, list(PORTFOLIO_LEDGER_COST_COLUMNS)
    ].sum().sum() == pytest.approx(100.0)
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "cash_mapping",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": "2023-01-06",
                "end": "2023-01-10",
                "access": "development",
            }
        },
    }
    root = write_test_sessions(
        tmp_path / "ledger",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        ledger_chunks=iter((ledger,)),
    )
    saved = read_portfolio_ledger_artifact(root)
    (batch,) = tuple(saved.iter_chunks())
    assert set(session_frame(batch).asset_id) == {"a", "b", "__cash__"}


def test_simple_producer_terminal_sessions_are_truncated_after_prediction_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
    )
    result = list(output)
    assert result
    assert max(
        pd.to_datetime(frame["decision_timestamp"]).max() for frame in result
    ) <= pd.Timestamp("2023-01-04")


@pytest.mark.parametrize(
    ("lookup", "error"),
    ((_PredictionLookup(extra=True), "misaligned_persisted_prediction_keys"),),
)
def test_simple_producer_prediction_coverage_fails_closed(
    tmp_path: Path, monkeypatch, lookup: _PredictionLookup, error: str
) -> None:
    frames, fee_frame, output_window = _inputs()
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        prediction_lookup=lookup,
    )
    with pytest.raises(ValueError, match=error):
        list(output)


@pytest.mark.parametrize("unknown_quote", (False, True))
def test_simple_producer_unknown_held_quote_carries_without_fabricated_trade(
    tmp_path: Path, monkeypatch, unknown_quote
) -> None:
    frames, fee_frame, output_window = _inputs()
    unknown = frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.eq(
        pd.Timestamp("2023-01-05")
    )
    if unknown_quote:
        frames["panel"].loc[unknown, ["amount", "open", "close"]] = np.nan
    else:
        frames["panel"] = frames["panel"].loc[~unknown].copy()
        for key in ("dataset", "execution"):
            frames[key] = (
                frames[key]
                .loc[
                    ~(
                        frames[key].asset_id.eq("b")
                        & frames[key].timestamp.eq(pd.Timestamp("2023-01-05"))
                    )
                ]
                .copy()
            )
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        typed_batches=True,
    )
    batches = list(output)
    assert batches
    marked = [
        row
        for batch in batches
        for row in batch.diagnostics
        if row.name.endswith("mark_kind")
    ]
    assert any(row.text == "ordinary_missing_quote" for row in marked)


def _mark_view(
    day,
    opening,
    closing,
    *,
    present=True,
    life="2020-01-01",
    trading=True,
    closed=False,
):
    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": [pd.Timestamp(day)],
            "listing_date": [pd.Timestamp(life)],
            "market_type": pd.Series([1], dtype="Int64"),
            "open": [opening],
            "close": [closing],
            "is_trading": pd.Series([trading], dtype="boolean"),
        }
    )
    if not present:
        frame = frame.iloc[:0].copy()
    return SimpleNamespace(
        timestamp=pd.Timestamp(day),
        panel=frame,
        valuation=frame.copy(),
        unproved_lifecycle_assets=frozenset(),
        confirmed_delisted_lifecycles=frozenset(),
        rights_events=pd.DataFrame(),
        corporate_events=pd.DataFrame(),
        closed_assets=frozenset({"a"}) if closed else frozenset(),
    )


def test_research_marks_are_phase_ordered_and_preserve_life_and_source():
    marks = production._ResearchValuationMarks()
    first = marks.project(_mark_view("2023-01-04", 10.0, 11.0))
    missing = _mark_view("2023-01-05", np.nan, np.nan, trading=False)
    original = missing.panel.copy(deep=True)
    gap = marks.project(missing)
    absent = marks.project(
        _mark_view("2023-01-06", np.nan, np.nan, present=False, closed=True)
    )
    restored = marks.project(_mark_view("2023-01-09", 12.0, 12.0))
    pd.testing.assert_frame_equal(missing.panel, original)
    assert first.open.iloc[0] == 10.0
    assert gap.open.iloc[0] == absent.open.iloc[0] == 11.0
    assert gap.research_open_mark_at.iloc[0] == pd.Timestamp("2023-01-04")
    assert gap.research_open_mark_kind.iloc[0] == "close"
    assert restored.open.iloc[0] == 12.0
    assert restored.research_open_mark_at.isna().all()
    assert gap.open.iloc[0] == 11.0
    new_life = marks.project(
        _mark_view("2023-01-10", np.nan, np.nan, life="2023-01-10")
    )
    assert new_life.open.isna().all()
    assert marks.last.empty


def test_missing_open_cannot_borrow_its_future_close():
    marks = production._ResearchValuationMarks()
    marks.project(_mark_view("2023-01-04", 10.0, 11.0))
    output = marks.project(_mark_view("2023-01-05", np.nan, 12.0))
    assert output.open.iloc[0] == 11.0
    assert output.close.iloc[0] == 12.0
    assert output.research_open_mark_kind.iloc[0] == "ordinary_missing_quote"
    fresh = production._ResearchValuationMarks().project(
        _mark_view("2023-01-05", np.nan, 12.0)
    )
    assert fresh.open.isna().all()


@pytest.mark.parametrize("invalid", (0.0, -1.0, np.inf, -np.inf))
def test_research_marks_reject_invalid_raw_price_before_projection(invalid):
    marks = production._ResearchValuationMarks()
    marks.project(_mark_view("2023-01-04", 10.0, 11.0))
    view = _mark_view("2023-01-05", invalid, np.nan)
    view.valuation.loc[:, ["open", "close"]] = np.nan
    with pytest.raises(ValueError, match="portfolio_return_invalid"):
        marks.project(view)


def test_multiple_unquoted_sessions_native_roundtrip_and_absolute_nav_stresses(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.portfolios.diagnostics import (
        _research_valuation_observation,
    )

    frames, fee, _window = _inputs()
    window = TimeWindow("2023-01-03", "2023-01-09")
    fee["rate"] = 0.0
    b = frames["panel"].asset_id.eq("b")
    frames["panel"].loc[b, ["open", "close"]] = 10.0
    frames["panel"].loc[b & frames["panel"].timestamp.eq("2023-01-04"), "close"] = 11.0
    frames["panel"].loc[
        b & frames["panel"].timestamp.ge("2023-01-09"), ["open", "close"]
    ] = 12.0
    suspended = frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.isin(
        pd.to_datetime(["2023-01-05", "2023-01-06"])
    )
    frames["panel"].loc[suspended, ["open", "close", "amount"]] = np.nan
    frames["panel"].loc[suspended, "is_trading"] = False
    frames["dataset"].loc[suspended, "eligible"] = False
    frames["execution"].loc[suspended, ["can_enter", "can_exit"]] = False
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee,
        output_window=window,
        typed_batches=True,
    )
    batches = list(chunks)
    assert len(batches) == 3
    rows = [session_frame(batch).set_index("asset_id") for batch in batches]
    assert [row.loc["b", "holding_return"] for row in rows] == pytest.approx(
        [0.1, 0.0, 12 / 11 - 1]
    )
    assert rows[-1].loc["b", "end_notional"] == pytest.approx(120.0)
    assert all(
        row.loc["__cash__", "end_notional"] == pytest.approx(0.0) for row in rows
    )
    assert all(not batch.trades and not batch.cashflows for batch in batches[1:])
    observations = [
        _research_valuation_observation(
            batch, accounting.portfolio_interval_economics(batch)
        )
        for batch in batches
    ]
    assert [row["nav_0pct"] for row in observations] == pytest.approx([0.0, 0.0, 120.0])
    assert [row["nav_50pct"] for row in observations] == pytest.approx(
        [55.0, 55.0, 120.0]
    )
    assert [row["nav_100pct"] for row in observations] == pytest.approx(
        [110.0, 110.0, 120.0]
    )
    assert observations[1]["maximum_mark_age_days"] == 2
    assert observations[2]["estimated_held_assets"] == 1
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "synthetic_research_valuation",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "access": "development",
            }
        },
    }

    def produce(emit):
        for batch in batches:
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "estimated",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    assert list(artifact.iter_chunks()) == batches

    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
    from tests.helpers.research_configuration import research_configuration
    from tests.helpers.statistics_artifacts import (
        portfolio_statistics_config,
        statistics_metadata,
    )

    configuration = research_configuration()
    configuration["portfolios"] = configuration["portfolios"][:1]
    configuration["primary_portfolio_id"] = configuration["portfolios"][0][
        "candidate_id"
    ]
    candidates, config = portfolio_statistics_config(configuration=configuration)
    ledgers = {}
    for candidate in candidates:
        owned = {
            "capital": PortfolioCapital(
                **{
                    **asdict(envelope["capital"]),
                    "candidate_id": candidate.candidate_id,
                }
            ),
            "metadata": {"window": envelope["metadata"]["window"]},
        }
        owned_root = write_portfolio_ledger_artifact(
            tmp_path / candidate.candidate_id,
            capital=owned["capital"],
            metadata=owned["metadata"],
            produce=produce,
        )
        ledgers[candidate.candidate_id] = read_portfolio_ledger_artifact(owned_root)
    intervals = production.holding_intervals(
        pd.bdate_range(window.start, window.end), window
    )
    holding_inputs = [
        HoldingFactorInputs(
            row.decision_at,
            row.entry_at,
            row.end_at,
            None,
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.DataFrame(columns=("benchmark_id", "asset_id", "weight")),
            "synthetic_no_risk_fit",
        )
        for row in intervals.itertuples(index=False)
    ]
    tables = _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        config=config,
        time_window=window,
        benchmark_opens=pd.DataFrame(columns=("benchmark_id", "timestamp", "open")),
        calendar=pd.DataFrame(
            {"calendar_date": pd.bdate_range(window.start, window.end)}
        ),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        holding_inputs={
            candidate.risk_style_columns: iter(holding_inputs)
            for candidate in candidates
        },
    )
    stat_metadata = statistics_metadata(
        window,
        candidate_ids=[c.candidate_id for c in candidates],
        comparison_portfolio_ids=config["statistics"]["group_comparison"][
            "portfolio_ids"
        ],
    )
    stat_root = write_statistics_artifact(
        tmp_path / "statistics",
        metadata=stat_metadata,
        tables=tables,
        table_group="portfolio",
    )
    saved = read_statistics_artifact(stat_root)
    with saved.table("portfolio_performance.parquet") as performance:
        metrics = performance.loc[
            performance.candidate_id.eq(candidates[0].candidate_id)
        ].set_index("metric_id")
        assert metrics.loc["research_valuation_0pct_max_drawdown", "value"] == -1.0
        assert metrics.loc[
            "research_valuation_50pct_max_drawdown", "value"
        ] == pytest.approx(-0.45)
        assert metrics.loc[
            "research_valuation_100pct_terminal_nav", "value"
        ] == pytest.approx(120.0)
        assert metrics.loc["maximum_estimated_mark_age_days", "value"] == 2
    with saved.table("portfolio_attribution.parquet") as attribution:
        unavailable = attribution.loc[
            ~attribution.component_id.isin(["benchmark", "cost"])
        ]
        assert unavailable.status.eq("UNDEFINED").all()
        assert unavailable.reason.eq("research_estimated_valuation").all()
        assert unavailable.value.isna().all()


def test_simple_producer_keeps_short_adv_history_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        input_window=TimeWindow("2023-01-03", "2023-01-06"),
        formal_dataset_window=TimeWindow("2023-01-03", "2023-01-06"),
    )
    batches = list(output)
    assert batches
    for batch in batches:
        requested = batch.loc[batch.asset_id.eq("b")].iloc[0]
        assert requested.block_reason == "adv_unavailable"
        assert pd.isna(requested.adv20)
        assert requested.target_notional == 100.0
        assert requested.executed_delta_notional == 0.0
        assert (
            batch.loc[batch.asset_id.eq("__cash__"), "post_trade_notional"].item()
            == 100.0
        )


def test_simple_producer_distinguishes_fee_coverage_from_malformed_fee(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    missing_coverage = fee_frame.copy()
    missing_coverage["end_date"] = pd.Timestamp("2023-01-03")
    output, _calls = _build_producer(
        tmp_path / "missing",
        monkeypatch,
        frames=frames,
        fee_frame=missing_coverage,
        output_window=output_window,
    )
    with pytest.raises(PortfolioTargetBlocked, match="portfolio_fee_coverage_missing"):
        list(output)
    malformed = fee_frame.copy()
    malformed["rate"] = malformed["rate"].astype(object)
    malformed.loc[0, "rate"] = "not-a-rate"
    output, _calls = _build_producer(
        tmp_path / "malformed",
        monkeypatch,
        frames=frames,
        fee_frame=malformed,
        output_window=output_window,
    )
    with pytest.raises(ValueError):
        list(output)


def test_simple_producer_is_invariant_to_asset_major_input_permutation(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    raw_bar_orders = []
    first, _calls = _build_producer(
        tmp_path / "first",
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        raw_bar_orders=raw_bar_orders,
    )
    left = [frame.reset_index(drop=True) for frame in first]
    permuted = {
        key: frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
        for key, frame in frames.items()
    }
    second, _calls = _build_producer(
        tmp_path / "second",
        monkeypatch,
        frames=permuted,
        fee_frame=fee_frame.sample(frac=1.0, random_state=23).reset_index(drop=True),
        output_window=output_window,
        raw_bar_orders=raw_bar_orders,
    )
    right = [frame.reset_index(drop=True) for frame in second]
    assert raw_bar_orders and not all(raw_bar_orders)
    for expected, actual in zip(left, right, strict=True):
        pd.testing.assert_frame_equal(expected, actual)


def test_simple_producer_retains_zero_activity_suspension_rows(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    sessions = pd.DatetimeIndex(frames["calendar"]["calendar_date"]).normalize()
    amount_by_date = {
        timestamp: float(index + 1) for index, timestamp in enumerate(sessions)
    }
    frames["panel"]["amount"] = frames["panel"]["timestamp"].map(amount_by_date)
    suspension_date = pd.Timestamp("2022-12-28")
    suspension = frames["panel"]["timestamp"].eq(suspension_date)
    frames["panel"].loc[suspension, "amount"] = 0.0
    output, calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
    )
    results = list(output)
    assert results
    adv = results[0].loc[results[0]["asset_id"].ne("__cash__"), "adv20"]
    output_position = int(
        np.flatnonzero(sessions >= pd.Timestamp(output_window.start))[0]
    )
    warmup_position = max(0, output_position - 19)
    expected_adv = np.mean(
        [
            0.0 if timestamp == suspension_date else amount_by_date[timestamp]
            for timestamp in sessions[warmup_position : output_position + 1]
        ]
    )
    assert adv.tolist() == pytest.approx([expected_adv])
    panel_reads = [
        window.start
        for target, _mode, window in calls
        if target == "panel" and window is not None
    ]
    assert panel_reads[0] == sessions[0]


def test_simple_producer_handles_universe_arrival_and_zero_weight_departure(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, output_window = _inputs()
    arrival_dates = pd.bdate_range("2023-01-04", "2023-01-05")
    frames["panel"] = pd.concat(
        [
            frames["panel"],
            pd.DataFrame(
                {
                    "asset_id": "c",
                    "timestamp": arrival_dates,
                    "open": 100.0,
                    "close": 100.0,
                    "amount": 1000.0,
                    "market_type": 1,
                }
            ),
        ],
        ignore_index=True,
    )
    frames["dataset"] = pd.concat(
        [
            frames["dataset"],
            pd.DataFrame(
                {
                    "asset_id": "c",
                    "timestamp": arrival_dates,
                    "eligible": False,
                    "total_market_cap": 1000.0,
                    "float_market_cap": 1000.0,
                }
            ),
        ],
        ignore_index=True,
    )
    frames["execution"] = pd.concat(
        [
            frames["execution"],
            pd.DataFrame(
                {
                    "asset_id": "c",
                    "timestamp": arrival_dates,
                    "can_enter": False,
                    "can_exit": False,
                }
            ),
        ],
        ignore_index=True,
    )
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
    )
    assert list(output)


def test_simple_producer_has_no_executable_output_without_two_lookahead_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    frames, fee_frame, _output_window = _inputs()
    lookup = _PredictionLookup()
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=TimeWindow("2023-01-09", "2023-01-10"),
        prediction_lookup=lookup,
    )
    with pytest.raises(
        PortfolioTargetBlocked, match="portfolio_execution_window_empty"
    ):
        list(output)
    assert {timestamp.date() for timestamp, _asset in lookup.matched} == {
        pd.Timestamp("2023-01-09").date(),
        pd.Timestamp("2023-01-10").date(),
    }


def test_enhanced_risk_fits_styles_on_eligible_and_retains_ineligible_asset_axis():
    from quant_research_platform.research.portfolios.risk import (
        FactorRiskState,
        preprocess_risk_styles,
    )
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS

    axis = np.arange(121, dtype="float64")
    assets = [f"asset_{index:03}" for index in range(121)]
    cross = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": assets,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": ["A", "B"] * 60 + ["A"],
            "total_market_cap": 10000 + axis,
            "float_market_cap": 10000 + axis,
            "eligible": [True] * 120 + [False],
            "asset_return": np.sin(axis) / 100,
            **{
                name: np.sin(axis / (index + 3)) + axis / 120
                for index, name in enumerate(RISK_STYLE_COLUMNS)
            },
        }
    )
    state = FactorRiskState(
        style_columns=RISK_STYLE_COLUMNS,
        config=FactorRiskConfig(
            factor_window=3,
            specific_window=3,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
        ),
    )
    for day in pd.bdate_range("2020-01-02", periods=3):
        state.observe_return_session(cross.assign(timestamp=day))
    scheduled = pd.DataFrame(
        {
            name: pd.Series(dtype="datetime64[ns]")
            for name in ("timestamp", "entry_at", "available_at")
        }
    )

    def decide(frame):
        return production._enhanced_decision(
            benchmark_weights=pd.Series(1 / 120, index=assets[:120]),
            alpha=pd.Series(0.0, index=assets),
            factor_risk=state,
            current_cross=frame,
            calibration_rows=[],
            calibration_intervals=scheduled,
            as_of=pd.Timestamp("2020-01-08"),
            risk_style_columns=RISK_STYLE_COLUMNS,
        ).exposures.set_index("asset_id")

    before = decide(cross)
    changed = cross.copy()
    changed.loc[~changed.eligible, list(RISK_STYLE_COLUMNS)] = 1000.0
    after = decide(changed)
    pd.testing.assert_frame_equal(
        before.loc[assets[:120]], after.loc[assets[:120]], check_exact=True
    )
    assert set(after.index) == set(assets)
    assert np.isfinite(
        after.loc[assets[-1], list(RISK_STYLE_COLUMNS)].to_numpy(dtype=float)
    ).all()
    fitted_only = preprocess_risk_styles(
        cross.loc[cross.eligible], style_columns=RISK_STYLE_COLUMNS
    )
    pd.testing.assert_frame_equal(
        before.loc[assets[:120], list(RISK_STYLE_COLUMNS)].reset_index(drop=True),
        fitted_only.loc[:, list(RISK_STYLE_COLUMNS)].reset_index(drop=True),
        check_exact=True,
    )


def test_zero_mad_risk_style_preserves_identifiable_factor_return():
    from quant_research_platform.research.portfolios.risk import (
        FactorRiskState,
        preprocess_risk_styles,
    )

    values = np.r_[np.zeros(60), np.ones(40)]
    frame = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": [f"synthetic_{i}" for i in range(100)],
            "timestamp": pd.Timestamp("2023-01-03"),
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "I",
            "total_market_cap": 1.0,
            "float_market_cap": 1.0,
            "style": values,
            "asset_return": 0.01 + 0.02 * values,
        }
    )
    actual = preprocess_risk_styles(frame, style_columns=("style",))
    np.testing.assert_allclose(actual["style"], (values - 0.4) / np.sqrt(0.24))
    fit = FactorRiskState(style_columns=("style",)).fit_return_session(frame)
    assert fit is not None
    assert fit.factor_returns["style"] == pytest.approx(0.02 * np.sqrt(0.24))
    np.testing.assert_allclose(fit.residuals, 0.0, atol=1e-15)


def test_risk_cross_uses_all_a_facts_and_persisted_styles():
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": ["a", "b"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "period_return": [0.01, -0.01],
            "total_market_cap": [100.0, 200.0],
            "float_market_cap": [100.0, 200.0],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": ["A", "B"],
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": False,
            "is_pt": False,
            "listing_age_trading_days": 300,
            "has_252_trading_sessions": True,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 100.0,
            "amount": 1000.0,
            "turnover": 0.01,
            "is_nonfinancial_industry": [True, False],
        }
    )
    facts = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "beta_252d": [0.25, 1.5],
            "residual_volatility_60d": [0.01, 0.02],
        }
    )
    cross = production._enhanced_risk_cross(
        panel=panel,
        risk_frames=[facts],
        style_columns=("beta_252d", "residual_volatility_60d"),
    )
    assert cross.eligible.tolist() == [True, True]
    assert cross.beta_252d.tolist() == [0.25, 1.5]
    panel.loc[1, "is_st"] = True
    assert production._enhanced_risk_cross(
        panel=panel, risk_frames=[facts], style_columns=("beta_252d",)
    ).eligible.tolist() == [True, False]


@pytest.mark.parametrize("can_exit", [True, False, None])
def test_feature_missing_held_asset_preserves_holdings_without_trading(
    tmp_path, monkeypatch, can_exit
):
    frames, fee_frame, output_window = _inputs()
    frames["dataset"]["feature_x"] = 1.0
    frames["dataset"]["feature_missing_x"] = 0.0
    second = pd.Timestamp(output_window.start) + pd.offsets.BDay(1)
    invalid = frames["dataset"].asset_id.eq("b") & frames["dataset"].timestamp.ge(
        second
    )
    frames["dataset"].loc[invalid, "feature_x"] = np.nan
    frames["dataset"].loc[invalid, "feature_missing_x"] = 1.0
    exit_rows = frames["execution"].asset_id.eq("b") & frames["execution"].timestamp.ge(
        second
    )
    frames["execution"]["can_exit"] = frames["execution"]["can_exit"].astype("boolean")
    frames["execution"].loc[exit_rows, "can_exit"] = can_exit
    lookup = _PredictionLookup()
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        prediction_lookup=lookup,
        model_inputs=("feature_missing_x",),
        model_algorithm="linear_regression",
    )
    output = pd.concat(list(chunks), ignore_index=True)
    held = output.loc[output.asset_id.eq("b")]
    assert held.iloc[0].end_notional > 0.0
    later = held.loc[held.decision_timestamp.ge(second)]
    assert later.prediction_rank_score.isna().all()
    assert all((asset != "b" for date, asset in lookup.matched if date >= second))
    assert later.post_trade_notional.gt(0.0).all()
    assert later.executed_delta_notional.eq(0.0).all()
    assert later.target_notional.gt(0.0).all()
    assert later[list(PORTFOLIO_LEDGER_COST_COLUMNS)].to_numpy().sum() == 0.0
    cash = output.loc[output.asset_id.eq("__cash__")]
    assert cash.end_notional.ge(0.0).all()
    for _date, session in output.groupby("decision_timestamp"):
        assert session.executed_delta_notional.sum() + session[
            list(PORTFOLIO_LEDGER_COST_COLUMNS)
        ].to_numpy().sum() == pytest.approx(0.0, abs=1e-10)


def test_risk_exposures_separate_same_code_in_different_standards():
    state = FactorRiskState(style_columns=("style",))
    cross = pd.DataFrame(
        {
            "asset_id": ["old", "new"],
            "industry_standard_id": ["CSRC2012", "CAPCO"],
            "industry_level1_code": ["A", "A"],
            "style": [0.0, 0.0],
        }
    )
    before = cross.copy(deep=True)
    exposures = state.current_exposures(cross)
    assert exposures["industry_CSRC2012:A"].tolist() == [1.0, 0.0]
    assert exposures["industry_CAPCO:A"].tolist() == [0.0, 1.0]
    assert exposures["industry_level1_code"].tolist() == ["A", "A"]
    assert exposures["industry_standard_id"].tolist() == ["CSRC2012", "CAPCO"]
    pd.testing.assert_frame_equal(cross, before)


def test_risk_exposures_leave_unknown_industry_assets_unavailable():
    state = FactorRiskState(style_columns=("style",))
    cross = pd.DataFrame(
        {
            "asset_id": ["known", "missing_standard", "missing_code"],
            "industry_standard_id": pd.Series(
                ["CSRC2012", pd.NA, "CAPCO"], dtype="string"
            ),
            "industry_level1_code": pd.Series(["A", "A", pd.NA], dtype="string"),
            "style": [0.0, 1.0, 2.0],
        }
    )
    before = cross.copy(deep=True)
    exposures = state.current_exposures(cross).set_index("asset_id")
    assert exposures.index.tolist() == ["known"]
    assert exposures.loc["known", "industry_CSRC2012:A"] == 1.0
    required = exposures.reindex(cross["asset_id"])
    assert required.loc[["missing_standard", "missing_code"]].isna().all().all()
    assert state.current_exposures(cross.iloc[1:]).empty
    pd.testing.assert_frame_equal(cross, before)


@pytest.mark.parametrize("unknown_asset", ("a", "b"))
def test_enhanced_producer_blocks_only_required_unknown_industry(
    tmp_path, monkeypatch, unknown_asset
):
    frames, fee_frame, output_window = _inputs()
    frames["panel"] = frames["panel"].assign(
        period_return=0.001,
        total_market_cap=frames["panel"].asset_id.map({"a": 1000.0, "b": 2000.0}),
        float_market_cap=frames["panel"].asset_id.map({"a": 1000.0, "b": 2000.0}),
        industry_standard_id="CSRC2012",
        industry_level1_code="A01",
        is_nonfinancial_industry=True,
    )
    frames["panel"].loc[
        frames["panel"].asset_id.eq(unknown_asset), "industry_standard_id"
    ] = pd.NA
    dates = pd.DatetimeIndex(frames["calendar"]["calendar_date"]).normalize()
    frames["benchmark"] = pd.DataFrame(
        {
            "benchmark_id": "csi500",
            "asset_id": "a",
            "timestamp": dates[
                (dates >= pd.Timestamp(output_window.start))
                & (dates <= pd.Timestamp(output_window.end))
            ],
            "weight": 1.0,
        }
    )
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="unknown_industry",
        construction="benchmark_enhanced",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        candidate=candidate,
    )
    if unknown_asset == "a":
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_industry_context_missing"
        ):
            list(chunks)
    else:
        result = pd.concat(list(chunks), ignore_index=True)
        assert set(result.asset_id) == {"a", "__cash__"}
        assert result.loc[result.asset_id.eq("a"), "benchmark_weight"].eq(1.0).all()


def test_final_alpha_includes_last_mature_session_and_persists_without_fake_trade(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.portfolios.accounting import AlphaCalibration

    frames, fee_frame, _ = _inputs()
    window = TimeWindow("2023-01-03", "2023-01-10")
    frames["panel"]["period_return"] = 0.001
    frames["panel"]["float_market_cap"] = frames["panel"].asset_id.map(
        {"a": 1000.0, "b": 2000.0}
    )
    frames["panel"]["industry_standard_id"] = "CSRC2012"
    frames["panel"]["industry_level1_code"] = "A01"
    frames["panel"]["is_nonfinancial_industry"] = True
    frames["benchmark"] = pd.DataFrame(
        [
            {
                "benchmark_id": "csi500",
                "asset_id": asset,
                "timestamp": date,
                "weight": 0.5,
            }
            for date in pd.bdate_range(window.start, window.end)
            for asset in ("a", "b")
        ]
    )
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        rebalance_schedule="daily",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    observed = []
    original = production.calibrate_expected_alpha

    def calibrate(frame, **kwargs):
        result = original(frame, **kwargs)
        observed.append((pd.Timestamp(kwargs["as_of"]), result))
        return result

    monkeypatch.setattr(production, "calibrate_expected_alpha", calibrate)
    final = []
    chunks, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=window,
        candidate=candidate,
        final_calibrations=final,
    )
    batches = list(chunks)
    assert len(final) == 1
    final_state = AlphaCalibration(**final[0]["calibration"])
    assert observed[-1][0] == pd.Timestamp(window.end)
    assert final_state.status == "DEFINED"
    assert final_state.valid_rebalance_dates > observed[-2][1].valid_rebalance_dates
    assert max(frame.decision_timestamp.max() for frame in batches) < pd.Timestamp(
        window.end
    )
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": candidate.candidate_id,
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {"window": {"start": str(window.start), "end": str(window.end)}},
    }
    root = write_test_sessions(
        tmp_path / "final-alpha",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        ledger_chunks=batches,
        final_calibration=lambda: final[0],
    )
    artifact = read_portfolio_ledger_artifact(root)
    restored, cutoff = accounting.frozen_alpha_calibration(
        artifact.final_alpha_calibration,
        cutoff=window.end,
        selection_group=candidate.selection_group,
        rebalance_schedule=candidate.rebalance_schedule,
    )
    assert restored == final_state
    assert cutoff == pd.Timestamp(window.end).date().isoformat()
    with pytest.raises(ValueError, match="test_alpha_calibration_binding_invalid"):
        accounting.frozen_alpha_calibration(
            artifact.final_alpha_calibration,
            cutoff="2023-01-09",
            selection_group=candidate.selection_group,
            rebalance_schedule=candidate.rebalance_schedule,
        )


@pytest.mark.parametrize("nav", [100.0, 1000000000.0])
@pytest.mark.parametrize(
    "weight,material",
    [(1e-15, False), (5e-14, False), (7.5e-14, True), (1e-08, True)],
)
@pytest.mark.parametrize("blocked", ["market", "adv"])
def test_tiny_blocked_orders_keep_reason_consistent_through_real_writer(
    tmp_path, nav, weight, material, blocked
):
    expected_reason = (
        None
        if not material
        else "market_blocked"
        if blocked == "market"
        else "adv_capacity"
    )
    _assert_execution_endpoints_roundtrip(
        tmp_path,
        nav=nav,
        current=0.0,
        target=weight,
        can_enter=blocked != "market",
        adv=0.0 if blocked == "adv" else nav,
        expected_delta=0.0,
        expected_reason=expected_reason,
    )


@pytest.mark.parametrize(
    "nav,current,target",
    [
        (100.0, 0.7, 0.7000000000000001),
        (1000000000.0, 0.4176785340482801, 0.41767853404828015),
        (1000000000.0, 0.41767853404828015, 0.4176785340482801),
    ],
)
def test_adjacent_weights_execute_the_stored_notional_difference(
    tmp_path, nav, current, target
):
    expected_delta = target * nav - current * nav
    assert expected_delta != (target - current) * nav
    _assert_execution_endpoints_roundtrip(
        tmp_path,
        nav=nav,
        current=current,
        target=target,
        can_enter=True,
        adv=nav,
        expected_delta=expected_delta,
        expected_reason=None,
    )


def _assert_execution_endpoints_roundtrip(
    tmp_path, *, nav, current, target, can_enter, adv, expected_delta, expected_reason
):
    from quant_research_platform.contracts.portfolios import TradeCostComponents
    from tests.helpers.portfolio_ledger import continuous_ledger_batch

    current_weights = pd.Series({"a": current})
    target_weights = pd.Series({"a": target})
    zero = pd.Series({"a": 0.0})
    assumptions = ExecutionAssumptions(nav, 0.0, 0.0, 1.0, (1.0,))
    account = execute_continuous_target(
        current_weights,
        target_weights,
        nav=nav,
        average_daily_volume=pd.Series({"a": adv}),
        can_enter=pd.Series({"a": can_enter}),
        can_exit=pd.Series({"a": True}),
        execution_date=pd.Timestamp("2023-01-04").date(),
        cost_for_trade=lambda _date, _asset, _side, amount, _volume, _settings: (
            TradeCostComponents(float(amount), 0.0, 0.0, 0.0, 0.0, 0.0, "f" * 64)
        ),
        assumptions=assumptions,
        cost_source_identity="f" * 64,
    )
    frame = ledger_frame(
        account=account,
        decision_timestamp=pd.Timestamp("2023-01-03"),
        returns=zero,
        prediction_rank_score=zero,
        adv20=pd.Series({"a": adv}),
        enhanced_decision=None,
        benchmark_weights=None,
        optimization_result=None,
        decision_reason=PORTFOLIO_DECISION_REASON_SIMPLE,
        decision_delta=target_weights - current_weights,
    )
    row = frame.set_index("asset_id").loc["a"]
    assert row.pre_trade_notional == current * nav
    assert row.target_notional == target * nav
    assert row.executed_delta_notional == expected_delta
    assert (None if pd.isna(row.block_reason) else row.block_reason) == expected_reason
    batch = continuous_ledger_batch(frame, 0, end_timestamp=pd.Timestamp("2023-01-05"))
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "roundoff_ledger",
                "initial_capital": nav,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": "2023-01-03",
                "end": "2023-01-05",
                "access": "development",
            }
        },
    }
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        produce=lambda emit: emit(batch),
    )
    artifact = read_portfolio_ledger_artifact(root)
    (saved,) = tuple(artifact.iter_chunks())
    assert saved == batch
    if expected_delta:
        assert len(saved.trades) == 1
        assert saved.trades[0].notional == expected_delta
    else:
        assert not saved.trades


@pytest.mark.parametrize("extra_style", (-10000.0, 10000.0))
def test_enhanced_risk_scaler_uses_complete_known_industry_domain(extra_style):
    from quant_research_platform.research.portfolios.accounting import AlphaCalibration
    from quant_research_platform.research.portfolios.production import (
        _enhanced_decision,
    )
    from quant_research_platform.research.portfolios.risk import FactorRiskConfig

    count = 120
    x = np.arange(count, dtype=float)
    cross = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": [f"a{i}" for i in range(count)],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "total_market_cap": 100.0 + x,
            "float_market_cap": 100.0 + x,
            "eligible": True,
            "style": x,
            "asset_return": np.sin(x) / 100,
        }
    )
    state = FactorRiskState(
        style_columns=("style",), config=FactorRiskConfig(minimum_cross_section=10)
    )
    calibration = AlphaCalibration(
        0.0, 0.0, None, None, None, None, None, None, 1.0, 1, 1, None
    )

    def decision(frame):
        return _enhanced_decision(
            benchmark_weights=pd.Series(dtype=float),
            alpha=pd.Series(0.0, index=frame.asset_id),
            factor_risk=state,
            current_cross=frame,
            calibration_rows=[],
            calibration_intervals=pd.DataFrame(),
            as_of=pd.Timestamp("2020-01-03"),
            risk_style_columns=("style",),
            frozen_calibration=calibration,
        ).exposures.set_index("asset_id")

    expected = decision(cross)
    changed = pd.concat(
        (
            cross,
            cross.iloc[[0]].assign(
                asset_id="unknown", industry_level1_code=pd.NA, style=extra_style
            ),
        ),
        ignore_index=True,
    )
    changed = changed.astype({"industry_level1_code": cross.industry_level1_code.dtype})
    pd.testing.assert_frame_equal(decision(changed), expected, check_exact=True)
    pd.testing.assert_frame_equal(
        state.fit_return_session(changed).exposures,
        state.fit_return_session(cross).exposures,
        check_exact=True,
    )
    projected = pd.concat(
        (
            cross,
            cross.iloc[[0]].assign(
                asset_id="benchmark_only", eligible=False, style=extra_style
            ),
        ),
        ignore_index=True,
    )
    assert "benchmark_only" in decision(projected).index
    pd.testing.assert_frame_equal(
        decision(projected).loc[expected.index], expected, check_exact=True
    )


@pytest.mark.parametrize("invalid", (-1.0, float("inf"), -float("inf")))
def test_amount_history_rejects_invalid_present_amounts(invalid):
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": ["unrelated"],
            "amount": [invalid],
        }
    )
    with pytest.raises(ValueError, match="portfolio_amount_invalid"):
        production._update_amount_history(panel, {}, frozenset())


def test_amount_history_keeps_unknown_session_until_it_leaves_adv_window():
    history = {}
    for index in range(40):
        amount = np.nan if index == 20 else 1000.0
        production._update_amount_history(
            pd.DataFrame(
                {
                    "listing_date": pd.Timestamp("2000-01-01"),
                    "asset_id": ["known", "unrelated"],
                    "amount": [1000.0, amount],
                }
            ),
            history,
            frozenset(),
        )
        assert np.mean(history[("known", pd.Timestamp("2000-01-01"))]) == 1000.0
        if index >= 20:
            assert len(history[("unrelated", pd.Timestamp("2000-01-01"))]) == 20
            assert np.isnan(np.mean(history[("unrelated", pd.Timestamp("2000-01-01"))]))
    production._update_amount_history(
        pd.DataFrame(
            {
                "listing_date": pd.Timestamp("2000-01-01"),
                "asset_id": ["known", "unrelated"],
                "amount": [1000.0, 1000.0],
            }
        ),
        history,
        frozenset(),
    )
    assert np.mean(history[("unrelated", pd.Timestamp("2000-01-01"))]) == 1000.0


@pytest.mark.parametrize("missing_count", (1, 12))
def test_simple_producer_retains_order_and_unknown_adv_window(
    tmp_path, monkeypatch, missing_count
):
    frames, fee_frame, output_window = _inputs()
    gap_dates = (
        pd.DatetimeIndex(["2022-12-28"])
        if missing_count == 1
        else pd.bdate_range(end=output_window.start, periods=20)[:12]
    )
    unknown = frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.isin(
        gap_dates
    )
    frames["panel"].loc[unknown, "amount"] = np.nan
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        typed_batches=True,
    )
    typed = list(output)
    facts = {(row.asset_id, row.name): row for row in typed[0].diagnostics}
    assert facts["b", "adv20_observation_count"].number == 20
    assert facts["b", "adv20_missing_observations"].number == missing_count
    batches = [session_frame(batch) for batch in typed]
    assert batches
    for batch in batches:
        requested = batch.loc[batch.asset_id.eq("b")].iloc[0]
        assert requested.block_reason == "adv_unavailable"
        assert pd.isna(requested.adv20)
        assert requested.target_notional == 100.0
        assert requested.executed_delta_notional == 0.0
        assert (
            batch.loc[batch.asset_id.eq("__cash__"), "post_trade_notional"].item()
            == 100.0
        )


def test_half_step_rebalance_uses_existing_ledger_and_preserves_residual_cash(
    tmp_path, monkeypatch
):
    frames, fee_frame, output_window = _inputs()
    output, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        construction="quantile_equal_weight_half_step",
        typed_batches=True,
    )
    batches = list(output)
    first = session_frame(batches[0]).set_index("asset_id")
    assert first.loc[
        first.index != "__cash__", "decision_delta_weight"
    ].sum() == pytest.approx(0.5)
    assert first.loc["__cash__", "decision_delta_weight"] == pytest.approx(-0.5)
    second = session_frame(batches[1]).set_index("asset_id")
    assert (
        0 < second.loc[second.index != "__cash__", "decision_delta_weight"].sum() < 0.5
    )
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": "half_step",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": "2023-01-03",
                "end": "2023-01-06",
                "access": "development",
            }
        },
    }

    def produce(emit):
        for batch in batches:
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "half-step",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        produce=produce,
    )
    assert list(read_portfolio_ledger_artifact(root).iter_chunks()) == batches


@pytest.mark.parametrize(
    "construction",
    ("quantile_equal_weight", "quantile_float_cap", "quantile_equal_weight_half_step"),
)
@pytest.mark.parametrize("empty_reason", ("ineligible", "structural"))
@pytest.mark.parametrize("state", ("cash", "sellable", "blocked", "closed", "unknown"))
def test_empty_selection_executes_zero_target_without_bypassing_market_facts(
    construction, empty_reason, state
):
    frames, fee_frame, _ = _inputs()
    stamp, entry, end = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    dataset = frames["dataset"].loc[frames["dataset"].timestamp.eq(stamp)].copy()
    dataset["feature_x"] = np.nan if empty_reason == "structural" else 1.0
    dataset["eligible"] = empty_reason != "ineligible"
    lookup = _PredictionLookup(missing=True)
    alpha = production._prediction_alpha(dataset, lookup, ("feature_x",))
    assert lookup.matched == []
    candidate = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="empty",
        construction=construction,
        rebalance_schedule="daily",
        top_quantile=0.5,
        benchmark_id=None,
    )

    def panel(at):
        return (
            frames["panel"]
            .loc[frames["panel"].timestamp.eq(at)]
            .assign(open=100.0, close=100.0)
        )

    execution = (
        frames["execution"]
        .loc[frames["execution"].timestamp.eq(stamp)]
        .assign(execution_at=entry)
    )
    execution["can_exit"] = pd.Series(
        state not in {"blocked", "unknown"}, index=execution.index, dtype="boolean"
    )
    if state == "unknown":
        execution["can_exit"] = pd.Series(pd.NA, index=execution.index, dtype="boolean")
    kwargs = dict(
        candidate=candidate,
        decision_timestamp=stamp,
        next_timestamp=entry,
        panel=panel(stamp),
        next_panel=panel(entry),
        endpoint_panel=panel(end),
        closed_assets=frozenset({"b"}) if state == "closed" else frozenset(),
        dataset=dataset,
        execution=execution,
        holdings=pd.Series(dtype="float64")
        if state == "cash"
        else pd.Series({"b": 50.0}),
        cash=100.0 if state == "cash" else 50.0,
        nav=100.0,
        amount_history={
            (asset, panel(stamp).listing_date.iloc[0]): deque([1000.0] * 20, maxlen=20)
            for asset in ("a", "b")
        },
        alpha=alpha,
        enhanced_decision=None,
        target_override=None,
        decision_reason=PORTFOLIO_DECISION_REASON_SIMPLE,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        fee_schedule=fee_schedule_from_frame(fee_frame, content_identity="f" * 64),
        fee_identity="f" * 64,
        prepare_trade_cost=prepare_trade_cost,
        market_name_for_type=lambda _: "shanghai",
    )
    kwargs["decision_receivable"] = 0.0
    kwargs["receivable"] = 0.0
    kwargs["locked_quantities"] = {}
    kwargs["recovery_cash"] = 0.0
    kwargs["held_units"] = kwargs["holdings"] / kwargs["panel"].set_index(
        "asset_id"
    ).close.reindex(kwargs["holdings"].index)
    account, _adv, _optimization, _delta, snapshot = production._build_session(**kwargs)
    ledger = account.asset_ledger
    if empty_reason == "structural":
        assert ledger.executed_delta_notional.eq(0.0).all()
        assert ledger[list(PORTFOLIO_LEDGER_COST_COLUMNS)].to_numpy().sum() == 0.0
        assert account.post_trade_cash == (100.0 if state == "cash" else 50.0)
        if state == "cash":
            assert ledger.empty
        else:
            assert ledger.target_notional.tolist() == [50.0]
            assert ledger.post_trade_notional.tolist() == [50.0]
        assert snapshot.attrs["decision_availability"]["decision_status"] == "SKIPPED"
        assert (
            snapshot.attrs["decision_availability"]["decision_skip_reason"]
            == "all_predictions_missing"
        )
        return
    assert ledger.target_notional.eq(0.0).all()
    if state == "cash":
        assert ledger.empty
    elif state == "sellable":
        assert ledger.executed_delta_notional.tolist() == [-50.0]
        assert ledger.post_trade_notional.tolist() == [0.0]
        assert ledger[list(PORTFOLIO_LEDGER_COST_COLUMNS)].to_numpy().sum() > 0.0
    else:
        assert ledger.executed_delta_notional.eq(0.0).all()
        assert ledger.post_trade_notional.tolist() == [50.0]
        if state == "unknown":
            assert ledger.reason.tolist() == ["execution_fact_unknown"]


@pytest.mark.parametrize("problem", ("nan", "infinite"))
def test_nonempty_selection_still_requires_valid_predictions(problem):
    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": pd.Timestamp("2023-01-03"),
            "eligible": True,
            "x": 1.0,
        }
    )

    class Lookup:
        def values_for_frame(self, frame, *, mark_matched):
            return (
                [np.inf if problem == "infinite" else np.nan],
                [problem != "missing"],
            )

    with pytest.raises(
        ValueError,
        match="portfolio_prediction_values_invalid",
    ):
        production._prediction_alpha(frame, Lookup(), ("x",))


@pytest.mark.parametrize(
    "construction", ("quantile_equal_weight", "quantile_equal_weight_half_step")
)
@pytest.mark.parametrize("fees", (False, True))
@pytest.mark.parametrize("blocked", (False, True))
def test_cash_exposure_is_post_execution_ledger_cash(
    tmp_path, monkeypatch, construction, fees, blocked
):
    from quant_research_platform.artifacts.statistics import (
        STATISTICS_TABLE_COLUMNS,
        _validated_frame,
    )
    from tests.helpers.statistics_artifacts import (
        statistics_metadata,
    )

    frames, fee_frame, window = _inputs()
    if not fees:
        fee_frame["rate"] = 0.0
    if blocked:
        frames["execution"]["can_enter"] = False
    output, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=window,
        construction=construction,
        typed_batches=True,
    )
    batches = list(output)
    first = batches[0]
    values = {
        value.asset_id: value.value
        for value in first.valuations
        if value.phase == "after"
    }
    row = accounting.portfolio_interval_economics(first)
    assert row["cash_post"] == values["__cash__"]
    assert row["cash_weight_post"] == values["__cash__"] / sum(values.values())
    if blocked:
        assert row["cash_weight_post"] == 1.0
    elif not fees:
        assert row["cash_weight_post"] == (
            0.5 if construction.endswith("half_step") else 0.0
        )
    row.update(
        candidate_id="cash_probe",
        sample_role="validation",
        return_status="DEFINED",
        return_reason="",
    )
    daily = pd.DataFrame(
        [row], columns=STATISTICS_TABLE_COLUMNS["portfolio_daily.parquet"]
    )
    metadata = statistics_metadata(window, candidate_ids=[row["candidate_id"]])
    _validated_frame(
        daily,
        member="portfolio_daily.parquet",
        metadata=metadata,
    )
    daily.loc[0, "cash_weight_post"] += 0.01
    with pytest.raises(ValueError, match="statistics_cash_exposure_invalid"):
        _validated_frame(
            daily,
            member="portfolio_daily.parquet",
            metadata=metadata,
        )


@pytest.mark.parametrize("styles", (RISK_STYLE_COLUMNS[:-1], RISK_STYLE_COLUMNS))
def test_holding_projection_does_not_change_fit_or_specific_risk_history(styles):
    from quant_research_platform.research.portfolios.risk import (
        HoldingFactorInputs,
        holding_attribution,
    )

    rng = np.random.default_rng(198)
    cross = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2023-01-03"),
            "asset_id": [f"asset_{i}" for i in range(120)],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": ["A"] * 60 + ["B"] * 60,
            "total_market_cap": np.geomspace(100.0, 100000.0, 120),
            "float_market_cap": np.geomspace(100.0, 100000.0, 120),
            "eligible": True,
            "asset_return": rng.normal(0, 0.02, 120),
            **{name: rng.normal(size=120) for name in styles},
        }
    )
    state = FactorRiskState(style_columns=styles)
    expected = state.fit_return_session(cross)
    state.observe_return_session(cross)
    for extreme in (-1e100, 1e100):
        held = cross.iloc[[0]].assign(
            asset_id="held_st",
            eligible=False,
            asset_return=0.07,
            total_market_cap=abs(extreme),
            float_market_cap=abs(extreme),
            **dict.fromkeys(styles, extreme),
        )
        extended = pd.concat((cross, held), ignore_index=True)
        changed = FactorRiskState(style_columns=styles)
        actual = changed.fit_return_session(
            extended, require_complete=True, project_ineligible=True
        )
        pd.testing.assert_series_equal(
            actual.factor_returns, expected.factor_returns, check_exact=True
        )
        pd.testing.assert_frame_equal(
            actual.exposures.loc[expected.exposures.index],
            expected.exposures,
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            actual.residuals.loc[expected.residuals.index],
            expected.residuals,
            check_exact=True,
        )
        changed.observe_return_session(extended)
        assert changed._factor_rows == state._factor_rows
        assert changed._residuals == state._residuals
        assert changed._residual_sessions == state._residual_sessions
        inputs = HoldingFactorInputs(
            pd.Timestamp("2023-01-03"),
            pd.Timestamp("2023-01-04"),
            pd.Timestamp("2023-01-05"),
            actual,
            pd.Series(100.0, index=actual.exposures.index),
            pd.Series(100.0, index=actual.exposures.index),
            pd.DataFrame(),
        )
        values, reasons, _ = holding_attribution(
            inputs,
            after_notionals=pd.Series({"held_st": 100.0}),
            nav_before=100.0,
            nav_post=100.0,
            benchmark_id=None,
            benchmark_return=None,
            holding_returns=actual.asset_returns,
        )
        assert not reasons
        assert sum(
            values[k] for k in ("industry", "style", "residual_unexplained")
        ) == pytest.approx(0.07)
        for fault in ("missing_style", "unknown_industry", "unidentified_industry"):
            invalid = extended.copy()
            field, value = {
                "missing_style": (styles[0], np.nan),
                "unknown_industry": ("industry_level1_code", None),
                "unidentified_industry": ("industry_level1_code", "never_fitted"),
            }[fault]
            invalid.loc[invalid.asset_id.eq("held_st"), field] = value
            unavailable = changed.fit_return_session(invalid, project_ineligible=True)
            assert "held_st" not in unavailable.exposures.index
            pd.testing.assert_series_equal(
                unavailable.factor_returns, expected.factor_returns, check_exact=True
            )
        deficient = extended.copy()
        deficient.loc[deficient.eligible, styles[0]] = 0.0
        assert changed.fit_return_session(deficient, project_ineligible=True) is None


def test_enhanced_candidates_preserve_ledgers_across_independent_repeats(
    tmp_path, monkeypatch
):
    frames, fee, window = _inputs()
    rng = np.random.default_rng(121)
    assets = [f"a{i:03}" for i in range(200)]
    for name in ("panel", "dataset", "execution"):
        seed = frames[name].loc[frames[name].asset_id.eq("a")].drop(columns="asset_id")
        frames[name] = seed.merge(pd.DataFrame({"asset_id": assets}), how="cross")
    panel = frames["panel"]
    panel["period_return"] = rng.normal(0, 0.01, len(panel))
    panel["float_market_cap"] = 1000.0
    panel["industry_standard_id"] = "CSRC2012"
    panel["industry_level1_code"] = "A01"
    panel["is_nonfinancial_industry"] = True
    for column in RISK_STYLE_COLUMNS:
        frames["dataset"][column] = rng.normal(size=len(panel))
    frames["benchmark"] = pd.DataFrame(
        [
            {
                "benchmark_id": benchmark,
                "asset_id": asset,
                "timestamp": date,
                "weight": 1 / 100,
            }
            for benchmark, selected in (
                ("csi500", assets[:100]),
                ("csi1000", assets[100:]),
            )
            for date in pd.bdate_range(window.start, window.end)
            for asset in selected
        ]
    )

    class Lookup(_PredictionLookup):
        def values_for_frame(self, frame, *, mark_matched):
            return np.array(
                [int(asset[1:]) / 200 for asset in frame.asset_id]
            ), bytearray([1] * len(frame))

    base = PortfolioCandidateSpec(
        selection_group="global",
        prediction_selector="champion",
        candidate_id="shared_500",
        construction="benchmark_enhanced",
        rebalance_schedule="daily",
        top_quantile=None,
        benchmark_id="csi500",
        risk_style_columns=RISK_STYLE_COLUMNS,
    )
    candidates = (
        base,
        replace(base, candidate_id="shared_1000", benchmark_id="csi1000"),
    )
    expected = []
    for candidate in candidates:
        chunks, _ = _build_producer(
            tmp_path / candidate.candidate_id,
            monkeypatch,
            frames=frames,
            fee_frame=fee,
            output_window=window,
            candidate=candidate,
            prediction_lookup=Lookup(),
        )
        expected.append(list(chunks))
    shared_lookup = Lookup()
    for candidate, baseline in zip(candidates, expected, strict=True):
        final = []
        chunks, _ = _build_producer(
            tmp_path / "shared_inputs",
            monkeypatch,
            frames=frames,
            fee_frame=fee,
            output_window=window,
            candidate=candidate,
            prediction_lookup=shared_lookup,
            final_calibrations=final,
        )
        actual = list(chunks)
        assert len(actual) == len(baseline)
        for left, right in zip(actual, baseline, strict=True):
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        assert final[0]["cutoff"] == window.end.isoformat()


@pytest.mark.parametrize("change_at", (None, "2022-12-19", "2022-12-22"))
def test_unheld_scoring_asset_calibration_binds_signal_entry_and_completion(
    tmp_path,
    monkeypatch,
    change_at,
):
    frames, fee, window, candidate = _enhanced_inputs(monkeypatch)
    for key in ("panel", "dataset", "execution"):
        extra = frames[key].loc[frames[key].asset_id.eq("b")].assign(asset_id="c")
        if key == "panel":
            extra = extra.assign(open=10.0, close=10.0, adj_open=10.0, adj_close=10.0)
            if change_at is not None:
                changed = extra.timestamp.ge(pd.Timestamp(change_at))
                extra.loc[changed, ["open", "close", "adj_open", "adj_close"]] = 100.0
                extra.loc[changed, "listing_date"] = pd.Timestamp(change_at)
        frames[key] = pd.concat((frames[key], extra), ignore_index=True)

    class Lookup(_PredictionLookup):
        def values_for_frame(self, frame, *, mark_matched):
            values, found = super().values_for_frame(frame, mark_matched=mark_matched)
            values = values.copy()
            values[frame.asset_id.eq("c").to_numpy()] = 2.0
            return values, found

    completed = {}
    original = production._completed_calibration_rows

    def observe(pending, panel):
        rows = original(pending, panel)
        completed[pending[2]] = rows
        return rows

    monkeypatch.setattr(production, "_completed_calibration_rows", observe)
    chunks, _ = _build_producer(
        tmp_path / "input",
        monkeypatch,
        frames=frames,
        fee_frame=fee,
        output_window=window,
        input_window=TimeWindow("2022-11-14", "2023-01-10"),
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
        candidate=candidate,
        prediction_lookup=Lookup(),
        typed_batches=True,
    )
    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": candidate.candidate_id,
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": str(window.start),
                "end": str(window.end),
                "access": "development",
            }
        },
    }

    def produce(emit):
        for batch in chunks:
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        produce=produce,
    )
    saved = read_portfolio_ledger_artifact(root)
    batches = list(saved.iter_chunks())
    assert batches
    assert not any(trade.asset_id == "c" for batch in batches for trade in batch.trades)
    rows = completed[pd.Timestamp("2022-12-16")]
    assert {row["prediction_rank_score"] for row in rows} == (
        {0.0, 1.0, 2.0} if change_at is None else {0.0, 1.0}
    )
    if change_at is None:
        assert (
            next(row for row in rows if row["prediction_rank_score"] == 2.0)[
                "holding_return"
            ]
            == 0.0
        )


def test_adv_history_restarts_for_new_listing_but_survives_same_life_closure():
    old, new = pd.to_datetime(["2020-01-02", "2024-01-02"])
    history = {}
    panel = pd.DataFrame({"asset_id": ["asset"], "listing_date": old, "amount": 10.0})
    for _ in range(20):
        production._update_amount_history(panel, history, frozenset())
    production._update_amount_history(panel.iloc[:0], history, frozenset({"asset"}))
    assert len(history[("asset", old)]) == 20
    production._update_amount_history(
        panel.assign(listing_date=new, amount=100.0), history, frozenset()
    )
    assert set(history) == {("asset", new)}
    assert list(history[("asset", new)]) == [100.0]


def test_confirmed_delisting_zero_mark_preserves_reliable_same_life_quote():
    marks = production._ResearchValuationMarks()
    marks.project(_mark_view("2023-01-04", 10.0, 11.0))
    unknown = _mark_view("2023-01-05", np.nan, np.nan, present=False)
    unknown.confirmed_delisted_lifecycles = frozenset(
        {("a", pd.Timestamp("2020-01-01"))}
    )
    zero = marks.project(unknown)
    assert zero.open.iloc[0] == zero.close.iloc[0] == 0.0
    assert zero.research_open_mark_kind.iloc[0] == "confirmed_delisting_zero"
    quote = _mark_view("2023-01-06", 2.0, 3.0)
    quote.confirmed_delisted_lifecycles = unknown.confirmed_delisted_lifecycles
    observed = marks.project(quote)
    assert observed.open.iloc[0] == 2.0
    assert observed.close.iloc[0] == 3.0
    assert observed.research_open_mark_at.isna().all()


def test_portfolio_delisting_zero_nav_retains_units_and_recovers_amounts(
    tmp_path, monkeypatch
):
    from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
    from quant_research_platform.contracts.portfolios import (
        PortfolioCapital,
        validate_portfolio_transition,
    )
    from quant_research_platform.research.portfolios.accounting import (
        portfolio_interval_economics,
    )

    frames, fee, _ = _inputs()
    fee["rate"] = 0.0
    listing = frames["panel"].listing_date.iloc[0]
    frames["stock_master"] = pd.DataFrame(
        {
            "source_asset_id": ["a", "b"],
            "timestamp": listing,
            "listing_date": listing,
            "delisting_date": [pd.NaT, pd.Timestamp("2023-01-05")],
        }
    )
    frames["panel"].loc[:, ["open", "close"]] = 10.0
    frames["panel"].loc[
        frames["panel"].timestamp.ge("2023-01-09"), ["open", "close"]
    ] = 2.0
    for name in ("panel", "dataset", "execution"):
        frame = frames[name]
        frames[name] = frame.loc[
            ~(
                frame.asset_id.eq("b")
                & frame.timestamp.isin(pd.to_datetime(["2023-01-05", "2023-01-06"]))
            )
        ].copy()
    chunks, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee,
        output_window=TimeWindow("2023-01-03", "2023-01-09"),
        typed_batches=True,
    )
    batches = list(chunks)
    capital = PortfolioCapital(
        "synthetic_quantile_equal_weight", 100.0, "CNY", "cash_equity"
    )
    previous = None
    economy = _LedgerEconomics(capital)
    intervals = []
    for batch in batches:
        previous = validate_portfolio_transition(batch, capital, previous)
        economy.consume(batch)
        intervals.append(portfolio_interval_economics(batch))
    assert intervals[0]["nav_end"] == 0.0
    assert intervals[1]["nav_before"] == intervals[1]["nav_end"] == 0.0
    assert np.isnan(intervals[1]["net_return"])
    assert intervals[2]["nav_before"] == 0.0
    assert intervals[2]["nav_end"] == pytest.approx(20.0)
    assert np.isnan(intervals[2]["net_return"])
    assert economy.summary()["nav"] == pytest.approx(20.0)
    for batch in batches[1:3]:
        assert not batch.trades and not batch.cashflows
        b = [row for row in batch.valuations if row.asset_id == "b"]
        assert all(row.quantity == pytest.approx(10.0) for row in b)
    assert all(
        row.value == 0.0
        for batch in batches[1:]
        for row in batch.valuations
        if row.asset_id == "__cash__"
    )

    envelope = {
        "capital": PortfolioCapital(
            **{
                "candidate_id": capital.candidate_id,
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        ),
        "metadata": {
            "window": {
                "start": "2023-01-03",
                "end": "2023-01-09",
                "access": "development",
            }
        },
    }
    root = write_portfolio_ledger_artifact(
        tmp_path / "zero-artifact",
        capital=envelope["capital"],
        metadata=envelope["metadata"],
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    artifact = read_portfolio_ledger_artifact(root)
    assert list(artifact.iter_chunks()) == batches
    assert sum(
        row.value
        for row in list(artifact.iter_chunks())[-1].valuations
        if row.phase == "end"
    ) == pytest.approx(20.0)
    changed_life = replace(
        batches[1],
        valuations=tuple(
            replace(row, lifecycle_id="2023-01-05") if row.asset_id == "b" else row
            for row in batches[1].valuations
        ),
    )
    previous = validate_portfolio_transition(batches[0], capital, None)
    with pytest.raises(ValueError, match="portfolio_held_lifecycle_unproven"):
        validate_portfolio_transition(changed_life, capital, previous)


def test_cancelled_life_stale_quote_cannot_revive_rights_or_execution():
    from fractions import Fraction

    marks = production._ResearchValuationMarks()
    marks.project(_mark_view("2023-01-04", 10.0, 11.0))
    cancelled = _mark_view("2023-01-05", 12.0, 13.0)
    cancelled.rights_events = pd.DataFrame(
        [
            {
                "event_id": "cancel",
                "asset_id": "a",
                "listing_date": pd.Timestamp("2020-01-01"),
                "kind": "cancel",
                "entitlement_proved": True,
                "remaining_old_units_per_old_unit": Fraction(),
                "successor_asset_id": None,
            }
        ]
    )
    first = marks.project(cancelled)
    later = marks.project(_mark_view("2023-01-06", 14.0, 15.0))
    assert first.open.iloc[0] == later.open.iloc[0] == 0.0
    assert production._unquoted_assets(later) == frozenset({"a"})
    new_life = marks.project(_mark_view("2023-01-07", 20.0, 21.0, life="2023-01-07"))
    assert new_life.open.iloc[0] == 20.0
    assert not production._unquoted_assets(new_life)


@pytest.mark.parametrize("corruption", ("known_adv", "complete_window", "fee", "trade"))
def test_persisted_adv_unavailable_rejects_fabricated_evidence(
    tmp_path, monkeypatch, corruption
):
    from fractions import Fraction

    from quant_research_platform.contracts.portfolios import (
        ExecutedTrade,
        PortfolioCapital,
        PortfolioCashFlow,
        PortfolioDiagnostic,
        validate_portfolio_transition,
    )

    frames, fee, window = _inputs()
    frames["panel"].loc[
        frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.eq("2022-12-28"),
        "amount",
    ] = np.nan
    producer, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee,
        output_window=window,
        typed_batches=True,
    )
    batches = list(producer)
    batch = batches[0]
    capital = PortfolioCapital(
        "synthetic_quantile_equal_weight", 100.0, "CNY", "cash_equity"
    )
    validate_portfolio_transition(batch, capital, None)
    timestamp = next(
        row.timestamp_ns for row in batch.valuations if row.phase == "after"
    )
    with pytest.raises(ValueError):
        if corruption == "known_adv":
            batch = replace(
                batch,
                diagnostics=(
                    *batch.diagnostics,
                    PortfolioDiagnostic("b", None, "adv20", number=1000.0),
                ),
            )
        elif corruption == "complete_window":
            batch = replace(
                batch,
                diagnostics=tuple(
                    replace(row, number=0.0)
                    if row.asset_id == "b" and row.name == "adv20_missing_observations"
                    else row
                    for row in batch.diagnostics
                ),
            )
        elif corruption == "fee":
            batch = replace(
                batch,
                cashflows=(
                    PortfolioCashFlow(
                        "fake_fee", timestamp, "b", None, "commission_cost", -1.0
                    ),
                ),
            )
        else:
            batch = replace(
                batch,
                trades=(
                    ExecutedTrade("fake", "fake", timestamp, "b", Fraction(1), 1.0),
                ),
            )
        validate_portfolio_transition(batch, capital, None)
