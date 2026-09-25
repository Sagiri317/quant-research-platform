from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    CONTINUOUS_EXECUTION_COLUMNS,
    PORTFOLIO_DECISION_REASON_ENHANCED,
    PORTFOLIO_DECISION_REASON_MAINTENANCE,
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    prepare_trade_cost,
    require_fee_coverage,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.accounting import (
    execute_continuous_target as _execute_continuous_target,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskEstimate,
    FactorRiskState,
)
from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets import DatasetReadMode
from quant_research_platform.storage.identity import filesystem_mkdir
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.portfolio_ledger import session_frame, write_test_sessions


def execute_continuous_target(*args, **kwargs):
    def totals(assets, amounts, adv):
        return np.asarray(
            [
                kwargs["cost_for_trade"](
                    kwargs["execution_date"],
                    asset,
                    "buy",
                    float(amount),
                    float(volume),
                    kwargs["assumptions"],
                ).total_cost
                if amount > 0
                else 0.0
                for asset, amount, volume in zip(assets, amounts, adv, strict=True)
            ]
        )

    return _execute_continuous_target(*args, cost_totals_for_buys=totals, **kwargs)


def test_enhanced_decision_requires_history_for_current_industry_domain() -> None:
    state = FactorRiskState(
        style_columns=("style",),
        config=FactorRiskConfig(
            minimum_cross_section=10,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
        ),
    )
    x = np.arange(20, dtype="float64")
    cross = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"a{index}" for index in range(20)],
            "industry_level1_code": "A",
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
    assert tuple(previous.factor_covariance.columns) == ("industry_A", "style")
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
        "industry_A",
        "industry_B",
        "style",
    )
    assert current.risk.factor_returns.timestamp.min() == pd.Timestamp("2020-01-04")
    assert tuple(previous.factor_covariance.columns) == ("industry_A", "style")


class _PredictionLookup:
    def __init__(self, *, missing: bool = False, extra: bool = False) -> None:
        self.missing = missing
        self.extra = extra
        self.matched: list[tuple[pd.Timestamp, str]] = []

    def values_for_frame(self, frame: pd.DataFrame, *, mark_matched: bool):
        if mark_matched:
            self.matched.extend(
                (
                    pd.Timestamp(timestamp),
                    str(asset_id),
                )
                for timestamp, asset_id in frame.loc[
                    :, ["timestamp", "asset_id"]
                ].itertuples(index=False, name=None)
            )
        values = frame["asset_id"].map({"a": 0.0, "b": 1.0}).to_numpy()
        found = bytearray([0 if self.missing else 1] * len(values))

        @contextmanager
        def result():
            yield values, found

        return result()

    def coverage_counts(self) -> tuple[int, int]:
        return 0, 1 if self.extra else 0


def _inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, TimeWindow]:
    dates = pd.bdate_range("2022-11-14", "2023-01-10")
    assets = ("a", "b")
    panel = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "adj_open": 100.0 + (timestamp.day * (1 if asset == "a" else 2)),
                "adj_close": 100.0 + (timestamp.day * (1 if asset == "a" else 2)),
                "amount": 1_000.0,
                "market_type": 1,
                "is_a_share_scope": True,
                "is_trading": True,
                "is_st": False,
                "is_pt": False,
                "listing_age_trading_days": 300,
                "close": 100.0,
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
                "float_market_cap": 1_000.0 if asset == "a" else 2_000.0,
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
    resource_policy=None,
    candidate: PortfolioCandidateSpec | None = None,
    prediction_window: TimeWindow | None = None,
):
    enhanced = candidate is not None and candidate.construction == "benchmark_enhanced"
    if enhanced:
        risk_columns = list(RISK_STYLE_COLUMNS)
        frames = {
            **frames,
            "risk_context": frames["dataset"].reindex(
                columns=["asset_id", "timestamp", *risk_columns]
            ),
            "dataset": frames["dataset"].drop(columns=risk_columns, errors="ignore"),
        }
    if candidate is None:
        candidate = PortfolioCandidateSpec(
            prediction_selector="champion",
            candidate_id="synthetic_top20_equal_weight",
            construction="top20_equal_weight",
            prediction_target_id="prediction",
            rebalance_schedule="daily",
            top_quantile=0.5,
            benchmark_id=None,
            benchmark_weights_target_id=None,
            research_dataset_id="dataset",
            market_panel_target_id="panel",
            execution_mask_target_id="execution",
            trading_calendar_target_id="calendar",
            fee_schedule_target_id="fee",
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
        )
    }
    roots = {}
    for target_id, identity in identities.items():
        root = tmp_path / target_id / identity
        filesystem_mkdir(root, parents=True)
        roots[target_id] = root
    layout = StorageLayout(project_root=tmp_path).with_target_roots(roots)
    calls: list[tuple[str, DatasetReadMode, TimeWindow | None]] = []

    def fake_read_dataset(
        root: Path,
        *,
        columns=None,
        time_window=None,
        mode=DatasetReadMode.MATERIALIZED,
        resource_policy=None,
    ):
        target_id = next(
            target_id
            for target_id, candidate_root in roots.items()
            if root == candidate_root
        )
        calls.append((target_id, mode, time_window))
        assert resource_policy is expected_policy
        if mode is DatasetReadMode.CHUNKED:
            assert isinstance(time_window, TimeWindow)
            selected = (
                frames[target_id]
                .loc[
                    frames[target_id].timestamp.between(
                        pd.Timestamp(time_window.start), pd.Timestamp(time_window.end)
                    ),
                    list(columns),
                ]
                .sort_values(["timestamp", "asset_id"], kind="mergesort")
            )
            return iter((selected.copy(),))
        if target_id == "fee":
            assert columns is None
            assert time_window is None
            return fee_frame.copy()
        if target_id == "calendar":
            assert columns == (
                "calendar_date",
                "source_market_type",
                "is_trading_day",
            )
            assert time_window is None
            return frames["calendar"].copy()
        if target_id == "benchmark":
            assert columns == ("benchmark_id", "asset_id", "timestamp", "weight")
            assert isinstance(time_window, TimeWindow)
            selected = frames[target_id].loc[
                pd.to_datetime(frames[target_id]["timestamp"]).dt.normalize()
                == pd.Timestamp(time_window.start).normalize()
            ]
            return selected.loc[:, list(columns)].copy()
        if target_id == "risk_context":
            selected = frames[target_id].loc[
                frames[target_id].timestamp.eq(pd.Timestamp(time_window.start))
            ]
            return (
                selected.copy()
                if columns is None
                else selected.loc[:, list(columns)].copy()
            )
        expected = {
            "dataset": ("asset_id", "timestamp", "eligible", "float_market_cap"),
            "panel": (
                production._ENHANCED_PANEL_COLUMNS
                if enhanced
                else (
                    "asset_id",
                    "timestamp",
                    "adj_open",
                    "adj_close",
                    "amount",
                    "market_type",
                )
            ),
            "execution": ("asset_id", "timestamp", "can_enter", "can_exit"),
        }[target_id]
        assert columns == expected
        assert isinstance(time_window, TimeWindow)
        assert time_window.start == time_window.end
        selected = frames[target_id].loc[
            pd.to_datetime(frames[target_id]["timestamp"]).dt.normalize()
            == pd.Timestamp(time_window.start).normalize()
        ]
        return selected.loc[:, list(columns)].copy()

    expected_policy = resource_policy

    @contextmanager
    def schema_session(root, *, columns, time_window, resource_policy):
        from types import SimpleNamespace

        target = next(key for key, value in roots.items() if value == root)
        assert columns is None
        assert resource_policy is expected_policy
        yield SimpleNamespace(
            arrow_schema=SimpleNamespace(names=tuple(frames[target].columns))
        )

    monkeypatch.setattr(production, "open_dataset_session", schema_session)
    monkeypatch.setattr(production, "read_dataset", fake_read_dataset)
    assumptions = ExecutionAssumptions(
        initial_nav=100.0,
        slippage_bps=0.0,
        impact_bps=0.0,
        adv_participation=1.0,
        adv_sensitivities=(1.0,),
    )

    def output():
        nonlocal expected_policy
        with runtime_resource_policy_scope(resource_policy) as bound_policy:
            expected_policy = bound_policy
            chunks = production.build_portfolio_ledger(
                candidate=candidate,
                layout=layout,
                upstream_identities={**identities, "prediction": "b" * 64},
                output_window=output_window,
                input_window=input_window or TimeWindow("2022-12-01", "2023-01-10"),
                formal_dataset_window=formal_dataset_window
                or TimeWindow("2022-12-01", "2023-01-10"),
                prediction_window=prediction_window,
                prediction_lookup=prediction_lookup or _PredictionLookup(),
                assumptions=assumptions,
                resource_policy=bound_policy,
                fee_schedule_from_frame=fee_schedule_from_frame,
                require_fee_coverage=require_fee_coverage,
                prepare_trade_cost=prepare_trade_cost,
                market_name_for_type=lambda _value: "shanghai",
                unified_trading_sessions=unified_trading_sessions,
            )
            for batch in chunks:
                yield session_frame(batch)

    return output(), calls


def test_simple_producer_publishes_one_bounded_ledger_and_reuses_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames, fee_frame, output_window = _inputs()
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
    )
    frames_out = list(chunks)
    assert frames_out
    assert all(
        tuple(frame.columns) == CONTINUOUS_EXECUTION_COLUMNS for frame in frames_out
    )
    assert all(frame["asset_id"].iloc[-1] == "__cash__" for frame in frames_out)

    envelope = build_business_manifest_envelope(
        logical_artifact_id="synthetic_portfolio",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-03",
            "end": "2023-01-06",
            "access": "development",
        },
        upstream_identities={"prediction": "b" * 64},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "synthetic_portfolio",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )
    artifact_root = write_test_sessions(
        tmp_path / "artifacts",
        envelope=envelope,
        ledger_chunks=iter(frames_out),
    )
    artifact = read_portfolio_ledger_artifact(
        artifact_root,
        expected_identity=artifact_root.name,
    )
    persisted = list(artifact.iter_chunks())
    assert len(persisted) == len(frames_out)
    for expected, batch in zip(frames_out, persisted, strict=True):
        actual = session_frame(batch)
        np.testing.assert_array_equal(
            actual.decision_delta_weight, expected.decision_delta_weight
        )
        assert abs(float(actual.decision_delta_weight.sum())) < 1e-14
    assert np.isclose(frames_out[0]["post_trade_notional"].sum(), 100.0 - 0.04)


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
        target for target, mode, _window in calls if mode is DatasetReadMode.CHUNKED
    )
    assert streams == {"panel": 3, "dataset": 1, "execution": 1}
    assert all(
        target in {"fee", "calendar"}
        for target, mode, _window in calls
        if mode is DatasetReadMode.MATERIALIZED
    )


def test_benchmark_stream_keeps_future_snapshot_across_missing_sessions(monkeypatch):
    sessions = pd.date_range("2020-01-01", periods=5)
    observed = []

    def snapshots(*args):
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
    with runtime_resource_policy_scope() as policy:
        weights = list(
            production._iter_benchmark_weights(None, "csi500", sessions, policy)
        )
        assert [item.empty for item in weights] == [True, False, True, False, True]
        for item in (weights[1], weights[3]):
            np.testing.assert_array_equal(item.to_numpy(), [0.25, 0.75])
        assert policy.used_memory_bytes == 0
    assert observed == ["open", "closed"]


def test_float_cap_roundoff_keeps_nonnegative_persisted_cash_target(tmp_path):
    assets = pd.Index([f"asset_{index:02}" for index in range(20)])
    caps = pd.Series(np.random.default_rng(4).uniform(1, 100, 20), index=assets)
    target = caps / caps.sum()
    nav = 100_000_000.0
    assert nav - float((target * nav).sum()) < 0
    zero = pd.Series(0.0, index=assets)
    adv = pd.Series(1e12, index=assets)
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
    frame = production._ledger_frame(
        account=account,
        decision_timestamp=pd.Timestamp("2023-01-03"),
        returns=zero,
        alpha_score=caps,
        adv20=adv,
        enhanced_decision=None,
        benchmark_weights=None,
        optimization_result=None,
        decision_reason=PORTFOLIO_DECISION_REASON_SIMPLE,
        decision_delta=target,
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="float_cap_ledger",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-03",
            "end": "2023-01-04",
            "access": "development",
        },
        upstream_identities={"prediction": "b" * 64},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "float_cap_ledger",
                "initial_capital": nav,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )
    root = write_test_sessions(
        tmp_path / "ledger", envelope=envelope, ledger_chunks=iter((frame,))
    )
    saved = read_portfolio_ledger_artifact(root, expected_identity=root.name)
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
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames, fee_frame, output_window = _inputs()
    frames["panel"]["period_return"] = 0.001
    frames["panel"]["float_market_cap"] = frames["panel"]["asset_id"].map(
        {"a": 1_000.0, "b": 2_000.0}
    )
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
        prediction_selector="champion",
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        prediction_target_id="prediction",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        benchmark_weights_target_id="benchmark",
        risk_style_columns=RISK_STYLE_COLUMNS,
        risk_context_target_ids=("risk_context",),
        research_dataset_id="dataset",
        market_panel_target_id="panel",
        execution_mask_target_id="execution",
        trading_calendar_target_id="calendar",
        fee_schedule_target_id="fee",
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
    assert first["pre_trade_notional"].iloc[:2].tolist() == pytest.approx([50.0, 50.0])
    assert first["target_notional"].iloc[:2].tolist() == pytest.approx([50.0, 50.0])
    assert first.loc[first["asset_id"].eq("__cash__"), "post_trade_notional"].iloc[
        0
    ] == pytest.approx(0.0)


@pytest.mark.parametrize("can_sell_removed", [False, True])
def test_enhanced_membership_maintenance_keeps_explicit_benchmark_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    can_sell_removed: bool,
) -> None:
    from tests.helpers.runtime_resources import capture_test_policy

    frames, fee_frame, _window = _inputs()
    output_window = TimeWindow("2023-01-03", "2023-01-06")
    frames["panel"] = frames["panel"].assign(
        adj_open=100.0,
        adj_close=100.0,
        period_return=0.0,
        float_market_cap=1_000.0,
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
        prediction_selector="champion",
        candidate_id="maintenance",
        construction="benchmark_enhanced",
        prediction_target_id="prediction",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        benchmark_weights_target_id="benchmark",
        risk_style_columns=RISK_STYLE_COLUMNS,
        risk_context_target_ids=("risk_context",),
        research_dataset_id="dataset",
        market_panel_target_id="panel",
        execution_mask_target_id="execution",
        trading_calendar_target_id="calendar",
        fee_schedule_target_id="fee",
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
        diagonal_shrinkage=0.20,
        annualization_factor=252.0,
    )
    original_calibrate = production.calibrate_expected_alpha
    monkeypatch.setattr(production.FactorRiskState, "estimate", lambda _self: risk)
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
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024**2, explicit_cap_bytes=64 * 1024**2
    )
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        candidate=candidate,
        resource_policy=policy,
    )
    sessions = list(chunks)
    assert policy.used_memory_bytes == 0
    maintenance = next(
        frame
        for frame in sessions
        if frame.decision_timestamp.iloc[0] == pd.Timestamp("2023-01-04")
    )
    indexed = maintenance.set_index("asset_id")
    assert (
        indexed.loc["__cash__", "decision_reason"]
        == PORTFOLIO_DECISION_REASON_MAINTENANCE
    )
    assert indexed.loc["a", "target_notional"] == 0.0
    assert indexed.loc["b", "target_notional"] == pytest.approx(100.0)
    assert indexed.loc["a", "decision_delta_weight"] == pytest.approx(-0.1)
    assert indexed.loc["b", "decision_delta_weight"] == pytest.approx(0.1)
    assert indexed.loc["a", "post_trade_notional"] == pytest.approx(
        0.0 if can_sell_removed else 10.0
    )
    assert indexed.loc["__cash__", "post_trade_notional"] >= 0.0
    costs = maintenance.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum()
    assert maintenance.post_trade_notional.sum() + costs == pytest.approx(100.0)
    if can_sell_removed:
        assert costs > 0.0
    else:
        assert indexed.loc["b", "executed_delta_notional"] == 0.0
    envelope = build_business_manifest_envelope(
        logical_artifact_id="maintenance",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": output_window.start.isoformat(),
            "end": output_window.end.isoformat(),
            "access": "development",
        },
        upstream_identities={"prediction": "b" * 64},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "maintenance",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )
    root = write_test_sessions(
        tmp_path / "ledger", envelope=envelope, ledger_chunks=iter(sessions)
    )
    saved = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    loaded = [session_frame(batch) for batch in saved.iter_chunks()]
    assert len(loaded) == len(sessions)
    for actual, expected in zip(loaded, sessions, strict=True):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_enhanced_producer_reaches_optimizer_and_publishes_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames, fee_frame, output_window = _inputs()
    output_window = TimeWindow("2023-01-03", "2023-01-10")
    frames["panel"]["period_return"] = 0.001
    dates = pd.DatetimeIndex(frames["calendar"]["calendar_date"]).normalize()
    date_positions = {
        timestamp: position
        for position, timestamp in enumerate(dates.drop_duplicates())
    }
    frames["panel"]["adj_open"] = [
        100.0 * (1.001 ** date_positions[timestamp])
        if asset == "a"
        else 100.0 * (1.003 ** date_positions[timestamp])
        for timestamp, asset in zip(
            frames["panel"]["timestamp"],
            frames["panel"]["asset_id"],
            strict=True,
        )
    ]
    frames["panel"]["adj_close"] = frames["panel"]["adj_open"].copy()
    frames["panel"]["float_market_cap"] = frames["panel"]["asset_id"].map(
        {"a": 1_000.0, "b": 2_000.0}
    )
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
    exposure_columns = ("industry_A01", *RISK_STYLE_COLUMNS)
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
        diagonal_shrinkage=0.20,
        annualization_factor=252.0,
    )

    frames["dataset"]["beta_252d"] = 1.0
    frames["dataset"]["residual_volatility_60d"] = 0.1
    monkeypatch.setattr(
        production.FactorRiskState,
        "estimate",
        lambda self: risk,
    )
    candidate = PortfolioCandidateSpec(
        prediction_selector="champion",
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        prediction_target_id="prediction",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        benchmark_weights_target_id="benchmark",
        risk_style_columns=RISK_STYLE_COLUMNS,
        risk_context_target_ids=("risk_context",),
        research_dataset_id="dataset",
        market_panel_target_id="panel",
        execution_mask_target_id="execution",
        trading_calendar_target_id="calendar",
        fee_schedule_target_id="fee",
    )
    from quant_research_platform.contracts.portfolios import (
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    observed = []
    actual_build = production._build_session

    def observe_decision(**kwargs):
        account, returns, adv, optimization, decision_delta = actual_build(**kwargs)
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
        return account, returns, adv, optimization, decision_delta

    monkeypatch.setattr(production, "_build_session", observe_decision)
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024**2, explicit_cap_bytes=64 * 1024**2
    )
    chunks, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        input_window=TimeWindow("2022-11-14", "2023-01-10"),
        candidate=candidate,
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
        resource_policy=policy,
    )
    with runtime_resource_policy_scope(policy):
        result = list(chunks)
        assert policy.used_memory_bytes == 0
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
    assert (
        result[0].loc[result[0].asset_id.eq("__cash__"), "post_trade_notional"].iloc[0]
        == 0.0
    )
    for row_frame in result:
        if row_frame.decision_timestamp.iloc[0] != pd.Timestamp("2023-01-06"):
            assert row_frame.executed_delta_notional.eq(0.0).all()
            assert row_frame.decision_delta_weight.eq(0.0).all()
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
    changed["panel"].loc[future, "adj_open"] *= 1.5
    second_policy = capture_test_policy(
        available_memory_bytes=256 * 1024**2, explicit_cap_bytes=64 * 1024**2
    )
    second, _ = _build_producer(
        tmp_path / "future",
        monkeypatch,
        frames=changed,
        fee_frame=fee_frame,
        output_window=output_window,
        input_window=TimeWindow("2022-11-14", "2023-01-10"),
        candidate=candidate,
        prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
        resource_policy=second_policy,
    )
    with runtime_resource_policy_scope(second_policy):
        list(second)
        assert second_policy.used_memory_bytes == 0
    assert len(observed) == 2
    pd.testing.assert_series_equal(observed[0][0], observed[1][0], check_exact=True)
    assert observed[0][1] == observed[1][1]
    pd.testing.assert_series_equal(observed[0][2], observed[1][2], check_exact=True)
    assert observed[0][3] != observed[1][3]
    pd.testing.assert_series_equal(observed[0][4], observed[1][4], check_exact=True)
    with monkeypatch.context() as patch:
        patch.setattr(production, "_build_session", actual_build)
        initial_policy = capture_test_policy(
            available_memory_bytes=256 * 1024**2, explicit_cap_bytes=64 * 1024**2
        )
        initial, _ = _build_producer(
            tmp_path / "initial",
            patch,
            frames=frames,
            fee_frame=fee_frame,
            output_window=TimeWindow("2023-01-06", "2023-01-10"),
            input_window=TimeWindow("2022-11-14", "2023-01-10"),
            candidate=candidate,
            prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
            resource_policy=initial_policy,
        )
        with runtime_resource_policy_scope(initial_policy):
            first_active = list(initial)[0]
            assert initial_policy.used_memory_bytes == 0
        assert first_active.pre_trade_notional.sum() == pytest.approx(100.0)
        assert first_active.loc[
            first_active.asset_id.ne("__cash__"), "pre_trade_notional"
        ].tolist() == pytest.approx([50.0, 50.0])
        assert (
            first_active.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum() > 0.0
        )

        original_calibrate = production.calibrate_expected_alpha
        calibrations = []

        def bounded_calibrate(frame, **kwargs):
            assert (
                kwargs["scheduled_intervals"]
                .timestamp.ge(pd.Timestamp("2022-12-01"))
                .all()
            )
            calibrated = original_calibrate(frame, **kwargs)
            calibrations.append(calibrated)
            return calibrated

        patch.setattr(production, "calibrate_expected_alpha", bounded_calibrate)
        lookup = _PredictionLookup()
        short_policy = capture_test_policy(
            available_memory_bytes=256 * 1024**2, explicit_cap_bytes=64 * 1024**2
        )
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
            resource_policy=short_policy,
        )
        with runtime_resource_policy_scope(short_policy):
            list(short)
            assert short_policy.used_memory_bytes == 0
        assert calibrations and calibrations[-1].scheduled_rebalance_dates == 4
        assert calibrations[-1].coverage == 1.0 and calibrations[-1].benchmark_only
        assert min(timestamp for timestamp, _asset in lookup.matched) >= pd.Timestamp(
            "2022-12-01"
        )

    for interrupt in (False, True):
        completed = []
        primary = KeyboardInterrupt("calibration_state_interrupted")
        actual_complete = production._completed_calibration_rows

        def observed_completion(
            *args,
            actual_complete=actual_complete,
            completed=completed,
            interrupt=interrupt,
            primary=primary,
            **kwargs,
        ):
            rows = actual_complete(*args, **kwargs)
            completed.append(len(rows))
            if interrupt and len(completed) == 2:
                rows = None
                raise primary
            return rows

        bounded = capture_test_policy(
            available_memory_bytes=256 * 1024**2, explicit_cap_bytes=128 * 1024
        )
        with monkeypatch.context() as patch, runtime_resource_policy_scope(bounded):
            patch.setattr(production, "_build_session", actual_build)
            patch.setattr(
                production, "_completed_calibration_rows", observed_completion
            )
            failing, _ = _build_producer(
                tmp_path / ("interrupt" if interrupt else "cap"),
                patch,
                frames=frames,
                fee_frame=fee_frame,
                output_window=output_window,
                input_window=TimeWindow("2022-11-14", "2023-01-10"),
                candidate=candidate,
                prediction_window=TimeWindow("2022-11-14", "2023-01-10"),
                resource_policy=bounded,
            )
            with pytest.raises(
                KeyboardInterrupt if interrupt else MemoryError
            ) as caught:
                list(failing)
            assert completed
            if interrupt:
                assert caught.value is primary
            assert bounded.used_memory_bytes == 0


@pytest.mark.parametrize("removed_member", [False, True])
def test_optimizer_assets_keep_residual_cash_as_explicit_current_weight(
    removed_member: bool,
) -> None:
    assets = pd.Index(
        ("a", "b", "old") if removed_member else ("a", "b"), dtype="string"
    )
    exposures = pd.DataFrame(
        {
            "asset_id": assets,
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
        expected_alpha=pd.Series(0.01, index=assets),
        exposures=exposures,
        specific_variance=pd.Series(0.01, index=assets),
        adv=pd.Series(1_000.0, index=assets),
        eligible=pd.Series(True, index=assets),
        panel=pd.DataFrame(
            {
                "asset_id": assets,
                "is_nonfinancial_industry": True,
            }
        ),
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
        assert optimized.weights.sum() == pytest.approx(1.0)
        assert optimized.target_turnover <= 0.20 + 1e-8


@pytest.mark.parametrize("retain_cash", [False, True])
def test_optimizer_explicit_cash_maps_to_stock_execution_and_persisted_cash(
    tmp_path: Path, retain_cash: bool
) -> None:
    frames, fee_frame, _window = _inputs()
    panel = frames["panel"].assign(
        adj_open=100.0,
        adj_close=100.0,
        period_return=0.0,
        float_market_cap=1000.0,
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
            index=("industry_A01", "style"),
            columns=("industry_A01", "style"),
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
            {"asset_id": ["a", "b"], "industry_A01": 1.0, "style": [0.1, -0.1]}
        ),
        risk,
        production.AlphaCalibration(
            1.0, 0.0, 1.0, 1.0, 0.5, -1.0, 1.0, 0.0, 1.0, 6, 6, None
        ),
    )
    candidate = PortfolioCandidateSpec(
        candidate_id="cash_mapping",
        construction="benchmark_enhanced",
        prediction_target_id="prediction",
        prediction_selector="champion",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        benchmark_weights_target_id="benchmark",
        research_dataset_id="dataset",
        market_panel_target_id="panel",
        execution_mask_target_id="execution",
        trading_calendar_target_id="calendar",
        fee_schedule_target_id="fee",
        risk_style_columns=("style",),
        risk_context_target_ids=("risk",),
    )
    assumptions = ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,))
    with runtime_resource_policy_scope() as policy:
        account, returns, adv, optimized, delta = production._build_session(
            candidate=candidate,
            decision_timestamp=pd.Timestamp("2023-01-06"),
            next_timestamp=pd.Timestamp("2023-01-09"),
            panel=at_date(panel, "2023-01-06"),
            next_panel=at_date(panel, "2023-01-09"),
            endpoint_panel=at_date(panel, "2023-01-10"),
            dataset=at_date(frames["dataset"], "2023-01-06"),
            execution=at_date(frames["execution"], "2023-01-06"),
            holdings=pd.Series({"a": 45.0, "b": 45.0}),
            cash=10.0,
            nav=100.0,
            initialize_at_execution=False,
            amount_history={
                asset: deque([1000.0] * 20, maxlen=20) for asset in ("a", "b")
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
        assert optimized is not None and "__cash__" in optimized.weights
        assert "__cash__" not in set(account.asset_ledger.asset_id)
        if retain_cash:
            assert optimized.weights.loc["__cash__"] > 0.0
        ledger = production._ledger_frame(
            account=account,
            decision_timestamp=pd.Timestamp("2023-01-06"),
            returns=returns,
            alpha_score=alpha,
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
        envelope = build_business_manifest_envelope(
            logical_artifact_id="cash_mapping",
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window={
                "start": "2023-01-06",
                "end": "2023-01-10",
                "access": "development",
            },
            upstream_identities={},
            semantic_metadata={
                "portfolio_capital": {
                    "candidate_id": "cash_mapping",
                    "initial_capital": 100.0,
                    "valuation_currency": "CNY",
                    "valuation_method": "adjusted_notional",
                }
            },
        )
        root = write_test_sessions(
            tmp_path / "ledger", envelope=envelope, ledger_chunks=iter((ledger,))
        )
        saved = read_portfolio_ledger_artifact(root, expected_identity=root.name)
        (batch,) = tuple(saved.iter_chunks())
        assert set(session_frame(batch).asset_id) == {"a", "b", "__cash__"}
        assert policy.used_memory_bytes == 0


def test_simple_producer_terminal_sessions_are_truncated_after_prediction_lookup(
    tmp_path: Path,
    monkeypatch,
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
    (
        (_PredictionLookup(missing=True), "missing_required_prediction_keys"),
        (_PredictionLookup(extra=True), "misaligned_persisted_prediction_keys"),
    ),
)
def test_simple_producer_prediction_coverage_fails_closed(
    tmp_path: Path,
    monkeypatch,
    lookup: _PredictionLookup,
    error: str,
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


def test_simple_producer_missing_held_return_is_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames, fee_frame, output_window = _inputs()
    frames["panel"] = (
        frames["panel"]
        .loc[
            ~(
                frames["panel"]["asset_id"].eq("b")
                & pd.to_datetime(frames["panel"]["timestamp"])
                .dt.normalize()
                .eq(pd.Timestamp("2023-01-05"))
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
    )
    with pytest.raises(PortfolioTargetBlocked, match="portfolio_held_return_missing"):
        list(output)


def test_simple_producer_blocks_when_adv_warmup_is_short(
    tmp_path: Path,
    monkeypatch,
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
    with pytest.raises(
        PortfolioTargetBlocked,
        match="portfolio_adv_warmup_insufficient",
    ):
        list(output)


def test_simple_producer_distinguishes_fee_coverage_from_malformed_fee(
    tmp_path: Path,
    monkeypatch,
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
    with pytest.raises(
        PortfolioTargetBlocked,
        match="portfolio_fee_coverage_missing",
    ):
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


def test_simple_producer_resource_state_lives_across_yield_and_closes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    frames, fee_frame, output_window = _inputs()
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    real_convert = production.continuous_ledger_batch
    conversion_leases = []

    def admitted_conversion(frame, sequence, *, end_timestamp):
        leases = [
            lease
            for lease in policy._leases
            if lease.owner == "research.portfolio.session_workspace"
        ]
        assert len(leases) == 1
        assert (
            leases[0].reserved_bytes
            >= len(frame) * len(CONTINUOUS_EXECUTION_COLUMNS) * 256
        )
        conversion_leases.append(leases[0])
        return real_convert(frame, sequence, end_timestamp=end_timestamp)

    monkeypatch.setattr(production, "continuous_ledger_batch", admitted_conversion)
    output, _calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
        resource_policy=policy,
    )
    with runtime_resource_policy_scope(policy):
        first = next(output)
        assert not first.empty
        assert policy.used_memory_bytes > 0
        assert conversion_leases and conversion_leases[0].active
        output.close()
        assert not conversion_leases[0].active
    assert policy.used_memory_bytes == 0


def test_simple_producer_is_invariant_to_asset_major_input_permutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames, fee_frame, output_window = _inputs()
    first, _calls = _build_producer(
        tmp_path / "first",
        monkeypatch,
        frames=frames,
        fee_frame=fee_frame,
        output_window=output_window,
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
    )
    right = [frame.reset_index(drop=True) for frame in second]
    for expected, actual in zip(left, right, strict=True):
        pd.testing.assert_frame_equal(expected, actual)


def test_simple_producer_retains_zero_activity_suspension_rows(
    tmp_path: Path,
    monkeypatch,
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
    assert panel_reads[0] == sessions[warmup_position]


def test_simple_producer_handles_universe_arrival_and_zero_weight_departure(
    tmp_path: Path,
    monkeypatch,
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
                    "adj_open": 100.0,
                    "adj_close": 100.0,
                    "amount": 1_000.0,
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
                    "float_market_cap": 1_000.0,
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
    tmp_path: Path,
    monkeypatch,
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
        PortfolioTargetBlocked,
        match="portfolio_execution_window_empty",
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
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": assets,
            "industry_level1_code": ["A", "B"] * 60 + ["A"],
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
            "asset_id": [f"synthetic_{i}" for i in range(100)],
            "timestamp": pd.Timestamp("2023-01-03"),
            "industry_level1_code": "I",
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
            "asset_id": ["a", "b"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "period_return": [0.01, -0.01],
            "float_market_cap": [100.0, 200.0],
            "industry_level1_code": ["A", "B"],
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": False,
            "is_pt": False,
            "listing_age_trading_days": 300,
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
