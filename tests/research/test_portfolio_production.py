from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_LEDGER_COLUMNS,
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
    require_fee_coverage,
    trade_cost,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.risk import (
    FactorRiskEstimate,
    MarketModelRiskStyles,
)
from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
from quant_research_platform.storage.datasets import DatasetReadMode
from quant_research_platform.storage.layout import StorageLayout


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
                "amount": 1_000.0,
                "market_type": 1,
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
):
    enhanced = candidate is not None and candidate.construction == "benchmark_enhanced"
    if candidate is None:
        candidate = PortfolioCandidateSpec(
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
        )
    }
    roots = {}
    for target_id, identity in identities.items():
        root = tmp_path / target_id / identity
        root.mkdir(parents=True)
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
        assert mode is DatasetReadMode.MATERIALIZED
        assert resource_policy is expected_policy
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
        expected = {
            "dataset": (
                (
                    "asset_id",
                    "timestamp",
                    "eligible",
                    "float_market_cap",
                    *(
                        column
                        for column in RISK_STYLE_COLUMNS
                        if enhanced
                        and column in frames["dataset"].columns
                        and column not in {"beta_252d", "residual_volatility_60d"}
                    ),
                )
            ),
            "panel": (
                production._ENHANCED_PANEL_COLUMNS
                if enhanced
                else ("asset_id", "timestamp", "adj_open", "amount", "market_type")
            ),
            "execution": ("asset_id", "timestamp", "can_enter", "can_exit"),
        }[target_id]
        if columns is None:
            assert target_id == "dataset"
            selected = frames[target_id].loc[
                pd.to_datetime(frames[target_id]["timestamp"]).dt.normalize()
                == pd.Timestamp(time_window.start).normalize()
            ]
            return selected.copy()
        assert columns == expected
        assert isinstance(time_window, TimeWindow)
        assert time_window.start == time_window.end
        selected = frames[target_id].loc[
            pd.to_datetime(frames[target_id]["timestamp"]).dt.normalize()
            == pd.Timestamp(time_window.start).normalize()
        ]
        return selected.loc[:, list(columns)].copy()

    expected_policy = resource_policy
    monkeypatch.setattr(production, "read_dataset", fake_read_dataset)
    assumptions = ExecutionAssumptions(
        initial_nav=100.0,
        slippage_bps=0.0,
        impact_bps=0.0,
        adv_participation=1.0,
        adv_sensitivities=(1.0,),
    )
    chunks = production.build_portfolio_ledger(
        candidate=candidate,
        layout=layout,
        upstream_identities={**identities, "prediction": "b" * 64},
        output_window=output_window,
        input_window=input_window or TimeWindow("2022-12-01", "2023-01-10"),
        formal_dataset_window=formal_dataset_window
        or TimeWindow("2022-12-01", "2023-01-10"),
        prediction_lookup=prediction_lookup or _PredictionLookup(),
        assumptions=assumptions,
        resource_policy=resource_policy,
        fee_schedule_from_frame=fee_schedule_from_frame,
        require_fee_coverage=require_fee_coverage,
        trade_cost=trade_cost,
        market_name_for_type=lambda _value: "shanghai",
        unified_trading_sessions=unified_trading_sessions,
    )
    return chunks, calls


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
    assert all(tuple(frame.columns) == PORTFOLIO_LEDGER_COLUMNS for frame in frames_out)
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
        semantic_metadata={"ledger_schema": list(PORTFOLIO_LEDGER_COLUMNS)},
    )
    artifact_root = write_portfolio_ledger_artifact(
        tmp_path / "artifacts",
        envelope=envelope,
        ledger_chunks=iter(frames_out),
    )
    artifact = read_portfolio_ledger_artifact(
        artifact_root,
        expected_identity=artifact_root.name,
    )
    assert len(list(artifact.iter_chunks())) == len(frames_out)
    assert np.isclose(frames_out[0]["post_trade_notional"].sum(), 100.0 - 0.04)


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
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        prediction_target_id="prediction",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        benchmark_weights_target_id="benchmark",
        risk_style_columns=RISK_STYLE_COLUMNS,
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

    def fixed_market_styles(self, panel):
        del self, panel
        return MarketModelRiskStyles(
            beta=pd.Series({"a": 1.0, "b": 1.0}),
            residual_volatility=pd.Series({"a": 0.1, "b": 0.1}),
        )

    monkeypatch.setattr(
        production.MarketModelRiskState,
        "observe",
        fixed_market_styles,
    )
    monkeypatch.setattr(
        production.FactorRiskState,
        "estimate",
        lambda self: risk,
    )
    candidate = PortfolioCandidateSpec(
        candidate_id="synthetic_csi500_enhanced",
        construction="benchmark_enhanced",
        prediction_target_id="prediction",
        rebalance_schedule="weekly-last-trading-day",
        top_quantile=None,
        benchmark_id="csi500",
        benchmark_weights_target_id="benchmark",
        risk_style_columns=RISK_STYLE_COLUMNS,
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
        input_window=TimeWindow("2022-11-14", "2023-01-10"),
        candidate=candidate,
    )
    result = list(chunks)
    optimized = [
        frame
        for frame in result
        if frame["decision_reason"].eq("enhanced_optimizer").any()
    ]
    assert optimized
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
        output.close()
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
