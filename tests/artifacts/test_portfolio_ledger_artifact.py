from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.artifacts.portfolio_ledger as ledger_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    CONTINUOUS_EXECUTION_COLUMNS,
    ExecutedTrade,
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
    PortfolioValuation,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.binance_hft import KNOWN_NO_FUNDING
from tests.helpers.portfolio_ledger import session_frame, write_test_sessions
from tests.helpers.runtime_resources import capture_test_policy


@pytest.mark.parametrize("boundary", ("batch", "cross_session"))
def test_perpetual_quantity_cannot_be_created_within_nav_amount_tolerance(
    tmp_path: Path, boundary: str
) -> None:
    capital = PortfolioCapital("quantity", 1e9, "USDT", "linear_perpetual")
    envelope = build_business_manifest_envelope(
        logical_artifact_id="quantity",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01",
            "end": "1970-01-02",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def batch(sequence, quantities):
        return PortfolioLedgerBatch(
            sequence,
            (),
            (),
            tuple(
                row
                for index, (phase, quantity) in enumerate(
                    zip(("before", "after", "end"), quantities, strict=True)
                )
                for row in (
                    PortfolioValuation(
                        sequence * 3 + index, "__cash__", phase, 1e9, 1e9
                    ),
                    PortfolioValuation(
                        sequence * 3 + index,
                        "asset",
                        phase,
                        quantity,
                        quantity * 100000.0,
                        100000.0,
                    ),
                )
            ),
            (),
        )

    baseline = replace(
        batch(0, (0.0, 0.0, 0.0)),
        trades=tuple(
            ExecutedTrade(
                str(index), str(index), 1, "asset", quantity, quantity * 100.0
            )
            for index, quantity in enumerate((0.1, 0.2, -0.3))
        ),
    )
    with runtime_resource_policy_scope() as policy:
        old_root = ledger_artifact.write_portfolio_ledger_artifact(
            tmp_path / "old", envelope=envelope, produce=lambda emit: emit(baseline)
        )
        old = read_portfolio_ledger_artifact(
            old_root, expected_identity=envelope.instance_identity
        )
        assert len(tuple(old.iter_chunks())) == 1
        assert ledger_artifact.summarize_portfolio_ledger(old)["net_pnl"] == 0.0
        old_bytes = {
            path.relative_to(old_root): path.read_bytes()
            for path in old_root.rglob("*")
            if path.is_file()
        }

        def invalid(emit):
            if boundary == "batch":
                emit(batch(0, (0.0, 1e-5, 1e-5)))
            else:
                emit(baseline)
                emit(batch(1, (1e-5, 1e-5, 1e-5)))

        error = (
            "portfolio_ledger_position_reconciliation_invalid"
            if boundary == "batch"
            else "portfolio_ledger_cross_session_state_invalid"
        )
        candidate = tmp_path / "candidate"
        with pytest.raises(ValueError, match=error):
            ledger_artifact.write_portfolio_ledger_artifact(
                candidate, envelope=envelope, produce=invalid
            )
        assert not (candidate / "instances" / envelope.instance_identity).exists()
        assert {
            path.relative_to(old_root): path.read_bytes()
            for path in old_root.rglob("*")
            if path.is_file()
        } == old_bytes
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("mode", ("maker", "taker"))
@pytest.mark.parametrize("funding_rate", (0.001, -0.001))
def test_perpetual_execution_roundtrip_keeps_collateral_funding_and_delayed_fill_links(
    tmp_path, mode, funding_rate
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
        simulate_taker,
    )
    from tests.markets.crypto.test_hft_execution import (
        _coverage,
        _maker_config,
        _quote,
        _state,
        _taker_config,
        _trade,
    )

    capital = PortfolioCapital(mode, 1000.0, "USDT", "linear_perpetual")
    envelope = build_business_manifest_envelope(
        logical_artifact_id=mode,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01T00:00:00",
            "end": "1970-01-01T00:00:01",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def run(emit, policy, rate):
        kwargs = dict(
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(FundingEvent(75_000_000, rate, 100.0),),
            on_ledger=emit,
            resource_policy=policy,
            observed_coverage={"s1": _coverage(0, 100)},
        )
        if mode == "taker":
            return simulate_taker(
                [_state(0), _state(100)],
                [
                    {"timestamp_ms": 0, "prediction": 0.02},
                    {"timestamp_ms": 50, "prediction": 0.02},
                ],
                funding_forecasts=(
                    FundingForecast(0, 1_000_000_000, ((75_000_000, None),)),
                ),
                prediction_horizon_ms=50,
                prediction_column="prediction",
                config=_taker_config(),
                **kwargs,
            )
        return simulate_maker(
            [_state(0), _state(100)],
            [_trade(50, price="99.4", quantity="1", maker_buyer=True)],
            [_quote(0, book_segment_id="s1")],
            config=_maker_config(),
            trade_coverage={"trade-connection-1": _coverage(0, 100)},
            **kwargs,
        )

    policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with runtime_resource_policy_scope(policy):
        root = write_portfolio_ledger_artifact(
            tmp_path / mode,
            envelope=envelope,
            produce=lambda emit: run(emit, policy, funding_rate),
        )
        artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
        summary = summarize_portfolio_ledger(artifact)
        batches = list(artifact.iter_chunks())
    expected_gross = -0.5 if mode == "taker" else 0.5
    expected_fee = 0.01005 if mode == "taker" else -0.00995
    assert summary["gross_pnl"] == pytest.approx(expected_gross)
    assert summary["fees"] == pytest.approx(expected_fee)
    assert summary["funding"] == pytest.approx(-100.0 * funding_rate)
    assert summary["net_pnl"] == pytest.approx(
        expected_gross - expected_fee - 100.0 * funding_rate
    )
    if mode == "taker":
        assert any(
            row.text == "portfolio_expected_funding_missing"
            for batch in batches
            for row in batch.diagnostics
        )
    trade_batch = next(batch for batch in batches if batch.trades)
    from quant_research_platform.contracts.portfolios import (
        validate_portfolio_transition,
    )

    phantom = replace(
        trade_batch,
        valuations=tuple(
            replace(row, cost_basis=row.cost_basis - 5.0, value=row.value + 5.0)
            if row.asset_id == "btc_usdt" and row.phase == "end"
            else row
            for row in trade_batch.valuations
        ),
    )
    with pytest.raises(ValueError, match="portfolio_ledger_cost_basis_invalid"):
        validate_portfolio_transition(phantom, capital, None)
    with pytest.raises(ValueError, match="portfolio_ledger_fill_timing_invalid"):
        replace(
            trade_batch,
            trades=tuple(
                replace(row, timestamp_ns=row.timestamp_ns + 1)
                for row in trade_batch.trades
            ),
        )
    with pytest.raises(ValueError, match="portfolio_ledger_cashflow_timing_invalid"):
        replace(
            trade_batch,
            cashflows=tuple(
                replace(row, timestamp_ns=row.timestamp_ns + 1)
                for row in trade_batch.cashflows
            ),
        )
    assert not any(
        flow.kind == "principal" for batch in batches for flow in batch.cashflows
    )
    assert next(
        row.value
        for row in trade_batch.valuations
        if row.asset_id == "__cash__" and row.phase == "after"
    ) == pytest.approx(1000.0 - expected_fee)
    fill_id = trade_batch.trades[0].fill_id
    assert any(
        batch.sequence > trade_batch.sequence
        and any(row.fill_id == fill_id for row in batch.diagnostics)
        for batch in batches
    )
    old_manifest = (root / "manifest.json").read_bytes()
    policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with (
        runtime_resource_policy_scope(policy),
        pytest.raises(
            PortfolioTargetBlocked, match="portfolio_required_funding_missing"
        ),
    ):
        write_portfolio_ledger_artifact(
            tmp_path / (mode + "_missing"),
            envelope=envelope,
            produce=lambda emit: run(emit, policy, None),
        )
    assert (root / "manifest.json").read_bytes() == old_manifest


def test_orphan_fill_and_unknown_delayed_fill_fail_closed(tmp_path):
    from quant_research_platform.artifacts.portfolio_ledger import (
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioDiagnostic
    from quant_research_platform.research.portfolios.accounting import (
        continuous_ledger_batch,
    )

    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    batch = continuous_ledger_batch(frame, 0, end_timestamp=pd.Timestamp("2019-01-02"))
    with pytest.raises(ValueError, match="portfolio_ledger_orphan_economic_fact"):
        replace(
            batch,
            trades=(
                *batch.trades,
                ExecutedTrade("orphan", "order", 1, "missing", 1.0, 1.0),
            ),
        )
    bad = replace(
        batch,
        diagnostics=(
            *batch.diagnostics,
            PortfolioDiagnostic("asset", "missing", "markout", number=0.5),
        ),
    )
    with pytest.raises(ValueError, match="portfolio_ledger_diagnostic_fill_mismatch"):
        write_portfolio_ledger_artifact(
            tmp_path / "bad", envelope=_envelope(), produce=lambda emit: emit(bad)
        )
    assert not (tmp_path / "bad" / "instances" / _envelope().instance_identity).exists()


def test_adjusted_return_terminal_nav_is_valued_at_actual_interval_end(
    tmp_path,
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        summarize_portfolio_ledger,
    )

    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    frame.loc[frame["asset_id"].eq("asset"), "holding_return"] = 0.1
    frame.loc[frame["asset_id"].eq("asset"), "end_notional"] = 110.0
    frame.attrs["end_timestamp"] = pd.Timestamp("2019-01-03")
    root = write_test_sessions(
        tmp_path / "interval", envelope=_envelope(), ledger_chunks=(frame,)
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    batch = next(artifact.iter_chunks())
    asset_end = next(
        row
        for row in batch.valuations
        if row.asset_id == "asset" and row.phase == "end"
    )
    assert pd.Timestamp(asset_end.timestamp_ns) == pd.Timestamp("2019-01-03")
    assert asset_end.value == pytest.approx(110.0)
    summary = summarize_portfolio_ledger(artifact)
    assert summary["nav"] == 110.0
    assert pd.Timestamp(summary["valuation_timestamp_ns"]) == pd.Timestamp("2019-01-03")
    assert sum(row.value for row in batch.valuations if row.phase == "after") == 100.0
    assert {flow.kind for flow in batch.cashflows} == {"principal"}
    assert not {"holding_return", "end_notional", "post_trade_notional"} & {
        row.name for row in batch.diagnostics
    }


def test_full_session_crossing_physical_batches_is_admitted_and_released(tmp_path):
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    assets = pd.concat([frame.iloc[[0]]] * 2000, ignore_index=True)
    assets["asset_id"] = [f"asset_{index:04}" for index in range(2000)]
    for column in (
        "target_notional",
        "executed_delta_notional",
        "post_trade_notional",
        "end_notional",
    ):
        assets[column] = 0.05
    frame = pd.concat([assets, frame.iloc[[-1]]], ignore_index=True)
    root = write_test_sessions(
        tmp_path / "large_session", envelope=_envelope(), ledger_chunks=(frame,)
    )
    policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with runtime_resource_policy_scope(policy):
        artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
        chunks = artifact.iter_chunks()
        batch = next(chunks)
        assert len(batch.trades) == 2000
        assert policy.used_memory_bytes > 4 * 1024**2
        chunks.close()
    assert policy.used_memory_bytes == 0
    limited = capture_test_policy(
        available_memory_bytes=128 * 1024**2, explicit_cap_bytes=8 * 1024**2
    )
    with runtime_resource_policy_scope(limited), pytest.raises(MemoryError):
        read_portfolio_ledger_artifact(root, expected_identity=root.name)
    assert limited.used_memory_bytes == 0


def _envelope() -> object:
    return build_business_manifest_envelope(
        logical_artifact_id="portfolio-ledger",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-01-03",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "synthetic_portfolio",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )


def _session(
    decision: str,
    execution: str,
    *,
    pre_asset: float,
    target_asset: float,
    post_asset: float,
    pre_cash: float,
    target_cash: float,
    post_cash: float,
) -> pd.DataFrame:
    rows = [
        {
            "decision_timestamp": decision,
            "execution_date": execution,
            "asset_id": "asset",
            "pre_trade_notional": pre_asset,
            "target_notional": target_asset,
            "executed_delta_notional": post_asset - pre_asset,
            "post_trade_notional": post_asset,
            "holding_return": 0.0,
            "end_notional": post_asset,
            "block_reason": None,
            "adv20": 1000.0,
            "alpha_score": 1.0,
            "expected_alpha": np.nan,
            "benchmark_weight": np.nan,
            "asset_specific_variance": np.nan,
            "decision_reason": None,
        },
        {
            "decision_timestamp": decision,
            "execution_date": execution,
            "asset_id": "__cash__",
            "pre_trade_notional": pre_cash,
            "target_notional": target_cash,
            "executed_delta_notional": post_cash - pre_cash,
            "post_trade_notional": post_cash,
            "holding_return": 0.0,
            "end_notional": post_cash,
            "block_reason": None,
            "adv20": np.nan,
            "alpha_score": np.nan,
            "expected_alpha": np.nan,
            "benchmark_weight": np.nan,
            "asset_specific_variance": np.nan,
            "decision_reason": "simple_rule",
        },
    ]
    frame = pd.DataFrame.from_records(rows)
    for column in CONTINUOUS_EXECUTION_COLUMNS:
        if column not in frame:
            frame[column] = 0.0 if column.endswith("_cost") else np.nan
    for column in (
        "commission_cost",
        "stamp_duty_cost",
        "transfer_fee_cost",
        "slippage_cost",
        "impact_cost",
    ):
        frame[column] = 0.0
    for column in (
        "alpha_slope",
        "alpha_slope_se",
        "alpha_slope_t",
        "alpha_slope_pvalue",
        "alpha_slope_ci_low",
        "alpha_slope_ci_high",
        "alpha_weighted_r2",
        "alpha_coverage",
        "alpha_valid_dates",
        "alpha_scheduled_dates",
        "predicted_tracking_error",
        "portfolio_factor_variance",
        "portfolio_specific_variance",
        "expected_transaction_cost",
    ):
        frame.loc[frame["asset_id"] != "__cash__", column] = np.nan
    return frame.loc[:, list(CONTINUOUS_EXECUTION_COLUMNS)]


@pytest.mark.parametrize("long_path", (False, True))
def test_portfolio_ledger_roundtrip_preserves_complete_sessions(
    tmp_path: Path,
    long_path: bool,
) -> None:
    if long_path:
        tmp_path = tmp_path / ("long" * 25) / ("path" * 25)
    first = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    second = _session(
        "2019-01-02",
        "2019-01-03",
        pre_asset=100.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=0.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_test_sessions(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame for frame in (first, second)),
    )
    with pytest.raises(ValueError, match="output_already_exists"):
        write_test_sessions(
            tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(first, second)
        )

    artifact = read_portfolio_ledger_artifact(
        root,
        expected_identity=root.name,
    )
    if long_path:
        assert len(str(root / "trades.parquet")) > 260
    chunks = [session_frame(batch) for batch in artifact.iter_chunks()]
    assert len(chunks) == 2
    assert [chunk["asset_id"].tolist() for chunk in chunks] == [
        ["asset", "__cash__"],
        ["asset", "__cash__"],
    ]


def test_portfolio_ledger_member_mutation_fails_closed(tmp_path: Path) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_test_sessions(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    with (root / "valuations.parquet").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_portfolio_ledger_artifact(root, expected_identity=root.name)


def test_portfolio_ledger_rejects_an_extra_persisted_member(tmp_path: Path) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_test_sessions(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    (root / "extra.parquet").write_bytes(b"not-a-ledger")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_portfolio_ledger_artifact(root, expected_identity=root.name)


def test_portfolio_ledger_reconciles_cash_rotation_costs_and_zero_asset_carry(
    tmp_path: Path,
) -> None:
    first = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=100.0,
        target_asset=0.0,
        post_asset=0.0,
        pre_cash=0.0,
        target_cash=0.0,
        post_cash=1.0,
    )
    first.loc[first["asset_id"].eq("asset"), "commission_cost"] = 1.0
    first.loc[first["asset_id"].eq("__cash__"), "executed_delta_notional"] = 1.0
    buyer = first.loc[first["asset_id"].eq("asset")].copy()
    buyer["asset_id"] = "buyer"
    buyer["pre_trade_notional"] = 0.0
    buyer["target_notional"] = 100.0
    buyer["executed_delta_notional"] = 98.0
    buyer["post_trade_notional"] = 98.0
    buyer["end_notional"] = 98.0
    buyer["commission_cost"] = 0.0
    buyer["block_reason"] = "cash_pro_rata"
    seller = first.loc[first["asset_id"].eq("asset")].copy()
    cash_row = first.loc[first["asset_id"].eq("__cash__")].copy()
    first = pd.concat([seller, buyer, cash_row], ignore_index=True)
    second = first.loc[first["asset_id"].isin(("buyer", "__cash__"))].copy()
    second["decision_timestamp"] = "2019-01-02"
    second["execution_date"] = "2019-01-03"
    second.loc[second["asset_id"].eq("buyer"), "pre_trade_notional"] = 98.0
    second.loc[second["asset_id"].eq("buyer"), "target_notional"] = 98.0
    second.loc[second["asset_id"].eq("buyer"), "post_trade_notional"] = 98.0
    second.loc[second["asset_id"].eq("buyer"), "end_notional"] = 98.0
    second.loc[second["asset_id"].eq("__cash__"), "pre_trade_notional"] = 1.0
    second.loc[second["asset_id"].eq("__cash__"), "target_notional"] = 1.0
    second["executed_delta_notional"] = 0.0
    second["post_trade_notional"] = second["target_notional"]
    second["end_notional"] = second["post_trade_notional"]
    second["block_reason"] = None
    root = write_test_sessions(
        tmp_path / "rotation",
        envelope=_envelope(),
        ledger_chunks=(first, second),
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    chunks = [session_frame(batch) for batch in artifact.iter_chunks()]
    assert float(chunks[0]["post_trade_notional"].sum()) == 99.0
    assert chunks[0]["target_notional"].sum() == 100.0
    assert (
        chunks[0]
        .loc[chunks[0]["asset_id"].eq("__cash__"), "executed_delta_notional"]
        .iloc[0]
        == 1.0
    )
    assert chunks[1]["asset_id"].tolist() == ["buyer", "__cash__"]


def test_portfolio_ledger_diagnostic_counts_are_integral(
    tmp_path: Path,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_valid_dates"] = 3
    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_scheduled_dates"] = 4
    root = write_test_sessions(
        tmp_path / "valid",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    read = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    result = session_frame(next(read.iter_chunks()))
    assert (
        result.loc[result["asset_id"].eq("__cash__"), "alpha_valid_dates"].iloc[0] == 3
    )

    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_valid_dates"] = 1.5
    with pytest.raises(ValueError, match="portfolio_ledger_count_invalid"):
        write_test_sessions(
            tmp_path / "invalid",
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )


def test_portfolio_ledger_writer_keeps_native_state_until_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    real_writer = ledger_artifact.pq.ParquetWriter
    close_usage: list[int] = []

    class ObservedWriter:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return self._writer.write_table(*args, **kwargs)

        def add_key_value_metadata(self, *args, **kwargs):
            return self._writer.add_key_value_metadata(*args, **kwargs)

        def close(self):  # noqa: ANN201
            close_usage.append(policy.used_memory_bytes)
            return self._writer.close()

    monkeypatch.setattr(ledger_artifact.pq, "ParquetWriter", ObservedWriter)
    with runtime_resource_policy_scope(policy):
        root = write_test_sessions(
            tmp_path / "portfolio",
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )
    assert policy.used_memory_bytes == 0
    assert close_usage and close_usage[0] > 0
    read_portfolio_ledger_artifact(root, expected_identity=root.name)


def test_portfolio_ledger_frame_lease_covers_validation_and_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    active = False
    observed = {"validated": False, "written": False}
    real_reservation = ledger_artifact._parquet_frame_reservation
    real_validate = ledger_artifact.validate_portfolio_transition
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    @contextmanager
    def reservation(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal active
        with real_reservation(*args, **kwargs):
            active = True
            try:
                yield
            finally:
                active = False

    def validate(*args, **kwargs):  # noqa: ANN002, ANN003
        assert policy.used_memory_bytes > 0
        observed["validated"] = True
        return real_validate(*args, **kwargs)

    real_writer = ledger_artifact.pq.ParquetWriter

    class ObservedWriter:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            assert active
            observed["written"] = True
            return self._writer.write_table(*args, **kwargs)

        def add_key_value_metadata(self, *args, **kwargs):
            return self._writer.add_key_value_metadata(*args, **kwargs)

        def close(self):  # noqa: ANN201
            return self._writer.close()

    monkeypatch.setattr(ledger_artifact, "_parquet_frame_reservation", reservation)
    monkeypatch.setattr(ledger_artifact, "validate_portfolio_transition", validate)
    monkeypatch.setattr(ledger_artifact.pq, "ParquetWriter", ObservedWriter)
    with runtime_resource_policy_scope(policy):
        write_test_sessions(
            tmp_path / "portfolio",
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )
    assert observed == {"validated": True, "written": True}


def test_portfolio_ledger_small_row_groups_preserve_complete_transitions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    real_table_state = ledger_artifact._parquet_table_write_state

    def one_row_physical_batches(*args, **kwargs):  # noqa: ANN002, ANN003
        persisted, state, _batch_rows, retained = real_table_state(*args, **kwargs)
        return persisted, state, 1, retained

    monkeypatch.setattr(
        ledger_artifact,
        "_parquet_table_write_state",
        one_row_physical_batches,
    )
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        root = write_test_sessions(
            tmp_path / "portfolio",
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )
        artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
        assert len(list(artifact.iter_chunks())) == 1
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("long_path", (False, True))
def test_portfolio_ledger_primary_error_survives_writer_close_failure(
    tmp_path: Path,
    monkeypatch,
    long_path: bool,
) -> None:
    if long_path:
        tmp_path = tmp_path / ("long" * 25) / ("path" * 25)
    first = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    second = first.drop(columns=["decision_reason"])
    real_writer = ledger_artifact.pq.ParquetWriter

    class FailingCloseWriter:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return self._writer.write_table(*args, **kwargs)

        def add_key_value_metadata(self, *args, **kwargs):
            return self._writer.add_key_value_metadata(*args, **kwargs)

        def close(self):  # noqa: ANN201
            self._writer.close()
            raise OSError("secondary_close_failure")

    monkeypatch.setattr(ledger_artifact.pq, "ParquetWriter", FailingCloseWriter)
    with pytest.raises(ValueError, match="portfolio_ledger_schema_invalid") as raised:
        write_test_sessions(
            tmp_path / "portfolio",
            envelope=_envelope(),
            ledger_chunks=(first, second),
        )
    assert any("close also failed" in note for note in raised.value.__notes__)
    from quant_research_platform.storage.identity import native_filesystem_path

    assert not list(
        Path(native_filesystem_path(tmp_path / "portfolio" / "instances")).iterdir()
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("negative_adv", "portfolio_ledger_adv_invalid"),
        ("zero_trade_adv", "portfolio_ledger_adv_invalid"),
        ("bool_numeric", "portfolio_ledger_numeric_invalid"),
        ("block_reason", "portfolio_ledger_block_reason_invalid"),
        ("decision_reason", "portfolio_ledger_decision_reason_invalid"),
    ),
)
def test_portfolio_ledger_trust_boundary_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    asset = frame["asset_id"].eq("asset")
    if mutation == "negative_adv":
        frame.loc[asset, "adv20"] = -1.0
    elif mutation == "zero_trade_adv":
        frame.loc[asset, "adv20"] = 0.0
    elif mutation == "bool_numeric":
        frame["adv20"] = frame["adv20"].astype(object)
        frame.loc[asset, "adv20"] = True
    elif mutation == "block_reason":
        frame.loc[asset, "block_reason"] = "unknown_reason"
    else:
        frame.loc[asset, "decision_reason"] = "unknown_reason"
    with pytest.raises(ValueError, match=error):
        write_test_sessions(
            tmp_path / mutation,
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )


def test_portfolio_ledger_rechecks_outer_manifest_and_member_set_each_iteration(
    tmp_path: Path,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_test_sessions(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"].append({"path": "extra.parquet", "size": 0, "sha256": "0" * 64})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        next(artifact.iter_chunks())


@pytest.mark.parametrize(
    ("mode", "case"),
    [
        ("taker", "late_quote"),
        ("maker", "late_quote"),
        ("taker", "no_quote"),
        ("maker", "no_quote"),
        ("taker", "closed_before_quote_expiry"),
    ],
)
def test_perpetual_terminal_marks_use_real_quote_or_cash_only(tmp_path, mode, case):
    from quant_research_platform.artifacts.portfolio_ledger import (
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
        simulate_taker,
    )
    from tests.markets.crypto.test_hft_execution import (
        _coverage,
        _maker_config,
        _quote,
        _state,
        _taker_config,
    )

    capital = PortfolioCapital(mode, 1000.0, "USDT", "linear_perpetual")
    envelope = build_business_manifest_envelope(
        logical_artifact_id=mode,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01T00:00:00",
            "end": "1970-01-01T00:00:01",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )
    books = (
        [_state(5)]
        if case == "late_quote"
        else []
        if case == "no_quote"
        else [_state(0)]
    )
    decisions = [{"timestamp_ms": 0, "prediction": 0.02}]
    if case == "closed_before_quote_expiry":
        decisions += [
            {"timestamp_ms": 10, "prediction": -0.02},
            {"timestamp_ms": 30, "prediction": 0.0},
        ]
    result = {}
    with runtime_resource_policy_scope() as policy:

        def produce(emit):
            kwargs = dict(
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                resource_policy=policy,
                on_ledger=emit,
                observed_coverage={"s1": _coverage(0, 20)},
            )
            if mode == "taker":
                result.update(
                    simulate_taker(
                        books,
                        decisions,
                        funding_forecasts=KNOWN_NO_FUNDING,
                        prediction_column="prediction",
                        prediction_horizon_ms=100,
                        config=replace(_taker_config(), position_size=1.0),
                        **kwargs,
                    )
                )
            else:
                result.update(
                    simulate_maker(
                        books,
                        [],
                        [_quote(0, book_segment_id="s1")],
                        config=_maker_config(),
                        trade_coverage={"trade-connection-1": _coverage(0, 20)},
                        **kwargs,
                    )
                )

        root = write_portfolio_ledger_artifact(
            tmp_path / mode, envelope=envelope, produce=produce
        )
        artifact = read_portfolio_ledger_artifact(
            root, expected_identity=envelope.instance_identity
        )
        chunks = list(artifact.iter_chunks())
        terminal = chunks[-1]
        if case == "late_quote":
            marks = [row for row in terminal.valuations if row.asset_id == "btc_usdt"]
            assert len(marks) == 3 and all(row.mark_price == 100.0 for row in marks)
            assert all(row.timestamp_ns == 5_000_000 for row in terminal.valuations)
        else:
            assert len(terminal.valuations) == 3
            assert {row.asset_id for row in terminal.valuations} == {"__cash__"}
        summary = summarize_portfolio_ledger(artifact)
        if case == "closed_before_quote_expiry":
            assert (
                result["filled_orders"] == 2
                and result["markout_undefined_fill_events"] == 2
            )
            assert summary["nav"] == pytest.approx(998.98)
            assert all(row.timestamp_ns == 30_000_000 for row in terminal.valuations)
            assert (
                sum(
                    row.fill_id is not None
                    for chunk in chunks[:-1]
                    for row in chunk.diagnostics
                )
                > 0
            )
        else:
            assert summary["nav"] == result["nav"] == 1000.0
        chunks = terminal = None
        assert policy.used_memory_bytes == 0
