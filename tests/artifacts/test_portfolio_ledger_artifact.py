from __future__ import annotations

import json
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
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_iterdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_replace,
    filesystem_stat,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)
from tests.helpers.binance_hft import KNOWN_NO_FUNDING
from tests.helpers.portfolio_ledger import session_frame, write_test_sessions


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


@pytest.mark.parametrize("boundary", ("batch", "cross_session"))
def test_perpetual_quantity_cannot_be_created_within_nav_amount_tolerance(
    tmp_path: Path, boundary: str
) -> None:
    capital = PortfolioCapital("quantity", 1000000000.0, "USDT", "linear_perpetual")
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
                (
                    row
                    for index, (phase, quantity) in enumerate(
                        zip(("before", "after", "end"), quantities, strict=True)
                    )
                    for row in (
                        PortfolioValuation(
                            sequence * 3 + index,
                            "__cash__",
                            phase,
                            1000000000.0,
                            1000000000.0,
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
                )
            ),
            (),
        )

    baseline = replace(
        batch(0, (0.0, 0.0, 0.0)),
        trades=tuple(
            (
                ExecutedTrade(
                    str(index), str(index), 1, "asset", quantity, quantity * 100.0
                )
                for index, quantity in enumerate((0.1, 0.2, -0.3))
            )
        ),
    )
    old_root = ledger_artifact.write_portfolio_ledger_artifact(
        tmp_path / "old", envelope=envelope, produce=lambda emit: emit(baseline)
    )
    old = read_portfolio_ledger_artifact(
        old_root, expected_identity=envelope.instance_identity
    )
    assert len(tuple(old.iter_chunks())) == 1
    assert (
        ledger_artifact.summarize_portfolio_ledger(old.capital, old.iter_chunks())[
            "net_pnl"
        ]
        == 0.0
    )
    old_bytes = _file_bytes(old_root)

    def invalid(emit):
        if boundary == "batch":
            emit(batch(0, (0.0, 1e-05, 1e-05)))
        else:
            emit(baseline)
            emit(batch(1, (1e-05, 1e-05, 1e-05)))

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
    assert not filesystem_exists(candidate / "instances" / envelope.instance_identity)
    assert _file_bytes(old_root) == old_bytes


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

    def run(emit, rate):
        kwargs = dict(
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(FundingEvent(75000000, rate, 100.0),),
            on_ledger=emit,
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
                    FundingForecast(0, 1000000000, ((75000000, None),)),
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

    root = write_portfolio_ledger_artifact(
        tmp_path / mode, envelope=envelope, produce=lambda emit: run(emit, funding_rate)
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    summary = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
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
    old_manifest = filesystem_read_bytes(root / "manifest.json")
    with pytest.raises(
        PortfolioTargetBlocked, match="portfolio_required_funding_missing"
    ):
        write_portfolio_ledger_artifact(
            tmp_path / (mode + "_missing"),
            envelope=envelope,
            produce=lambda emit: run(emit, None),
        )
    assert filesystem_read_bytes(root / "manifest.json") == old_manifest


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
    assert not filesystem_exists(
        tmp_path / "bad" / "instances" / _envelope().instance_identity
    )


def test_adjusted_return_terminal_nav_is_valued_at_actual_interval_end(tmp_path):
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
    summary = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
    assert summary["nav"] == 110.0
    assert pd.Timestamp(summary["valuation_timestamp_ns"]) == pd.Timestamp("2019-01-03")
    assert sum(row.value for row in batch.valuations if row.phase == "after") == 100.0
    assert {flow.kind for flow in batch.cashflows} == {"principal"}
    assert not {"holding_return", "end_notional", "post_trade_notional"} & {
        row.name for row in batch.diagnostics
    }


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
    tmp_path: Path, long_path: bool, monkeypatch
) -> None:
    if long_path:
        while (
            len(
                str(
                    tmp_path / "portfolio" / "instances" / ("f" * 64) / "trades.parquet"
                )
            )
            <= 300
        ):
            tmp_path /= "segment_" + "a" * 32
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
    inspected = []
    original_batches = ledger_artifact._iter_batches

    def counted_batches(*args):
        for batch in original_batches(*args):
            inspected.append(batch.sequence)
            yield batch

    def consume_first(_artifact, batches):
        with closing_iterator(batches):
            assert next(batches).sequence == 0

    with monkeypatch.context() as patch:
        patch.setattr(ledger_artifact, "_iter_batches", counted_batches)
        artifact = read_portfolio_ledger_artifact(
            root, expected_identity=root.name, consume_batches=consume_first
        )
    assert inspected == [0, 1]

    def invalid_tail(*args):
        with closing_iterator(original_batches(*args)) as batches:
            yield next(batches)
            raise ValueError("synthetic_tail_invalid")

    with monkeypatch.context() as patch:
        patch.setattr(ledger_artifact, "_iter_batches", invalid_tail)
        with pytest.raises(ValueError, match="synthetic_tail_invalid"):
            read_portfolio_ledger_artifact(
                root, expected_identity=root.name, consume_batches=consume_first
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
        tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(frame,)
    )
    with open_filesystem_file(root / "valuations.parquet", "ab") as handle:
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
        tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(frame,)
    )
    filesystem_write_bytes(root / "extra.parquet", b"not-a-ledger")
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
        tmp_path / "rotation", envelope=_envelope(), ledger_chunks=(first, second)
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


def test_portfolio_ledger_diagnostic_counts_are_integral(tmp_path: Path) -> None:
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
        tmp_path / "valid", envelope=_envelope(), ledger_chunks=(frame,)
    )
    read = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    result = session_frame(next(read.iter_chunks()))
    assert (
        result.loc[result["asset_id"].eq("__cash__"), "alpha_valid_dates"].iloc[0] == 3
    )
    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_valid_dates"] = 1.5
    with pytest.raises(ValueError, match="portfolio_ledger_count_invalid"):
        write_test_sessions(
            tmp_path / "invalid", envelope=_envelope(), ledger_chunks=(frame,)
        )


def test_portfolio_ledger_small_row_groups_preserve_complete_transitions(
    tmp_path: Path, monkeypatch
) -> None:
    import pyarrow.parquet as pq

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
    real_write = pq.ParquetWriter.write_table

    def one_row_physical_batches(self, table, **kwargs):
        return real_write(self, table, row_group_size=1)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", one_row_physical_batches)
    root = write_test_sessions(
        tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(frame,)
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    assert len(list(artifact.iter_chunks())) == 1


@pytest.mark.parametrize("long_path", (False, True))
def test_portfolio_ledger_primary_error_survives_writer_close_failure(
    tmp_path: Path, monkeypatch, long_path: bool
) -> None:
    if long_path:
        while (
            len(
                str(
                    tmp_path / "portfolio" / "instances" / ("f" * 64) / "trades.parquet"
                )
            )
            <= 300
        ):
            tmp_path /= "segment_" + "a" * 32
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
    import pyarrow.parquet as pq

    real_writer = pq.ParquetWriter

    class FailingCloseWriter:
        def __init__(self, *args, **kwargs):
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):
            return self._writer.write_table(*args, **kwargs)

        def add_key_value_metadata(self, *args, **kwargs):
            return self._writer.add_key_value_metadata(*args, **kwargs)

        def close(self):
            self._writer.close()
            raise OSError("secondary_close_failure")

    monkeypatch.setattr(pq, "ParquetWriter", FailingCloseWriter)
    monkeypatch.setattr(ledger_artifact, "_BUFFER_ROWS", 1)
    with pytest.raises(ValueError, match="portfolio_ledger_schema_invalid") as raised:
        write_test_sessions(
            tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(first, second)
        )
    assert any("close also failed" in note for note in raised.value.__notes__)
    assert not list(filesystem_iterdir(tmp_path / "portfolio" / "instances"))


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
    tmp_path: Path, mutation: str, error: str
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
            tmp_path / mutation, envelope=_envelope(), ledger_chunks=(frame,)
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("manifest", "dataset_metadata_file_identity_changed"),
        ("member", "semantic_artifact_member_integrity_invalid"),
    ),
)
def test_portfolio_ledger_rechecks_outer_manifest_and_member_set_each_iteration(
    tmp_path: Path, mutation: str, error: str
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
        tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(frame,)
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    if mutation == "manifest":
        manifest_path = root / "manifest.json"
        manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
        manifest["members"].append(
            {"path": "extra.parquet", "size": 0, "sha256": "0" * 64}
        )
        filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    else:
        filesystem_write_bytes(root / "extra.parquet", b"extra")
    with pytest.raises(ValueError, match=f"^{error}$"):
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

    def produce(emit):
        kwargs = dict(
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
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
        assert all(row.timestamp_ns == 5000000 for row in terminal.valuations)
    else:
        assert len(terminal.valuations) == 3
        assert {row.asset_id for row in terminal.valuations} == {"__cash__"}
    summary = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
    if case == "closed_before_quote_expiry":
        assert (
            result["filled_orders"] == 2
            and result["markout_undefined_fill_events"] == 2
        )
        assert summary["nav"] == pytest.approx(998.98)
        assert all(row.timestamp_ns == 30000000 for row in terminal.valuations)
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


@pytest.mark.parametrize("consumer", ("close", "keyboard"))
def test_portfolio_reader_early_close_reports_identity_failures(tmp_path, consumer):
    import shutil

    from quant_research_platform.policy.lifecycle import closing_iterator

    sessions = tuple(
        _session(
            f"2019-01-0{index + 1}",
            f"2019-01-0{index + 2}",
            pre_asset=100.0 if index else 0.0,
            target_asset=100.0,
            post_asset=100.0,
            pre_cash=0.0 if index else 100.0,
            target_cash=0.0,
            post_cash=0.0,
        )
        for index in range(2)
    )
    root = write_test_sessions(
        tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=sessions
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    original = {
        path.name: filesystem_read_bytes(path) for path in filesystem_iterdir(root)
    }
    copied = tmp_path / "copy" / root.name
    shutil.copytree(native_filesystem_path(root), native_filesystem_path(copied))
    artifact = read_portfolio_ledger_artifact(copied, expected_identity=copied.name)

    def mutate_member():
        path = artifact.root / "valuations.parquet"
        before = filesystem_stat(path)
        with open_filesystem_file(path, "ab") as handle:
            handle.write(b"x")
        assert filesystem_stat(path).st_size == before.st_size + 1

    primary = KeyboardInterrupt("ledger_consumer_primary")
    chunks = artifact.iter_chunks()
    with pytest.raises(
        KeyboardInterrupt if consumer == "keyboard" else ValueError
    ) as caught:
        if consumer == "keyboard":
            with closing_iterator(chunks):
                assert next(chunks).valuations
                mutate_member()
                raise primary
        else:
            assert next(chunks).valuations
            mutate_member()
            chunks.close()
    if consumer == "keyboard":
        assert caught.value is primary
        assert any("ValueError" in note for note in getattr(primary, "__notes__", ()))
    else:
        assert str(caught.value) == "dataset_file_identity_changed"
    assert {
        path.name: filesystem_read_bytes(path) for path in filesystem_iterdir(root)
    } == original


@pytest.mark.parametrize("observed_dates", [1, 8])
def test_unidentified_alpha_survives_publication_and_blocks_freezing(
    tmp_path, observed_dates
):
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
        calibrate_expected_alpha,
    )
    from quant_research_platform.research.portfolios.production import (
        frozen_alpha_calibration,
    )

    sessions = pd.bdate_range("2018-12-10", periods=10)
    intervals = alpha_calibration_intervals(sessions, "daily")
    calibration = calibrate_expected_alpha(
        pd.DataFrame(
            {
                "timestamp": np.repeat(intervals.timestamp.iloc[:observed_dates], 10),
                "alpha_score": 0.0,
                "holding_return": np.arange(observed_dates * 10) / 10000,
            }
        ),
        as_of=sessions[-1],
        scheduled_intervals=intervals,
    )
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=0.0,
        post_asset=0.0,
        pre_cash=100.0,
        target_cash=100.0,
        post_cash=100.0,
    )
    frame["alpha_status"] = None
    frame["alpha_undefined_reason"] = None
    cash = frame.asset_id.eq("__cash__")
    frame.loc[cash, "alpha_status"] = calibration.status
    frame.loc[cash, "alpha_undefined_reason"] = calibration.undefined_reason
    frame.loc[cash, "alpha_coverage"] = calibration.coverage
    frame.loc[cash, "alpha_valid_dates"] = calibration.valid_rebalance_dates
    frame.loc[cash, "alpha_scheduled_dates"] = calibration.scheduled_rebalance_dates
    envelope = _alpha_envelope("2023-12-31")
    root = write_test_sessions(
        tmp_path / "undefined",
        envelope=envelope,
        ledger_chunks=(frame,),
        final_calibration=lambda: _alpha_payload(calibration, envelope),
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    summary = ledger_artifact.summarize_portfolio_ledger(
        artifact.capital, artifact.iter_chunks()
    )
    assert summary["alpha_calibration"]["status"] == "UNDEFINED"
    assert (
        summary["alpha_calibration"]["undefined_reason"] == calibration.undefined_reason
    )
    assert "slope" not in summary["alpha_calibration"]
    assert summary["nav"] == 100.0
    with pytest.raises(
        PortfolioTargetBlocked, match="holdout_alpha_calibration_undefined"
    ):
        frozen_alpha_calibration(
            artifact.final_alpha_calibration,
            cutoff=pd.Timestamp("2023-12-31"),
            selection_group="global",
        )


@pytest.mark.parametrize("alpha_exponent", (0, 600, -600))
@pytest.mark.parametrize("slope,intercept", [(0.0, 0.0), (0.0, 0.25), (0.125, 0.0)])
def test_degenerate_alpha_diagnostics_survive_ledger_and_freeze(
    tmp_path, slope, intercept, alpha_exponent
):
    from quant_research_platform.research.portfolios.accounting import (
        calibrate_expected_alpha,
    )
    from quant_research_platform.research.portfolios.production import (
        _ALPHA_DIAGNOSTICS,
        frozen_alpha_calibration,
    )

    dates = pd.date_range("2019-01-01", periods=6, freq="MS")
    inputs = pd.DataFrame(
        {"timestamp": dates.repeat(2), "alpha_score": [-0.5, 0.5] * 6}
    )
    inputs["holding_return"] = intercept + slope * inputs.alpha_score
    alpha_scale = float(np.ldexp(1.0, alpha_exponent))
    inputs["alpha_score"] *= alpha_scale
    intervals = pd.DataFrame(
        {
            "timestamp": dates,
            "entry_at": dates + pd.Timedelta(days=1),
            "available_at": dates + pd.Timedelta(days=2),
        }
    )
    calibration = calibrate_expected_alpha(
        inputs, as_of=pd.Timestamp("2019-12-31"), scheduled_intervals=intervals
    )
    assert calibration.status == "DEFINED"
    assert calibration.slope == pytest.approx(slope / alpha_scale, rel=1e-12, abs=0)
    assert calibration.slope_standard_error == 0.0
    assert (
        calibration.slope_t_statistic
        is calibration.slope_pvalue
        is calibration.slope_ci_low
        is calibration.slope_ci_high
        is None
    )
    assert calibration.inference_status == "UNDEFINED"
    assert calibration.r2_status == ("UNDEFINED" if slope == 0.0 else "DEFINED")
    if slope:
        assert calibration.weighted_r2 == 1.0
    frame = _session(
        "2019-07-01",
        "2019-07-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    for column, field in _ALPHA_DIAGNOSTICS.items():
        frame[column] = pd.Series([None] * len(frame), dtype=object)
        frame.loc[frame.asset_id.eq("__cash__"), column] = getattr(calibration, field)
    envelope = _alpha_envelope("2019-12-31")
    root = write_test_sessions(
        tmp_path / "portfolio",
        envelope=envelope,
        ledger_chunks=(frame,),
        final_calibration=lambda: _alpha_payload(calibration, envelope),
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    frozen, _ = frozen_alpha_calibration(
        artifact.final_alpha_calibration,
        cutoff=pd.Timestamp("2019-12-31"),
        selection_group="global",
    )
    assert frozen == calibration
    summary = ledger_artifact.summarize_portfolio_ledger(
        artifact.capital, artifact.iter_chunks()
    )
    assert summary["alpha_calibration"]["inference_status"] == "UNDEFINED"
    assert (
        summary["alpha_calibration"]["inference_reason"]
        == "alpha_calibration_zero_standard_error"
    )
    assert "slope_t" not in summary["alpha_calibration"]
    from quant_research_platform.artifacts.research import (
        read_research_report_artifact,
        write_research_report_artifact,
    )
    from tests.artifacts.test_research_artifact import _report

    envelope, payload = _report()
    payload["sections"]["portfolio"] = {"status": "DEFINED", "results": [summary]}
    report_root = write_research_report_artifact(
        tmp_path / "report", envelope=envelope, payload=payload
    )
    saved = read_research_report_artifact(
        report_root, expected_identity=envelope.instance_identity
    )
    assert (
        saved["sections"]["portfolio"]["results"][0]["alpha_calibration"]
        == summary["alpha_calibration"]
    )
    assert "alpha_calibration_zero_standard_error" in filesystem_read_text(
        report_root / "report.md", encoding="utf-8"
    )


@pytest.mark.parametrize("engine", ("taker", "maker"))
@pytest.mark.parametrize("end_ms", (70, 75, 100))
def test_flat_eof_preserves_historical_fill_diagnostics_and_cash_ledger(
    tmp_path, engine, end_ms
):
    from quant_research_platform.markets.implementations.crypto.execution import (
        TakerConfig,
        simulate_maker,
        simulate_taker,
    )
    from tests.markets.crypto.test_hft_execution import (
        _maker_config,
        _quote,
        _state,
        _trade,
    )

    capital = PortfolioCapital("flat_eof", 1000.0, "USDT", "linear_perpetual")
    envelope = build_business_manifest_envelope(
        logical_artifact_id="flat_eof",
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
    batches = []

    def produce(emit):

        def observe(batch):
            batches.append(batch)
            emit(batch)

        shared = dict(
            observed_coverage={"s1": (0, 75000000)},
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            funding_forecasts=(FundingForecast(0, 1000000000, ()),),
            decision_windows_ns=((0, end_ms * 1000000),),
            on_ledger=observe,
        )
        if engine == "taker":
            return simulate_taker(
                [_state(0)],
                [
                    {"timestamp_ms": 10, "prediction": 0.05, "book_segment_id": "s1"},
                    {"timestamp_ms": 20, "prediction": -0.05, "book_segment_id": "s1"},
                ],
                prediction_column="prediction",
                prediction_horizon_ms=500,
                config=TakerConfig(1.0, 1.0, 0, 0.0, 500),
                **shared,
            )
        return simulate_maker(
            [_state(0)],
            [
                _trade(50, price="99.4", quantity="1", maker_buyer=True),
                _trade(60, price="100.6", quantity="1", maker_buyer=False),
            ],
            [_quote(0, book_segment_id="s1")],
            config=_maker_config(),
            trade_coverage={"trade-connection-1": (0, 101000000)},
            **shared,
        )

    published = ledger_artifact.write_portfolio_ledger_artifact(
        tmp_path / "flat_eof", envelope=envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(
        published, expected_identity=envelope.instance_identity
    )
    persisted = list(artifact.iter_chunks())
    trades = [trade for batch in persisted for trade in batch.trades]
    assert len(trades) == 2 and sum(trade.quantity for trade in trades) == 0
    summary = ledger_artifact.summarize_portfolio_ledger(
        artifact.capital, artifact.iter_chunks()
    )
    assert summary["net_pnl"] == pytest.approx(-1.0 if engine == "taker" else 1.02)
    diagnostics = [
        row
        for batch in persisted
        for row in batch.diagnostics
        if row.fill_id is not None
    ]
    assert diagnostics and {row.fill_id for row in diagnostics} == {
        row.fill_id for row in trades
    }
    assert len(persisted) == len(batches)


@pytest.mark.parametrize("scale", [1.0, 2.0 ** (-700)])
@pytest.mark.parametrize("direction", [1.0, -1.0])
@pytest.mark.parametrize("closing_units", [0.5, 1.0, 2.0])
def test_perpetual_close_sign_is_independent_of_quantity_units(
    tmp_path, scale, direction, closing_units
):
    from quant_research_platform.research.portfolios.accounting import (
        PortfolioEconomics,
    )

    capital = PortfolioCapital("units", 1000.0, "USDT", "linear_perpetual")
    envelope = build_business_manifest_envelope(
        logical_artifact_id="units",
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
    expected_realized = direction * min(1.0, closing_units)
    expected_basis = (
        direction * (1.0 - closing_units) * (1.0 if closing_units < 1.0 else 2.0)
    )

    def produce(emit):
        economics = PortfolioEconomics(capital, "asset", emit)
        economics.fill(
            timestamp_ns=1,
            order_id="open",
            quantity=direction * scale,
            price=1.0 / scale,
            mark_price=1.0 / scale,
            fee=0.0,
        )
        economics.fill(
            timestamp_ns=2,
            order_id="close",
            quantity=-direction * closing_units * scale,
            price=2.0 / scale,
            mark_price=2.0 / scale,
            fee=0.0,
        )
        assert economics.cash == 1000.0 + expected_realized
        assert economics.quantity == direction * (1.0 - closing_units) * scale
        assert economics.cost_basis == expected_basis

    root = ledger_artifact.write_portfolio_ledger_artifact(
        tmp_path / "units", envelope=envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    batches = tuple(artifact.iter_chunks())
    assert (
        sum(
            flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "realized_pnl"
        )
        == expected_realized
    )
    end = {row.asset_id: row for row in batches[-1].valuations if row.phase == "end"}
    assert end["__cash__"].value == 1000.0 + expected_realized
    assert end["asset"].cost_basis == expected_basis
    summary = ledger_artifact.summarize_portfolio_ledger(
        artifact.capital, artifact.iter_chunks()
    )
    assert summary["net_pnl"] == summary["gross_pnl"] == direction


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_tiny_trade_validity_uses_sign_without_multiplication(direction):
    from quant_research_platform.research.portfolios.accounting import (
        PortfolioEconomics,
    )

    batches = []
    economics = PortfolioEconomics(
        PortfolioCapital("tiny", 1000.0, "USDT", "linear_perpetual"),
        "asset",
        batches.append,
    )
    quantity = direction * 2.0 ** (-700)
    economics.fill(
        timestamp_ns=1,
        order_id="open",
        quantity=quantity,
        price=1.0,
        mark_price=1.0,
        fee=0.0,
    )
    batch = batches[0]
    assert batch.trades[0].quantity == batch.trades[0].notional == quantity
    for q, n in ((0.0, quantity), (quantity, 0.0), (quantity, -quantity)):
        with pytest.raises(ValueError, match="portfolio_ledger_trade_invalid"):
            replace(batch, trades=(replace(batch.trades[0], quantity=q, notional=n),))


def test_tiny_adjusted_notional_trade_cannot_reverse_target_direction(tmp_path):
    value = 2.0 ** (-700)
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=2 * value,
        target_asset=value,
        post_asset=3 * value,
        pre_cash=1.0,
        target_cash=1.0,
        post_cash=1.0,
    )
    payload = _envelope().payload()
    del payload["instance_identity"], payload["status"]
    payload["semantic_metadata"]["portfolio_capital"]["initial_capital"] = 1.0
    envelope = build_business_manifest_envelope(**payload)
    with pytest.raises(ValueError, match="portfolio_ledger_direction_invalid"):
        write_test_sessions(
            tmp_path / "wrong_direction", envelope=envelope, ledger_chunks=(frame,)
        )
    assert not filesystem_exists(
        tmp_path / "wrong_direction" / "instances" / envelope.instance_identity
    )


@pytest.mark.parametrize("entry", ("read", "iterate"))
def test_verified_member_identity_reaches_ledger_decoder(tmp_path, monkeypatch, entry):
    import pyarrow as pa
    import pyarrow.parquet as pq

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
        tmp_path / "portfolio", envelope=_envelope(), ledger_chunks=(frame,)
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    native = ledger_artifact.read_semantic_artifact_record

    def replace_after_verification(*args, **kwargs):
        result = native(*args, **kwargs)
        member = root / "valuations.parquet"
        table = pq.read_table(native_filesystem_path(member))
        rows = table.to_pylist()
        for row in rows:
            if row["phase"] == "end" and row["asset_id"] == "asset":
                row["value"] = 120.0
                row["mark_price"] = 1.2
        replacement = tmp_path / "replacement.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema),
            native_filesystem_path(replacement),
        )
        filesystem_replace(replacement, member)
        return result

    monkeypatch.setattr(
        ledger_artifact, "read_semantic_artifact_record", replace_after_verification
    )
    with pytest.raises(ValueError, match="identity_changed"):
        if entry == "read":
            read_portfolio_ledger_artifact(root, expected_identity=root.name)
        else:
            tuple(artifact.iter_chunks())


def _alpha_envelope(cutoff):
    base = _envelope().payload()
    return build_business_manifest_envelope(
        logical_artifact_id=base["logical_artifact_id"],
        schema_identity=base["schema_identity"],
        protocol_or_spec_identity=base["protocol_or_spec_identity"],
        software_identity=base["software_identity"],
        research_window={**base["research_window"], "end": cutoff},
        upstream_identities={"prediction": "d" * 64},
        semantic_metadata={
            **base["semantic_metadata"],
            "target_contract": {
                "protocol": {"protocol_id": "a_share_low_frequency"},
                "portfolio_candidate": {
                    "construction": "benchmark_enhanced",
                    "selection_group": "global",
                },
            },
        },
    )


def _alpha_payload(calibration, envelope):
    from dataclasses import asdict

    from quant_research_platform.contracts.targets import unmeasured_operation_evidence

    return {
        "schema_id": "qrp-final-alpha-calibration-v1",
        "selection_group": "global",
        "cutoff": envelope.research_window["end"],
        "upstream_identities": dict(envelope.upstream_identities),
        "calibration": asdict(calibration),
        "observations": unmeasured_operation_evidence("test_calibration"),
    }
