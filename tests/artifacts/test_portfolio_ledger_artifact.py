from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.artifacts.portfolio_ledger as ledger_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
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
    filesystem_walk,
    native_filesystem_path,
    open_filesystem_file,
)
from tests.helpers.binance_hft import KNOWN_NO_FUNDING
from tests.helpers.portfolio_ledger import (
    CONTINUOUS_EXECUTION_COLUMNS,
    session_frame,
    write_test_sessions,
)


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): Path(
            native_filesystem_path(parent / name)
        ).read_bytes()
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


@pytest.mark.parametrize("boundary", ("batch", "cross_session"))
def test_perpetual_quantity_cannot_be_created_within_nav_amount_tolerance(
    tmp_path: Path, boundary: str
) -> None:
    capital = PortfolioCapital("quantity", 1000000000.0, "USDT", "linear_perpetual")
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01",
                "end": "1970-01-02",
                "access": "validation",
            }
        },
    }

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
        tmp_path / "old", **envelope, produce=lambda emit: emit(baseline)
    )
    old = read_portfolio_ledger_artifact(old_root)
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
            candidate, **envelope, produce=invalid
        )
    assert not filesystem_exists(candidate)
    assert _file_bytes(old_root) == old_bytes


@pytest.mark.parametrize("mode", ("maker", "taker"))
@pytest.mark.parametrize("tail", (False, True))
def test_markout_roundtrip_uses_final_book_at_exact_target(tmp_path, mode, tail):
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
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01",
                "end": "1970-01-02",
                "access": "validation",
            }
        },
    }
    target = 150 if mode == "maker" else 100
    books = [
        _state(0),
        _state(target, bid_index=1005, ask_index=1015),
        replace(
            _state(target, bid_index=1015, ask_index=1025),
            source_sequence=target + 2,
            update_id=target + 2,
        ),
    ]
    if tail:
        books.append(_state(200, bid_index=1095, ask_index=1105))

    def produce(emit):
        common = dict(
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=emit,
            observed_coverage={"s1": _coverage(0, 201)},
        )
        if mode == "maker":
            return simulate_maker(
                books,
                [_trade(50, price="99.4", quantity="1", maker_buyer=True)],
                [_quote(0, book_segment_id="s1")],
                config=_maker_config(),
                trade_coverage={"trade-connection-1": _coverage(0, 201)},
                **common,
            )
        return simulate_taker(
            books,
            [{"timestamp_ms": 0, "prediction": 0.1}],
            prediction_column="prediction",
            prediction_horizon_ms=500,
            funding_forecasts=KNOWN_NO_FUNDING,
            config=_taker_config(),
            **common,
        )

    root = ledger_artifact.write_portfolio_ledger_artifact(
        tmp_path / mode, **envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    markouts = [
        row.number
        for batch in artifact.iter_chunks()
        for row in batch.diagnostics
        if row.name == "markout_per_unit"
    ]
    assert markouts == pytest.approx([2.5 if mode == "maker" else 1.5])


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
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01T00:00:00",
                "end": "1970-01-01T00:00:01",
                "access": "validation",
            }
        },
    }

    def run(emit, rate):
        kwargs = dict(
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(FundingEvent(75000000, rate, 100.0),),
            on_ledger=emit,
            decision_windows_ns=((0, 100_000_000),),
            observed_coverage={"s1": _coverage(0, 101)},
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
            trade_coverage={"trade-connection-1": _coverage(0, 101)},
            **kwargs,
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / mode, **envelope, produce=lambda emit: run(emit, funding_rate)
    )
    artifact = read_portfolio_ledger_artifact(root)
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
    old_manifest = Path(native_filesystem_path(root / "manifest.json")).read_bytes()
    missing_root = write_portfolio_ledger_artifact(
        tmp_path / (mode + "_missing"),
        **envelope,
        produce=lambda emit: run(emit, None),
    )
    from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger

    partial = read_portfolio_ledger_artifact(missing_root)
    missing_summary = summarize_crypto_ledger(partial, partial.iter_chunks())
    assert missing_summary["execution"]["account_path_status"] == "PARTIAL"
    assert (
        missing_summary["execution"]["account_stop_reason"]
        == "portfolio_required_funding_missing"
    )
    assert missing_summary["economics"]["net_pnl"] is None
    assert missing_summary["economics"]["funding"] is None
    assert missing_summary["execution"]["known_prefix_funding"] == 0.0
    assert missing_summary["economics"]["fill_events"] == 1
    assert (
        Path(native_filesystem_path(root / "manifest.json")).read_bytes()
        == old_manifest
    )


def test_orphan_fill_and_unknown_delayed_fill_fail_closed(tmp_path):
    from quant_research_platform.artifacts.portfolio_ledger import (
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioDiagnostic
    from tests.helpers.portfolio_ledger import continuous_ledger_batch

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
            tmp_path / "bad", **_ledger_spec(), produce=lambda emit: emit(bad)
        )
    assert not filesystem_exists(tmp_path / "bad")


def test_raw_return_terminal_nav_is_valued_at_actual_interval_end(tmp_path):
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
        tmp_path / "interval", **_ledger_spec(), ledger_chunks=(frame,)
    )
    artifact = read_portfolio_ledger_artifact(root)
    batch = next(artifact.iter_chunks())
    asset_end = next(
        row
        for row in batch.valuations
        if row.asset_id == "asset" and row.phase == "end"
    )
    assert pd.Timestamp(asset_end.timestamp_ns) == pd.Timestamp("2019-01-03")
    assert asset_end.value == pytest.approx(110.0)
    summary = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
    assert summary["nav"] == pytest.approx(110.0, rel=1e-14)
    assert pd.Timestamp(summary["valuation_timestamp_ns"]) == pd.Timestamp("2019-01-03")
    assert sum(row.value for row in batch.valuations if row.phase == "after") == 100.0
    assert {flow.kind for flow in batch.cashflows} == {"principal"}
    assert not {"holding_return", "end_notional", "post_trade_notional"} & {
        row.name for row in batch.diagnostics
    }


def _ledger_spec() -> object:
    return {
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
                "start": "2019-01-01",
                "end": "2019-01-03",
                "access": "validation",
            }
        },
    }


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
            "prediction_rank_score": 1.0,
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
            "prediction_rank_score": np.nan,
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
        while len(str(tmp_path / "portfolio" / "trades.parquet")) <= 300:
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
        **_ledger_spec(),
        ledger_chunks=(frame for frame in (first, second)),
    )
    with pytest.raises(ValueError, match="output_already_exists"):
        write_test_sessions(
            tmp_path / "portfolio", **_ledger_spec(), ledger_chunks=(first, second)
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
        artifact = read_portfolio_ledger_artifact(root, consume_batches=consume_first)
    assert inspected == [0]

    def invalid_tail(*args):
        with closing_iterator(original_batches(*args)) as batches:
            yield next(batches)
            raise ValueError("synthetic_tail_invalid")

    with monkeypatch.context() as patch:
        patch.setattr(ledger_artifact, "_iter_batches", invalid_tail)
        read_portfolio_ledger_artifact(root, consume_batches=consume_first)
    if long_path:
        assert len(str(root / "trades.parquet")) > 260
    chunks = [session_frame(batch) for batch in artifact.iter_chunks()]
    assert len(chunks) == 2
    assert [chunk["asset_id"].tolist() for chunk in chunks] == [
        ["asset", "__cash__"],
        ["asset", "__cash__"],
    ]


def test_portfolio_ledger_corrupt_selected_table_fails_decoding(tmp_path: Path) -> None:
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
        tmp_path / "portfolio", **_ledger_spec(), ledger_chunks=(frame,)
    )
    with open_filesystem_file(root / "valuations.parquet", "ab") as handle:
        handle.write(b"mutation")
    import pyarrow as pa

    with pytest.raises((pa.ArrowInvalid, ValueError)):
        list(read_portfolio_ledger_artifact(root).iter_chunks())


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
        tmp_path / "rotation", **_ledger_spec(), ledger_chunks=(first, second)
    )
    artifact = read_portfolio_ledger_artifact(root)
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
        tmp_path / "valid", **_ledger_spec(), ledger_chunks=(frame,)
    )
    read = read_portfolio_ledger_artifact(root)
    result = session_frame(next(read.iter_chunks()))
    assert (
        result.loc[result["asset_id"].eq("__cash__"), "alpha_valid_dates"].iloc[0] == 3
    )
    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_valid_dates"] = 1.5
    with pytest.raises(ValueError, match="portfolio_ledger_count_invalid"):
        write_test_sessions(
            tmp_path / "invalid", **_ledger_spec(), ledger_chunks=(frame,)
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
        tmp_path / "portfolio", **_ledger_spec(), ledger_chunks=(frame,)
    )
    artifact = read_portfolio_ledger_artifact(root)
    assert len(list(artifact.iter_chunks())) == 1


@pytest.mark.parametrize("long_path", (False, True))
def test_portfolio_ledger_primary_error_survives_writer_close_failure(
    tmp_path: Path, monkeypatch, long_path: bool
) -> None:
    if long_path:
        while len(str(tmp_path / "portfolio" / "trades.parquet")) <= 300:
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
    monkeypatch.setattr(ledger_artifact, "DEFAULT_BATCH_ROWS", 1)
    with pytest.raises(ValueError, match="portfolio_ledger_schema_invalid") as raised:
        write_test_sessions(
            tmp_path / "portfolio", **_ledger_spec(), ledger_chunks=(first, second)
        )
    assert any("close also failed" in note for note in raised.value.__notes__)
    assert not filesystem_exists(tmp_path / "portfolio")


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
            tmp_path / mutation, **_ledger_spec(), ledger_chunks=(frame,)
        )


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
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01T00:00:00",
                "end": "1970-01-01T00:00:01",
                "access": "validation",
            }
        },
    }
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

    root = write_portfolio_ledger_artifact(tmp_path / mode, **envelope, produce=produce)
    artifact = read_portfolio_ledger_artifact(root)
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


@pytest.mark.parametrize("observed_dates", [1, 8])
def test_unidentified_alpha_survives_publication_and_blocks_freezing(
    tmp_path, observed_dates
):
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
        calibrate_expected_alpha,
        frozen_alpha_calibration,
    )

    sessions = pd.bdate_range("2018-12-10", periods=10)
    intervals = alpha_calibration_intervals(sessions, "daily")
    calibration = calibrate_expected_alpha(
        pd.DataFrame(
            {
                "timestamp": np.repeat(intervals.timestamp.iloc[:observed_dates], 10),
                "prediction_rank_score": 0.0,
                "holding_return": np.arange(observed_dates * 10) / 10000,
            }
        ),
        trading_sessions=sessions,
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
    envelope = _alpha_metadata("2023-12-31")
    root = write_test_sessions(
        tmp_path / "undefined",
        **envelope,
        ledger_chunks=(frame,),
        final_calibration=lambda: _alpha_payload(calibration, envelope),
    )
    artifact = read_portfolio_ledger_artifact(root)
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
        PortfolioTargetBlocked, match="test_alpha_calibration_undefined"
    ):
        frozen_alpha_calibration(
            artifact.final_alpha_calibration,
            cutoff=pd.Timestamp("2023-12-31"),
            selection_group="global",
            rebalance_schedule="daily",
        )


@pytest.mark.parametrize("alpha_exponent", (0, 600, -600))
@pytest.mark.parametrize("slope,intercept", [(0.0, 0.0), (0.0, 0.25), (0.125, 0.0)])
def test_degenerate_alpha_diagnostics_survive_ledger_and_freeze(
    tmp_path, slope, intercept, alpha_exponent
):
    from quant_research_platform.research.portfolios.accounting import (
        calibrate_expected_alpha,
        frozen_alpha_calibration,
    )
    from quant_research_platform.research.portfolios.production import (
        _ALPHA_DIAGNOSTICS,
    )

    dates = pd.date_range("2019-07-01", periods=6, freq="MS")
    inputs = pd.DataFrame(
        {"timestamp": dates.repeat(2), "prediction_rank_score": [-0.5, 0.5] * 6}
    )
    inputs["holding_return"] = intercept + slope * inputs.prediction_rank_score
    alpha_scale = float(np.ldexp(1.0, alpha_exponent))
    inputs["prediction_rank_score"] *= alpha_scale
    intervals = pd.DataFrame(
        {
            "timestamp": dates,
            "entry_at": dates + pd.Timedelta(days=1),
            "available_at": dates + pd.Timedelta(days=2),
        }
    )
    calibration = calibrate_expected_alpha(
        inputs,
        trading_sessions=pd.date_range("2019-01-01", "2019-12-31"),
        as_of=pd.Timestamp("2019-12-31"),
        scheduled_intervals=intervals,
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
    envelope = _alpha_metadata("2019-12-31")
    root = write_test_sessions(
        tmp_path / "portfolio",
        **envelope,
        ledger_chunks=(frame,),
        final_calibration=lambda: _alpha_payload(calibration, envelope),
    )
    artifact = read_portfolio_ledger_artifact(root)
    frozen, _ = frozen_alpha_calibration(
        artifact.final_alpha_calibration,
        cutoff=pd.Timestamp("2019-12-31"),
        selection_group="global",
        rebalance_schedule="daily",
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
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01T00:00:00",
                "end": "1970-01-01T00:00:01",
                "access": "validation",
            }
        },
    }
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
        tmp_path / "flat_eof", **envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(published)
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
    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
    )

    capital = PortfolioCapital("units", 1000.0, "USDT", "linear_perpetual")
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01",
                "end": "1970-01-02",
                "access": "validation",
            }
        },
    }
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
        tmp_path / "units", **envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
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
    from quant_research_platform.markets.implementations.crypto.execution import (
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


def test_tiny_cash_equity_trade_cannot_reverse_target_direction(tmp_path):
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
    envelope = _ledger_spec()
    envelope["capital"] = replace(envelope["capital"], initial_capital=1.0)
    with pytest.raises(ValueError, match="portfolio_ledger_direction_invalid"):
        write_test_sessions(
            tmp_path / "wrong_direction", **envelope, ledger_chunks=(frame,)
        )
    assert not filesystem_exists(tmp_path / "wrong_direction")


def _alpha_metadata(cutoff):
    base = _ledger_spec()
    base["metadata"]["window"]["end"] = cutoff
    return base


def _alpha_payload(calibration, envelope):
    return {
        "schema_id": "qrp-final-alpha-calibration",
        "interval_semantics": {
            "rebalance_schedule": "daily",
            "return_convention": (
                "next_execution_open_to_following_rebalance_execution_open"
            ),
            "calendar_convention": "a_share_unified_trading_sessions",
        },
        "selection_group": "global",
        "cutoff": envelope["metadata"]["window"]["end"],
        "inputs": {},
        "calibration": asdict(calibration),
    }


@pytest.mark.parametrize(
    "quantity, entry, mark",
    [(0.7, 30000.0, 30000.01), (0.7, 30000.0, 30001.0), (2.0, 8e307, 9e307)],
)
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_perpetual_rounded_cost_preserves_exact_pnl(
    tmp_path, quantity, entry, mark, direction
):
    from fractions import Fraction
    from math import ulp

    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
    )

    quantity *= direction
    expected = float(Fraction(quantity) * (Fraction(mark) - Fraction(entry)))
    capital = PortfolioCapital("rounded_cost", 1000.0, "USDT", "linear_perpetual")
    envelope = {
        "capital": PortfolioCapital(**asdict(capital)),
        "metadata": {
            "window": {
                "start": "1970-01-01",
                "end": "1970-01-02",
                "access": "validation",
            }
        },
    }

    def produce(emit):
        account = PortfolioEconomics(capital, "asset", emit)
        account.fill(
            timestamp_ns=1,
            order_id="open",
            quantity=quantity,
            price=entry,
            mark_price=entry,
            fee=0.0,
        )
        assert account.mark(2, mark)["net_pnl"] == expected

    root = ledger_artifact.write_portfolio_ledger_artifact(
        tmp_path / "ledger", **envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    batches = tuple(artifact.iter_chunks())
    assert (
        ledger_artifact.summarize_portfolio_ledger(artifact.capital, iter(batches))[
            "net_pnl"
        ]
        == expected
    )
    last = batches[-1]
    value = next(
        row for row in last.valuations if row.asset_id == "asset" and row.phase == "end"
    )
    assert value.value == expected
    corrupted = replace(
        value, value=value.value + 16 * max(ulp(value.cost_basis), ulp(value.value))
    )
    with pytest.raises(ValueError, match="portfolio_ledger_valuation_invalid"):
        replace(
            last,
            valuations=tuple(
                corrupted if row is value else row for row in last.valuations
            ),
        )
