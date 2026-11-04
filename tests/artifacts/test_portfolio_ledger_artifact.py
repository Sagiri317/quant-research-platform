from __future__ import annotations

from dataclasses import asdict, fields, replace
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
    PortfolioCapital,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
    PortfolioValuation,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.identity import (
    filesystem_exists,
    open_filesystem_file,
)
from tests.helpers.portfolio_ledger import (
    CONTINUOUS_EXECUTION_COLUMNS,
    session_frame,
    write_test_sessions,
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
    elif mutation == "block_reason":
        frame.loc[asset, "block_reason"] = "unknown_reason"
    else:
        frame.loc[asset, "decision_reason"] = "unknown_reason"
    with pytest.raises(ValueError, match=error):
        write_test_sessions(
            tmp_path / mutation, **_ledger_spec(), ledger_chunks=(frame,)
        )


@pytest.mark.parametrize("observed_dates", [1, 8])
def test_unidentified_alpha_survives_publication_and_blocks_freezing(
    tmp_path, observed_dates
):
    from quant_research_platform.research.portfolios.calibration import (
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


@pytest.mark.parametrize("slope,intercept", [(0.0, 0.0), (0.0, 0.25), (0.125, 0.0)])
def test_degenerate_alpha_diagnostics_survive_ledger_and_freeze(
    tmp_path, slope, intercept
):
    from quant_research_platform.research.portfolios.calibration import (
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
    assert calibration.slope == pytest.approx(slope, rel=1e-12, abs=0)
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


def test_tiny_cash_equity_trade_cannot_reverse_target_direction(tmp_path):
    value = 1e-8
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=2 * value,
        target_asset=value,
        post_asset=3 * value,
        pre_cash=1.0,
        target_cash=1.0 + value,
        post_cash=1.0 - value,
    )
    envelope = _ledger_spec()
    envelope["capital"] = replace(envelope["capital"], initial_capital=1.0 + 2 * value)
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


def test_repeated_partial_fills_preserve_registered_rights_through_native_roundtrip(
    tmp_path,
):

    from quant_research_platform.contracts.portfolios import (
        PortfolioCashFlow,
        PortfolioDiagnostic,
        PortfolioRightsAdjustment,
    )
    from quant_research_platform.research.portfolios.accounting import (
        EquityEntitlements,
    )

    dates = pd.bdate_range("2024-01-02", periods=6) + pd.Timedelta(milliseconds=1)
    life = pd.Timestamp("2000-01-01")
    entitlements = EquityEntitlements()
    units, lives = pd.Series({"a": 0.0}, dtype=object), pd.Series({"a": life})
    cash = 100_000.0
    events = pd.DataFrame(
        [
            dict(
                event_id=kind,
                asset_id="a",
                listing_date=life,
                record_at=dates[3],
                effective_at=dates[4],
                settlement_at=dates[5],
                available_at=dates[0],
                entitlement_proved=True,
                evidence_identity="a" * 64,
                kind=kind,
                cash_per_share=0.25,
                additional_shares_per_share=0.5,
            )
            for kind in ("cash_dividend", "stock_distribution")
        ]
    )
    batches = []

    def valuation(phase, at, price):
        receivable = entitlements.receivables.get("a", 0.0)
        return (
            PortfolioValuation(
                at.value,
                "a",
                phase,
                units.a,
                float(units.a * float(price)) + receivable,
                price,
                lifecycle_id="2000-01-01",
                receivable=receivable,
                locked_quantity=entitlements.locked_quantities.get("a", 0.0),
            ),
            PortfolioValuation(at.value, "__cash__", phase, cash, cash),
        )

    for sequence in range(5):
        start, end = dates[sequence : sequence + 2]
        price = round(100.01 + min(sequence, 3) / 100, 2)
        next_price = round(100.01 + min(sequence + 1, 3) / 100, 2)
        before = valuation("before", start, price)
        trades, flows = (), ()
        if sequence < 3:
            quantity = 100.0
            units.a += quantity
            cash -= quantity * price
            fill = f"fill{sequence}"
            trades = (
                ExecutedTrade(fill, fill, start.value, "a", quantity, quantity * price),
            )
            flows = (
                PortfolioCashFlow(
                    fill, start.value, "a", fill, "principal", -quantity * price
                ),
            )
        after = valuation("after", start, price)
        rights = entitlements.register(events, units, lives, end, phase="end")
        units, lives, paid, adjustments = entitlements.process(
            events, pd.DataFrame(), units, lives, end, phase="end"
        )
        rights += adjustments
        if sequence == 4:
            rights += (
                PortfolioRightsAdjustment(
                    "recovery",
                    end.value,
                    "a",
                    "2000-01-01",
                    units.a,
                    units.a,
                    0.0,
                    None,
                    None,
                    0.0,
                    "b" * 64,
                    phase="end",
                ),
            )
        cash += paid
        if paid:
            flows += (
                PortfolioCashFlow(
                    "dividend:cash_dividend:settlement",
                    end.value,
                    "a",
                    None,
                    "dividend",
                    paid,
                ),
            )
        diagnostics = (
            PortfolioDiagnostic(
                "a", None, "target_notional", number=after[0].share_value
            ),
            PortfolioDiagnostic("a", None, "adv20", number=1_000_000.0),
            PortfolioDiagnostic(
                "__cash__", None, "target_notional", number=after[1].value
            ),
            PortfolioDiagnostic(
                "__cash__", None, "decision_reason", text="simple_rule"
            ),
        )
        batches.append(
            PortfolioLedgerBatch(
                sequence,
                trades,
                flows,
                before + after + valuation("end", end, next_price),
                diagnostics,
                rights,
            )
        )
    registered = 300.0
    root = ledger_artifact.write_portfolio_ledger_artifact(
        tmp_path / "portfolio",
        capital=PortfolioCapital("ordinary", 100_000.0, "CNY", "cash_equity"),
        metadata={
            "window": {"start": str(dates[0].date()), "end": str(dates[-1].date())}
        },
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    restored = list(read_portfolio_ledger_artifact(root).iter_chunks())
    assert restored == batches
    rights = [row for batch in restored for row in batch.rights_adjustments]
    assert {row.kind for row in rights} == {
        "registration",
        "dividend",
        "stock_distribution",
        "unlock",
        "recovery",
    }
    assert all(
        row.entitlement_quantity == registered
        for row in rights
        if row.kind != "recovery"
    )
    final = next(
        row
        for row in restored[-1].valuations
        if row.asset_id == "a" and row.phase == "end"
    )
    assert final.quantity == registered * (3 / 2)
    assert final.locked_quantity == final.receivable == 0


@pytest.mark.parametrize("invalid", [True, None, np.nan, np.inf, "1.0"])
def test_typed_portfolio_ledger_rejects_invalid_numeric_fields(invalid):
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
    for table in ("trades", "cashflows", "valuations", "diagnostics"):
        rows = getattr(batch, table)
        assert rows
        for field in fields(rows[0]):
            if field.type not in ("float", "float | None"):
                continue
            if invalid is None and field.type == "float | None":
                continue
            changed = replace(rows[0], **{field.name: invalid})
            with pytest.raises(ValueError, match="portfolio_ledger_numeric_invalid"):
                replace(batch, **{table: (changed, *rows[1:])})
