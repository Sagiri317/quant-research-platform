"""Systemic checks for the continuous portfolio economic ledger."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeRule,
    FeeSchedule,
    fee_schedule_from_frame,
    require_fee_coverage,
    trade_cost,
)
from quant_research_platform.research.portfolios.accounting import (
    drift_continuous_holdings,
)
from quant_research_platform.research.portfolios.accounting import (
    execute_continuous_target as _execute_continuous_target,
)
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.sources.reference.a_share_fees import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SIDECAR_NAME,
    FEE_CSV_COLUMNS,
    load_a_share_fee_source,
)
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_stat,
    filesystem_write_bytes,
    filesystem_write_text,
    open_filesystem_file,
)
from tests.helpers.portfolio_ledger import execute_continuous_target


def _cash_constrained_inputs(
    order: tuple[str, ...],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    current = pd.Series(0.0, index=order, dtype="float64")
    target = pd.Series(1.0 / 3.0, index=order, dtype="float64")
    adv = pd.Series(1_000.0, index=order, dtype="float64")
    return current, target, adv


def _schedule(
    *,
    broker_rate: float = 0.0003,
    transfer_rate: float = 0.0001,
    stamp_rate: float = 0.0005,
) -> FeeSchedule:
    rules = [
        FeeRule(
            fee_code="BROKER_COMMISSION",
            effective_date=date(2023, 1, 1),
            end_date=date(2025, 12, 31),
            market="a_share",
            side=side,
            basis="amount",
            rate=broker_rate,
            included_in_ledger=True,
        )
        for side in ("buy", "sell")
    ]
    rules.extend(
        FeeRule(
            fee_code="TRANSFER_FEE",
            effective_date=date(2023, 1, 1),
            end_date=date(2025, 12, 31),
            market="a_share",
            side=side,
            basis="amount",
            rate=transfer_rate,
            included_in_ledger=True,
        )
        for side in ("buy", "sell")
    )
    rules.append(
        FeeRule(
            fee_code="STAMP_DUTY",
            effective_date=date(2023, 1, 1),
            end_date=date(2025, 12, 31),
            market="a_share",
            side="sell",
            basis="amount",
            rate=stamp_rate,
            included_in_ledger=True,
        )
    )
    return FeeSchedule(
        rules=tuple(rules),
        content_identity="0" * 64,
    )


def _assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(
        initial_nav=100.0,
        slippage_bps=0.0,
        impact_bps=0.0,
        adv_participation=1.0,
        adv_sensitivities=(1.0,),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (("initial_nav", "100.0"), ("adv_sensitivities", ("1.0",))),
)
def test_execution_assumptions_reject_text_numbers(
    field: str,
    value: object,
) -> None:
    parameters: dict[str, object] = {
        "initial_nav": 100.0,
        "slippage_bps": 0.0,
        "impact_bps": 0.0,
        "adv_participation": 1.0,
        "adv_sensitivities": (1.0,),
    }
    parameters[field] = value
    with pytest.raises(ValueError, match="execution_assumptions_invalid"):
        ExecutionAssumptions(**parameters)


def _a_share_cost(
    schedule: FeeSchedule,
    markets: dict[str, str],
):
    def cost_for_trade(
        execution_date: date,
        asset: str,
        side: str,
        notional: float,
        adv20: float,
        assumptions: ExecutionAssumptions,
    ):
        return trade_cost(
            execution_date,
            markets[asset],
            side,
            notional,
            adv20,
            schedule,
            assumptions,
        )

    return cost_for_trade


def _zero_cost_transition():
    return execute_continuous_target(
        pd.Series({"a": 1.0, "c": 0.0}),
        pd.Series({"b": 1.0, "c": 0.0}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 100.0, "b": 100.0, "c": 100.0}),
        can_enter=pd.Series({"a": True, "b": True, "c": True}),
        can_exit=pd.Series({"a": True, "b": True, "c": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(transfer_rate=0.0, broker_rate=0.0, stamp_rate=0.0),
            {"a": "shanghai", "b": "shanghai", "c": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )


def _real_fee_schedule(tmp_path: Path) -> FeeSchedule:
    csv_path, sidecar_path = _copy_fee_source(tmp_path)
    source = load_a_share_fee_source(reference_root=csv_path.parent.parent)
    frame = pd.DataFrame(
        [
            {
                field: getattr(row, field)
                for field in (
                    "fee_code",
                    "effective_date",
                    "end_date",
                    "market",
                    "side",
                    "basis",
                    "rate",
                    "included_in_ledger",
                    "included_in",
                )
            }
            for row in source.rows
        ]
    )
    return fee_schedule_from_frame(
        frame,
        content_identity=source.source_identity,
    )


def _copy_fee_source(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "a_share_fee_schedule"
    filesystem_mkdir(source_root, parents=True, exist_ok=True)
    csv_path = source_root / A_SHARE_FEE_CSV_NAME
    sidecar_path = source_root / A_SHARE_FEE_SIDECAR_NAME
    row_specs = [
        (
            "BROKER_COMMISSION",
            "2014-01-01",
            "2025-12-31",
            "沪深",
            "买入+卖出",
            "成交金额",
            0.0003,
        ),
        ("STAMP_DUTY", "2014-01-01", "2023-08-27", "沪深", "卖出", "成交金额", 0.001),
        ("STAMP_DUTY", "2023-08-28", "2025-12-31", "沪深", "卖出", "成交金额", 0.0005),
        (
            "TRANSFER_FEE",
            "2014-01-01",
            "2015-07-31",
            "沪市",
            "买入+卖出",
            "成交面值",
            0.0006,
        ),
        (
            "TRANSFER_FEE",
            "2014-01-01",
            "2015-07-31",
            "深市",
            "买入+卖出",
            "成交金额",
            0.0000255,
        ),
        (
            "TRANSFER_FEE",
            "2015-08-01",
            "2022-04-28",
            "沪深",
            "买入+卖出",
            "成交金额",
            0.00002,
        ),
        (
            "TRANSFER_FEE",
            "2022-04-29",
            "2025-12-31",
            "沪深",
            "买入+卖出",
            "成交金额",
            0.00001,
        ),
        (
            "EXCHANGE_HANDLING_FEE",
            "2014-01-01",
            "2025-12-31",
            "沪深",
            "买入+卖出",
            "成交金额",
            0.0000341,
            "否",
            "BROKER_COMMISSION",
        ),
    ]
    rows = [
        _source_row(
            *spec[:7],
            included=spec[7] if len(spec) > 7 else "是",
            included_in=spec[8] if len(spec) > 8 else "",
        )
        for spec in row_specs
    ]
    with open_filesystem_file(
        csv_path, "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FEE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "file_sha256": hashlib.sha256(filesystem_read_bytes(csv_path)).hexdigest(),
        "file_bytes": filesystem_stat(csv_path).st_size,
        "row_count": len(rows),
        "fields_raw": list(FEE_CSV_COLUMNS),
    }
    filesystem_write_text(
        sidecar_path,
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return csv_path, sidecar_path


def _source_row(
    fee_code: str,
    start: str,
    end: str,
    market: str,
    direction: str,
    basis: str,
    rate: float,
    *,
    included: str = "是",
    included_in: str = "",
) -> dict[str, str]:
    row = dict.fromkeys(FEE_CSV_COLUMNS, "")
    row.update(
        {
            "费用代码": fee_code,
            "生效日": start,
            "截止日": end,
            "市场": market,
            "方向": direction,
            "计费基数": basis,
            "十进制费率": str(rate),
            "费率(‰)": str(rate * 1_000.0),
            "费率(bps)": str(rate * 10_000.0),
            "进入正式账本": included,
            "包含于": included_in,
        }
    )
    return row


def _rewrite_sidecar(sidecar_path: Path, **changes: object) -> None:
    metadata = json.loads(filesystem_read_text(sidecar_path, encoding="utf-8"))
    metadata.update(changes)
    filesystem_write_text(
        sidecar_path,
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def test_fee_source_preserves_identity_and_exact_counts(
    tmp_path: Path,
) -> None:
    csv_path, sidecar_path = _copy_fee_source(tmp_path)
    source = load_a_share_fee_source(reference_root=csv_path.parent.parent)

    assert len(source.rows) == 14
    assert len(source.csv_sha256) == 64
    assert source.source_identity != source.csv_sha256
    frame = pd.DataFrame(
        [
            {
                field: getattr(row, field)
                for field in (
                    "fee_code",
                    "effective_date",
                    "end_date",
                    "market",
                    "side",
                    "basis",
                    "rate",
                    "included_in_ledger",
                    "included_in",
                )
            }
            for row in source.rows
        ]
    )
    require_fee_coverage(
        fee_schedule_from_frame(
            frame,
            content_identity=source.source_identity,
        ),
        date(2019, 1, 1),
        date(2025, 12, 31),
        markets=("shanghai", "shenzhen"),
    )


@pytest.mark.parametrize("field", ("row_count", "file_bytes"))
def test_fee_source_rejects_sidecar_count_and_byte_mutations(
    tmp_path: Path,
    field: str,
) -> None:
    _csv_path, sidecar_path = _copy_fee_source(tmp_path)
    metadata = json.loads(filesystem_read_text(sidecar_path, encoding="utf-8"))
    _rewrite_sidecar(sidecar_path, **{field: metadata[field] + 1})

    with pytest.raises(ValueError, match="a_share_fee_source_(row_count|metadata)"):
        load_a_share_fee_source(reference_root=_csv_path.parent.parent)


def test_fee_source_rejects_csv_hash_and_schema_mutations(tmp_path: Path) -> None:
    csv_path, sidecar_path = _copy_fee_source(tmp_path)
    filesystem_write_bytes(csv_path, filesystem_read_bytes(csv_path) + b"\n")
    with pytest.raises(ValueError, match="a_share_fee_source_metadata_invalid"):
        load_a_share_fee_source(reference_root=csv_path.parent.parent)

    csv_path, sidecar_path = _copy_fee_source(tmp_path / "schema")
    raw = filesystem_read_text(csv_path, encoding="utf-8-sig")
    filesystem_write_text(
        csv_path, raw.replace("市场", "市场X", 1), encoding="utf-8-sig"
    )
    digest = hashlib.sha256(filesystem_read_bytes(csv_path)).hexdigest()
    _rewrite_sidecar(
        sidecar_path,
        file_sha256=digest,
        file_bytes=filesystem_stat(csv_path).st_size,
    )
    with pytest.raises(ValueError, match="a_share_fee_source_schema_invalid"):
        load_a_share_fee_source(reference_root=csv_path.parent.parent)


def test_fee_source_sidecar_change_changes_combined_identity(tmp_path: Path) -> None:
    csv_path, sidecar_path = _copy_fee_source(tmp_path)
    first = load_a_share_fee_source(reference_root=csv_path.parent.parent)
    _rewrite_sidecar(sidecar_path, notes="sidecar identity mutation")
    second = load_a_share_fee_source(reference_root=csv_path.parent.parent)

    assert first.csv_sha256 == second.csv_sha256
    assert first.source_identity != second.source_identity


def test_fee_schedule_adapter_rejects_text_boolean(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "fee_code": ["BROKER_COMMISSION"],
            "effective_date": [pd.Timestamp("2023-01-01")],
            "end_date": [pd.Timestamp("2025-12-31")],
            "market": ["a_share"],
            "side": ["buy"],
            "basis": ["amount"],
            "rate": [0.0003],
            "included_in_ledger": ["False"],
            "included_in": ["BROKER_COMMISSION"],
        }
    )
    with pytest.raises(ValueError, match="a_share_fee_schedule_frame_invalid"):
        fee_schedule_from_frame(frame, content_identity="0" * 64)


def test_fee_rules_reject_overlap_gap_and_face_value_without_notional(
    tmp_path: Path,
) -> None:
    schedule = _real_fee_schedule(tmp_path)
    with pytest.raises(ValueError, match="a_share_fee_rule_invalid"):
        replace(schedule.rules[0], included_in="BROKER_COMMISSION")
    stamps = [rule for rule in schedule.rules if rule.fee_code == "STAMP_DUTY"]
    assert len(stamps) == 2
    overlap = tuple(
        replace(stamps[1], effective_date=stamps[0].end_date)
        if rule is stamps[1]
        else rule
        for rule in schedule.rules
    )
    with pytest.raises(ValueError, match="interval_overlap"):
        FeeSchedule(
            rules=overlap,
            content_identity=schedule.content_identity,
        )

    gap = tuple(
        replace(stamps[0], end_date=stamps[0].end_date - timedelta(days=1))
        if rule is stamps[0]
        else rule
        for rule in schedule.rules
    )
    gap_schedule = FeeSchedule(
        rules=gap,
        content_identity=schedule.content_identity,
    )
    with pytest.raises(ValueError, match="coverage_missing"):
        require_fee_coverage(
            gap_schedule,
            date(2023, 8, 27),
            date(2023, 8, 28),
            markets=("shanghai",),
        )

    transfer = next(
        rule
        for rule in schedule.rules
        if rule.fee_code == "TRANSFER_FEE"
        and rule.market == "a_share"
        and rule.effective_date == date(2015, 8, 1)
    )
    basis_mutation = tuple(
        replace(transfer, basis="face_value") if rule is transfer else rule
        for rule in schedule.rules
    )
    basis_schedule = FeeSchedule(
        rules=basis_mutation,
        content_identity=schedule.content_identity,
    )
    with pytest.raises(ValueError, match="face_value_notional_required"):
        trade_cost(
            date(2019, 1, 1),
            "shanghai",
            "buy",
            1_000.0,
            1_000_000.0,
            basis_schedule,
            ExecutionAssumptions(
                initial_nav=1_000_000_000.0,
                slippage_bps=5.0,
                impact_bps=25.0,
                adv_participation=0.05,
                adv_sensitivities=(0.01, 0.02, 0.05, 0.10),
            ),
        )
    with pytest.raises(ValueError, match="face_value_notional_invalid"):
        trade_cost(
            date(2019, 1, 1),
            "shanghai",
            "buy",
            1_000.0,
            1_000_000.0,
            basis_schedule,
            _assumptions(),
            face_value_notional=0.0,
        )

    ambiguous = tuple(
        (
            *schedule.rules,
            FeeRule(
                fee_code="TRANSFER_FEE",
                effective_date=date(2019, 1, 1),
                end_date=date(2025, 12, 31),
                market="shanghai",
                side="buy",
                basis="amount",
                rate=0.0001,
                included_in_ledger=True,
            ),
        )
    )
    with pytest.raises(ValueError, match="interval_overlap"):
        FeeSchedule(
            rules=ambiguous,
            content_identity=schedule.content_identity,
        )


def test_fee_costs_are_side_aware_and_stamp_transition_is_exact(
    tmp_path: Path,
) -> None:
    schedule = _real_fee_schedule(tmp_path)
    assumptions = ExecutionAssumptions(
        initial_nav=1_000_000_000.0,
        slippage_bps=0.0,
        impact_bps=0.0,
        adv_participation=0.05,
        adv_sensitivities=(0.01, 0.02, 0.05, 0.10),
    )
    buy = trade_cost(
        date(2023, 8, 28),
        "shanghai",
        "buy",
        100_000.0,
        10_000_000.0,
        schedule,
        assumptions,
    )
    old_sell = trade_cost(
        date(2023, 8, 27),
        "shanghai",
        "sell",
        100_000.0,
        10_000_000.0,
        schedule,
        assumptions,
    )
    new_sell = trade_cost(
        date(2023, 8, 28),
        "shanghai",
        "sell",
        100_000.0,
        10_000_000.0,
        schedule,
        assumptions,
    )
    assert buy.commission_cost == pytest.approx(30.0)
    assert buy.stamp_duty_cost == 0.0
    assert buy.transfer_fee_cost == pytest.approx(1.0)
    old_transfer = trade_cost(
        date(2022, 4, 28),
        "shanghai",
        "buy",
        100_000.0,
        10_000_000.0,
        schedule,
        assumptions,
    )
    new_transfer = trade_cost(
        date(2022, 4, 29),
        "shanghai",
        "buy",
        100_000.0,
        10_000_000.0,
        schedule,
        assumptions,
    )
    assert old_transfer.transfer_fee_cost == pytest.approx(2.0)
    assert new_transfer.transfer_fee_cost == pytest.approx(1.0)
    assert old_sell.stamp_duty_cost == pytest.approx(100.0)
    assert new_sell.stamp_duty_cost == pytest.approx(50.0)
    for cost in (buy, old_sell, new_sell):
        assert cost.total_cost == pytest.approx(
            cost.commission_cost
            + cost.stamp_duty_cost
            + cost.transfer_fee_cost
            + cost.slippage_cost
            + cost.impact_cost
        )


def test_zero_cost_sell_buy_transition_is_exact_and_normalized() -> None:
    result = _zero_cost_transition()
    assert result.nav_after == pytest.approx(100.0)
    assert result.post_trade_cash == pytest.approx(0.0)
    assert (
        result.asset_ledger.set_index("asset_id")["post_trade_notional"]
        / result.nav_after
    ).to_dict() == {"a": 0.0, "b": 1.0, "c": 0.0}
    assert result.cash_weight == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(0.0)
    ledger = result.asset_ledger
    assert ledger["asset_id"].tolist() == ["a", "b", "c"]
    assert (
        ledger["pre_trade_notional"] + ledger["executed_delta_notional"]
        == ledger["post_trade_notional"]
    ).all()
    assert ledger.loc[ledger["asset_id"].eq("b"), "executed_delta_notional"].iloc[
        0
    ] == pytest.approx(100.0)
    view = result.asset_ledger
    view.loc[view.asset_id.eq("b"), "post_trade_notional"] = 0.0
    assert result.asset_ledger.set_index("asset_id").loc[
        "b", "post_trade_notional"
    ] == pytest.approx(100.0)


@pytest.mark.parametrize("mutation", ("wrong_direction", "overshoot", "phantom_cost"))
def test_execution_accounting_rejects_canonical_ledger_mutations(
    mutation: str,
) -> None:
    result = _zero_cost_transition()
    ledger = result.asset_ledger
    if mutation == "wrong_direction":
        ledger.loc[ledger["asset_id"].eq("a"), "executed_delta_notional"] = 1.0
        ledger.loc[ledger["asset_id"].eq("a"), "post_trade_notional"] = 101.0
    elif mutation == "overshoot":
        ledger.loc[ledger["asset_id"].eq("b"), "executed_delta_notional"] = 101.0
        ledger.loc[ledger["asset_id"].eq("b"), "post_trade_notional"] = 101.0
    else:
        ledger.loc[ledger["asset_id"].eq("c"), "commission_cost"] = 1.0
    with pytest.raises(
        ValueError,
        match="execution_ledger_(direction|phantom_cost)_invalid",
    ):
        replace(result, _asset_ledger=ledger)


def test_zero_adv_is_legal_zero_fill_and_missing_adv_only_blocks_demand() -> None:
    zero = execute_continuous_target(
        pd.Series({"a": 1.0}),
        pd.Series({"b": 1.0}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 0.0, "b": 0.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(_schedule(), {"a": "shanghai", "b": "shanghai"}),
        cost_source_identity="0" * 64,
    )
    assert zero.asset_ledger["reason"].notna().all()
    assert set(zero.asset_ledger["reason"]) == {"adv_capacity"}
    assert zero.total_cost == 0.0

    for missing_adv in (pd.Series(dtype="float64"), pd.Series({"a": float("nan")})):
        no_trade = execute_continuous_target(
            pd.Series({"a": 1.0}),
            pd.Series({"a": 1.0}),
            nav=100.0,
            average_daily_volume=missing_adv,
            can_enter=pd.Series({"a": True}),
            can_exit=pd.Series({"a": True}),
            execution_date=date(2023, 8, 28),
            assumptions=_assumptions(),
            cost_for_trade=_a_share_cost(_schedule(), {"a": "shanghai"}),
            cost_source_identity="0" * 64,
        )
        assert no_trade.asset_ledger["executed_delta_notional"].eq(0.0).all()
        assert no_trade.asset_ledger["asset_id"].tolist() == ["a"]
    with pytest.raises(ValueError, match="execution_adv_invalid"):
        execute_continuous_target(
            pd.Series({"a": 1.0}),
            pd.Series({"a": 1.0}),
            nav=100.0,
            average_daily_volume=pd.Series({"a": -1.0}),
            can_enter=pd.Series({"a": True}),
            can_exit=pd.Series({"a": True}),
            execution_date=date(2023, 8, 28),
            assumptions=_assumptions(),
            cost_for_trade=_a_share_cost(_schedule(), {"a": "shanghai"}),
            cost_source_identity="0" * 64,
        )
    with pytest.raises(ValueError, match="execution_adv_invalid"):
        execute_continuous_target(
            pd.Series({"a": 0.0}),
            pd.Series({"a": 1.0}),
            nav=100.0,
            average_daily_volume=pd.Series(dtype="float64"),
            can_enter=pd.Series({"a": True}),
            can_exit=pd.Series({"a": True}),
            execution_date=date(2023, 8, 28),
            assumptions=_assumptions(),
            cost_for_trade=_a_share_cost(_schedule(), {"a": "shanghai"}),
            cost_source_identity="0" * 64,
        )


def test_empty_asset_transition_is_a_cash_only_ledger() -> None:
    result = execute_continuous_target(
        pd.Series(dtype="float64"),
        pd.Series(dtype="float64"),
        nav=100.0,
        average_daily_volume=pd.Series(dtype="float64"),
        can_enter=pd.Series(dtype="bool"),
        can_exit=pd.Series(dtype="bool"),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(_schedule(), {}),
        cost_source_identity="0" * 64,
    )
    assert result.asset_ledger.empty
    assert result.pre_trade_cash == pytest.approx(100.0)
    assert result.post_trade_cash == pytest.approx(100.0)
    assert result.nav_after == pytest.approx(100.0)
    assert result.total_cost == pytest.approx(0.0)


def test_blocked_sell_cannot_fund_buy_and_cash_partial_is_pro_rata() -> None:
    blocked = execute_continuous_target(
        pd.Series({"a": 0.5, "b": 0.0}),
        pd.Series({"a": 0.0, "b": 1.0}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 100.0, "b": 100.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": False, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(transfer_rate=0.0, broker_rate=0.0, stamp_rate=0.0),
            {"a": "shanghai", "b": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    assert blocked.post_trade_cash == pytest.approx(0.0)
    assert (
        blocked.asset_ledger.set_index("asset_id")["post_trade_notional"]
        / blocked.nav_after
    )["b"] == pytest.approx(0.5)
    assert (
        blocked.asset_ledger.loc[
            blocked.asset_ledger["asset_id"].eq("a"), "reason"
        ].iloc[0]
        == "market_blocked"
    )

    blocked_without_cash = execute_continuous_target(
        pd.Series({"a": 1.0, "b": 0.0}),
        pd.Series({"a": 0.0, "b": 1.0}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 100.0, "b": 100.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": False, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(transfer_rate=0.0, broker_rate=0.0, stamp_rate=0.0),
            {"a": "shanghai", "b": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    assert (
        blocked_without_cash.asset_ledger.set_index("asset_id")["post_trade_notional"]
        / blocked_without_cash.nav_after
    )["b"] == pytest.approx(0.0)
    assert (
        blocked_without_cash.asset_ledger.loc[
            blocked_without_cash.asset_ledger["asset_id"].eq("b"), "reason"
        ].iloc[0]
        == "cash_pro_rata"
    )

    partial = execute_continuous_target(
        pd.Series({"a": 0.0, "b": 0.0, "c": 0.0}),
        pd.Series({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 100.0, "b": 100.0, "c": 100.0}),
        can_enter=pd.Series({"a": True, "b": True, "c": True}),
        can_exit=pd.Series({"a": True, "b": True, "c": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(),
            {"a": "shanghai", "b": "shanghai", "c": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    buys = partial.asset_ledger.loc[
        partial.asset_ledger["target_notional"]
        > partial.asset_ledger["pre_trade_notional"]
    ]
    assert buys["executed_delta_notional"].nunique() == 1
    assert buys["reason"].eq("cash_pro_rata").all()


def test_cash_rotation_with_cost_keeps_target_cash_and_reconciles_delta() -> None:
    result = execute_continuous_target(
        pd.Series({"a": 0.5}),
        pd.Series({"b": 0.5}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 1_000.0, "b": 1_000.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(),
            {"a": "shanghai", "b": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    assert result.pre_trade_cash == pytest.approx(50.0)
    assert result.target_weights.sum() == pytest.approx(0.5)
    assert result.total_cost > 0.0
    assert result.post_trade_cash == pytest.approx(50.0 - result.total_cost)
    assert result.asset_ledger["executed_delta_notional"].sum() == pytest.approx(0.0)


def test_turnover_excludes_cost_and_drift_feeds_next_rebalance() -> None:
    schedule = _schedule()
    assumptions = _assumptions()
    first = execute_continuous_target(
        pd.Series({"a": 1.0}),
        pd.Series({"b": 1.0}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 1_000.0, "b": 1_000.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=assumptions,
        cost_for_trade=_a_share_cost(schedule, {"a": "shanghai", "b": "shanghai"}),
        cost_source_identity=schedule.content_identity,
    )
    assert first.realized_turnover == pytest.approx(1.0)
    first_ledger = first.asset_ledger
    assert first.total_cost > 0.0
    assert first.total_cost == pytest.approx(
        float(
            first_ledger.loc[
                :,
                [
                    "commission_cost",
                    "stamp_duty_cost",
                    "transfer_fee_cost",
                    "slippage_cost",
                    "impact_cost",
                ],
            ]
            .sum()
            .sum()
        )
    )
    assert first.nav_after == pytest.approx(first.nav_before - first.total_cost)
    drifted, drift_cash, drift_nav = drift_continuous_holdings(
        pd.Series({"a": 60.0, "b": 30.0}),
        10.0,
        pd.Series({"a": 0.10, "b": -0.10}),
    )
    assert drifted.to_dict() == {"a": 66.0, "b": 27.0}
    assert drift_cash == 10.0
    assert drift_nav == pytest.approx(103.0)
    second = execute_continuous_target(
        drifted / drift_nav,
        pd.Series({"a": 0.5, "b": 0.5}),
        nav=drift_nav,
        average_daily_volume=pd.Series({"a": 1_000.0, "b": 1_000.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=date(2023, 8, 29),
        assumptions=assumptions,
        cost_for_trade=_a_share_cost(
            _schedule(transfer_rate=0.0, broker_rate=0.0, stamp_rate=0.0),
            {"a": "shanghai", "b": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    assert not second.asset_ledger["executed_delta_notional"].eq(0.0).all()
    assert (
        second.asset_ledger.set_index("asset_id")["pre_trade_notional"]
        / second.nav_before
    )["a"] == pytest.approx(66.0 / 103.0)
    assert second.pre_trade_cash == pytest.approx(drift_cash)


def test_future_mask_mutation_changes_execution_not_past_target() -> None:
    signal = pd.Series({"a": 1.0, "b": 2.0})
    target = build_simple_target_weights(
        signal,
        eligible=pd.Series({"b": True, "a": True}),
        top_quantile=0.5,
    )
    args = dict(
        nav=100.0,
        average_daily_volume=pd.Series({"a": 100.0, "b": 100.0}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(transfer_rate=0.0, broker_rate=0.0, stamp_rate=0.0),
            {"a": "shanghai", "b": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    allowed = execute_continuous_target(
        pd.Series({"a": 1.0, "b": 0.0}),
        target,
        can_enter=pd.Series({"a": True, "b": True}),
        **args,
    )
    blocked = execute_continuous_target(
        pd.Series({"a": 1.0, "b": 0.0}),
        target,
        can_enter=pd.Series({"a": True, "b": False}),
        **args,
    )
    pd.testing.assert_series_equal(
        target,
        build_simple_target_weights(
            signal,
            eligible=pd.Series({"a": True, "b": True}),
            top_quantile=0.5,
        ),
    )
    assert (
        allowed.asset_ledger.set_index("asset_id")["post_trade_notional"]
        / allowed.nav_after
    )["b"] > (
        blocked.asset_ledger.set_index("asset_id")["post_trade_notional"]
        / blocked.nav_after
    )["b"]


def test_cash_constrained_buy_fills_are_asset_order_invariant() -> None:
    first_current, first_target, first_adv = _cash_constrained_inputs(("a", "b", "c"))
    second_current, second_target, second_adv = _cash_constrained_inputs(
        ("c", "b", "a")
    )

    first = execute_continuous_target(
        first_current,
        first_target,
        nav=100.0,
        average_daily_volume=first_adv,
        can_enter=pd.Series(True, index=first_adv.index),
        can_exit=pd.Series(True, index=first_adv.index),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(),
            {asset: "shanghai" for asset in first_adv.index},
        ),
        cost_source_identity="0" * 64,
    )
    second = execute_continuous_target(
        second_current,
        second_target,
        nav=100.0,
        average_daily_volume=second_adv,
        can_enter=pd.Series(True, index=second_adv.index),
        can_exit=pd.Series(True, index=second_adv.index),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(),
            {asset: "shanghai" for asset in second_adv.index},
        ),
        cost_source_identity="0" * 64,
    )

    pd.testing.assert_series_equal(
        (
            first.asset_ledger.set_index("asset_id")["post_trade_notional"]
            / first.nav_after
        ).sort_index(),
        (
            second.asset_ledger.set_index("asset_id")["post_trade_notional"]
            / second.nav_after
        ).sort_index(),
    )


def test_buy_costs_are_admitted_before_cash_sizing() -> None:
    result = execute_continuous_target(
        pd.Series({"asset": 0.0}),
        pd.Series({"asset": 1.0}),
        nav=1.0,
        average_daily_volume=pd.Series({"asset": 1_000_000.0}),
        can_enter=pd.Series({"asset": True}),
        can_exit=pd.Series({"asset": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(broker_rate=2.0, transfer_rate=0.0, stamp_rate=0.0),
            {"asset": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )

    assert result.cash_weight >= -1e-12

    tiny = execute_continuous_target(
        pd.Series({"asset": 0.0}),
        pd.Series({"asset": 1e-13}),
        nav=1.0,
        average_daily_volume=pd.Series({"asset": 1_000_000.0}),
        can_enter=pd.Series({"asset": True}),
        can_exit=pd.Series({"asset": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(broker_rate=2.0, transfer_rate=0.0, stamp_rate=0.0),
            {"asset": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    tiny_delta = float(tiny.asset_ledger["executed_delta_notional"].iloc[0])
    assert 0.0 < tiny_delta < 1e-12
    assert tiny.total_cost > 0.0


def test_simple_target_includes_all_boundary_ties() -> None:
    signal = pd.Series(
        {"a": 1.0, "b": 2.0, "c": 2.0, "d": 2.0, "e": 3.0},
        dtype="float64",
    )
    target = build_simple_target_weights(
        signal,
        eligible=pd.Series(True, index=signal.index),
        top_quantile=0.4,
    )

    assert target.loc["a"] == 0.0
    assert target.loc[["b", "c", "d", "e"]].tolist() == [
        0.25,
        0.25,
        0.25,
        0.25,
    ]


def test_simple_target_float_cap_and_long_short_weights() -> None:
    signal = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, dtype="float64")
    caps = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, dtype="float64")
    long_only = build_simple_target_weights(
        signal,
        eligible=pd.Series(True, index=signal.index),
        method="float_cap_weight",
        top_quantile=0.5,
        float_market_cap=caps,
    )
    assert long_only.tolist() == [0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0]

    long_short = build_simple_target_weights(
        signal,
        eligible=pd.Series(True, index=signal.index),
        portfolio_mode="long_short",
        top_quantile=0.25,
        bottom_quantile=0.25,
    )
    assert long_short.tolist() == [-0.5, 0.0, 0.0, 0.5]
    assert long_short.sum() == 0.0
    assert long_short.abs().sum() == 1.0


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_simple_target_rejects_missing_eligible_signal(bad: float) -> None:
    with pytest.raises(ValueError, match="target_signal_invalid"):
        build_simple_target_weights(
            pd.Series({"a": bad, "b": 2.0}),
            eligible=pd.Series({"a": True, "b": True}),
        )


def test_cash_search_stops_at_float_stagnation_and_preserves_old_low_exactly():
    assumptions = _assumptions()
    when = date(2023, 8, 28)
    actual_cost = _a_share_cost(_schedule(), {"a": "shanghai"})
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        amount = 100.0 * middle
        cost = actual_cost(when, "a", "buy", amount, 1000.0, assumptions)
        if amount + cost.total_cost <= 100.0:
            low = middle
        else:
            high = middle
    expected_amount = 100.0 * low
    expected_cost = actual_cost(when, "a", "buy", expected_amount, 1000.0, assumptions)
    calls = []

    def observed_cost(*args):
        calls.append(args[3])
        return actual_cost(*args)

    result = execute_continuous_target(
        pd.Series({"a": 0.0}),
        pd.Series({"a": 1.0}),
        nav=100.0,
        average_daily_volume=pd.Series({"a": 1000.0}),
        can_enter=pd.Series({"a": True}),
        can_exit=pd.Series({"a": True}),
        execution_date=when,
        assumptions=assumptions,
        cost_for_trade=observed_cost,
        cost_source_identity="0" * 64,
    )
    assert result.asset_ledger.executed_delta_notional.item() == expected_amount
    assert result.post_trade_cash == max(
        0.0, 100.0 - expected_amount - expected_cost.total_cost
    )
    assert len(calls) <= np.finfo(float).nmant + 3


@pytest.mark.parametrize(
    "notional,adv,bps",
    (
        (1e307, 1e307, 25.0),
        (1e308, 1e308, 25.0),
        (1e-308, 1e-308, 1e308),
        (1e308, 1e308, 1e-320),
        (1e-100, 1e308, 1e308),
        (1e308, 1e-100, 1e-320),
        (1e308, 1e-100, 0.0),
    ),
)
def test_trade_cost_preserves_representable_scaled_costs(notional, adv, bps):
    from decimal import Decimal, localcontext

    assumptions = replace(_assumptions(), slippage_bps=bps, impact_bps=bps)
    with localcontext() as context:
        context.prec = 100
        amount = Decimal.from_float(notional)
        rate = Decimal.from_float(bps) / 10000
        expected_slippage = float(amount * rate)
        expected_impact = float(
            amount * rate * (amount / Decimal.from_float(adv)).sqrt()
        )
    result = trade_cost(
        date(2023, 8, 28),
        "shanghai",
        "buy",
        notional,
        adv,
        _schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0),
        assumptions,
    )
    assert result.slippage_cost == pytest.approx(expected_slippage, rel=1e-14, abs=0)
    assert result.impact_cost == pytest.approx(expected_impact, rel=1e-14, abs=0)


@pytest.mark.parametrize("nav", (1e8, 1e307))
def test_scaled_execution_costs_reconcile_cash(nav):
    assumptions = replace(
        _assumptions(), initial_nav=nav, slippage_bps=5.0, impact_bps=25.0
    )
    schedule = _schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0)
    result = execute_continuous_target(
        pd.Series({"a": 1.0}),
        pd.Series({"a": 0.0}),
        nav=nav,
        average_daily_volume=pd.Series({"a": nav}),
        can_enter=pd.Series({"a": True}),
        can_exit=pd.Series({"a": True}),
        execution_date=date(2023, 8, 28),
        assumptions=assumptions,
        cost_for_trade=_a_share_cost(schedule, {"a": "shanghai"}),
        cost_source_identity=schedule.content_identity,
    )
    # Full liquidation at unit participation costs 5 + 25 bps, leaving cash only.
    assert result.total_cost / nav == pytest.approx(0.003, rel=1e-14, abs=0)
    assert result.nav_after / nav == pytest.approx(0.997, rel=1e-14, abs=0)
    assert result.post_trade_cash == result.nav_after
    assert result.asset_ledger["post_trade_notional"].tolist() == [0.0]


def test_cash_search_vector_costs_build_trade_details_only_once():
    from quant_research_platform.markets.implementations.a_share.execution import (
        prepare_trade_cost,
    )

    schedule = _schedule(broker_rate=0.1, transfer_rate=0.0, stamp_rate=0.0)
    assumptions = replace(
        _assumptions(), slippage_bps=0.0, impact_bps=0.0, adv_participation=1.0
    )
    when = date(2023, 8, 28)
    evaluator = prepare_trade_cost(when, "shanghai", "buy", schedule, assumptions)
    details, vectors = [], []

    def trade(_date, asset, side, amount, adv, _assumptions):
        assert side == "buy"
        details.append(asset)
        return evaluator.trade(amount, adv)

    def totals(assets, amounts, adv):
        vectors.append(len(assets))
        return evaluator.totals(amounts, adv)

    result = _execute_continuous_target(
        pd.Series({"a": 0.0, "b": 0.0}),
        pd.Series({"a": 0.5, "b": 0.5}),
        nav=88.0,
        average_daily_volume=pd.Series({"a": 1000.0, "b": 1000.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=when,
        cost_for_trade=trade,
        cost_totals_for_buys=totals,
        assumptions=assumptions,
        cost_source_identity=schedule.content_identity,
    )
    assert details == ["a", "b"]
    assert len(vectors) > 1
    assert result.asset_ledger.executed_delta_notional.tolist() == pytest.approx(
        [40.0, 40.0]
    )
    assert result.total_cost == pytest.approx(8.0)
    assert result.post_trade_cash == pytest.approx(0.0, abs=1e-12)
