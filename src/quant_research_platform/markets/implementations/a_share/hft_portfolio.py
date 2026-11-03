"""Independent stock-day inventory accounts using observed subsequent quotes."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    canonical_json_hash,
)
from quant_research_platform.contracts.models import _validate_model_prediction_window
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    ExecutedTrade,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.time import PredictionWindow, TimeWindow
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    require_fee_coverage,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
    read_dataset,
)


def _parameters(value):
    params = _strict_config(
        value,
        {
            "initial_capital",
            "initial_inventory",
            "maximum_inventory",
            "lot_size",
            "buy_threshold",
            "sell_threshold",
            "latency_ms",
            "participation_rate",
            "minimum_commission",
        },
        set(),
    )
    for name in ("initial_inventory", "maximum_inventory", "lot_size"):
        if type(params[name]) is not int or params[name] < (
            1 if name == "lot_size" else 0
        ):
            raise ValueError("hft_inventory_invalid")
    for name in set(params) - {"initial_inventory", "maximum_inventory", "lot_size"}:
        params[name] = strict_finite_float(params[name], error="hft_parameter_invalid")
    if (
        params["initial_capital"] <= 0
        or not 0 <= params["initial_inventory"] <= params["maximum_inventory"]
        or params["initial_inventory"] % params["lot_size"]
        or params["maximum_inventory"] % params["lot_size"]
        or params["buy_threshold"] <= params["sell_threshold"]
        or params["latency_ms"] < 0
        or not 0 < params["participation_rate"] <= 1
        or params["minimum_commission"] < 0
    ):
        raise ValueError("hft_parameter_invalid")
    return params


def inventory_batches(quotes, predictions, schedule, params):
    """T+1 applies to sellable opening inventory; buys never replenish it today."""
    quotes = quotes.sort_values("timestamp").reset_index(drop=True)
    if (
        quotes.empty
        or quotes.asset_id.nunique() != 1
        or quotes.session_date.nunique() != 1
    ):
        raise ValueError("hft_account_requires_one_stock_day")
    asset = quotes.asset_id.iloc[0]
    day = pd.Timestamp(quotes.session_date.iloc[0]).date()
    market = {"SSE": "shanghai", "SZSE": "shenzhen"}[quotes.exchange.iloc[0]]
    require_fee_coverage(schedule, day, day, markets=(market,))
    # Invalid books do not trade; the last observed valid mid remains an explicit mark.
    first_valid = np.flatnonzero(quotes.eligible & np.isfinite(quotes.mid))
    if not len(first_valid):
        raise ValueError("hft_account_initial_mark_missing")
    quotes = quotes.iloc[first_valid[0] :].reset_index(drop=True)
    times = quotes.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
    signal_times = predictions.timestamp.to_numpy(dtype="datetime64[ns]").astype(
        "int64"
    )
    decisions = {}
    for row, signal_at in zip(
        predictions.itertuples(index=False), signal_times, strict=True
    ):
        eligible_at = signal_at + int(params["latency_ms"] * 1_000_000)
        index = max(
            np.searchsorted(times, eligible_at),
            np.searchsorted(times, signal_at, side="right"),
        )
        decision = np.searchsorted(times, signal_at)
        if (
            index < len(times)
            and decision < len(times)
            and times[decision] == signal_at
            and quotes.segment_id.iloc[index] == quotes.segment_id.iloc[decision]
        ):
            decisions[index] = (int(signal_at), float(row.prediction_score))
    quantity = params["initial_inventory"]
    sellable = quantity
    mark = float(quotes.mid.iloc[0])
    cash = params["initial_capital"] - quantity * mark
    if cash < 0:
        raise ValueError("hft_initial_inventory_exceeds_capital")
    previous_time = int(times[0])
    lifecycle = f"{asset}:{day.isoformat()}"
    lot = params["lot_size"]
    fee_names = {
        "BROKER_COMMISSION": "commission_cost",
        "STAMP_DUTY": "stamp_duty_cost",
        "TRANSFER_FEE": "transfer_fee_cost",
    }

    def valuation(phase, at, held, money, price):
        return (
            PortfolioValuation(
                at, asset, phase, held, held * price, price, lifecycle_id=lifecycle
            ),
            PortfolioValuation(at, PORTFOLIO_CASH_ASSET_ID, phase, money, money),
        )

    for index, quote in enumerate(quotes.itertuples(index=False)):
        timestamp = int(times[index])
        before = valuation("before", previous_time, quantity, cash, mark)
        trades, flows, diagnostics = (), (), []
        if quote.eligible and np.isfinite(quote.mid):
            mark = float(quote.mid)
        diagnostics.append(
            PortfolioDiagnostic(
                asset,
                None,
                "mark_source",
                text="current_mid" if quote.eligible else "last_valid_mid",
            )
        )
        if index in decisions and quote.eligible:
            decision_at, score = decisions[index]
            desired = (
                params["maximum_inventory"] - quantity
                if score > params["buy_threshold"]
                else -sellable
                if score < params["sell_threshold"]
                else 0
            )
            side = "buy" if desired > 0 else "sell"
            price = float(quote.ask_px1 if desired > 0 else quote.bid_px1)
            depth = quote.ask_vol1 if desired > 0 else quote.bid_vol1
            amount = min(
                abs(desired), int(depth * params["participation_rate"] // lot) * lot
            )

            def fees(units, side=side, price=price):
                result = {}
                for rule in schedule.rules_for(day, market=market, side=side):
                    if rule.basis != "amount":
                        raise ValueError("hft_fee_requires_notional_basis")
                    charge = units * price * rule.rate
                    if rule.fee_code == "BROKER_COMMISSION" and units:
                        charge = max(charge, params["minimum_commission"])
                    result[fee_names[rule.fee_code]] = charge
                return result

            if desired > 0:
                amount = min(amount, int(cash / price // lot) * lot)
                # Fee plus principal is monotone in lots, including minimum commission.
                lo, hi = 0, amount // lot
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if mid * lot * price + sum(fees(mid * lot).values()) <= cash:
                        lo = mid
                    else:
                        hi = mid - 1
                amount = lo * lot
            elif amount and sum(fees(amount).values()) > cash + amount * price:
                amount = 0
            if amount:
                units = amount if desired > 0 else -amount
                costs = fees(amount)
                notional = units * price
                fill = str(index)
                trades = (
                    ExecutedTrade(
                        fill, str(decision_at), timestamp, asset, units, notional
                    ),
                )
                flows = tuple(
                    PortfolioCashFlow(
                        f"{fill}:{name}", timestamp, asset, fill, name, value
                    )
                    for name, value in (
                        ("principal", -notional),
                        *((key, -value) for key, value in costs.items()),
                    )
                )
                cash += sum(item.amount for item in flows)
                quantity += units
                if units < 0:
                    sellable += units
                diagnostics.append(
                    PortfolioDiagnostic(
                        asset, fill, "decision_timestamp_ns", text=str(decision_at)
                    )
                )
        diagnostics.append(
            PortfolioDiagnostic(
                asset, None, "sellable_quantity", number=float(sellable)
            )
        )
        yield PortfolioLedgerBatch(
            index,
            trades,
            flows,
            (
                *before,
                *valuation("after", timestamp, quantity, cash, mark),
                *valuation("end", timestamp, quantity, cash, mark),
            ),
            tuple(diagnostics),
        )
        previous_time = timestamp


def backtest(config):
    """Publish independent stock-day ledgers, never a fabricated overnight NAV chain."""
    config = _strict_config(
        config,
        {"panel", "predictions", "fee_schedule", "window", "parameters", "output"},
        set(),
    )
    params = _parameters(config["parameters"])
    window = TimeWindow(**config["window"])
    panel = open_dataset_session(
        **_dataset_input_options(config["panel"], time_window=window)
    )
    if panel.spec.get("market_id") != "a_share" or panel.spec["data_id"] != "hft_panel":
        raise ValueError("hft_panel_required")
    fee_frame = read_dataset(Path(config["fee_schedule"]))
    schedule = fee_schedule_from_frame(
        fee_frame,
        content_identity=canonical_json_hash(fee_frame.to_json(date_format="iso")),
    )

    def write(staging):
        records = []
        for day in pd.date_range(
            pd.Timestamp(window.start).normalize(), pd.Timestamp(window.end).normalize()
        ):
            day_window = TimeWindow(day.date(), day.date())
            quotes = materialize_dataset_session(panel.filtered_view(day_window))
            if quotes.empty:
                continue
            predictions, facts = read_predictions(
                Path(config["predictions"]), time_window=day_window
            )
            if facts["model_facts"]["market_id"] != "a_share":
                raise ValueError("hft_prediction_market_mismatch")
            _validate_model_prediction_window(
                facts["model_facts"], PredictionWindow(window.start, window.end)
            )
            for account_index, (asset, group) in enumerate(
                quotes.groupby("asset_id", sort=True)
            ):
                signals = predictions.loc[predictions.asset_id.eq(asset)]
                member = f"{day:%Y%m%d}/{account_index}"
                capital = PortfolioCapital(
                    asset, params["initial_capital"], "CNY", "cash_equity"
                )

                def produce(emit, group=group, signals=signals):
                    for batch in inventory_batches(group, signals, schedule, params):
                        emit(batch)

                path = write_portfolio_ledger_artifact(
                    staging / member,
                    capital=capital,
                    metadata={
                        "market_id": "a_share",
                        "kind": "a_share_hft",
                        "window": {
                            "start": day.date().isoformat(),
                            "end": day.date().isoformat(),
                        },
                        "parameters": params,
                        "account_semantics": (
                            "independent_stock_day_prior_inventory_no_overnight_carry"
                        ),
                        "execution": (
                            "strictly_subsequent_observed_top_of_book_no_queue_or_receive_clock"
                        ),
                        "fee_schedule_identity": schedule.content_identity,
                    },
                    produce=produce,
                )
                ledger = read_portfolio_ledger_artifact(path)
                records.append(
                    {
                        "member": member,
                        "session_date": day.date().isoformat(),
                        "asset_id": asset,
                        **summarize_portfolio_ledger(capital, ledger.iter_chunks()),
                    }
                )
        if not records:
            raise ValueError("hft_backtest_empty")
        atomic_write_json(
            staging / "summary.json",
            {
                "schema_id": "qrp-a-share-hft-accounts",
                "accounts": records,
                "account_semantics": "independent_stock_day_no_compounding",
                "net_pnl": sum(row["net_pnl"] for row in records),
                "fees": sum(row["fees"] for row in records),
                "turnover": sum(row["turnover"] for row in records),
                "capital_return": sum(row["net_pnl"] for row in records)
                / sum(row["initial_capital"] for row in records),
            },
        )

    atomic_write_directory(Path(config["output"]), write)
    return Path(config["output"])
