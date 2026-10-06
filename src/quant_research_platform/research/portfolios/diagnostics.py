"""Production portfolio diagnostics builders."""

from __future__ import annotations

from contextlib import ExitStack
from math import comb

import numpy as np
import pandas as pd

from quant_research_platform.contracts.experiments import _model_inference_blocker
from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share import ids as a_share_ids
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.accounting import (
    capacity_quantiles,
    portfolio_interval_economics,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
    HoldingFactorInputs,
    holding_attribution,
)
from quant_research_platform.research.portfolios.schedules import holding_intervals
from quant_research_platform.research.statistics import (
    _return_sharpes,
    cscv_pbo,
    probabilistic_and_deflated_sharpe,
    return_standard_deviation,
    stationary_bootstrap_mean_ci,
    time_series_inference_summary,
)


def _nav_return_roundoff(start: object, end: object, returns: object) -> np.ndarray:
    """Enclose NAV-ratio rounding using adjacent float64 operand values.

    One outward ULP encloses each rounded NAV. Outward division/subtraction
    bounds cover both (end / start - 1) and ((end - start) / start).
    """
    before = np.asarray(start, dtype="float64")
    after = np.asarray(end, dtype="float64")
    with np.errstate(over="ignore", divide="ignore"):
        lower = np.nextafter(
            np.nextafter(after, -np.inf)
            / np.nextafter(before, np.where(after >= 0, np.inf, -np.inf)),
            -np.inf,
        )
        upper = np.nextafter(
            np.nextafter(after, np.inf)
            / np.nextafter(before, np.where(after >= 0, -np.inf, np.inf)),
            np.inf,
        )
        lower = np.nextafter(lower - 1.0, -np.inf)
        upper = np.nextafter(upper - 1.0, np.inf)
        data = np.asarray(returns, dtype="float64")
        return np.maximum(np.abs(data - lower), np.abs(upper - data))


def fixed_turnover_cost_sensitivity(
    *,
    turnover,
    net_pnl,
    base_fee_bps,
    fee_scenarios_bps,
    additional_spread_scenarios_bps,
):
    """Project explicit fee/spread stresses onto the same actually filled turnover."""
    turnover = strict_finite_float(
        turnover, error="hft_cost_sensitivity_result_invalid"
    )
    net_pnl = strict_finite_float(net_pnl, error="hft_cost_sensitivity_result_invalid")
    base_fee_bps = strict_finite_float(
        base_fee_bps, error="hft_cost_sensitivity_result_invalid"
    )
    result = []
    for fee in fee_scenarios_bps:
        fee = strict_finite_float(fee, error="hft_cost_sensitivity_invalid")
        for spread in additional_spread_scenarios_bps:
            spread = strict_finite_float(spread, error="hft_cost_sensitivity_invalid")
            if min(fee, spread) < 0:
                raise ValueError("hft_cost_sensitivity_invalid")
            change = turnover * (fee - base_fee_bps + spread) / 10000.0
            result.append(
                {
                    "fee_bps": fee,
                    "additional_spread_bps": spread,
                    "cost_delta": change,
                    "net_pnl": net_pnl - change,
                    "method": "constant_actual_turnover_additive_sensitivity",
                }
            )
    return result


def _crypto_observations(ledger, windows, limit, *, observation_window_ns=None):
    bounds = [
        (fold, pd.Timestamp(window.start).value, pd.Timestamp(window.end).value)
        for fold, window in windows
    ]
    day_ns = 86400000000000
    utc = {}
    previous_utc_timestamp = None
    endpoints = {}
    previous_timestamp = (
        bounds[0][1] if observation_window_ns is None else observation_window_ns[0]
    )
    previous_position = 0.0
    inventory = {
        scope: {
            "duration": 0,
            "mean": 0.0,
            "m2": 0.0,
            "at_limit": 0,
            "max_abs": 0.0,
            "observations": 0,
        }
        for scope in ("oos", "gap")
    }
    peak = ledger.capital.initial_capital
    drawdown = 0.0

    def observe(batch):
        nonlocal \
            previous_utc_timestamp, \
            previous_timestamp, \
            previous_position, \
            peak, \
            drawdown
        endings = [row for row in batch.valuations if row.phase == "end"]
        timestamp = endings[0].timestamp_ns
        nav = sum(row.value for row in endings)
        position = sum(
            row.quantity for row in endings if row.asset_id != PORTFOLIO_CASH_ASSET_ID
        )
        duration = max(0, timestamp - previous_timestamp)
        oos_duration = sum(
            (
                max(0, min(timestamp, end) - max(previous_timestamp, start))
                for _fold, start, end in bounds
            )
        )
        for scope, weight in (("oos", oos_duration), ("gap", duration - oos_duration)):
            if not weight:
                continue
            state = inventory[scope]
            new_duration = state["duration"] + weight
            delta = previous_position - state["mean"]
            state["mean"] += delta * weight / new_duration
            state["m2"] += weight * delta * (previous_position - state["mean"])
            state["duration"] = new_duration
            state["at_limit"] += weight * int(abs(previous_position) >= limit)
            state["max_abs"] = max(state["max_abs"], abs(previous_position))
        for diagnostic in batch.diagnostics:
            if diagnostic.name == "utc_day_boundary":
                if timestamp % day_ns or timestamp == previous_utc_timestamp:
                    raise ValueError("crypto_utc_day_boundary_invalid")
                previous_utc_timestamp = timestamp
                if any((start <= timestamp <= end for _fold, start, end in bounds)):
                    utc[timestamp] = nav
            elif diagnostic.name == "oos_window_start":
                endpoints[int(diagnostic.number), "start"] = nav
            elif diagnostic.name == "oos_window_end":
                endpoints[int(diagnostic.number), "end"] = nav
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1.0)
        scope = (
            "oos"
            if any((start <= timestamp <= end for _fold, start, end in bounds))
            else "gap"
        )
        state = inventory[scope]
        running_position = sum(
            row.quantity
            for row in batch.valuations
            if row.phase == "before" and row.asset_id != PORTFOLIO_CASH_ASSET_ID
        )
        state["max_abs"] = max(state["max_abs"], abs(running_position), abs(position))
        for trade in batch.trades:
            running_position += trade.quantity
            state["max_abs"] = max(state["max_abs"], abs(running_position))
        state["observations"] += 1
        previous_timestamp, previous_position = (timestamp, position)

    def finish():
        daily = []
        for index, (fold, start, end) in enumerate(bounds):
            observed_start, observed_end = (
                (start, end) if observation_window_ns is None else observation_window_ns
            )
            point = (max(start, observed_start) + day_ns - 1) // day_ns * day_ns
            while point + day_ns <= min(end, observed_end):
                start_nav = (
                    endpoints.get((index, "start"))
                    if point == start
                    else utc.get(point)
                )
                end_nav = (
                    endpoints.get((index, "end"))
                    if point + day_ns == end
                    else utc.get(point + day_ns)
                )
                reason = (
                    "utc_boundary_valuation_missing"
                    if start_nav is None or end_nav is None
                    else "nonpositive_start_nav"
                    if start_nav <= 0
                    else ""
                )
                daily.append(
                    {
                        "fold_id": fold,
                        "start_at": pd.Timestamp(point),
                        "end_at": pd.Timestamp(point + day_ns),
                        "start_nav": np.nan if start_nav is None else start_nav,
                        "end_nav": np.nan if end_nav is None else end_nav,
                        "net_return": (end_nav - start_nav) / start_nav
                        if not reason
                        else np.nan,
                        "benchmark_return": 0.0,
                        "status": "UNDEFINED" if reason else "DEFINED",
                        "reason": reason,
                    }
                )
                point += day_ns
        return (daily, inventory, drawdown)

    return (observe, finish)


def build_crypto_portfolio_statistics_tables(
    *,
    layout,
    protocol_manifest,
    time_window,
    upstream_identities,
    read_ledger,
    read_episodes,
    summarize_ledger,
    table_columns,
):
    from quant_research_platform.research.protocols.crypto_hft import (
        CryptoMakerCandidate,
        execution_windows,
        portfolio_candidates,
    )

    (candidate,) = portfolio_candidates(protocol_manifest["portfolio_candidates"])
    mode = protocol_manifest["execution_mode"]
    target_id = protocol_manifest["execution_target_id"]
    plan = protocol_manifest["research_plan"]
    statistics_policy = protocol_manifest["portfolio_statistics"]
    windows = execution_windows(
        plan, time_window, access_window=protocol_manifest["access_window"]
    )
    bounds = tuple(
        (
            (pd.Timestamp(window.start).value, pd.Timestamp(window.end).value)
            for _fold, window in windows
        )
    )
    interval_rows, daily_rows, metrics, diagnostics, costs, coverage = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    covered_ns = episode_count = truncated_count = 0
    costs_total = {
        key: 0.0
        for key in ("fees", "funding", "turnover", "fill_events", "fill_quantity")
    }
    cursor = bounds[0][0]
    envelope_stop = bounds[-1][1] + 1
    candidate_key = {"candidate_id": candidate.candidate_id, "execution_mode": mode}

    def cover(start, stop, kind):
        coverage.append(
            {
                **candidate_key,
                "start_at": pd.Timestamp(start),
                "stop_at": pd.Timestamp(stop),
                "kind": kind,
                "reason": ""
                if kind == "covered"
                else "not_covered_by_qualified_episode",
            }
        )

    def metric(name, value, count, reason=""):
        defined = value is not None and np.isfinite(value)
        metrics.append(
            {
                **row_key,
                "metric_id": name,
                "value": float(value) if defined else np.nan,
                "observation_count": count,
                "status": "DEFINED" if defined else "UNDEFINED",
                "reason": "" if defined else reason,
            }
        )

    row_key = {}

    def consume_episode(episode, artifact, batches):
        nonlocal covered_ns, episode_count, truncated_count, cursor, row_key
        observed_start, observed_end = (
            (bounds[0][0], bounds[-1][1])
            if episode is None
            else (episode["start_ns"], episode["terminal_valuation_ns"])
        )
        episode_count += 1
        episode_id = "continuous" if episode is None else episode["episode_id"]
        row_key = {**candidate_key, "episode_id": episode_id}
        observed = (
            None
            if episode is None
            else (episode["start_ns"], episode["terminal_valuation_ns"])
        )
        observe, finish = _crypto_observations(
            artifact,
            windows,
            plan["exposure_limits"][candidate.asset_id]["max_position_quantity"],
            observation_window_ns=observed,
        )
        summary = summarize_ledger(
            artifact,
            batches,
            windows_ns=bounds if episode is None else None,
            consume_batch=observe,
        )
        economic = summary["economics"]
        if episode is None:
            intervals = summary["intervals"]
            start, stop = (bounds[0][0], envelope_stop)
        else:
            start, stop = (episode["start_ns"], episode["stop_ns"])
            observed = (start, episode["terminal_valuation_ns"])
            intervals = [
                {
                    "kind": "episode",
                    "start_ns": start,
                    "end_ns": observed[1],
                    "start_nav": economic["initial_capital"],
                    "end_nav": economic["nav"],
                    **{
                        name: economic[name]
                        for name in (
                            "gross_pnl",
                            "net_pnl",
                            "fees",
                            "funding",
                            "turnover",
                            "fill_events",
                        )
                    },
                }
            ]
            truncated_count += episode["termination"] == "TRUNCATED"
        if cursor < start:
            cover(cursor, start, "missing")
        cover(start, stop, "covered")
        cursor = stop
        covered_ns += stop - start
        for name in costs_total:
            costs_total[name] += economic[name]
        execution = summary["execution"]
        if episode is not None:
            execution = {
                **execution,
                "termination": episode["termination"],
                "stop_reasons": ",".join(episode["stop_reasons"]),
            }
        daily, inventory, drawdown = finish()
        for interval in intervals:
            interval_rows.append(
                {
                    **row_key,
                    "kind": interval["kind"],
                    "start_at": pd.Timestamp(interval["start_ns"]),
                    "end_at": pd.Timestamp(interval["end_ns"]),
                    **{
                        key: interval[key]
                        for key in (
                            "start_nav",
                            "end_nav",
                            "gross_pnl",
                            "net_pnl",
                            "fees",
                            "funding",
                            "turnover",
                            "fill_events",
                        )
                    },
                }
            )
        daily_rows.extend({**row_key, **row} for row in daily)
        for key in (
            "initial_capital",
            "gross_pnl",
            "net_pnl",
            "capital_return",
            "fees",
            "funding",
            "turnover",
            "fill_events",
            "fill_quantity",
        ):
            metric(key, economic[key], int(economic["fill_events"]))
        metric("terminal_nav", economic["nav"], len(intervals))
        metric(
            "continuous_max_drawdown" if episode is None else "episode_max_drawdown",
            drawdown,
            len(intervals),
        )
        for scope in ("oos", "gap"):
            if episode is None:
                metric(
                    f"{scope}_net_pnl",
                    sum(row["net_pnl"] for row in intervals if row["kind"] == scope),
                    sum(row["kind"] == scope for row in intervals),
                )
            state = inventory[scope]
            for name, value in (
                ("inventory_mean", state["mean"]),
                (
                    "inventory_variance",
                    state["m2"] / state["duration"] if state["duration"] else None,
                ),
                (
                    "inventory_time_at_limit_fraction",
                    state["at_limit"] / state["duration"]
                    if state["duration"]
                    else None,
                ),
            ):
                metric(
                    f"{scope}_{name}",
                    value if state["duration"] else None,
                    int(state["duration"] > 0),
                    f"no_{scope}_duration",
                )
            metric(
                f"{scope}_inventory_max_abs",
                state["max_abs"] if state["observations"] else None,
                state["observations"],
                f"no_{scope}_observations",
            )
        values = np.array(
            [row["net_return"] for row in daily if row["status"] == "DEFINED"],
            dtype=np.float64,
        )
        count = len(values)
        metric("planned_complete_utc_days", len(daily), len(daily))
        metric("defined_complete_utc_days", count, count)
        daily_missing = any(row["status"] != "DEFINED" for row in daily)
        day_reason = (
            "planned_utc_day_valuation_undefined"
            if daily_missing
            else "insufficient_complete_utc_days"
        )
        metric(
            "oos_mean_daily_return",
            float(values.mean()) if count and (not daily_missing) else None,
            count,
            day_reason,
        )
        if (
            not daily_missing
            and count >= statistics_policy["annualization_minimum_observations"]
        ):
            defined = [row for row in daily if row["status"] == "DEFINED"]
            deviation = float(
                return_standard_deviation(
                    values,
                    roundoff=_nav_return_roundoff(
                        [row["start_nav"] for row in defined],
                        [row["end_nav"] for row in defined],
                        values,
                    ),
                )
            )
            metric(
                "annualized_volatility",
                deviation * np.sqrt(statistics_policy["annualization_periods"]),
                count,
                "return_risk_unresolved",
            )
            metric(
                "sharpe",
                float(
                    _return_sharpes(
                        values,
                        roundoff=_nav_return_roundoff(
                            [row["start_nav"] for row in defined],
                            [row["end_nav"] for row in defined],
                            values,
                        ),
                    )
                    * np.sqrt(statistics_policy["annualization_periods"])
                ),
                count,
                "zero_daily_return_variance"
                if deviation == 0
                else "return_risk_unresolved",
            )
        else:
            metric("annualized_volatility", None, count, day_reason)
            metric("sharpe", None, count, day_reason)
        metric(
            "pnl_per_fill",
            economic["net_pnl"] / economic["fill_events"]
            if economic["fill_events"]
            else None,
            int(economic["fill_events"]),
            "no_fills",
        )
        metric(
            "pnl_per_notional",
            economic["net_pnl"] / economic["turnover"]
            if economic["turnover"]
            else None,
            int(economic["fill_events"]),
            "no_turnover",
        )
        for name, value in execution.items():
            diagnostics.append(
                {
                    **row_key,
                    "metric_id": name,
                    "value": float(value)
                    if type(value) in (float, int, bool)
                    else np.nan,
                    "text": value if type(value) is str else "",
                    "status": "DEFINED" if value is not None else "UNDEFINED",
                    "reason": "" if value is not None else f"{name}_undefined",
                }
            )
        base_fee = (
            plan["execution"]["maker"]["maker_fee_bps"]
            if isinstance(candidate, CryptoMakerCandidate)
            else plan["execution"]["taker_fee_bps"]
        )
        costs.extend(
            {**row_key, **row}
            for row in fixed_turnover_cost_sensitivity(
                turnover=economic["turnover"],
                net_pnl=economic["net_pnl"],
                base_fee_bps=base_fee,
                fee_scenarios_bps=plan["execution"]["fee_scenarios_bps"],
                additional_spread_scenarios_bps=plan["execution"][
                    "additional_spread_scenarios_bps"
                ],
            )
        )

    root = layout.target_root(target_id, None)
    identity = upstream_identities[target_id]
    if mode == "continuous":
        read_ledger(
            root,
            expected_identity=identity,
            consume_batches=lambda artifact, batches: consume_episode(
                None, artifact, batches
            ),
        )
    else:
        read_episodes(root, expected_identity=identity, consume_episode=consume_episode)
    if cursor < envelope_stop:
        cover(cursor, envelope_stop, "missing")
    row_key = {**candidate_key, "episode_id": "all"}
    for name, value in {
        "covered_seconds": covered_ns / 1000000000,
        "missing_seconds": (envelope_stop - bounds[0][0] - covered_ns) / 1000000000,
        "coverage_fraction": covered_ns / (envelope_stop - bounds[0][0]),
        "episode_count": episode_count,
        "truncated_episode_count": truncated_count,
        **costs_total,
    }.items():
        metric(name, value, episode_count)
    records = {
        "crypto_portfolio_intervals": interval_rows,
        "crypto_portfolio_daily": daily_rows,
        "crypto_portfolio_statistics": metrics,
        "crypto_execution_diagnostics": diagnostics,
        "crypto_cost_sensitivity": costs,
        "crypto_portfolio_coverage": coverage,
    }
    tables = {
        name: pd.DataFrame.from_records(rows, columns=table_columns[f"{name}.parquet"])
        for name, rows in records.items()
    }

    return tables


def iter_holding_factor_inputs(
    *,
    layout,
    calendar,
    time_window,
    unified_trading_sessions,
    style_columns,
    risk_context_target_ids,
    benchmark_required,
    risk_config: FactorRiskConfig | None = None,
):
    """Stream aligned inputs, retaining only decision/entry/end prices."""
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.production import (
        _ENHANCED_PANEL_COLUMNS,
        _enhanced_risk_cross,
        _iter_input_sessions,
        _risk_context_input_sessions,
        _validate_input_keys,
    )

    sessions = pd.DatetimeIndex(unified_trading_sessions(calendar))
    intervals = holding_intervals(sessions, time_window)
    if intervals.empty:
        return
    position = int(sessions.get_loc(intervals.decision_at.iloc[0]))
    history = TimeWindow(sessions[position], pd.Timestamp(time_window.end))
    risk = FactorRiskState(style_columns=style_columns, config=risk_config)
    panel_window = {}
    context_frames = {}

    def stamp(frame):
        return pd.Timestamp(frame.timestamp.iloc[0])

    with ExitStack() as channels:

        def stream(root, projection, window):
            return channels.enter_context(
                closing_iterator(_iter_input_sessions(root, projection, window))
            )

        panels = stream(
            layout.target_root("market_panel_1d", None),
            _ENHANCED_PANEL_COLUMNS,
            history,
        )
        risk_inputs = _risk_context_input_sessions(
            roots=tuple(
                layout.target_root(target, None) for target in risk_context_target_ids
            ),
            style_columns=style_columns,
            time_window=history,
            retained=channels,
        )
        contexts = {root: frames for root, (_columns, frames) in risk_inputs.items()}
        context_frames = {root: next(source, None) for root, source in contexts.items()}
        benchmarks = (
            stream(
                layout.target_root("benchmark_weights", None),
                ("benchmark_id", "asset_id", "timestamp", "weight"),
                time_window,
            )
            if benchmark_required
            else iter(())
        )
        panel, benchmark = (next(panels, None), next(benchmarks, None))

        def take_panels(through):
            nonlocal panel
            while panel is not None and stamp(panel) <= through:
                timestamp = stamp(panel)
                if panel.asset_id.duplicated().any() or timestamp not in sessions:
                    raise ValueError("holding_panel_keys_invalid")
                panel_window[timestamp] = panel.copy(deep=True)
                panel = next(panels, None)

        for interval in intervals.itertuples(index=False):
            while (
                position < len(sessions) and sessions[position] <= interval.decision_at
            ):
                timestamp = sessions[position]
                take_panels(timestamp)
                current = panel_window.get(timestamp)
                for context in context_frames.values():
                    if context is not None and stamp(context) < timestamp:
                        raise ValueError("holding_risk_context_calendar_misaligned")
                if current is None:
                    cross = None
                else:
                    risk_frames = []
                    for context in context_frames.values():
                        if context is None or stamp(context) != timestamp:
                            raise ValueError("holding_risk_context_session_missing")
                        if context.asset_id.duplicated().any():
                            raise ValueError("holding_risk_context_keys_invalid")
                        _validate_input_keys(current, context)
                        risk_frames.append(context)
                    cross = _enhanced_risk_cross(
                        panel=current,
                        risk_frames=risk_frames,
                        style_columns=style_columns,
                    ).copy(deep=True)

                for root, context in context_frames.items():
                    if context is not None and stamp(context) == timestamp:
                        context_frames[root] = next(contexts[root], None)

                if timestamp < interval.decision_at:
                    panel_window.pop(timestamp, None)
                position += 1
            take_panels(interval.end_at)
            if len(panel_window) > 3:
                raise ValueError("holding_price_window_invalid")
            if benchmark is not None and stamp(benchmark) < interval.decision_at:
                raise ValueError("holding_benchmark_calendar_misaligned")
            weights = (
                benchmark
                if benchmark is not None and stamp(benchmark) == interval.decision_at
                else pd.DataFrame(
                    columns=("benchmark_id", "asset_id", "timestamp", "weight")
                )
            )
            if weights.duplicated(["benchmark_id", "asset_id"]).any():
                raise ValueError("holding_benchmark_keys_invalid")
            prices = [
                panel_window.get(timestamp)
                for timestamp in (
                    interval.decision_at,
                    interval.entry_at,
                    interval.end_at,
                )
            ]
            fitted = None
            reason = "attribution_pit_history_missing" if cross is None else None
            close = entry = pd.Series(dtype="float64")
            holding_cross = None
            if any(frame is None for frame in prices):
                reason = "holding_prices_missing"
            else:
                close = prices[0].set_index("asset_id").adj_close
                entry = prices[1].set_index("asset_id").adj_open
                endpoint = prices[2].set_index("asset_id").adj_open
                if reason is None:
                    holding_cross = risk.prepare_exposures(cross)
                    holding_cross["asset_return"] = holding_cross.asset_id.map(
                        endpoint.div(entry) - 1.0
                    )
                    try:
                        fitted = risk.fit_return_session(
                            holding_cross, require_complete=True
                        )
                    except ValueError as error:
                        if str(error) != "holding_returns_incomplete":
                            raise
                        reason = str(error)
                    if fitted is None and reason is None:
                        reason = "holding_factor_fit_unidentified"
            result = HoldingFactorInputs(
                interval.decision_at,
                interval.entry_at,
                interval.end_at,
                fitted,
                close,
                entry,
                weights,
                reason,
            )
            try:
                yield result
            except GeneratorExit:
                return
            panel_window.pop(interval.decision_at, None)
            if benchmark is not None and stamp(benchmark) == interval.decision_at:
                benchmark = next(benchmarks, None)


def build_portfolio_statistics_tables(
    *,
    ledgers,
    candidates,
    protocol_manifest,
    time_window,
    unified_trading_sessions,
    layout,
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.types import DatasetReadMode

    history = TimeWindow(
        pd.Timestamp(protocol_manifest["formal_dataset_window"]["start"]),
        pd.Timestamp(time_window.end),
    )
    pieces = []

    calendar = read_dataset(
        layout.target_root("trading_calendar", None),
        columns=("calendar_date", "source_market_type", "is_trading_day"),
        time_window=history,
        mode=DatasetReadMode.MATERIALIZED,
    )
    codes = protocol_manifest["benchmark_total_return_indices"]
    price_codes = {
        name: code
        for code, (name, _label) in a_share_ids.FORMAL_BENCHMARKS.items()
        if name in codes
    }
    if codes:
        endpoint_dates = (
            holding_intervals(
                pd.DatetimeIndex(unified_trading_sessions(calendar)), time_window
            )
            .loc[:, ["entry_at", "end_at"]]
            .to_numpy()
            .ravel()
        )
        with closing_iterator(
            read_dataset(
                layout.target_root("index_market", None),
                columns=("source_asset_id", "timestamp", "open", "close"),
                time_window=time_window,
                mode=DatasetReadMode.CHUNKED,
            )
        ) as chunks:
            for frame in chunks:
                piece = frame.loc[
                    frame.source_asset_id.isin((*codes.values(), *price_codes.values()))
                ].copy()
                if not piece.empty:
                    pieces.append(piece)

    benchmark = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=("source_asset_id", "timestamp", "open", "close"))
    )

    for name, price_code in price_codes.items():
        missing = (
            benchmark.source_asset_id.eq(codes[name])
            & benchmark.open.isna()
            & benchmark.timestamp.isin(endpoint_dates)
        )
        if not missing.any():
            continue
        prices = benchmark.loc[
            benchmark.source_asset_id.eq(price_code), ["timestamp", "open", "close"]
        ].set_index("timestamp")
        required = (
            benchmark.loc[missing, ["timestamp", "close"]]
            .set_index("timestamp")
            .rename(columns={"close": "total_close"})
            .join(prices)
        )
        if ((~np.isfinite(required) | required.le(0)) & required.notna()).any(
            axis=None
        ):
            raise ValueError("benchmark_return_open_invalid")
        benchmark.loc[missing, "open"] = (
            required.open * required.total_close / required.close
        ).to_numpy()
    benchmark = benchmark.loc[
        benchmark.source_asset_id.isin(codes.values()),
        ["source_asset_id", "timestamp", "open"],
    ].copy()
    benchmark["benchmark_id"] = benchmark.source_asset_id.map(
        {code: name for name, code in codes.items()}
    )
    inputs = iter_holding_factor_inputs(
        layout=layout,
        calendar=calendar,
        time_window=time_window,
        unified_trading_sessions=unified_trading_sessions,
        style_columns=tuple(
            dict.fromkeys(
                column
                for candidate in candidates
                for column in candidate.risk_style_columns
            )
        ),
        risk_config=FactorRiskConfig(**protocol_manifest["risk"]),
        benchmark_required=bool(codes),
        risk_context_target_ids=("price_volume_facts",)
        if not any(candidate.risk_context_target_ids for candidate in candidates)
        else tuple(
            dict.fromkeys(
                target_id
                for candidate in candidates
                for target_id in candidate.risk_context_target_ids
            )
        ),
    )
    return _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        protocol_manifest=protocol_manifest,
        time_window=time_window,
        benchmark_opens=benchmark,
        calendar=calendar,
        unified_trading_sessions=unified_trading_sessions,
        holding_inputs=inputs,
    )


def _portfolio_statistics_tables(
    *,
    ledgers,
    candidates,
    protocol_manifest,
    time_window,
    benchmark_opens,
    calendar,
    unified_trading_sessions,
    holding_inputs,
):
    """Project the protocol-selected ledgers into interval-aligned diagnostics."""
    ids = tuple(candidate.candidate_id for candidate in candidates)
    if tuple(protocol_manifest["portfolio_candidates"]) != ids or set(ledgers) != set(
        ids
    ):
        raise ValueError("portfolio_statistics_candidate_set_invalid")
    expected_intervals = holding_intervals(
        pd.DatetimeIndex(unified_trading_sessions(calendar)), time_window
    )
    access = protocol_manifest["access_window"]
    (
        daily_rows,
        capacity_rows,
        summary_rows,
        capacity_summary,
        attribution_rows,
        robustness_rows,
    ) = ([], [], [], [], [], [])
    exposure_rows, factor_rows = ([], [])
    benchmarks = {}
    for benchmark_id in dict.fromkeys(
        candidate.benchmark_id
        for candidate in candidates
        if candidate.benchmark_id is not None
    ):
        benchmark = benchmark_opens.loc[
            benchmark_opens.benchmark_id.eq(benchmark_id), ["timestamp", "open"]
        ].set_index("timestamp")["open"]
        if benchmark.index.has_duplicates:
            raise ValueError("benchmark_return_duplicate_keys")
        benchmarks[benchmark_id] = benchmark
    previous = {
        candidate_id: (None, ledgers[candidate_id].capital.initial_capital)
        for candidate_id in ids
    }
    with ExitStack() as sources:
        streams = [
            sources.enter_context(
                closing_iterator(iter(ledgers[candidate_id].iter_chunks()))
            )
            for candidate_id in ids
        ]
        fit_stream = sources.enter_context(closing_iterator(iter(holding_inputs)))
        sentinel = object()

        for interval_index in range(len(expected_intervals) + 1):
            group = tuple(next(stream, sentinel) for stream in (*streams, fit_stream))
            if all(value is sentinel for value in group):
                break
            if any(value is sentinel for value in group):
                raise ValueError("portfolio_statistics_candidate_dates_misaligned")
            if interval_index >= len(expected_intervals):
                raise ValueError("portfolio_statistics_window_incomplete")
            batch_group, holding_input = (group[:-1], group[-1])
            expected = expected_intervals.iloc[interval_index]
            if (
                holding_input.decision_at,
                holding_input.entry_at,
                holding_input.end_at,
            ) != tuple(expected):
                raise ValueError("holding_factor_interval_misaligned")
            if holding_input.fitted is not None:
                factor_rows.extend(
                    (
                        dict(
                            decision_at=holding_input.decision_at,
                            entry_at=holding_input.entry_at,
                            end_at=holding_input.end_at,
                            factor_id=name,
                            value=float(value),
                            observation_count=len(holding_input.fitted.asset_returns),
                        )
                        for name, value in (holding_input.fitted.factor_returns.items())
                    )
                )
            for candidate, batch in zip(candidates, batch_group, strict=True):
                candidate_id = candidate.candidate_id
                previous_end, previous_nav = previous[candidate_id]
                row = portfolio_interval_economics(batch)
                expected = expected_intervals.iloc[interval_index]
                if (row["entry_at"], row["end_at"]) != (
                    expected.entry_at,
                    expected.end_at,
                ):
                    raise ValueError("portfolio_statistics_window_incomplete")
                if previous_end is not None and row["entry_at"] != previous_end:
                    raise ValueError("portfolio_statistics_interval_gap")
                if not np.isclose(
                    previous_nav,
                    row["nav_before"],
                    rtol=0,
                    atol=max(1e-10, abs(previous_nav) * 1e-12),
                ):
                    raise ValueError("portfolio_statistics_nav_discontinuity")
                previous_end, previous_nav = (row["end_at"], row["nav_end"])
                previous[candidate_id] = (previous_end, previous_nav)
                row = {"candidate_id": candidate_id, **row}
                row.update(
                    benchmark_return=np.nan,
                    active_return=np.nan,
                    relative_wealth=np.nan,
                    benchmark_status="NOT_APPLICABLE",
                    benchmark_reason="",
                    access_window=access,
                )
                if candidate.benchmark_id is not None:
                    benchmark = benchmarks[candidate.benchmark_id]
                    values = benchmark.reindex([row["entry_at"], row["end_at"]])
                    if values.isna().any():
                        row.update(
                            benchmark_status="BLOCKED",
                            benchmark_reason="benchmark_total_return_open_missing",
                        )
                    elif not np.isfinite(values).all() or values.le(0).any():
                        raise ValueError("benchmark_return_open_invalid")
                    else:
                        value = float(values.iloc[1] / values.iloc[0] - 1)
                        row.update(
                            benchmark_return=value,
                            active_return=row["net_return"] - value,
                            benchmark_status="DEFINED",
                        )
                daily_rows.append(row)
                after = pd.Series(
                    {
                        value.asset_id: value.value
                        for value in batch.valuations
                        if value.phase == "after"
                        and value.asset_id != PORTFOLIO_CASH_ASSET_ID
                    }
                )
                endpoint = pd.Series(
                    {
                        value.asset_id: value.value
                        for value in batch.valuations
                        if value.phase == "end"
                        and value.asset_id != PORTFOLIO_CASH_ASSET_ID
                    }
                )
                held_returns = endpoint.div(after.where(after.ne(0))) - 1.0
                components, reasons, exposures = holding_attribution(
                    holding_input,
                    after_notionals=after,
                    nav_before=row["nav_before"],
                    nav_post=row["nav_post"],
                    benchmark_id=candidate.benchmark_id,
                    benchmark_return=None
                    if pd.isna(row["benchmark_return"])
                    else row["benchmark_return"],
                    holding_returns=held_returns,
                )
                components["cost"] = -row["cost_drag"]
                for component, value in components.items():
                    reason = reasons.get(component, "")
                    if value is None and component == "benchmark":
                        reason = "benchmark_total_return_open_missing"
                    attribution_rows.append(
                        dict(
                            candidate_id=candidate_id,
                            decision_at=holding_input.decision_at,
                            entry_at=row["entry_at"],
                            end_at=row["end_at"],
                            component_id=component,
                            component_type="detail"
                            if component
                            in {
                                "stock_residual_difference",
                                "benchmark_replication_gap",
                            }
                            else "additive",
                            value=np.nan if value is None else value,
                            status="DEFINED"
                            if value is not None
                            else "BLOCKED"
                            if "missing" in reason or "incomplete" in reason
                            else "UNDEFINED",
                            reason=reason,
                        )
                    )
                exposure_rows.extend(
                    (
                        dict(
                            candidate_id=candidate_id,
                            decision_at=holding_input.decision_at,
                            entry_at=row["entry_at"],
                            factor_id=name,
                            value=float(value),
                        )
                        for name, value in exposures.items()
                    )
                )

                facts = {(item.asset_id, item.name): item for item in batch.diagnostics}
                deltas = {
                    item.asset_id: item.number
                    for item in batch.diagnostics
                    if item.name == "decision_delta_weight"
                    and item.asset_id != PORTFOLIO_CASH_ASSET_ID
                    and (item.number != 0.0)
                }
                if deltas:
                    decision_date = pd.Timestamp(
                        facts[next(iter(deltas)), "decision_timestamp"].text
                    )
                    for participation in protocol_manifest["execution_assumptions"][
                        "adv_sensitivities"
                    ]:
                        available = {
                            asset: facts[asset, "adv20"].number
                            if (asset, "adv20") in facts
                            else None
                            for asset in deltas
                        }
                        adv_missing = any(value is None for value in available.values())
                        capacities = (
                            {
                                asset: available[asset] * participation / abs(delta)
                                for asset, delta in deltas.items()
                            }
                            if not adv_missing
                            else {}
                        )
                        bottleneck = (
                            min(
                                capacities,
                                key=lambda asset: (capacities[asset], asset),
                            )
                            if capacities
                            else ""
                        )
                        capacity_rows.append(
                            dict(
                                candidate_id=candidate_id,
                                decision_timestamp=decision_date,
                                entry_at=row["entry_at"],
                                participation=participation,
                                capacity=capacities[bottleneck]
                                if capacities
                                else np.nan,
                                bottleneck_asset_id=bottleneck,
                                status="BLOCKED" if adv_missing else "DEFINED",
                                reason="capacity_required_adv_missing"
                                if adv_missing
                                else "",
                            )
                        )

    for candidate in candidates:
        candidate_id = candidate.candidate_id
        artifact = ledgers[candidate_id]
        previous_end, previous_nav = previous[candidate_id]
        selected = pd.DataFrame(
            [row for row in daily_rows if row["candidate_id"] == candidate_id]
        )
        if selected.empty:
            raise ValueError("portfolio_statistics_empty_candidate")
        wealth = (1 + selected.net_return).cumprod()
        if not np.isclose(
            float(wealth.iloc[-1]),
            previous_nav / artifact.capital.initial_capital,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("portfolio_statistics_compounding_invalid")
        count = len(selected)
        deviation = float(
            return_standard_deviation(
                selected.net_return,
                roundoff=_nav_return_roundoff(
                    selected.nav_before, selected.nav_end, selected.net_return
                ),
            )
        )
        values = {
            "terminal_nav": previous_nav,
            "capital_return": float(wealth.iloc[-1] - 1),
            "cagr": float(wealth.iloc[-1] ** (252 / count) - 1),
            "annualized_volatility": deviation * np.sqrt(252),
            "sharpe": float(
                _return_sharpes(
                    selected.net_return,
                    roundoff=_nav_return_roundoff(
                        selected.nav_before, selected.nav_end, selected.net_return
                    ),
                )
                * np.sqrt(252)
            ),
            "max_drawdown": float((wealth / wealth.cummax().clip(lower=1) - 1).min()),
            "target_turnover": float(selected.target_turnover.mean()),
            "requested_turnover": float(selected.requested_turnover.mean()),
            "realized_turnover": float(selected.realized_turnover.mean()),
            "gross_pnl": float((selected.nav_end - selected.nav_post).sum()),
            "net_pnl": previous_nav - artifact.capital.initial_capital,
            "cost": float((selected.cost_drag * selected.nav_before).sum()),
        }
        if candidate.benchmark_id is not None:
            if selected.benchmark_status.eq("DEFINED").all():
                relative = wealth / (1 + selected.benchmark_return).cumprod()
                for row, value in zip(
                    (r for r in daily_rows if r["candidate_id"] == candidate_id),
                    relative,
                    strict=True,
                ):
                    row["relative_wealth"] = float(value)
                benchmark = benchmarks[candidate.benchmark_id]
                active_roundoff = (
                    _nav_return_roundoff(
                        selected.nav_before, selected.nav_end, selected.net_return
                    )
                    + _nav_return_roundoff(
                        benchmark.reindex(selected.entry_at).to_numpy(),
                        benchmark.reindex(selected.end_at).to_numpy(),
                        selected.benchmark_return,
                    )
                    + np.spacing(np.abs(selected.active_return.to_numpy()))
                )
                active_sd = float(
                    return_standard_deviation(
                        selected.active_return, roundoff=active_roundoff
                    )
                )
                values.update(
                    annualized_active_return=float(selected.active_return.mean() * 252),
                    tracking_error=active_sd * np.sqrt(252),
                    net_information_ratio=float(
                        _return_sharpes(
                            selected.active_return, roundoff=active_roundoff
                        )
                        * np.sqrt(252)
                    ),
                    relative_wealth=float(relative.iloc[-1]),
                    max_active_drawdown=float(
                        (relative / relative.cummax().clip(lower=1) - 1).min()
                    ),
                )
            else:
                values.update(
                    {
                        name: np.nan
                        for name in (
                            "annualized_active_return",
                            "tracking_error",
                            "net_information_ratio",
                            "relative_wealth",
                            "max_active_drawdown",
                        )
                    }
                )
        for name, value in values.items():
            blocked = (
                name
                in {
                    "annualized_active_return",
                    "tracking_error",
                    "net_information_ratio",
                    "relative_wealth",
                    "max_active_drawdown",
                }
                and selected.benchmark_status.ne("DEFINED").any()
            )
            summary_rows.append(
                dict(
                    candidate_id=candidate_id,
                    metric_id=name,
                    value=value,
                    status="BLOCKED"
                    if blocked
                    else "DEFINED"
                    if np.isfinite(value)
                    else "UNDEFINED",
                    reason="benchmark_total_return_open_missing"
                    if blocked
                    else ""
                    if np.isfinite(value)
                    else "statistic_not_identified",
                    observation_count=count,
                    access_window=access,
                )
            )
    daily = pd.DataFrame(daily_rows)
    if daily.duplicated(["candidate_id", "entry_at", "end_at"]).any():
        raise ValueError("portfolio_statistics_duplicate_intervals")
    matrix = daily.pivot(
        index=["entry_at", "end_at"], columns="candidate_id", values="net_return"
    ).reindex(columns=ids)
    if tuple(matrix.index) != tuple(
        expected_intervals.loc[:, ["entry_at", "end_at"]].itertuples(
            index=False, name=None
        )
    ):
        raise ValueError("portfolio_statistics_window_incomplete")
    if matrix.isna().any().any():
        raise ValueError("portfolio_statistics_candidate_dates_misaligned")
    paired_comparison = _group_portfolio_comparison(matrix, protocol_manifest)
    if access == "development":
        blockers = {
            candidate.candidate_id: _model_inference_blocker(
                access_window=access, selection_groups=(candidate.selection_group,)
            )
            for candidate in candidates
        }
        inference_ids = tuple(
            candidate_id for candidate_id in ids if blockers[candidate_id] is None
        )
        inference_matrix = matrix.loc[:, list(inference_ids)]
        before = daily.pivot(
            index=["entry_at", "end_at"],
            columns="candidate_id",
            values="nav_before",
        ).reindex(columns=inference_ids)
        after = daily.pivot(
            index=["entry_at", "end_at"], columns="candidate_id", values="nav_end"
        ).reindex(columns=inference_ids)
        roundoff = _nav_return_roundoff(before, after, inference_matrix)
        trial_sharpes = _return_sharpes(inference_matrix, roundoff=roundoff)
        for candidate_id in ids:
            if blockers[candidate_id] is not None:
                outputs = (
                    ("psr", None, blockers[candidate_id]),
                    ("dsr", None, blockers[candidate_id]),
                )
            else:
                try:
                    result = probabilistic_and_deflated_sharpe(
                        inference_matrix[candidate_id],
                        benchmark_sharpe=0.0,
                        trial_sharpes=trial_sharpes,
                        return_roundoff=roundoff[:, inference_ids.index(candidate_id)],
                    )
                    outputs = (
                        ("psr", result.psr, ""),
                        ("dsr", result.dsr, result.dsr_reason or ""),
                    )
                except ValueError as error:
                    if str(error) not in {
                        "finite_vector_required",
                        "sharpe_zero_variance",
                        "sharpe_variance_invalid",
                    }:
                        raise
                    outputs = (("psr", None, str(error)), ("dsr", None, str(error)))
            for test, value, reason in outputs:
                robustness_rows.append(
                    dict(
                        candidate_id=candidate_id,
                        test_id=test,
                        value=np.nan if value is None else value,
                        status="UNDEFINED" if value is None else "DEFINED",
                        reason=reason,
                        observation_count=len(matrix),
                        candidate_count=len(inference_ids),
                        access_window=access,
                    )
                )
        blocks = protocol_manifest["statistics"]["cscv_blocks"]
        if not inference_ids:
            value, status, reason = (
                np.nan,
                "UNDEFINED",
                next(iter(blockers.values())),
            )
        elif len(matrix) >= blocks:
            result = cscv_pbo(
                inference_matrix,
                slice_count=blocks,
                max_combinations=comb(blocks, blocks // 2),
                return_roundoff=roundoff,
            )
            value, status, reason = (result.pbo, result.status, result.reason or "")
        else:
            value, status, reason = (
                np.nan,
                "UNDEFINED",
                "cscv_slice_count_invalid",
            )
        robustness_rows.append(
            dict(
                candidate_id="joint",
                test_id="cscv_pbo",
                value=value,
                status=status,
                reason=reason,
                observation_count=len(matrix),
                candidate_count=len(inference_ids),
                access_window=access,
            )
        )
    else:
        robustness_rows.append(
            dict(
                candidate_id="joint",
                test_id="development_only",
                value=np.nan,
                status="UNDEFINED",
                reason="holdout_excluded",
                observation_count=0,
                candidate_count=len(ids),
                access_window=access,
            )
        )
    capacity = pd.DataFrame(
        capacity_rows,
        columns=(
            "candidate_id",
            "decision_timestamp",
            "entry_at",
            "participation",
            "capacity",
            "bottleneck_asset_id",
            "status",
            "reason",
        ),
    )
    for candidate_id in ids:
        for participation in protocol_manifest["execution_assumptions"][
            "adv_sensitivities"
        ]:
            sample = capacity.loc[
                capacity.candidate_id.eq(candidate_id)
                & capacity.participation.eq(participation)
            ]
            periods = {"all": sample}
            if not sample.empty:
                periods.update(
                    {
                        str(year): part
                        for year, part in sample.groupby(
                            sample.decision_timestamp.dt.year
                        )
                    }
                )
            for period, part in periods.items():
                defined = not part.empty and part.status.eq("DEFINED").all()
                quantiles = (
                    capacity_quantiles(
                        pd.Series(
                            part.capacity.to_numpy(),
                            index=pd.DatetimeIndex(part.decision_timestamp),
                        )
                    )
                    if defined
                    else {}
                )
                for metric, key in (
                    ("p05", "formal_p05"),
                    ("median", "median"),
                    ("minimum", "minimum"),
                ):
                    capacity_summary.append(
                        dict(
                            candidate_id=candidate_id,
                            participation=participation,
                            period=period,
                            metric_id=metric,
                            value=quantiles[key] if defined else np.nan,
                            status="DEFINED"
                            if defined
                            else "UNDEFINED"
                            if part.empty
                            else "BLOCKED",
                            reason=""
                            if defined
                            else "no_rebalance_requests"
                            if part.empty
                            else "capacity_required_adv_missing",
                            observation_count=len(part),
                        )
                    )
    tables = {
        "portfolio_daily": daily,
        "portfolio_group_comparison": paired_comparison,
        "portfolio_performance": pd.DataFrame(summary_rows),
        "portfolio_capacity": capacity,
        "portfolio_capacity_summary": pd.DataFrame(capacity_summary),
        "portfolio_attribution": pd.DataFrame(attribution_rows),
        "portfolio_robustness": pd.DataFrame(robustness_rows),
        "portfolio_exposures": pd.DataFrame(
            exposure_rows,
            columns=(
                "candidate_id",
                "decision_at",
                "entry_at",
                "factor_id",
                "value",
            ),
        ),
        "holding_factor_returns": pd.DataFrame(
            factor_rows,
            columns=(
                "decision_at",
                "entry_at",
                "end_at",
                "factor_id",
                "value",
                "observation_count",
            ),
        ),
    }

    return tables


def _group_portfolio_comparison(matrix, protocol_manifest):
    """Compare both frozen groups on the complete, already validated daily grid."""
    statistics = protocol_manifest["statistics"]
    declaration = statistics["group_comparison"]
    candidate_ids = declaration["portfolio_ids"]
    columns = [
        f"{group}_{candidate}"
        for candidate in candidate_ids
        for group in ("global", "quarterly")
    ]
    if (
        set(columns) != set(matrix.columns)
        or not np.isfinite(matrix.loc[:, columns].to_numpy(dtype="float64")).all()
    ):
        raise ValueError("portfolio_group_comparison_coverage_invalid")
    differences = pd.DataFrame(
        {
            "timestamp": matrix.index.get_level_values("end_at"),
            **{
                candidate: (
                    matrix[f"quarterly_{candidate}"] - matrix[f"global_{candidate}"]
                ).to_numpy()
                for candidate in candidate_ids
            },
        }
    )
    access = protocol_manifest["access_window"]
    inference_blocker = _model_inference_blocker(
        access_window=access, selection_groups=("global", "quarterly")
    )
    inference = (
        time_series_inference_summary(
            differences,
            identity_column="timestamp",
            max_lags=statistics["hac_lag"],
            fdr_method=declaration["fdr_method"],
            fdr_alpha=declaration["alpha"],
            fdr_family_size=len(candidate_ids),
        ).set_index("timestamp")
        if inference_blocker is None
        else None
    )
    rows = []
    for candidate in candidate_ids:
        values = differences[candidate].to_numpy(dtype="float64")
        row = {
            "portfolio_id": candidate,
            "comparison_role": "primary"
            if candidate == declaration["primary_portfolio_id"]
            else "secondary",
            "access_window": access,
            "observation_count": len(values),
            "global_mean": float(matrix[f"global_{candidate}"].mean()),
            "quarterly_mean": float(matrix[f"quarterly_{candidate}"].mean()),
            "mean_difference": float(values.mean()),
            "global_terminal_wealth": float((1 + matrix[f"global_{candidate}"]).prod()),
            "quarterly_terminal_wealth": float(
                (1 + matrix[f"quarterly_{candidate}"]).prod()
            ),
            "hac_lag": statistics["hac_lag"],
            "bootstrap_block_size": statistics["bootstrap_block"],
            "bootstrap_repetitions": statistics["bootstrap_repetitions"],
            "bootstrap_seed": declaration["seed"],
            "confidence": declaration["confidence"],
            "fdr_family_size": len(candidate_ids),
            "fdr_method": declaration["fdr_method"],
            "fdr_alpha": declaration["alpha"],
            **dict.fromkeys(
                (
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                    "fdr_adjusted_p_value",
                    "bootstrap_lower",
                    "bootstrap_upper",
                ),
                np.nan,
            ),
            "status": "UNDEFINED",
            "reason": inference_blocker,
            "bootstrap_status": "UNDEFINED",
            "bootstrap_reason": inference_blocker,
        }
        if inference is not None:
            estimate = inference.loc[candidate]
            for key in (
                "status",
                "reason",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
                "fdr_adjusted_p_value",
            ):
                row[key] = estimate[key]
            if (
                estimate["status"] == "DEFINED"
                and len(values) > statistics["bootstrap_block"]
            ):
                interval = stationary_bootstrap_mean_ci(
                    values,
                    block_size=statistics["bootstrap_block"],
                    repetitions=statistics["bootstrap_repetitions"],
                    seed=declaration["seed"],
                    confidence=declaration["confidence"],
                )
                row.update(
                    bootstrap_lower=interval.lower,
                    bootstrap_upper=interval.upper,
                    bootstrap_status="DEFINED",
                    bootstrap_reason=None,
                )
            else:
                row["bootstrap_reason"] = (
                    estimate["reason"]
                    if estimate["status"] != "DEFINED"
                    else "bootstrap_block_exceeds_observations"
                )
        rows.append(row)
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "portfolio_id",
            "comparison_role",
            "access_window",
            "status",
            "reason",
            "bootstrap_status",
            "bootstrap_reason",
            "fdr_method",
            "observation_count",
            "hac_lag",
            "bootstrap_block_size",
            "bootstrap_repetitions",
            "bootstrap_seed",
            "fdr_family_size",
            "global_mean",
            "quarterly_mean",
            "mean_difference",
            "global_terminal_wealth",
            "quarterly_terminal_wealth",
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
            "bootstrap_lower",
            "bootstrap_upper",
            "confidence",
            "fdr_alpha",
            "fdr_adjusted_p_value",
        ),
    )
