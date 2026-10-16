"""Production portfolio diagnostics builders."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
from fractions import Fraction
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
from quant_research_platform.contracts.validation import (
    finite_exact_float,
    strict_finite_float,
)
from quant_research_platform.markets.implementations.a_share import ids as a_share_ids
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.accounting import (
    capacity_quantiles,
    portfolio_interval_economics,
)
from quant_research_platform.research.portfolios.comparison import (
    COMPARISON_ANALYSES,
    _nav_return_roundoff,
    comparison_tables,
    selected_portfolio_analyses,
    validate_comparison_config,
)
from quant_research_platform.research.portfolios.comparison import (
    endpoint_return as comparison_endpoint_return,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
    HoldingFactorInputs,
    holding_attribution,
)
from quant_research_platform.research.portfolios.schedules import holding_intervals
from quant_research_platform.research.protocols.a_share import _inference_blocker
from quant_research_platform.research.statistics import (
    _return_sharpes,
    cscv_pbo,
    probabilistic_and_deflated_sharpe,
    return_standard_deviation,
    stationary_bootstrap_mean_ci,
    time_series_inference_summary,
)

_PORTFOLIO_ANALYSES = (
    "performance",
    "attribution",
    "capacity",
    "sharpe_inference",
    "cscv",
    "group_comparison",
    *COMPARISON_ANALYSES,
)
_CRYPTO_ANALYSES = (
    "performance",
    "annualized",
    "execution",
    "cost_sensitivity",
    *COMPARISON_ANALYSES,
)


def _research_valuation_observation(batch, economics):
    """Summarize explicit research marks without changing the execution path."""
    from quant_research_platform.contracts.portfolios import (
        RESEARCH_VALUATION_POLICY,
    )

    facts = {(item.asset_id, item.name): item for item in batch.diagnostics}
    policy = facts.get((PORTFOLIO_CASH_ASSET_ID, "research_valuation_policy"))
    if policy is None:
        return None
    if policy.text != RESEARCH_VALUATION_POLICY:
        raise ValueError("portfolio_research_valuation_policy_invalid")
    values = {
        phase: {
            row.asset_id: row.value
            for row in batch.valuations
            if row.phase == phase and row.asset_id != PORTFOLIO_CASH_ASSET_ID
        }
        for phase in ("after", "end")
    }
    held = {(row.phase, row.asset_id) for row in batch.valuations if row.quantity > 0}
    marked = {
        phase: {
            asset
            for asset, value in values[valuation_phase].items()
            if (asset, f"research_{phase}_mark_at") in facts
            and (valuation_phase, asset) in held
        }
        for phase, valuation_phase in (("entry", "after"), ("end", "end"))
    }
    amount = sum(values["end"][asset] for asset in marked["end"])
    return {
        "entry_at": economics["entry_at"],
        "end_at": economics["end_at"],
        "estimated_held_assets": len(marked["entry"] | marked["end"]),
        "estimated_value_end": amount,
        "estimated_weight_end": amount / economics["nav_end"]
        if economics["nav_end"] > 0
        else np.nan,
        "maximum_mark_age_days": max(
            (
                facts[asset, f"research_{phase}_mark_age_days"].number
                for phase, assets in marked.items()
                for asset in assets
            ),
            default=0.0,
        ),
        **{
            reason: sum(
                item.name == "block_reason" and item.text == reason
                for item in batch.diagnostics
            )
            for reason in ("execution_fact_unknown", "research_unquoted_valuation")
        },
        **{
            f"nav_{percent}pct": facts[
                PORTFOLIO_CASH_ASSET_ID, f"research_valuation_nav_{percent}pct"
            ].number
            for percent in (0, 50, 100)
        },
    }


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
    if net_pnl is not None:
        net_pnl = strict_finite_float(
            net_pnl, error="hft_cost_sensitivity_result_invalid"
        )
    base_fee_bps = strict_finite_float(
        base_fee_bps, error="hft_cost_sensitivity_result_invalid"
    )
    result = []
    for fee in fee_scenarios_bps:
        fee = strict_finite_float(fee, error="hft_cost_sensitivity_invalid")
        for spread in additional_spread_scenarios_bps:
            spread = strict_finite_float(spread, error="hft_cost_sensitivity_invalid")
            if spread < 0:
                raise ValueError("hft_cost_sensitivity_invalid")
            exact_change = (
                Fraction(turnover)
                * (Fraction(fee) - Fraction(base_fee_bps) + Fraction(spread))
                / 10000
            )
            try:
                change = float(exact_change)
            except OverflowError:
                change = -np.inf if exact_change < 0 else np.inf
            result.append(
                {
                    "fee_bps": fee,
                    "additional_spread_bps": spread,
                    "cost_delta": change,
                    "net_pnl": None if net_pnl is None else net_pnl - change,
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
            "scale": 0.0,
            "mean": 0.0,
            "variance": 0.0,
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
            scale = max(state["scale"], abs(previous_position))
            if scale:
                # Merge normalized population moments, without ns-sized products.
                rescale = state["scale"] / scale
                old_mean = state["mean"] * rescale
                value = previous_position / scale
                old_weight = state["duration"] / new_duration
                new_weight = weight / new_duration
                state["mean"] = old_weight * old_mean + new_weight * value
                state["variance"] = (
                    old_weight * state["variance"] * rescale * rescale
                    + old_weight * new_weight * (value - old_mean) ** 2
                )
                state["scale"] = scale
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

    def finish(*, observation_end_ns=None):
        daily = []
        for index, (fold, start, end) in enumerate(bounds):
            observed_start, observed_end = (
                (start, end) if observation_window_ns is None else observation_window_ns
            )
            if observation_end_ns is not None:
                observed_end = observation_end_ns
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
                net_return = None
                if not reason:
                    net_return = finite_exact_float(
                        Fraction(end_nav) / Fraction(start_nav) - 1
                    )
                    if net_return is None:
                        reason = "nonfinite_result"
                daily.append(
                    {
                        "fold_id": fold,
                        "start_at": pd.Timestamp(point),
                        "end_at": pd.Timestamp(point + day_ns),
                        "start_nav": np.nan if start_nav is None else start_nav,
                        "end_nav": np.nan if end_nav is None else end_nav,
                        "net_return": np.nan if net_return is None else net_return,
                        "status": "UNDEFINED" if reason else "DEFINED",
                        "reason": reason,
                    }
                )
                point += day_ns
        return (daily, inventory, drawdown)

    return (observe, finish)


def _crypto_comparison_axis(
    *, windows, episode, summary, daily, capital, row_key, sample_role
):
    """Partition OOS/episode windows using summary endpoints without overlap."""
    execution = summary["execution"]
    partial_reason = (
        execution.get("account_stop_reason")
        if execution.get("account_path_status") == "PARTIAL"
        else ""
    )
    prefix_end = (
        execution.get("known_prefix_valuation_timestamp_ns") if partial_reason else None
    )
    marks = {}
    for interval in summary["intervals"]:
        for side in ("start", "end"):
            marks[interval[f"{side}_ns"]] = interval[f"{side}_nav"]
    for row in daily:
        marks[row["start_at"].value] = row["start_nav"]
        marks[row["end_at"].value] = row["end_nav"]
    if episode is None:
        marks[pd.Timestamp(windows[0][1].start).value] = capital.initial_capital
    else:
        marks[episode["start_ns"]] = capital.initial_capital
        if not partial_reason:
            marks[episode["terminal_valuation_ns"]] = summary["economics"]["nav"]
    rows = []
    for fold, window in windows:
        start, end = pd.Timestamp(window.start).value, pd.Timestamp(window.end).value
        if episode is not None:
            start, end = (
                max(start, episode["start_ns"]),
                min(end, episode["stop_ns"] - 1),
            )
        if start >= end:
            continue
        points = sorted(
            {
                start,
                end,
                *(
                    row[column].value
                    for row in daily
                    if row["fold_id"] == fold
                    for column in ("start_at", "end_at")
                    if start < row[column].value < end
                ),
            }
        )
        for begin, stop in zip(points, points[1:], strict=False):
            before, after = marks.get(begin), marks.get(stop)
            reason = (
                partial_reason
                if partial_reason and (prefix_end is None or stop > prefix_end)
                else "portfolio_endpoint_missing"
                if before is None or after is None or pd.isna(before) or pd.isna(after)
                else "nonpositive_start_nav"
                if before <= 0
                else ""
            )
            value = (
                None
                if reason
                else finite_exact_float(
                    Fraction(float(after)) / Fraction(float(before)) - 1
                )
            )
            if value is None and not reason:
                reason = "nonfinite_result"
            rows.append(
                {
                    **row_key,
                    "fold_id": fold,
                    "sample_role": sample_role,
                    "start_at": pd.Timestamp(begin),
                    "end_at": pd.Timestamp(stop),
                    "start_nav": np.nan if before is None else before,
                    "end_nav": np.nan if after is None else after,
                    "portfolio_return": np.nan if value is None else value,
                    "portfolio_reason": reason,
                }
            )
    return rows


def build_crypto_portfolio_statistics_tables(
    *,
    ledger_path,
    config,
    windows,
    read_ledger,
    read_episodes,
    summarize_ledger,
    table_columns,
):
    analyses = selected_portfolio_analyses(config, _CRYPTO_ANALYSES)
    mode = config["execution_mode"]
    execution_config = config.get("execution", {})
    statistics_policy = config.get("statistics", {})
    if not windows:
        raise ValueError("crypto_statistics_windows_required")
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
    comparison_rows = []
    comparison_requested = bool(set(analyses).intersection(COMPARISON_ANALYSES))
    covered_ns = episode_count = truncated_count = 0
    funding_required = funding_covered = 0
    costs_total = {
        key: 0.0
        for key in ("fees", "funding", "turnover", "fill_events", "fill_quantity")
    }
    costs_reasons = {}
    observed_total_names = {"fees", "turnover", "fill_events", "fill_quantity"}
    partial_observed_totals = False
    cursor = bounds[0][0]
    envelope_stop = bounds[-1][1] + 1
    candidate_key = {"candidate_id": config["candidate_id"], "execution_mode": mode}

    def cover(start, stop, kind, reason="not_covered_by_qualified_episode"):
        coverage.append(
            {
                **candidate_key,
                "start_at": pd.Timestamp(start),
                "stop_at": pd.Timestamp(stop),
                "kind": kind,
                "reason": "" if kind == "covered" else reason,
            }
        )

    def metric(name, value, count, reason=""):
        annualized = name in {"annualized_volatility", "sharpe"}
        day_count = name in {
            "planned_complete_utc_days",
            "defined_complete_utc_days",
            "oos_mean_daily_return",
        }
        if (annualized and "annualized" not in analyses) or (
            not annualized
            and "performance" not in analyses
            and not (day_count and "annualized" in analyses)
        ):
            return
        if isinstance(value, Fraction):
            value = finite_exact_float(value)
        defined = value is not None and np.isfinite(value)
        metrics.append(
            {
                **row_key,
                "metric_id": name,
                "value": float(value) if defined else np.nan,
                "observation_count": count,
                "status": "DEFINED" if defined else "UNDEFINED",
                "reason": "" if defined else reason or "nonfinite_result",
            }
        )

    row_key = {}

    def consume_episode(episode, artifact, batches):
        nonlocal covered_ns, episode_count, truncated_count, cursor, row_key
        nonlocal funding_required, funding_covered
        nonlocal partial_observed_totals
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
        observe = finish = None
        if {"performance", "annualized"} & set(analyses) or comparison_requested:
            observe, finish = _crypto_observations(
                artifact,
                windows,
                execution_config["max_position"]
                if config["execution_style"] == "maker"
                else execution_config["position_size"],
                observation_window_ns=observed,
            )
        summary = summarize_ledger(
            artifact,
            batches,
            windows_ns=bounds if episode is None else None,
            consume_batch=observe,
        )
        economic = summary["economics"]
        execution = summary["execution"]
        partial_reason = (
            execution["account_stop_reason"]
            if execution.get("account_path_status") == "PARTIAL"
            else ""
        )
        observed_prefix = execution.get(
            "observed_totals_scope"
        ) == "known_prefix" or bool(partial_reason)
        partial_observed_totals |= observed_prefix
        drawdown_reason = partial_reason or (
            execution["market_observation_reason"]
            if execution.get("market_observation_status") == "PARTIAL"
            else ""
        )
        if episode is None:
            intervals = summary["intervals"]
            start, stop = (bounds[0][0], envelope_stop)
        else:
            start, stop = (episode["start_ns"], episode["stop_ns"])
            observed = (start, episode["terminal_valuation_ns"])
            intervals = (
                []
                if partial_reason
                else [
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
            )
            truncated_count += episode["termination"] == "TRUNCATED"
        planned_stop = stop
        if partial_reason:
            stop = min(stop, execution["known_prefix_valuation_timestamp_ns"] + 1)
            if episode is not None:
                observed_end = episode["stop_ns"] - 1
        if cursor < start:
            cover(cursor, start, "missing")
        if start < stop:
            cover(start, stop, "covered")
        if stop < planned_stop:
            cover(stop, planned_stop, "missing", partial_reason)
        cursor = planned_stop
        covered_ns += stop - start
        for name in costs_total:
            costs_total[name] = (
                None
                if economic[name] is None or costs_total[name] is None
                else costs_total[name] + economic[name]
            )
            if economic[name] is None:
                costs_reasons[name] = economic.get(f"{name}_reason") or partial_reason
        if (
            funding_required is not None
            and "funding_required_decisions" in execution
            and "funding_covered_decisions" in execution
        ):
            funding_required += execution["funding_required_decisions"]
            funding_covered += execution["funding_covered_decisions"]
        else:
            funding_required = funding_covered = None
        if episode is not None:
            execution = {
                **execution,
                "termination": episode["termination"],
                "stop_reasons": ",".join(episode["stop_reasons"]),
            }
        if finish is not None:
            daily, inventory, drawdown = finish(observation_end_ns=observed_end)
            if comparison_requested:
                comparison_rows.extend(
                    _crypto_comparison_axis(
                        windows=windows,
                        episode=episode,
                        summary=summary,
                        daily=daily,
                        capital=artifact.capital,
                        row_key=row_key,
                        sample_role=config["sample_role"],
                    )
                )
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
                metric(
                    f"known_prefix_{key}"
                    if observed_prefix and key in observed_total_names
                    else key,
                    economic[key],
                    int(economic["fill_events"]),
                    economic.get(f"{key}_reason") or "",
                )
            metric(
                "terminal_nav",
                economic["nav"],
                len(intervals),
                economic.get("nav_reason") or "",
            )
            metric(
                "continuous_max_drawdown"
                if episode is None
                else "episode_max_drawdown",
                None if drawdown_reason else drawdown,
                len(intervals),
                drawdown_reason,
            )
            for scope in ("oos", "gap"):
                state = inventory[scope]
                expected_duration = sum(
                    max(0, min(end, observed_end) - max(start, observed_start))
                    for start, end in bounds
                )
                if scope == "gap":
                    expected_duration = (
                        observed_end - observed_start - expected_duration
                    )
                incomplete = (
                    bool(partial_reason) and state["duration"] < expected_duration
                )
                stop_at = execution.get("account_stop_ns")
                unknown_extreme = (
                    partial_reason == "portfolio_pending_execution_unknown"
                    and stop_at is not None
                    and observed_start <= stop_at <= observed_end
                    and (any(start <= stop_at <= end for start, end in bounds))
                    == (scope == "oos")
                )
                if episode is None:
                    scope_intervals = [row for row in intervals if row["kind"] == scope]
                    pnl_incomplete = (
                        bool(partial_reason)
                        and sum(
                            row["end_ns"] - row["start_ns"] for row in scope_intervals
                        )
                        < expected_duration
                    )
                    metric(
                        f"{scope}_net_pnl",
                        None
                        if pnl_incomplete
                        else sum(row["net_pnl"] for row in scope_intervals),
                        len(scope_intervals),
                        partial_reason if pnl_incomplete else "",
                    )
                inventory_mean = state["mean"] * state["scale"]
                mean_underflow = state["mean"] != 0 and inventory_mean == 0
                variance = state["variance"] * state["scale"] * state["scale"]
                variance_underflow = state["variance"] > 0 and variance == 0
                for name, value, underflow in (
                    ("inventory_mean", inventory_mean, mean_underflow),
                    ("inventory_variance", variance, variance_underflow),
                    (
                        "inventory_time_at_limit_fraction",
                        state["at_limit"] / state["duration"]
                        if state["duration"]
                        else None,
                        False,
                    ),
                ):
                    metric(
                        f"{scope}_{name}",
                        value
                        if state["duration"] and not incomplete and not underflow
                        else None,
                        int(state["duration"] > 0),
                        partial_reason
                        if incomplete
                        else f"no_{scope}_duration"
                        if not state["duration"]
                        else f"nonzero_{name.removeprefix('inventory_')}_underflow"
                        if underflow
                        else "",
                    )
                metric(
                    f"{scope}_inventory_max_abs",
                    state["max_abs"]
                    if state["observations"] and not (incomplete or unknown_extreme)
                    else None,
                    state["observations"],
                    partial_reason
                    if incomplete or unknown_extreme
                    else f"no_{scope}_observations",
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
                float(mean(values)) if count and (not daily_missing) else None,
                count,
                day_reason if daily_missing or not count else "",
            )
            if "annualized" in analyses and (
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
            elif "annualized" in analyses:
                metric("annualized_volatility", None, count, day_reason)
                metric("sharpe", None, count, day_reason)
            metric(
                "pnl_per_fill",
                economic["net_pnl"] / economic["fill_events"]
                if economic["fill_events"] and economic["net_pnl"] is not None
                else None,
                int(economic["fill_events"]),
                economic.get("net_pnl_reason") or "no_fills",
            )
            metric(
                "pnl_per_notional",
                economic["net_pnl"] / economic["turnover"]
                if economic["turnover"] and economic["net_pnl"] is not None
                else None,
                int(economic["fill_events"]),
                economic.get("net_pnl_reason") or "no_turnover",
            )
        if "execution" in analyses:
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
        if "cost_sensitivity" in analyses:
            base_fee = (
                execution_config["maker_fee_bps"]
                if config["execution_style"] == "maker"
                else execution_config["taker_fee_bps"]
            )
            costs.extend(
                {
                    **row_key,
                    **row,
                    "method": "constant_known_prefix_turnover_additive_sensitivity"
                    if observed_prefix
                    else row["method"],
                }
                for row in fixed_turnover_cost_sensitivity(
                    turnover=economic["turnover"],
                    net_pnl=economic["net_pnl"],
                    base_fee_bps=base_fee,
                    fee_scenarios_bps=statistics_policy["fee_scenarios_bps"],
                    additional_spread_scenarios_bps=statistics_policy[
                        "additional_spread_scenarios_bps"
                    ],
                )
            )

    root = ledger_path
    if mode == "continuous":
        read_ledger(
            root,
            consume_batches=lambda artifact, batches: consume_episode(
                None, artifact, batches
            ),
        )
    else:
        read_episodes(root, consume_episode=consume_episode)
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
        metric(
            f"known_prefix_{name}"
            if partial_observed_totals and name in observed_total_names
            else name,
            value,
            episode_count,
            costs_reasons.get(name, ""),
        )
    if "execution" in analyses:
        for name, value in {
            "funding_required_decisions": funding_required,
            "funding_covered_decisions": funding_covered,
            "funding_decision_coverage": funding_covered / funding_required
            if funding_required
            else None,
            "funding_selection_status": "PARTIAL"
            if funding_required and 0 < funding_covered < funding_required
            else "BLOCKED"
            if funding_required and not funding_covered
            else "COMPLETE"
            if funding_required
            else "UNDEFINED",
        }.items():
            diagnostics.append(
                {
                    **row_key,
                    "metric_id": name,
                    "value": float(value) if type(value) in (float, int) else np.nan,
                    "text": value if type(value) is str else "",
                    "status": "DEFINED" if value is not None else "UNDEFINED",
                    "reason": ""
                    if value is not None
                    else "funding_decision_counts_unrecorded"
                    if funding_required is None
                    else "no_funding_required_decisions",
                }
            )
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

    requested = {"crypto_portfolio_coverage"}
    if "performance" in analyses:
        requested.add("crypto_portfolio_intervals")
    if {"performance", "annualized"} & set(analyses):
        requested.update(("crypto_portfolio_daily", "crypto_portfolio_statistics"))
    if "execution" in analyses:
        requested.add("crypto_execution_diagnostics")
    if "cost_sensitivity" in analyses:
        requested.add("crypto_cost_sensitivity")
    result = {name: table for name, table in tables.items() if name in requested}
    if comparison_requested:
        axis = pd.DataFrame.from_records(
            comparison_rows,
            columns=(
                "candidate_id",
                "execution_mode",
                "episode_id",
                "fold_id",
                "sample_role",
                "start_at",
                "end_at",
                "portfolio_return",
                "portfolio_reason",
                "start_nav",
                "end_nav",
            ),
        )
        result.update(
            comparison_tables(
                axis,
                config=config,
                analyses=analyses,
                currencies={config["candidate_id"]: config["valuation_currency"]},
                clock="utc",
                metadata=config,
            )
        )
    return result


def iter_holding_factor_inputs(
    *,
    inputs,
    calendar,
    time_window,
    unified_trading_sessions,
    style_columns,
    benchmark_required,
    risk_config: FactorRiskConfig | None = None,
    iter_portfolio_sessions,
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
                closing_iterator(
                    _iter_input_sessions(
                        root, projection, window, business_columns=("weight",)
                    )
                )
            )

        panels = channels.enter_context(
            closing_iterator(
                iter_portfolio_sessions(
                    market_bars_root=inputs["market_bars"],
                    stock_master_root=inputs["stock_master"],
                    equity_events_root=inputs.get("equity_events"),
                    panel_root=inputs["panel"],
                    panel_columns=(*_ENHANCED_PANEL_COLUMNS, risk.config.weight_column),
                    calendar=calendar,
                    time_window=history,
                )
            )
        )
        risk_inputs = _risk_context_input_sessions(
            roots=tuple(inputs["risk_context"]),
            style_columns=style_columns,
            time_window=history,
            retained=channels,
        )
        contexts = {root: frames for root, (_columns, frames) in risk_inputs.items()}
        context_frames = {root: next(source, None) for root, source in contexts.items()}
        benchmarks = (
            stream(
                inputs["benchmark"],
                ("benchmark_id", "asset_id", "timestamp", "weight"),
                time_window,
            )
            if benchmark_required
            else iter(())
        )
        panel, benchmark = (next(panels, None), next(benchmarks, None))

        def take_panels(through):
            nonlocal panel
            while panel is not None and panel.timestamp <= through:
                timestamp = panel.timestamp
                if panel.panel.asset_id.duplicated().any() or timestamp not in sessions:
                    raise ValueError("holding_panel_keys_invalid")
                panel_window[timestamp] = panel
                panel = next(panels, None)

        for interval in intervals.itertuples(index=False):
            while (
                position < len(sessions) and sessions[position] <= interval.decision_at
            ):
                timestamp = sessions[position]
                take_panels(timestamp)
                current_view = panel_window.get(timestamp)
                current = current_view.panel if current_view is not None else None
                for context in context_frames.values():
                    if context is not None and stamp(context) < timestamp:
                        raise ValueError("holding_risk_context_calendar_misaligned")
                if current is None or current.empty:
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
                        weight_column=risk.config.weight_column,
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
                panel_window[timestamp].valuation if timestamp in panel_window else None
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
            decision_exposure_assets = frozenset()
            if any(frame is None for frame in prices):
                reason = "holding_prices_missing"
            else:
                close = prices[0].set_index("asset_id").adj_close
                entry = prices[1].set_index("asset_id").adj_open
                endpoint = prices[2].set_index("asset_id").adj_open
                if reason is None:
                    holding_cross = risk.prepare_exposures(
                        cross, include_ineligible=True
                    )
                    decision_exposure_assets = frozenset(holding_cross.asset_id)
                    life_keys = ["asset_id", "listing_date"]
                    real_pairs = pd.MultiIndex.from_frame(
                        panel_window[interval.entry_at].panel[life_keys]
                    ).intersection(
                        pd.MultiIndex.from_frame(
                            panel_window[interval.end_at].panel[life_keys]
                        )
                    )
                    real_pairs = real_pairs.intersection(
                        pd.MultiIndex.from_frame(cross[life_keys])
                    )
                    holding_cross = holding_cross.loc[
                        holding_cross.asset_id.isin(
                            real_pairs.get_level_values("asset_id")
                        )
                    ].copy()
                    holding_cross["asset_return"] = holding_cross.asset_id.map(
                        endpoint.div(entry) - 1.0
                    )
                    try:
                        fitted = risk.fit_return_session(
                            holding_cross,
                            require_complete=True,
                            project_ineligible=True,
                        )
                    except ValueError as error:
                        if str(error) != "holding_returns_incomplete":
                            raise
                        reason = str(error)
                    if fitted is None and reason is None:
                        reason = "holding_factor_fit_unidentified"
            decision_closed = panel_window[interval.decision_at].closed_assets
            closed_domain = (
                decision_closed
                | panel_window[interval.entry_at].closed_assets
                | panel_window[interval.end_at].closed_assets
            )
            closed_assets = pd.Index(
                closed_domain & (decision_exposure_assets | decision_closed)
            )
            if any(frame is None for frame in prices):
                closed_assets = closed_assets[:0]
            else:
                for frame, column in zip(
                    prices, ("adj_close", "adj_open", "adj_open"), strict=True
                ):
                    mark = (
                        frame.set_index("asset_id")[column]
                        .reindex(closed_assets)
                        .astype("float64")
                    )
                    closed_assets = closed_assets[np.isfinite(mark) & mark.gt(0)]
            result = HoldingFactorInputs(
                interval.decision_at,
                interval.entry_at,
                interval.end_at,
                fitted,
                close,
                entry,
                weights,
                reason,
                frozenset(closed_assets),
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
    config,
    time_window,
    unified_trading_sessions,
    inputs,
    iter_portfolio_sessions,
    comparison_metadata=None,
):
    from quant_research_platform.research.portfolios.production import (
        _iter_input_sessions,
        _read_materialized,
    )

    analyses = selected_portfolio_analyses(config, _PORTFOLIO_ANALYSES)
    if "group_comparison" in analyses:
        declaration = config["statistics"].get("group_comparison")
        if declaration is not None:
            by_id = {candidate.candidate_id: candidate for candidate in candidates}
            for portfolio_id in declaration["portfolio_ids"]:
                pair = [
                    by_id.get(f"{group}_{portfolio_id}")
                    for group in ("global", "quarterly")
                ]
                if any(
                    candidate is None or candidate.selection_group != group
                    for candidate, group in zip(
                        pair, ("global", "quarterly"), strict=True
                    )
                ):
                    raise ValueError(
                        "portfolio_group_comparison_selection_group_invalid"
                    )
                if any(
                    getattr(pair[0], field) != getattr(pair[1], field)
                    for field in (
                        "construction",
                        "rebalance_schedule",
                        "top_quantile",
                        "benchmark_id",
                        "neutrality",
                        "risk_style_columns",
                    )
                ) or (
                    pair[0].neutrality in {"size", "industry_size"}
                    and pair[0].size_cap_column != pair[1].size_cap_column
                ):
                    raise ValueError("portfolio_group_comparison_strategy_mismatch")
    pieces = []

    calendar = (
        _read_materialized(
            inputs["calendar"],
            ("calendar_date", "source_market_type", "is_trading_day"),
        )
        if "attribution" in analyses
        else None
    )
    codes = (
        config.get("benchmark_total_return_indices", {})
        if {"attribution"} & set(analyses)
        else {}
    )
    price_codes = {
        name: code
        for code, (name, _label) in a_share_ids.FORMAL_BENCHMARKS.items()
        if name in codes
    }
    if codes:
        endpoint_dates = (
            (
                holding_intervals(
                    pd.DatetimeIndex(unified_trading_sessions(calendar)), time_window
                )
                .loc[:, ["entry_at", "end_at"]]
                .to_numpy()
                .ravel()
            )
            if calendar is not None
            else None
        )
        with closing_iterator(
            _iter_input_sessions(
                inputs["index_market"],
                ("source_asset_id", "timestamp", "open", "close"),
                time_window,
                business_columns=("open", "close"),
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
            & (
                benchmark.timestamp.isin(endpoint_dates)
                if endpoint_dates is not None
                else True
            )
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
        required = required.where(np.isfinite(required) & required.gt(0))
        open_mantissa, open_exponent = np.frexp(required.open.to_numpy())
        total_mantissa, total_exponent = np.frexp(required.total_close.to_numpy())
        close_mantissa, close_exponent = np.frexp(required.close.to_numpy())
        benchmark.loc[missing, "open"] = np.ldexp(
            open_mantissa * total_mantissa / close_mantissa,
            open_exponent + total_exponent - close_exponent,
        )
    benchmark = benchmark.loc[
        benchmark.source_asset_id.isin(codes.values()),
        ["source_asset_id", "timestamp", "open"],
    ].copy()
    benchmark["benchmark_id"] = benchmark.source_asset_id.map(
        {code: name for name, code in codes.items()}
    )
    if "attribution" in analyses and calendar is None:
        raise ValueError("portfolio_statistics_attribution_calendar_required")
    holding_inputs = (
        {
            styles: iter_holding_factor_inputs(
                iter_portfolio_sessions=iter_portfolio_sessions,
                inputs=inputs,
                calendar=calendar,
                time_window=time_window,
                unified_trading_sessions=unified_trading_sessions,
                style_columns=styles,
                risk_config=FactorRiskConfig(**config["risk"]),
                benchmark_required=any(
                    candidate.benchmark_id is not None
                    for candidate in candidates
                    if candidate.risk_style_columns == styles
                ),
            )
            for styles in dict.fromkeys(
                candidate.risk_style_columns for candidate in candidates
            )
        }
        if "attribution" in analyses
        else {}
    )
    return _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        config=config,
        time_window=time_window,
        benchmark_opens=benchmark,
        calendar=calendar,
        unified_trading_sessions=unified_trading_sessions,
        holding_inputs=holding_inputs,
        comparison_metadata=comparison_metadata,
    )


def _portfolio_statistics_tables(
    *,
    ledgers,
    candidates,
    config,
    time_window,
    benchmark_opens,
    calendar,
    unified_trading_sessions,
    holding_inputs,
    comparison_metadata=None,
):
    """Project the protocol-selected ledgers into interval-aligned diagnostics."""
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS

    analyses = selected_portfolio_analyses(config, _PORTFOLIO_ANALYSES)
    benchmark_requested = bool({"attribution"} & set(analyses))
    ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(ids) != len(set(ids)) or set(ledgers) != set(ids):
        raise ValueError("portfolio_statistics_candidate_set_invalid")
    expected_intervals = (
        holding_intervals(
            pd.DatetimeIndex(unified_trading_sessions(calendar)), time_window
        )
        if calendar is not None
        else None
    )
    access = config["sample_role"]
    (
        daily_rows,
        capacity_rows,
        summary_rows,
        capacity_summary,
        attribution_rows,
        robustness_rows,
    ) = ([], [], [], [], [], [])
    exposure_rows, factor_rows = ([], [])
    daily_by_candidate = {candidate_id: [] for candidate_id in ids}
    valuation_by_candidate = {candidate_id: [] for candidate_id in ids}
    adv_gaps = {
        candidate_id: dict(
            adv_unavailable_intervals=0.0,
            adv_unavailable_buy_orders=0.0,
            adv_unavailable_sell_orders=0.0,
            adv_unavailable_buy_target_notional=Fraction(),
            adv_unavailable_sell_target_notional=Fraction(),
        )
        for candidate_id in ids
    }
    benchmarks = {}
    for benchmark_id in dict.fromkeys(
        candidate.benchmark_id
        for candidate in candidates
        if benchmark_requested and candidate.benchmark_id is not None
    ):
        benchmark = benchmark_opens.loc[
            benchmark_opens.benchmark_id.eq(benchmark_id), ["timestamp", "open"]
        ].set_index("timestamp")["open"]
        if benchmark.index.has_duplicates:
            raise ValueError("benchmark_return_duplicate_keys")
        benchmarks[benchmark_id] = benchmark
    economic_totals = {
        candidate_id: _LedgerEconomics(ledgers[candidate_id].capital)
        for candidate_id in ids
    }
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
        fit_streams = {
            styles: sources.enter_context(closing_iterator(iter(inputs)))
            for styles, inputs in holding_inputs.items()
            if "attribution" in analyses
        }
        sentinel = object()

        interval_index = 0
        while True:
            group = tuple(
                next(stream, sentinel) for stream in (*streams, *fit_streams.values())
            )
            if all(value is sentinel for value in group):
                break
            if any(value is sentinel for value in group):
                raise ValueError("portfolio_statistics_candidate_dates_misaligned")
            if expected_intervals is not None and interval_index >= len(
                expected_intervals
            ):
                raise ValueError("portfolio_statistics_window_incomplete")
            batch_group = group[: len(streams)]
            interval_inputs = dict(zip(fit_streams, group[len(streams) :], strict=True))
            if expected_intervals is not None:
                expected = expected_intervals.iloc[interval_index]
            else:
                first = portfolio_interval_economics(batch_group[0])
                expected = pd.Series(
                    {"entry_at": first["entry_at"], "end_at": first["end_at"]}
                )
            for styles, holding_input in interval_inputs.items():
                if (
                    holding_input.decision_at,
                    holding_input.entry_at,
                    holding_input.end_at,
                ) != tuple(expected):
                    raise ValueError("holding_factor_interval_misaligned")
                if holding_input.fitted is not None:
                    factor_rows.extend(
                        dict(
                            attribution_group=canonical_json_hash(list(styles)),
                            decision_at=holding_input.decision_at,
                            entry_at=holding_input.entry_at,
                            end_at=holding_input.end_at,
                            factor_id=name,
                            value=float(value),
                            observation_count=len(holding_input.fitted.asset_returns),
                        )
                        for name, value in holding_input.fitted.factor_returns.items()
                    )
            for candidate, batch in zip(candidates, batch_group, strict=True):
                holding_input = interval_inputs.get(candidate.risk_style_columns)
                candidate_id = candidate.candidate_id
                previous_end, previous_nav = previous[candidate_id]
                economic_totals[candidate_id].consume(batch)
                row = portfolio_interval_economics(batch)
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
                    return_status="DEFINED"
                    if np.isfinite(row["net_return"])
                    else "UNDEFINED",
                    return_reason=""
                    if np.isfinite(row["net_return"])
                    else "nonpositive_start_nav"
                    if row["nav_before"] == 0
                    else "nonfinite_result",
                    benchmark_return=np.nan,
                    benchmark_reason="",
                    sample_role=access,
                )
                if benchmark_requested and candidate.benchmark_id is not None:
                    benchmark = benchmarks[candidate.benchmark_id]
                    value, _reason = comparison_endpoint_return(
                        benchmark.get(row["entry_at"], np.nan),
                        benchmark.get(row["end_at"], np.nan),
                    )
                    row["benchmark_return"] = value
                    row["benchmark_reason"] = (
                        "benchmark_total_return_open_missing"
                        if _reason == "reference_endpoint_missing"
                        else _reason
                    )
                daily_rows.append(row)
                daily_by_candidate[candidate_id].append(row)
                valuation = _research_valuation_observation(batch, row)
                if valuation is not None:
                    valuation_by_candidate[candidate_id].append(valuation)
                if "attribution" in analyses:
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
                    if valuation is not None and valuation["estimated_held_assets"]:
                        holding_input = replace(
                            holding_input,
                            fitted=None,
                            reason="research_estimated_valuation",
                        )
                    if (
                        batch.rights_adjustments
                        or ((after == 0) & (endpoint > 0)).any()
                    ):
                        holding_input = replace(
                            holding_input, fitted=None, reason="zero_valuation_recovery"
                        )
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
                    components["cost"] = (
                        -row["cost_drag"] if np.isfinite(row["cost_drag"]) else None
                    )
                    if components["cost"] is None:
                        reasons["cost"] = (
                            "nonpositive_start_nav"
                            if row["nav_before"] == 0
                            else "nonfinite_result"
                        )
                    for component, value in components.items():
                        reason = reasons.get(component, "")
                        if value is None and component == "benchmark":
                            reason = (
                                row["benchmark_reason"]
                                or "benchmark_total_return_open_missing"
                            )
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
                if "attribution" in analyses:
                    exposure_rows.extend(
                        dict(
                            candidate_id=candidate_id,
                            decision_at=holding_input.decision_at,
                            entry_at=row["entry_at"],
                            factor_id=item.name,
                            value=item.number,
                        )
                        for item in batch.diagnostics
                        if item.asset_id == PORTFOLIO_CASH_ASSET_ID
                        and item.name.startswith(
                            (
                                "target_exposure:",
                                "realized_exposure:",
                                "benchmark_exposure:",
                            )
                        )
                        and item.number is not None
                    )
                missing_orders = {
                    item.asset_id
                    for item in batch.diagnostics
                    if item.name == "block_reason" and item.text == "adv_unavailable"
                }
                adv_gaps[candidate_id]["adv_unavailable_intervals"] += bool(
                    missing_orders
                )
                for value in batch.valuations:
                    if value.phase == "after" and value.asset_id in missing_orders:
                        desired = Fraction(
                            facts[value.asset_id, "target_notional"].number
                        ) - Fraction(value.value)
                        side = "buy" if desired > 0 else "sell"
                        adv_gaps[candidate_id][f"adv_unavailable_{side}_orders"] += 1
                        adv_gaps[candidate_id][
                            f"adv_unavailable_{side}_target_notional"
                        ] += abs(desired)
                if "capacity" in analyses:
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
                        available = {
                            asset: facts[asset, "adv20"].number
                            if (asset, "adv20") in facts
                            else None
                            for asset in deltas
                        }
                        adv_missing = any(value is None for value in available.values())
                        for participation in config["execution_assumptions"][
                            "adv_sensitivities"
                        ]:
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
                                    status="UNDEFINED" if adv_missing else "DEFINED",
                                    reason="capacity_required_adv_missing"
                                    if adv_missing
                                    else "",
                                )
                            )
            interval_index += 1

    for candidate in candidates:
        candidate_id = candidate.candidate_id
        artifact = ledgers[candidate_id]
        previous_end, previous_nav = previous[candidate_id]
        candidate_rows = daily_by_candidate[candidate_id]
        selected = pd.DataFrame(candidate_rows)
        if selected.empty:
            raise ValueError("portfolio_statistics_empty_candidate")
        if not {"performance"} & set(analyses):
            continue
        returns_defined = selected.return_status.eq("DEFINED").all()
        count = len(selected)
        deviation = float(
            return_standard_deviation(
                selected.net_return,
                roundoff=_nav_return_roundoff(
                    selected.nav_before, selected.nav_end, selected.net_return
                ),
            )
        )
        economic = economic_totals[candidate_id].summary()
        endpoint_return = finite_exact_float(
            (Fraction(previous_nav) - Fraction(artifact.capital.initial_capital))
            / Fraction(artifact.capital.initial_capital)
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_growth = (
                np.log1p(endpoint_return)
                if endpoint_return is not None and endpoint_return > -1
                else np.log(previous_nav) - np.log(artifact.capital.initial_capital)
            )
            cagr = float(np.expm1(log_growth * (252 / count)))
        if not np.isfinite(cagr):
            cagr = np.nan
        values = {
            "terminal_nav": previous_nav,
            "capital_return": np.nan
            if economic["capital_return"] is None
            else economic["capital_return"],
            "cagr": cagr,
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
            "max_drawdown": float(
                (
                    selected.nav_end
                    / selected.nav_end.cummax().clip(
                        lower=artifact.capital.initial_capital
                    )
                    - 1
                ).min()
            ),
            "mean_cash_weight_post": float(
                selected.cash_weight_post.mean(skipna=False)
            ),
            "terminal_cash_weight_post": float(selected.cash_weight_post.iloc[-1]),
            "terminal_cash_post": float(selected.cash_post.iloc[-1]),
            "target_turnover": float(selected.target_turnover.mean(skipna=False)),
            "requested_turnover": float(selected.requested_turnover.mean(skipna=False)),
            "realized_turnover": float(selected.realized_turnover.mean(skipna=False)),
            "gross_pnl": economic["gross_pnl"],
            "net_pnl": economic["net_pnl"],
            "cost": economic["fees"],
        }
        for name, value in adv_gaps[candidate_id].items():
            number = finite_exact_float(value) if isinstance(value, Fraction) else value
            values[name] = np.nan if number is None else number
        valuation = pd.DataFrame(valuation_by_candidate[candidate_id])
        if not valuation.empty:
            values.update(
                unknown_execution_no_fill_orders=float(
                    valuation.execution_fact_unknown.sum()
                ),
                unquoted_valuation_no_fill_orders=float(
                    valuation.research_unquoted_valuation.sum()
                ),
                estimated_held_asset_intervals=float(
                    valuation.estimated_held_assets.sum()
                ),
                terminal_estimated_value=float(valuation.estimated_value_end.iloc[-1]),
                terminal_estimated_weight=float(
                    valuation.estimated_weight_end.iloc[-1]
                ),
                maximum_estimated_weight=float(
                    valuation.estimated_weight_end.max(skipna=False)
                ),
                maximum_estimated_mark_age_days=float(
                    valuation.maximum_mark_age_days.max()
                ),
            )
            for percent in (0, 50, 100):
                stressed = valuation[f"nav_{percent}pct"]
                prefix = f"research_valuation_{percent}pct"
                values[f"{prefix}_terminal_nav"] = float(
                    valuation[f"nav_{percent}pct"].iloc[-1]
                )
                ratio = finite_exact_float(
                    (
                        Fraction(float(stressed.iloc[-1]))
                        - Fraction(artifact.capital.initial_capital)
                    )
                    / Fraction(artifact.capital.initial_capital)
                )
                values[f"{prefix}_capital_return"] = np.nan if ratio is None else ratio
                values[f"{prefix}_max_drawdown"] = float(
                    (
                        stressed
                        / stressed.cummax().clip(lower=artifact.capital.initial_capital)
                        - 1
                    ).min()
                )
        undefined_reason = (
            "nonpositive_start_nav"
            if selected.nav_before.eq(0).any()
            else "nonfinite_result"
            if not returns_defined
            or selected[
                [
                    "gross_return",
                    "cost_drag",
                    "target_turnover",
                    "requested_turnover",
                    "realized_turnover",
                    "cash_weight_post",
                ]
            ]
            .isna()
            .any()
            .any()
            else "statistic_not_identified"
        )
        for name, value in values.items():
            summary_rows.append(
                dict(
                    candidate_id=candidate_id,
                    metric_id=name,
                    value=value if np.isfinite(value) else np.nan,
                    status="DEFINED" if np.isfinite(value) else "UNDEFINED",
                    reason=""
                    if np.isfinite(value)
                    else "nonfinite_result"
                    if np.isinf(value)
                    or name
                    in {
                        "capital_return",
                        "cagr",
                    }
                    or name.endswith(("_capital_return", "_target_notional"))
                    else undefined_reason,
                    observation_count=count,
                    sample_role=access,
                )
            )
    daily = pd.DataFrame(daily_rows)
    if daily.duplicated(["candidate_id", "entry_at", "end_at"]).any():
        raise ValueError("portfolio_statistics_duplicate_intervals")
    matrix = daily.pivot(
        index=["entry_at", "end_at"], columns="candidate_id", values="net_return"
    ).reindex(columns=ids)
    if expected_intervals is not None and tuple(matrix.index) != tuple(
        expected_intervals.loc[:, ["entry_at", "end_at"]].itertuples(
            index=False, name=None
        )
    ):
        raise ValueError("portfolio_statistics_window_incomplete")
    paired_comparison = (
        _group_portfolio_comparison(matrix, config)
        if "group_comparison" in analyses
        else pd.DataFrame(columns=_PORTFOLIO_COMPARISON_COLUMNS)
    )
    for index, pair in paired_comparison.iterrows():
        if pair.reason == "nonpositive_start_nav":
            pair_rows = daily.loc[
                daily.candidate_id.isin(
                    [f"global_{pair.portfolio_id}", f"quarterly_{pair.portfolio_id}"]
                )
            ]
            if not pair_rows.nav_before.eq(0).any():
                paired_comparison.loc[index, ["reason", "bootstrap_reason"]] = (
                    "nonfinite_result"
                )
    for group in ("global", "quarterly"):
        paired_comparison[f"{group}_terminal_wealth"] = [
            finite_exact_float(
                Fraction(previous[f"{group}_{portfolio_id}"][1])
                / Fraction(ledgers[f"{group}_{portfolio_id}"].capital.initial_capital)
            )
            for portfolio_id in paired_comparison.portfolio_id
        ]
        paired_comparison[f"{group}_terminal_wealth_reason"] = np.where(
            paired_comparison[f"{group}_terminal_wealth"].isna(), "nonfinite_result", ""
        )
    if {"sharpe_inference", "cscv"} & set(analyses):
        inference_ids = ids
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
        trial_sharpes = (
            _return_sharpes(inference_matrix, roundoff=roundoff)
            if "sharpe_inference" in analyses
            else None
        )
        for candidate_id in ids if "sharpe_inference" in analyses else ():
            if inference_matrix[candidate_id].isna().any():
                outputs = tuple(
                    (
                        name,
                        None,
                        "nonpositive_start_nav"
                        if before[candidate_id].eq(0).any()
                        else "nonfinite_result",
                    )
                    for name in ("psr", "dsr")
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
                        sample_role=access,
                    )
                )
        if "cscv" in analyses:
            blocks = config["statistics"]["cscv_blocks"]
            if not np.isfinite(inference_matrix.to_numpy()).all():
                value, status, reason = (
                    np.nan,
                    "UNDEFINED",
                    "nonpositive_start_nav"
                    if before.eq(0).any().any()
                    else "nonfinite_result",
                )
            elif len(matrix) >= blocks:
                result = cscv_pbo(
                    inference_matrix,
                    slice_count=blocks,
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
                    sample_role=access,
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
    capacity_groups = capacity.groupby(
        ["candidate_id", "participation"], sort=False
    ).indices
    for candidate_id in ids:
        for participation in (
            config.get("execution_assumptions", {}).get("adv_sensitivities", ())
            if "capacity" in analyses
            else ()
        ):
            sample = capacity.iloc[
                capacity_groups.get((candidate_id, participation), [])
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
                            status="DEFINED" if defined else "UNDEFINED",
                            reason=""
                            if defined
                            else "no_rebalance_requests"
                            if part.empty
                            else "capacity_required_adv_missing",
                            observation_count=len(part),
                        )
                    )
    tables = {
        "portfolio_daily": daily.reindex(
            columns=STATISTICS_TABLE_COLUMNS["portfolio_daily.parquet"]
        ),
        "portfolio_group_comparison": paired_comparison,
        "portfolio_performance": pd.DataFrame.from_records(
            summary_rows,
            columns=STATISTICS_TABLE_COLUMNS["portfolio_performance.parquet"],
        ),
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
                "attribution_group",
                "decision_at",
                "entry_at",
                "end_at",
                "factor_id",
                "value",
                "observation_count",
            ),
        ),
    }

    requested = set()
    if {"performance"} & set(analyses):
        requested.update(("portfolio_daily", "portfolio_performance"))
    if "capacity" in analyses:
        requested.update(("portfolio_capacity", "portfolio_capacity_summary"))
    if "attribution" in analyses:
        requested.update(
            ("portfolio_attribution", "portfolio_exposures", "holding_factor_returns")
        )
    if {"sharpe_inference", "cscv"} & set(analyses):
        requested.add("portfolio_robustness")
    if "group_comparison" in analyses:
        requested.add("portfolio_group_comparison")
    result = {name: table for name, table in tables.items() if name in requested}
    if set(analyses).intersection(COMPARISON_ANALYSES):
        axis = daily.rename(
            columns={
                "entry_at": "start_at",
                "net_return": "portfolio_return",
                "return_reason": "portfolio_reason",
            }
        ).assign(
            execution_mode="continuous",
            episode_id="continuous",
            fold_id="",
            start_nav=daily.nav_before,
            end_nav=daily.nav_end,
        )
        result.update(
            comparison_tables(
                axis,
                config=config,
                analyses=analyses,
                currencies={
                    key: ledger.capital.valuation_currency
                    for key, ledger in ledgers.items()
                },
                clock="trading_day_open",
                metadata={} if comparison_metadata is None else comparison_metadata,
            )
        )
    return result


_PORTFOLIO_COMPARISON_COLUMNS = (
    "portfolio_id",
    "comparison_role",
    "sample_role",
    "status",
    "reason",
    "bootstrap_status",
    "bootstrap_reason",
    "fdr_method",
    "global_terminal_wealth_reason",
    "quarterly_terminal_wealth_reason",
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
)


def _group_portfolio_comparison(matrix, config):
    """Compare both frozen groups on the complete, already validated daily grid."""
    statistics = config["statistics"]
    declaration = statistics.get("group_comparison")
    if declaration is None:
        return pd.DataFrame(columns=_PORTFOLIO_COMPARISON_COLUMNS)
    candidate_ids = declaration["portfolio_ids"]
    columns = [
        f"{group}_{candidate}"
        for candidate in candidate_ids
        for group in ("global", "quarterly")
    ]
    if (
        set(columns) != set(matrix.columns)
        or np.isinf(matrix.to_numpy(dtype="float64")).any()
    ):
        raise ValueError("portfolio_group_comparison_coverage_invalid")
    differences = pd.DataFrame(
        {
            "timestamp": matrix.index.get_level_values("end_at"),
            **{
                index: (
                    matrix[f"quarterly_{candidate}"] - matrix[f"global_{candidate}"]
                ).to_numpy()
                for index, candidate in enumerate(candidate_ids)
            },
        }
    )
    access = config["sample_role"]
    inference_blocker = None
    fdr_method = (
        None if _inference_blocker(config) is not None else declaration["fdr_method"]
    )
    undefined_pairs = {
        index for index in range(len(candidate_ids)) if differences[index].isna().any()
    }
    inference_differences = differences.copy()
    for index in undefined_pairs:
        inference_differences[index] = np.nan
    inference = (
        time_series_inference_summary(
            inference_differences,
            identity_column="timestamp",
            max_lags=statistics["hac_lag"],
            fdr_method=fdr_method,
            fdr_alpha=declaration["alpha"],
            fdr_family_size=len(candidate_ids),
        )
        .set_index("timestamp")
        .set_axis(candidate_ids)
        if inference_blocker is None
        else None
    )
    rows = []
    for index, candidate in enumerate(candidate_ids):
        values = differences[index].to_numpy(dtype="float64")
        row = {
            "portfolio_id": candidate,
            "comparison_role": "primary"
            if candidate == declaration["primary_portfolio_id"]
            else "secondary",
            "sample_role": access,
            "observation_count": len(values),
            "global_mean": float(mean(matrix[f"global_{candidate}"])),
            "quarterly_mean": float(mean(matrix[f"quarterly_{candidate}"])),
            "mean_difference": float(mean(values)),
            "global_terminal_wealth": float(
                (1 + matrix[f"global_{candidate}"]).prod(skipna=False)
            ),
            "quarterly_terminal_wealth": float(
                (1 + matrix[f"quarterly_{candidate}"]).prod(skipna=False)
            ),
            "hac_lag": statistics["hac_lag"],
            "bootstrap_block_size": statistics["bootstrap_block"],
            "bootstrap_repetitions": statistics["bootstrap_repetitions"],
            "bootstrap_seed": declaration["seed"],
            "confidence": declaration["confidence"],
            "fdr_family_size": len(candidate_ids),
            "fdr_method": fdr_method or "NOT_APPLICABLE",
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
        if index in undefined_pairs:
            row["reason"] = row["bootstrap_reason"] = "nonpositive_start_nav"
        elif inference is not None:
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
        for group in ("global", "quarterly"):
            column = f"{group}_terminal_wealth"
            row[f"{column}_reason"] = (
                ""
                if np.isfinite(row[column])
                else "nonpositive_start_nav"
                if matrix[f"{group}_{candidate}"].isna().any()
                else "nonfinite_result"
            )
            if not np.isfinite(row[column]):
                row[column] = np.nan
        rows.append(row)
    return pd.DataFrame.from_records(
        rows,
        columns=_PORTFOLIO_COMPARISON_COLUMNS,
    )


def portfolio_statistics(config):
    """Publish statistics for explicitly chosen A-share ledgers and market inputs."""
    from pathlib import Path

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import write_statistics_artifact
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.a_share import (
        portfolio_sessions,
    )
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )
    from quant_research_platform.storage.atomic import reject_if_exists

    config = qrp_json_mapping_snapshot(config)
    required = {
        "ledgers",
        "output",
        "window",
    }
    optional = {
        "sample_role",
        "inputs",
        "benchmark_total_return_indices",
        "risk",
        "adv_sensitivities",
        "statistics",
        "analyses",
        "selection_is_exploratory",
        "expectations",
        "feature_recipe",
        "training_weighting",
        "comparison_benchmarks",
        "comparison_controls",
        "market_exposure",
        "regime_comparison",
    }
    analyses = selected_portfolio_analyses(config, _PORTFOLIO_ANALYSES)
    if "attribution" in analyses:
        required.update(("inputs", "risk", "benchmark_total_return_indices"))
    if "capacity" in analyses:
        required.add("adv_sensitivities")
    if "cscv" in analyses:
        required.add("statistics")
    if required - set(config) or set(config) - required - optional:
        raise ValueError("portfolio_statistics_config_invalid")
    validate_comparison_config(config, analyses)
    config.setdefault("sample_role", "exploratory")
    if type(config["sample_role"]) is not str or not config["sample_role"].strip():
        raise ValueError("statistics_sample_role_invalid")
    if not isinstance(config["ledgers"], list) or not config["ledgers"]:
        raise ValueError("portfolio_statistics_ledgers_required")
    policy = config.get("statistics", {})
    paired = isinstance(policy, dict) and "group_comparison" in policy
    expected = ({"cscv_blocks"} if "cscv" in analyses else set()) | (
        {"hac_lag", "bootstrap_block", "bootstrap_repetitions", "group_comparison"}
        if paired and "group_comparison" in analyses
        else set()
    )
    known = {
        "cscv_blocks",
        "hac_lag",
        "bootstrap_block",
        "bootstrap_repetitions",
        "group_comparison",
    }
    if "group_comparison" in analyses and not paired:
        raise ValueError("portfolio_statistics_comparison_policy_required")
    if not isinstance(policy, dict) or expected - set(policy) or set(policy) - known:
        raise ValueError("portfolio_statistics_policy_invalid")
    if (
        paired
        and "group_comparison" in analyses
        and (
            not isinstance(policy["group_comparison"], dict)
            or set(policy["group_comparison"])
            != {
                "portfolio_ids",
                "primary_portfolio_id",
                "fdr_method",
                "alpha",
                "seed",
                "confidence",
            }
        )
    ):
        raise ValueError("portfolio_statistics_comparison_policy_invalid")
    participation = config.get("adv_sensitivities", [])
    if "capacity" in analyses and (
        not isinstance(participation, list)
        or not participation
        or any(
            type(value) not in {int, float}
            or not np.isfinite(value)
            or not 0 < value <= 1
            for value in participation
        )
        or len(set(participation)) != len(participation)
    ):
        raise ValueError("portfolio_statistics_adv_sensitivities_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    window = TimeWindow(**config["window"])
    ledgers, candidates = {}, []
    for path in config["ledgers"]:
        ledger = read_portfolio_ledger_artifact(Path(path))
        values = dict(ledger.metadata["candidate"])
        values["risk_style_columns"] = tuple(values.get("risk_style_columns", ()))
        candidate = PortfolioCandidateSpec(**values)
        if (
            candidate.candidate_id in ledgers
            or ledger.capital.candidate_id != candidate.candidate_id
        ):
            raise ValueError("portfolio_statistics_candidate_set_invalid")
        if TimeWindow(**ledger.metadata["window"]) != window:
            raise ValueError("portfolio_statistics_ledger_window_mismatch")
        ledgers[candidate.candidate_id] = ledger
        candidates.append(candidate)
    comparison_metadata = {}
    tables = build_portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=tuple(candidates),
        config={
            **config,
            "statistics": policy,
            "execution_assumptions": {"adv_sensitivities": participation},
        },
        time_window=window,
        unified_trading_sessions=unified_trading_sessions,
        inputs=config.get("inputs", {}),
        iter_portfolio_sessions=portfolio_sessions.iter_portfolio_sessions,
        comparison_metadata=comparison_metadata,
    )
    metadata = {key: value for key, value in config.items() if key != "output"}
    if {"sharpe_inference", "cscv", "group_comparison"}.intersection(analyses):
        metadata["inference_scope"] = "exploratory"
        metadata["formal_selection_qualification"] = "not_established"
    if "attribution" in analyses:
        metadata["risk"] = asdict(FactorRiskConfig(**config["risk"]))
    metadata.update(comparison_metadata)
    metadata["analyses"] = list(analyses)
    metadata["execution_mode"] = "continuous"
    metadata["candidate_ids"] = list(ledgers)
    metadata["comparison_portfolio_ids"] = policy.get("group_comparison", {}).get(
        "portfolio_ids", []
    )
    return write_statistics_artifact(
        output, metadata=metadata, tables=tables, table_group="portfolio"
    )


def crypto_statistics(config):
    """Calculate economic statistics from one explicit native Crypto account."""
    from pathlib import Path

    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
        summarize_crypto_ledger,
    )
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import (
        STATISTICS_TABLE_COLUMNS,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.contracts.portfolios import PortfolioCapital
    from quant_research_platform.contracts.time import (
        TimeWindow,
        time_window_timestamp_interval,
    )
    from quant_research_platform.storage.atomic import reject_if_exists
    from quant_research_platform.storage.manifests import read_qrp_json_file

    config = qrp_json_mapping_snapshot(config)
    if {"ledger", "output"} - set(config) or set(config) - {
        "ledger",
        "output",
        "statistics",
        "analyses",
        "comparison_benchmarks",
        "comparison_controls",
        "market_exposure",
        "regime_comparison",
    }:
        raise ValueError("crypto_statistics_config_invalid")
    analyses = selected_portfolio_analyses(config, _CRYPTO_ANALYSES)
    validate_comparison_config(config, analyses)
    policy = config.get("statistics", {})
    expected = set()
    if "annualized" in analyses:
        expected.update(("annualization_minimum_observations", "annualization_periods"))
    if "cost_sensitivity" in analyses:
        expected.update(("fee_scenarios_bps", "additional_spread_scenarios_bps"))
    known = {
        "annualization_minimum_observations",
        "annualization_periods",
        "fee_scenarios_bps",
        "additional_spread_scenarios_bps",
    }
    if not isinstance(policy, dict) or expected - set(policy) or set(policy) - known:
        raise ValueError("crypto_statistics_policy_invalid")
    if (
        "annualized" in analyses or set(analyses).intersection(COMPARISON_ANALYSES)
    ) and (
        type(policy.get("annualization_minimum_observations", 2)) is not int
        or policy.get("annualization_minimum_observations", 2) < 2
        or type(policy.get("annualization_periods", 365.0)) not in {int, float}
        or not np.isfinite(policy.get("annualization_periods", 365.0))
        or policy.get("annualization_periods", 365.0) <= 0
    ):
        raise ValueError("crypto_statistics_annualization_invalid")
    output = Path(config["output"])
    reject_if_exists(output)
    source = Path(config["ledger"])
    _, manifest = read_qrp_json_file(source / "manifest.json")
    metadata = manifest["metadata"]
    capital = PortfolioCapital(**manifest["capital"])
    mode = metadata["execution_mode"]
    expected = "qrp-portfolio-ledger" if mode == "continuous" else "qrp-crypto-episodes"
    if (
        mode not in {"continuous", "independent_segments"}
        or manifest.get("schema_id") != expected
    ):
        raise ValueError("crypto_statistics_execution_mode_invalid")
    window = TimeWindow(**metadata["window"])
    start, stop, inclusive = time_window_timestamp_interval(window)
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(stop).value - (0 if inclusive else 1)
    declared_windows = metadata.get("oos_windows_ns", [[start_ns, end_ns]])
    if (
        not isinstance(declared_windows, list)
        or not declared_windows
        or any(
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not start_ns <= bounds[0] <= bounds[1] <= end_ns
            for bounds in declared_windows
        )
        or any(
            left[1] >= right[0]
            for left, right in zip(declared_windows, declared_windows[1:], strict=False)
        )
    ):
        raise ValueError("crypto_statistics_windows_invalid")
    folds = (metadata.get("chronology") or {}).get("folds", [])
    windows = tuple(
        (
            folds[-1]["fold_id"]
            if len(declared_windows) == 1 and folds
            else f"window_{index}",
            TimeWindow(pd.Timestamp(begin), pd.Timestamp(end)),
        )
        for index, (begin, end) in enumerate(declared_windows)
    )
    science = {
        **metadata,
        **{
            key: value
            for key, value in config.items()
            if key
            in {
                "comparison_benchmarks",
                "comparison_controls",
                "market_exposure",
                "regime_comparison",
            }
        },
        "candidate_id": capital.candidate_id,
        "valuation_currency": capital.valuation_currency,
        "statistics": policy,
        "analyses": list(analyses),
    }
    tables = build_crypto_portfolio_statistics_tables(
        ledger_path=source,
        config=science,
        windows=windows,
        read_ledger=read_portfolio_ledger_artifact,
        read_episodes=read_crypto_episode_artifact,
        summarize_ledger=summarize_crypto_ledger,
        table_columns=STATISTICS_TABLE_COLUMNS,
    )
    return write_statistics_artifact(
        output,
        metadata={
            "ledger": str(source),
            "window": metadata["window"],
            "sample_role": metadata["sample_role"],
            "execution_mode": mode,
            "candidate_ids": [capital.candidate_id],
            "statistics": policy,
            "analyses": list(analyses),
            **{
                key: value
                for key, value in science.items()
                if key
                in {
                    "comparison_benchmarks",
                    "comparison_controls",
                    "market_exposure",
                    "regime_comparison",
                    "comparison_method",
                }
            },
        },
        tables=tables,
        table_group="crypto_portfolio",
    )
