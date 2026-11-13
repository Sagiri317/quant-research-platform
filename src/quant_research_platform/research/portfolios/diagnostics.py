"""Compute ledger-based performance, attribution, valuation, and portfolio
comparisons.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS
from quant_research_platform.compute.portfolio_economics import (
    _LedgerEconomics,
    _nav_return_roundoff,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
from quant_research_platform.contracts.risk import FactorRiskConfig, FactorRiskModel
from quant_research_platform.markets.implementations.a_share import ids as a_share_ids
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.accounting import (
    capacity_quantiles,
    portfolio_interval_economics,
)
from quant_research_platform.research.portfolios.attribution import (
    HoldingFactorInputs,
    holding_attribution,
)
from quant_research_platform.research.portfolios.comparison import (
    COMPARISON_ANALYSES,
    _complete_daily,
    comparison_tables,
    selected_portfolio_analyses,
    validate_comparison_config,
)
from quant_research_platform.research.portfolios.comparison import (
    endpoint_return as comparison_endpoint_return,
)
from quant_research_platform.research.portfolios.performance import performance_rows
from quant_research_platform.research.portfolios.schedules import holding_intervals
from quant_research_platform.research.protocols.a_share import _inference_blocker
from quant_research_platform.research.risk import (
    FactorRiskState,
)
from quant_research_platform.research.statistics import (
    _paired_mean_inference,
    _return_sharpes,
    _validate_bootstrap_configuration,
    _validate_cscv_slice_count,
    cscv_pbo,
    probabilistic_and_deflated_sharpe,
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


def _research_valuation_observation(batch, economics):
    """Summarize explicit research marks without changing the execution path."""
    facts = {(item.asset_id, item.name): item for item in batch.diagnostics}
    policy = facts.get((PORTFOLIO_CASH_ASSET_ID, "research_valuation_policy"))
    if policy is None:
        return None
    values = {
        phase: {
            row.asset_id: row.share_value
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
    from quant_research_platform.research.portfolios.daily import (
        _ENHANCED_PANEL_COLUMNS,
        _iter_input_sessions,
        _risk_context_input_sessions,
        _risk_exposure_cross,
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
                    panel_columns=(
                        *(
                            column
                            for column in _ENHANCED_PANEL_COLUMNS
                            if column
                            not in {
                                "open",
                                "close",
                                "amount",
                                "period_return",
                                "is_nonfinancial_industry",
                            }
                        ),
                        risk.config.weight_column,
                    ),
                    mark_price_columns=("adj_open", "adj_close"),
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
                    cross = _risk_exposure_cross(
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
                    prepared = risk.prepare_exposure_session(
                        cross, include_ineligible=True
                    )
                    holding_cross = prepared.frame
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
                            prepared=prepared,
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
            yield result
            panel_window.pop(interval.decision_at, None)
            if benchmark is not None and stamp(benchmark) == interval.decision_at:
                benchmark = next(benchmarks, None)


def _portfolio_risk_models(ledgers, candidates, config):
    """Inherit realized risk definitions; never silently overwrite ledger facts."""
    declared = (
        FactorRiskModel.from_mapping(config["risk_model"])
        if "risk_model" in config
        else None
    )
    legacy = FactorRiskConfig(**config["risk"]) if "risk" in config else None
    result = {}
    for candidate in candidates:
        metadata = ledgers[candidate.candidate_id].metadata
        if "risk_model" in metadata:
            model = FactorRiskModel.from_mapping(metadata["risk_model"])
        elif "risk_config" in metadata:
            model = FactorRiskModel(
                candidate.risk_style_columns,
                FactorRiskConfig(**metadata["risk_config"]),
            )
        else:
            if (
                candidate.construction == "benchmark_enhanced"
                and declared is None
                and legacy is None
            ):
                raise ValueError("portfolio_attribution_risk_model_required")
            model = declared or FactorRiskModel(
                candidate.risk_style_columns, legacy or FactorRiskConfig()
            )
        if (
            model.style_columns != candidate.risk_style_columns
            or declared is not None
            and model != declared
            or legacy is not None
            and model.config != legacy
        ):
            raise ValueError("portfolio_attribution_risk_model_conflict")
        result[candidate.candidate_id] = model
    return result


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
    from quant_research_platform.research.portfolios.daily import (
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
                ) or any(
                    getattr(pair[0], field) != getattr(pair[1], field)
                    for field, modes in (
                        ("size_cap_column", {"size", "industry_size"}),
                        ("size_neutrality_limit", {"size", "industry_size"}),
                        ("industry_neutrality_limit", {"industry", "industry_size"}),
                    )
                    if pair[0].neutrality in modes
                ):
                    raise ValueError("portfolio_group_comparison_strategy_mismatch")
    pieces = []

    calendar = (
        _read_materialized(
            inputs["calendar"],
            ("calendar_date", "source_market_type", "is_trading_day"),
        )
        if "calendar" in inputs
        and set(analyses).intersection(
            {
                "attribution",
                "performance",
                "sharpe_inference",
                "cscv",
                "group_comparison",
                "benchmark_comparison",
                "control_comparison",
                "market_exposure",
            }
        )
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
        benchmark.loc[missing, "open"] = (
            required.total_close * (required.open / required.close)
        ).to_numpy()
    benchmark = benchmark.loc[
        benchmark.source_asset_id.isin(codes.values()),
        ["source_asset_id", "timestamp", "open"],
    ].copy()
    benchmark["benchmark_id"] = benchmark.source_asset_id.map(
        {code: name for name, code in codes.items()}
    )
    if "attribution" in analyses and calendar is None:
        raise ValueError("portfolio_statistics_attribution_calendar_required")
    risk_models = (
        _portfolio_risk_models(ledgers, candidates, config)
        if "attribution" in analyses
        else {}
    )
    if comparison_metadata is not None and risk_models:
        comparison_metadata["risk_models"] = {
            candidate_id: model.to_dict() for candidate_id, model in risk_models.items()
        }
    holding_inputs = {
        model: iter_holding_factor_inputs(
            iter_portfolio_sessions=iter_portfolio_sessions,
            inputs=inputs,
            calendar=calendar,
            time_window=time_window,
            unified_trading_sessions=unified_trading_sessions,
            style_columns=model.style_columns,
            risk_config=model.config,
            benchmark_required=any(
                candidate.benchmark_id is not None
                for candidate in candidates
                if risk_models[candidate.candidate_id] == model
            ),
        )
        for model in dict.fromkeys(risk_models.values())
    }
    return _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        config=config,
        time_window=time_window,
        benchmark_opens=benchmark,
        calendar=calendar,
        unified_trading_sessions=unified_trading_sessions,
        holding_inputs=holding_inputs,
        risk_models=risk_models,
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
    risk_models=None,
):
    """Project the protocol-selected ledgers into interval-aligned diagnostics."""

    analyses = selected_portfolio_analyses(config, _PORTFOLIO_ANALYSES)
    benchmark_requested = bool({"attribution"} & set(analyses))
    ids = tuple(ledgers)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    sessions = (
        pd.DatetimeIndex(unified_trading_sessions(calendar))
        if calendar is not None
        else None
    )
    expected_intervals = (
        holding_intervals(sessions, time_window)
        if "attribution" in analyses and sessions is not None
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
            adv_unavailable_buy_target_notional=[],
            adv_unavailable_sell_target_notional=[],
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
            interval_economics = tuple(map(portfolio_interval_economics, batch_group))
            interval_inputs = dict(zip(fit_streams, group[len(streams) :], strict=True))
            if expected_intervals is not None:
                expected = expected_intervals.iloc[interval_index]
            else:
                first = interval_economics[0]
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
                            attribution_group=canonical_json_hash(
                                styles.to_dict()
                                if isinstance(styles, FactorRiskModel)
                                else list(styles)
                            ),
                            decision_at=holding_input.decision_at,
                            entry_at=holding_input.entry_at,
                            end_at=holding_input.end_at,
                            factor_id=name,
                            value=float(value),
                            observation_count=len(holding_input.fitted.asset_returns),
                        )
                        for name, value in holding_input.fitted.factor_returns.items()
                    )
            for candidate_id, batch, row in zip(
                ids, batch_group, interval_economics, strict=True
            ):
                if "attribution" in analyses:
                    candidate = by_id[candidate_id]
                    holding_input = interval_inputs[
                        risk_models[candidate_id]
                        if risk_models is not None
                        else candidate.risk_style_columns
                    ]
                previous_end, previous_nav = previous[candidate_id]
                economic_totals[candidate_id].consume(batch)
                if (row["entry_at"], row["end_at"]) != (
                    expected.entry_at,
                    expected.end_at,
                ):
                    raise ValueError("portfolio_statistics_window_incomplete")
                if previous_end is not None and row["entry_at"] != previous_end:
                    raise ValueError("portfolio_statistics_interval_gap")
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
                            value.asset_id: value.share_value
                            for value in batch.valuations
                            if value.phase == "after"
                            and value.asset_id != PORTFOLIO_CASH_ASSET_ID
                        }
                    )
                    endpoint = pd.Series(
                        {
                            value.asset_id: value.share_value
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
                    # Validated registration records only snapshot entitlement:
                    # shares, cash, locked shares and receivables are unchanged.
                    if any(
                        adjustment.kind != "registration"
                        for adjustment in batch.rights_adjustments
                    ):
                        holding_input = replace(
                            holding_input,
                            fitted=None,
                            reason="corporate_action_adjustment",
                        )
                    if (
                        any(
                            adjustment.kind == "recovery"
                            for adjustment in batch.rights_adjustments
                        )
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
                                if component.startswith("factor:")
                                or component
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
                        and (
                            item.name.startswith(
                                (
                                    "target_exposure:",
                                    "realized_exposure:",
                                    "benchmark_exposure:",
                                )
                            )
                            or item.name
                            in {
                                "target_unknown_industry_weight",
                                "realized_unknown_industry_weight",
                                "benchmark_unknown_industry_weight",
                            }
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
                        desired = (
                            facts[value.asset_id, "target_notional"].number
                            - value.share_value
                        )
                        side = "buy" if desired > 0 else "sell"
                        adv_gaps[candidate_id][f"adv_unavailable_{side}_orders"] += 1
                        adv_gaps[candidate_id][
                            f"adv_unavailable_{side}_target_notional"
                        ].append(abs(desired))
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

    for candidate_id in ids:
        artifact = ledgers[candidate_id]
        previous_end, previous_nav = previous[candidate_id]
        candidate_rows = daily_by_candidate[candidate_id]
        selected = pd.DataFrame(candidate_rows)
        if selected.empty:
            raise ValueError("portfolio_statistics_empty_candidate")
        if not {"performance"} & set(analyses):
            continue
        daily_complete = _complete_daily(
            selected.rename(columns={"entry_at": "start_at"}),
            "trading_day_open",
            sessions,
        )
        values = {}
        for name, value in adv_gaps[candidate_id].items():
            values[name] = float(np.sum(value)) if isinstance(value, list) else value
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
                ratio = (
                    float(stressed.iloc[-1]) - artifact.capital.initial_capital
                ) / artifact.capital.initial_capital
                values[f"{prefix}_capital_return"] = (
                    ratio if np.isfinite(ratio) else np.nan
                )
                values[f"{prefix}_max_drawdown"] = float(
                    (
                        stressed
                        / stressed.cummax().clip(lower=artifact.capital.initial_capital)
                        - 1
                    ).min()
                )
        summary_rows.extend(
            performance_rows(
                selected,
                capital=artifact.capital,
                economic=economic_totals[candidate_id].summary(),
                daily_complete=daily_complete,
                sample_role=access,
                extra_values=values,
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
    daily_complete = _complete_daily(
        matrix.index.to_frame(index=False).rename(columns={"entry_at": "start_at"}),
        "trading_day_open",
        sessions,
    )
    paired_comparison = (
        _group_portfolio_comparison(matrix, config, daily_complete=daily_complete)
        if "group_comparison" in analyses
        else pd.DataFrame(
            columns=STATISTICS_TABLE_COLUMNS["portfolio_group_comparison.parquet"]
        )
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
            previous[f"{group}_{portfolio_id}"][1]
            / ledgers[f"{group}_{portfolio_id}"].capital.initial_capital
            for portfolio_id in paired_comparison.portfolio_id
        ]
        column = f"{group}_terminal_wealth"
        valid = np.isfinite(paired_comparison[column])
        paired_comparison.loc[~valid, column] = np.nan
        paired_comparison[f"{column}_reason"] = np.where(valid, "", "nonfinite_result")
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
            if not daily_complete:
                outputs = tuple(
                    (name, None, "incomplete_daily_axis") for name in ("psr", "dsr")
                )
            elif inference_matrix[candidate_id].isna().any():
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
            if not daily_complete:
                value, status, reason = np.nan, "UNDEFINED", "incomplete_daily_axis"
            elif not np.isfinite(inference_matrix.to_numpy()).all():
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
        columns=STATISTICS_TABLE_COLUMNS["portfolio_capacity.parquet"],
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
            columns=STATISTICS_TABLE_COLUMNS["portfolio_exposures.parquet"],
        ),
        "holding_factor_returns": pd.DataFrame(
            factor_rows,
            columns=STATISTICS_TABLE_COLUMNS["holding_factor_returns.parquet"],
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
                sessions=sessions,
                metadata={} if comparison_metadata is None else comparison_metadata,
            )
        )
    return result


def _group_portfolio_comparison(matrix, config, *, daily_complete):
    """Keep paired descriptions; infer only on a caller-verified daily grid."""
    columns = STATISTICS_TABLE_COLUMNS["portfolio_group_comparison.parquet"]
    statistics = config["statistics"]
    declaration = statistics.get("group_comparison")
    if declaration is None:
        return pd.DataFrame(columns=columns)
    candidate_ids = declaration["portfolio_ids"]
    expected = [
        f"{group}_{candidate}"
        for candidate in candidate_ids
        for group in ("global", "quarterly")
    ]
    if (
        set(expected) != set(matrix.columns)
        or np.isinf(matrix.to_numpy(dtype="float64")).any()
    ):
        raise ValueError("portfolio_group_comparison_coverage_invalid")
    differences = pd.DataFrame(
        {
            candidate: (
                matrix[f"quarterly_{candidate}"] - matrix[f"global_{candidate}"]
            ).to_numpy()
            for candidate in candidate_ids
        },
        index=matrix.index.get_level_values("end_at"),
    )
    inference = _paired_mean_inference(
        differences,
        blockers={
            candidate: "incomplete_daily_axis"
            if not daily_complete
            else "nonpositive_start_nav"
            if differences[candidate].isna().any()
            else None
            for candidate in candidate_ids
        },
        max_lags=statistics["hac_lag"],
        block_size=statistics["bootstrap_block"],
        repetitions=statistics["bootstrap_repetitions"],
        seed=declaration["seed"],
        confidence=declaration["confidence"],
        fdr_method=None
        if _inference_blocker(config) is not None
        else declaration["fdr_method"],
        fdr_alpha=declaration["alpha"],
    )
    rows = []
    for candidate in candidate_ids:
        pair = matrix[[f"global_{candidate}", f"quarterly_{candidate}"]]
        paired = np.isfinite(pair).all(axis=1)
        pair = pair.loc[paired]
        values = differences[candidate].to_numpy(dtype="float64")
        rows.append(
            {
                **inference[candidate],
                "portfolio_id": candidate,
                "comparison_role": "primary"
                if candidate == declaration["primary_portfolio_id"]
                else "secondary",
                "sample_role": config["sample_role"],
                "observation_count": len(values),
                "global_mean": float(pair.iloc[:, 0].mean()),
                "quarterly_mean": float(pair.iloc[:, 1].mean()),
                "mean_difference": float(values[paired].mean())
                if paired.any()
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


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
        "risk_model",
        "risk_exposure_input",
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
        required.add("inputs")
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
    if paired and "group_comparison" in analyses:
        policy.setdefault("bootstrap_repetitions", 2000)
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
    if "cscv" in analyses:
        _validate_cscv_slice_count(policy["cscv_blocks"])
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
    if paired and "group_comparison" in analyses:
        declaration = policy["group_comparison"]
        if (
            type(policy["hac_lag"]) is not int
            or policy["hac_lag"] < 0
            or type(declaration["confidence"]) is not float
            or not 0 < declaration["confidence"] < 1
            or type(declaration["alpha"]) is not float
            or not 0 < declaration["alpha"] < 1
            or declaration["fdr_method"] not in ("BH", "BY")
        ):
            raise ValueError("portfolio_statistics_comparison_policy_invalid")
        _validate_bootstrap_configuration(
            policy["bootstrap_block"],
            policy["bootstrap_repetitions"],
            declaration["seed"],
        )
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
        if TimeWindow(**ledger.metadata["window"]) != window:
            raise ValueError("portfolio_statistics_ledger_window_mismatch")
        candidate_id = ledger.capital.candidate_id
        if candidate_id in ledgers:
            raise ValueError("portfolio_statistics_candidate_set_invalid")
        ledgers[candidate_id] = ledger
        if ledger.metadata.get("kind") == "a_share_hft" or not {
            "attribution",
            "group_comparison",
        }.intersection(analyses):
            continue
        if "candidate" not in ledger.metadata:
            raise ValueError("portfolio_statistics_candidate_metadata_required")
        values = dict(ledger.metadata["candidate"])
        values["risk_style_columns"] = tuple(values.get("risk_style_columns", ()))
        candidate = PortfolioCandidateSpec(**values)
        if ledger.capital.candidate_id != candidate.candidate_id:
            raise ValueError("portfolio_statistics_candidate_set_invalid")
        candidates.append(candidate)
    comparison_metadata = {}
    hft_clocks = {
        ledger.metadata.get("kind") == "a_share_hft" for ledger in ledgers.values()
    }
    if len(hft_clocks) != 1:
        raise ValueError("portfolio_statistics_account_clocks_mixed")
    if True in hft_clocks:
        from quant_research_platform.research.portfolios.high_frequency import (
            statistics_tables,
        )

        tables = statistics_tables(
            ledgers, window=window, config=config, metadata=comparison_metadata
        )
        comparison_metadata.update(
            observation_clock="trading_day_close",
            cash_observation="end_of_day",
            turnover_semantics="sum_of_intraday_requests_and_fills_over_start_nav",
            annualization="complete_trading_days_only",
        )
    else:
        if "attribution" in analyses and "benchmark_total_return_indices" not in config:
            raise ValueError("portfolio_statistics_config_invalid")
        if "risk_exposure_input" in config:
            raise ValueError("portfolio_statistics_config_invalid")
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
    if "attribution" in analyses and "risk" in config:
        metadata["risk"] = asdict(FactorRiskConfig(**config["risk"]))
    metadata.update(comparison_metadata)
    metadata["analyses"] = list(analyses)
    metadata["candidate_ids"] = list(ledgers)
    metadata["selection_modes"] = {
        candidate_id: ledger.metadata.get("selection_mode")
        for candidate_id, ledger in ledgers.items()
    }
    industry_domains = {
        candidate_id: ledger.metadata["industry_neutrality_domain"]
        for candidate_id, ledger in ledgers.items()
        if "industry_neutrality_domain" in ledger.metadata
    }
    if industry_domains:
        metadata["industry_neutrality_domains"] = industry_domains
    metadata["comparison_portfolio_ids"] = policy.get("group_comparison", {}).get(
        "portfolio_ids", []
    )
    return write_statistics_artifact(
        output, metadata=metadata, tables=tables, table_group="portfolio"
    )
