"""Advance A-share cash, shares, entitlements, and valuation across market sessions.

Decision inputs use bounded historical reads. Subsequent sessions own execution
and settlement, and every transition yields a native portfolio ledger batch."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.policies import PolicyDecision, policy_target
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_DECISION_REASON_BENCHMARK,
    PORTFOLIO_DECISION_REASON_ENHANCED,
    PORTFOLIO_DECISION_REASON_MAINTENANCE,
    PORTFOLIO_DECISION_REASON_POLICY,
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    RESEARCH_VALUATION_POLICY,
    ExecutedTrade,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioTargetBlocked,
    PortfolioValuation,
    _portfolio_session_frame,
    portfolio_notional_tolerance,
)
from quant_research_platform.contracts.time import (
    DECISION_CONTEXT_COLUMNS,
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_finite_float,
    strict_real_numeric_series,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    stock_order_limits,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    _industry_group_codes,
)
from quant_research_platform.research.filtering import (
    A_SHARE_ELIGIBILITY_FACTS,
    a_share_eligibility,
)
from quant_research_platform.research.portfolios.accounting import (
    EquityEntitlements,
    ExecutionAccounting,
    ResearchValuationMarks,
    execute_stock_target,
)
from quant_research_platform.research.portfolios.calibration import (
    AlphaCalibration,
    alpha_calibration_intervals,
    alpha_calibration_start,
    alpha_interval_semantics,
    calibrate_expected_alpha,
)
from quant_research_platform.research.portfolios.construction import (
    _neutrality_context,
    _research_neutrality_domain,
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    _neutrality_bands,
    _neutrality_risk_limits,
    _portfolio_risk_limits,
    _project_neutral_target,
    optimize_portfolio,
)
from quant_research_platform.research.portfolios.schedules import (
    holding_intervals,
    is_rebalance_session,
)
from quant_research_platform.research.risk import (
    FactorExposureSession,
    FactorRiskConfig,
    FactorRiskEstimate,
    FactorRiskState,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    _iter_dataset_session_frames,
    materialize_dataset_session,
    open_dataset_session,
)

_CALENDAR_COLUMNS = ("calendar_date", "source_market_type", "is_trading_day")


_FEE_COLUMNS = (
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


_DATASET_COLUMNS = ("asset_id", "timestamp", "eligible")


_PANEL_COLUMNS = (
    "asset_id",
    "listing_date",
    "timestamp",
    "open",
    "close",
    "amount",
    "market_type",
    "is_trading",
)


_ENHANCED_PANEL_COLUMNS = (
    "asset_id",
    "listing_date",
    "timestamp",
    "open",
    "close",
    "adj_open",
    "adj_close",
    "amount",
    "market_type",
    "period_return",
    "industry_standard_id",
    "industry_level1_code",
    "is_nonfinancial_industry",
    "is_trading",
    *tuple(
        column
        for column in A_SHARE_ELIGIBILITY_FACTS
        if column not in {"adj_close", "close", "amount", "is_trading"}
    ),
)


_BENCHMARK_COLUMNS = ("benchmark_id", "asset_id", "timestamp", "weight")


_SESSION_KEY = ("timestamp", "asset_id")


_NEUTRALITY_REFERENCE_DOMAIN = "capitalization_available_research_pool"


_INDUSTRY_NEUTRALITY_DOMAIN = "known_industry_subportfolio"


_ALPHA_DIAGNOSTICS = {
    "alpha_status": "status",
    "alpha_undefined_reason": "undefined_reason",
    "alpha_slope": "slope",
    "alpha_intercept": "intercept",
    "alpha_slope_se": "slope_standard_error",
    "alpha_slope_t": "slope_t_statistic",
    "alpha_slope_pvalue": "slope_pvalue",
    "alpha_slope_ci_low": "slope_ci_low",
    "alpha_slope_ci_high": "slope_ci_high",
    "alpha_weighted_r2": "weighted_r2",
    "alpha_inference_status": "inference_status",
    "alpha_inference_reason": "inference_reason",
    "alpha_r2_status": "r2_status",
    "alpha_r2_reason": "r2_reason",
    "alpha_coverage": "coverage",
    "alpha_valid_dates": "valid_rebalance_dates",
    "alpha_scheduled_dates": "scheduled_rebalance_dates",
}


def _repair_stock_allocation(target, hold, benchmark, minimum, increment, evaluate):
    """Keep risk bands first, then stay close to the requested allocation."""
    requested = evaluate(target)
    if requested[2][:2] == (0.0, 0.0):
        return requested
    alternatives = [
        (target, requested),
        (hold, evaluate(hold)),
        (benchmark, evaluate(benchmark)),
    ]
    benchmark_account = alternatives[-1][1][0]
    change = benchmark - hold
    moving = change.ne(0.0)
    if moving.any():
        reached = (
            benchmark_account.asset_ledger.set_index("asset_id").post_trade_notional
            / benchmark_account.nav_before
        ).reindex(target.index)
        scale = ((reached - hold).loc[moving] / change.loc[moving]).min()
        if 0.0 < scale < 1.0:
            # Preserve the stock-leg mix when per-name capacities differ.
            proportional = hold + scale * change
            alternatives.append((proportional, evaluate(proportional)))
    best_weights, best = min(alternatives, key=lambda item: item[1][2])
    if not np.isfinite(best[2][0]):
        return alternatives[1][1]
    if best[2][0] == 0.0:
        # Search the feasible endpoint toward the signal, pricing every rounded fill.
        low, high = 0.0, 1.0
        anchor = best_weights
        steps = increment.where(increment > 0, np.inf)
        while ((target - anchor).abs() * (high - low) >= steps).any():
            middle = (low + high) / 2.0
            trial = evaluate(anchor + middle * (target - anchor))
            if trial[2][0] == 0.0:
                low = middle
                if trial[2] < best[2]:
                    best = trial
            else:
                high = middle
        return best
    # shortcut: local lot improvements are not a global integer feasibility proof;
    # add an integer solver only when an exact discrete optimum is required.
    while True:
        prior_risk = best[2][:2]
        for asset in target.index:
            if minimum[asset] <= 0.0:
                continue
            for direction in (-1, 1):
                lots = 1
                while True:
                    rows = best[0].asset_ledger.set_index("asset_id")
                    weights = (
                        rows.post_trade_notional.reindex(target.index)
                        / best[0].nav_before
                    )
                    step = (
                        increment[asset]
                        if rows.loc[asset, "executed_quantity"] != 0
                        else minimum[asset]
                    )
                    weights.loc[asset] = max(
                        0.0, weights[asset] + direction * step * lots
                    )
                    stock_budget = 1.0 - best[0].receivable / best[0].nav_before
                    shortage = weights.sum() - stock_budget
                    if shortage > 1e-12:
                        # Fund a corrective buy from the largest excess over the signal.
                        donors = (weights - target).drop(asset)
                        donors = donors.loc[minimum.reindex(donors.index).gt(0)]
                        if donors.empty or donors.max() <= 0:
                            break
                        donor = donors.idxmax()
                        donor_step = (
                            increment[donor]
                            if rows.loc[donor, "executed_quantity"] != 0
                            else minimum[donor]
                        )
                        reduction = np.ceil(shortage / donor_step) * donor_step
                        if reduction > weights[donor]:
                            break
                        weights.loc[donor] -= reduction
                    trial = evaluate(weights)
                    if trial[2] >= best[2]:
                        break
                    best = trial
                    if best[2][0] == 0.0:
                        return best
                    # Cross long improving stretches without scanning every lot.
                    lots *= 2
        if best[2][:2] == prior_risk:
            return best


def _unit_values(units, prices):
    """Value actual shares at the supplied raw prices."""
    values = units.astype("float64") * prices.reindex(units.index).astype("float64")
    if not np.isfinite(values).all():
        raise ValueError("portfolio_valuation_unrepresentable")
    return values


def _unquoted_assets(frame, phase="open"):
    return frozenset(
        frame.loc[frame[f"research_{phase}_mark_at"].notna(), "asset_id"].astype(str)
    )


@dataclass(frozen=True, slots=True)
class _EnhancedDecision:
    benchmark_weights: pd.Series
    expected_alpha: pd.Series
    exposures: pd.DataFrame
    risk: FactorRiskEstimate | None
    calibration: AlphaCalibration


def backtest_a_share(
    *,
    candidate: PortfolioCandidateSpec,
    inputs: Mapping[str, object],
    output_window: TimeWindow,
    input_window: TimeWindow,
    formal_dataset_window: TimeWindow,
    prediction_window: TimeWindow | None = None,
    risk_config: FactorRiskConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    frozen_calibration: AlphaCalibration | None = None,
    on_final_calibration=None,
    prediction_lookup: object,
    assumptions: ExecutionAssumptions,
    decision_columns: tuple[str, ...] | None = None,
) -> Iterator[PortfolioLedgerBatch]:
    """Yield daily ledger transitions from bounded market and decision inputs.

    With decision_columns, yield a PolicyDecision before advancing to execution;
    the caller sends an asset-indexed target-weight Series back into the iterator.
    Closing the iterator releases all retained input streams."""
    from quant_research_platform.markets.implementations.a_share import (
        portfolio_sessions,
    )
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )
    from quant_research_platform.markets.implementations.a_share.execution import (
        fee_schedule_from_frame,
        market_name_for_type,
        prepare_trade_cost,
        require_fee_coverage,
    )

    if not isinstance(candidate, PortfolioCandidateSpec):
        raise ValueError("portfolio_candidate_invalid")
    if not isinstance(assumptions, ExecutionAssumptions):
        raise ValueError("portfolio_assumptions_invalid")
    _validate_windows(output_window, input_window, formal_dataset_window)
    enhanced = candidate.construction == "benchmark_enhanced"
    if decision_columns is not None and enhanced:
        raise ValueError("policy_and_enhanced_optimizer_are_separate_decisions")
    risk_style_columns = candidate.risk_style_columns
    calendar_root = inputs["calendar"]
    calendar = _read_materialized(calendar_root, _CALENDAR_COLUMNS)
    try:
        sessions = pd.DatetimeIndex(unified_trading_sessions(calendar))
    except (TypeError, ValueError) as exc:
        raise ValueError("portfolio_trading_calendar_invalid") from exc
    sessions = sessions.normalize().drop_duplicates().sort_values()
    trading_sessions = sessions
    session_start = pd.Timestamp(input_window.start).normalize()
    session_end = min(
        pd.Timestamp(input_window.end).normalize(),
        pd.Timestamp(output_window.end).normalize(),
    )
    output_start = pd.Timestamp(output_window.start).normalize()
    # Consume the declared history. Union-session slicing can discard required
    # own-market observations when exchanges have asynchronous holidays.
    calibration_intervals = alpha_calibration_intervals(
        sessions, candidate.rebalance_schedule
    )
    sessions = sessions[(sessions >= session_start) & (sessions <= session_end)]
    output_intervals = holding_intervals(sessions, output_window).set_index(
        "decision_at"
    )

    if sessions.empty or output_intervals.empty:
        raise PortfolioTargetBlocked("portfolio_execution_window_empty")
    prediction_window = prediction_window or TimeWindow(
        max(
            pd.Timestamp(formal_dataset_window.start),
            alpha_calibration_start(trading_sessions, as_of=output_start),
        ).date(),
        pd.Timestamp(output_window.end).date(),
    )
    dataset_options = _dataset_input_options(inputs["dataset"])
    dataset_options["columns"] = ("timestamp", "asset_id")
    dataset_session = open_dataset_session(**dataset_options)
    declared_columns = dataset_session.spec["column_roles"]
    explicit_decisions = "signal_session" in declared_columns
    dataset_business_columns = (
        ("float_market_cap",)
        if candidate.construction == "quantile_float_cap"
        or (
            candidate.neutrality != "none"
            and candidate.benchmark_id is None
            and "float_market_cap" in declared_columns
        )
        else ()
    )
    if (
        candidate.neutrality in {"size", "industry_size"}
        and candidate.benchmark_id is None
        and candidate.size_cap_column in declared_columns
    ):
        dataset_business_columns = tuple(
            dict.fromkeys((*dataset_business_columns, candidate.size_cap_column))
        )
    if decision_columns is not None:
        if any(
            declared_columns.get(column) not in {"feature", "factor"}
            for column in decision_columns
        ):
            raise ValueError("policy_observation_feature_role_required")
        dataset_business_columns = tuple(
            dict.fromkeys(
                (
                    *dataset_business_columns,
                    *decision_columns,
                )
            )
        )
    if explicit_decisions:
        if not set(DECISION_CONTEXT_COLUMNS) <= set(declared_columns):
            raise ValueError("portfolio_decision_clock_context_missing")
        dataset_business_columns = (
            *dataset_business_columns,
            *DECISION_CONTEXT_COLUMNS,
        )
    dataset_input_columns = (*_DATASET_COLUMNS, *dataset_business_columns)
    dataset_columns = (
        *dataset_input_columns,
        *(("decision_at",) if explicit_decisions else ()),
    )
    alpha_start = max(session_start, pd.Timestamp(prediction_window.start))
    fee_root = inputs["fees"]
    fee_identity = str(_dataset_input_options(fee_root)["dataset_root"])
    fee_frame = _read_materialized(fee_root, _FEE_COLUMNS, business_columns=("rate",))
    fee_schedule = fee_schedule_from_frame(fee_frame, content_identity=fee_identity)

    try:
        require_fee_coverage(
            fee_schedule,
            pd.Timestamp(output_window.start).date(),
            pd.Timestamp(output_window.end).date(),
            markets=("shanghai", "shenzhen"),
        )
    except ValueError as exc:
        raise PortfolioTargetBlocked("portfolio_fee_coverage_missing") from exc
    dataset_root = inputs["dataset"]
    panel_root = inputs["panel"]
    execution_root = inputs["execution"]
    benchmark_root = inputs["benchmark"] if candidate.benchmark_id is not None else None
    holdings = pd.Series(dtype="float64")
    entitlements = EquityEntitlements()
    persisted_endpoint_events = set()
    capital_allocated = False
    held_units = pd.Series(dtype="float64")
    held_lives = pd.Series(dtype="datetime64[ns]")
    previous_values = {}
    cash = float(assumptions.initial_nav)
    nav = cash
    amount_history: dict[tuple[str, pd.Timestamp], deque[float]] = {}
    factor_risk = (
        FactorRiskState(style_columns=risk_style_columns, config=risk_config)
        if enhanced
        else None
    )
    previous_risk_cross: pd.DataFrame | None = None
    previous_exposures: FactorExposureSession | None = None
    calibration_rows: list[dict[str, object]] = []
    pending_calibration: tuple[pd.Series, pd.Series, pd.Timestamp] | None = None
    waiting_alpha: tuple[pd.Series, pd.Timestamp] | None = None
    calibration_dates = calibration_intervals.set_index("timestamp")
    previous_benchmark_assets: frozenset[str] | None = None
    emitted = False
    sequence = 0
    research_marks = ResearchValuationMarks()
    with ExitStack() as input_channels:
        history = TimeWindow(sessions[0], sessions[-1])
        dataset_history = history
        if explicit_decisions:
            successor = trading_sessions.searchsorted(sessions[-1], side="right")
            if successor == len(trading_sessions):
                raise ValueError("portfolio_decision_clock_requires_successor_session")
            dataset_history = TimeWindow(
                sessions[0],
                trading_sessions[successor] + pd.Timedelta(hours=1, minutes=25),
            )
        panel_columns = (
            (*_ENHANCED_PANEL_COLUMNS, factor_risk.config.weight_column)
            if enhanced
            else _PANEL_COLUMNS
        )
        if candidate.neutrality in {"industry", "industry_size"}:
            panel_columns = tuple(
                dict.fromkeys(
                    (*panel_columns, "industry_standard_id", "industry_level1_code")
                )
            )
        if candidate.benchmark_id is not None and candidate.neutrality in {
            "size",
            "industry_size",
        }:
            panel_columns = tuple(
                dict.fromkeys((*panel_columns, candidate.size_cap_column))
            )
        panel_frames = input_channels.enter_context(
            closing_iterator(
                portfolio_sessions.iter_portfolio_sessions(
                    market_bars_root=inputs["market_bars"],
                    stock_master_root=inputs["stock_master"],
                    corporate_actions_root=inputs["corporate_actions"],
                    equity_events_root=inputs.get("equity_events"),
                    panel_root=panel_root,
                    panel_columns=panel_columns,
                    mark_price_columns=("open", "close"),
                    calendar=calendar,
                    time_window=history,
                    execution_root=execution_root,
                )
            )
        )
        dataset_frames = input_channels.enter_context(
            closing_iterator(
                _iter_input_sessions(
                    dataset_root,
                    dataset_input_columns,
                    dataset_history,
                    business_columns=dataset_business_columns,
                    session=dataset_session,
                )
            )
        )
        if explicit_decisions:
            dataset_frames = input_channels.enter_context(
                closing_iterator(
                    _decision_session_frames(dataset_frames, trading_sessions)
                )
            )
        dataset_at = _session_reader(dataset_frames, dataset_columns)
        _, prediction_end, prediction_end_closed = time_window_timestamp_interval(
            prediction_window
        )

        def alpha_for_dataset(dataset):
            clock = dataset.decision_at if explicit_decisions else dataset.timestamp
            selected = clock.ge(alpha_start) & (
                clock.le(prediction_end)
                if prediction_end_closed
                else clock.lt(prediction_end)
            )
            return _prediction_alpha(dataset.loc[selected], prediction_lookup).reindex(
                pd.Index(dataset.asset_id, name="asset_id")
            )

        risk_inputs = (
            _risk_context_input_sessions(
                roots=tuple(inputs.get("risk_context", ())),
                style_columns=risk_style_columns,
                time_window=history,
                retained=input_channels,
            )
            if enhanced
            else {}
        )
        risk_inputs = {
            position: _session_reader(frames, columns)
            for position, (columns, frames) in risk_inputs.items()
        }
        panel_window = {}

        def release_panels(before):
            for timestamp in tuple(panel_window):
                if timestamp < before:
                    del panel_window[timestamp]

        def panel_at(timestamp):
            if timestamp not in panel_window:
                for view in panel_frames:
                    view = replace(view, valuation=research_marks.project(view))
                    panel_window[view.timestamp] = view
                    if view.timestamp == timestamp:
                        break
            return panel_window[timestamp]

        benchmark_frames = (
            input_channels.enter_context(
                closing_iterator(
                    _iter_benchmark_weights(
                        benchmark_root, candidate.benchmark_id, sessions
                    )
                )
            )
            if benchmark_root is not None
            else None
        )
        for position, decision_timestamp in enumerate(sessions):
            next_timestamp = (
                sessions[position + 1]
                if position + 1 < len(sessions)
                else decision_timestamp
            )

            release_panels(decision_timestamp)
            view = panel_at(decision_timestamp)
            panel = view.panel
            if panel.empty and view.valuation.empty:
                raise PortfolioTargetBlocked("portfolio_panel_session_missing")
            _update_amount_history(
                panel,
                amount_history,
                view.closed_assets,
                view.unknown_market_assets,
            )
            benchmark_weights = pd.Series(dtype="float64")
            produces_interval = decision_timestamp in output_intervals.index
            alpha: pd.Series | None = None
            current_cross: pd.DataFrame | None = None
            if enhanced:
                assert factor_risk is not None
                # The mark producer separates normal carries from accounting-only marks.
                calibration_valuation = view.valuation.loc[
                    view.valuation.open.gt(0)
                    & view.valuation.adj_open.notna()
                    & (
                        view.valuation.research_open_mark_at.isna()
                        | view.valuation.research_open_mark_kind.isin(
                            ("open", "close", "corporate_action_carry")
                        )
                    )
                ]
                risk_frames = []
                for context_at in risk_inputs.values() if not panel.empty else ():
                    context = context_at(decision_timestamp)
                    _validate_input_keys(panel, context)
                    risk_frames.append(context)
                current_cross = _enhanced_risk_cross(
                    panel=panel,
                    risk_frames=risk_frames,
                    style_columns=risk_style_columns,
                    weight_column=factor_risk.config.weight_column,
                )
                risk_frames.clear()

                if previous_risk_cross is not None:
                    risk_cross = _factor_return_cross(
                        exposures=previous_risk_cross, panel=panel
                    )
                    factor_risk.observe_return_session(
                        risk_cross,
                        closed_assets=view.closed_assets,
                        prepared=previous_exposures,
                    )
                if previous_risk_cross is not None:
                    carried = previous_risk_cross.loc[
                        previous_risk_cross.asset_id.isin(view.closed_assets)
                    ].copy()
                    carried["timestamp"] = decision_timestamp
                    current_cross = pd.concat(
                        (current_cross, carried), ignore_index=True
                    )
                previous_risk_cross = current_cross.copy(deep=True)
                previous_exposures = factor_risk.prepare_exposure_session(
                    current_cross, include_ineligible=True
                )
                calibration_start = alpha_calibration_start(
                    trading_sessions, as_of=decision_timestamp
                )
                if (
                    calibration_rows
                    and calibration_rows[0]["timestamp"] < calibration_start
                ):
                    calibration_rows[:] = [
                        row
                        for row in calibration_rows
                        if row["timestamp"] >= calibration_start
                    ]
                if (
                    pending_calibration is not None
                    and decision_timestamp
                    == calibration_dates.loc[pending_calibration[2], "available_at"]
                ):
                    calibration_rows.extend(
                        _completed_calibration_rows(
                            pending_calibration, calibration_valuation
                        )
                    )
                    pending_calibration = None
                if (
                    waiting_alpha is not None
                    and decision_timestamp
                    == calibration_dates.loc[waiting_alpha[1], "entry_at"]
                ):
                    pending_calibration = (
                        waiting_alpha[0],
                        calibration_valuation.set_index(["asset_id", "listing_date"])[
                            "adj_open"
                        ].astype("float64"),
                        waiting_alpha[1],
                    )
                    waiting_alpha = None
                calibration_candidate = (
                    frozen_calibration is None
                    and decision_timestamp in calibration_dates.index
                    and calibration_dates.loc[decision_timestamp, "available_at"]
                    <= sessions[-1]
                )
                if produces_interval or (
                    calibration_candidate
                    and (explicit_decisions or decision_timestamp >= alpha_start)
                ):
                    dataset = (
                        dataset_at(decision_timestamp)
                        if not panel.empty
                        else pd.DataFrame(columns=dataset_columns).astype(
                            {"eligible": "bool"}
                        )
                    )
                    if (
                        dataset.empty
                        and not panel.empty
                        and (
                            not explicit_decisions or decision_timestamp >= output_start
                        )
                    ):
                        raise PortfolioTargetBlocked(
                            "portfolio_research_dataset_session_missing"
                        )
                    if not dataset.empty or panel.empty:
                        _validate_input_keys(dataset, panel)
                        alpha = alpha_for_dataset(dataset)
                    if alpha is not None and calibration_candidate:
                        calibration_alpha = alpha.copy(deep=True)
                        calibration_alpha.index = pd.MultiIndex.from_arrays(
                            (
                                alpha.index,
                                panel.set_index("asset_id").listing_date.reindex(
                                    alpha.index
                                ),
                            ),
                            names=("asset_id", "listing_date"),
                        )
                        waiting_alpha = (calibration_alpha, decision_timestamp)
            if benchmark_frames is not None:
                benchmark_weights = next(benchmark_frames)
            if not produces_interval:
                continue
            if not enhanced:
                dataset = (
                    dataset_at(decision_timestamp)
                    if not panel.empty
                    else pd.DataFrame(columns=dataset_columns).astype(
                        {"eligible": "bool"}
                    )
                )
                if dataset.empty and not panel.empty:
                    raise PortfolioTargetBlocked(
                        "portfolio_research_dataset_session_missing"
                    )
            assert dataset is not None
            execution = view.execution
            _validate_input_keys(dataset, panel)
            _validate_input_keys(
                dataset, execution.loc[execution.timestamp.eq(decision_timestamp)]
            )
            if decision_columns is not None:
                alpha = pd.Series(
                    np.nan, index=pd.Index(dataset.asset_id, name="asset_id")
                )
            elif not enhanced:
                alpha = alpha_for_dataset(dataset)
            endpoint_timestamp = output_intervals.loc[decision_timestamp, "end_at"]
            next_view = panel_at(next_timestamp)
            next_panel = next_view.valuation
            endpoint_view = panel_at(endpoint_timestamp)
            endpoint_panel = endpoint_view.valuation
            enhanced_decision: _EnhancedDecision | None = None
            target_override: pd.Series | None = None
            decision_reason = PORTFOLIO_DECISION_REASON_SIMPLE
            decision_skip_reason = None
            missing_industry_context = False
            if not held_units.empty:
                prices = view.valuation.set_index("asset_id").reindex(held_units.index)
                active = held_units.ne(0)
                if not prices.listing_date.loc[active].eq(held_lives.loc[active]).all():
                    raise PortfolioTargetBlocked("portfolio_held_lifecycle_unproven")
                if prices.close.loc[active].isna().any():
                    raise PortfolioTargetBlocked("portfolio_held_return_missing")
                holdings = _unit_values(held_units, prices.close.fillna(0.0))
                nav = float(
                    holdings.sum() + cash + sum(entitlements.receivables.values())
                )
            if enhanced:
                assert alpha is not None and current_cross is not None
                decision_reason = PORTFOLIO_DECISION_REASON_BENCHMARK
                scheduled = is_rebalance_session(
                    decision_timestamp, next_timestamp, candidate.rebalance_schedule
                )
                membership_changed = (
                    not benchmark_weights.empty
                    and previous_benchmark_assets is not None
                    and (
                        frozenset(benchmark_weights.index) != previous_benchmark_assets
                    )
                )
                if not benchmark_weights.empty:
                    previous_benchmark_assets = frozenset(benchmark_weights.index)
                initializing = not capital_allocated
                maintenance = membership_changed and (not scheduled)
                has_predictions = bool(
                    (dataset.set_index("asset_id").eligible & alpha.notna()).any()
                )
                needs_target = has_predictions and (
                    scheduled or maintenance or initializing
                )
                required_assets = benchmark_weights.index[
                    benchmark_weights.ne(0.0)
                ].union(holdings.index[holdings.ne(0.0)])
                industry = _industry_group_codes(
                    view.valuation.set_index("asset_id"), "industry_level1_code"
                ).reindex(required_assets)
                missing_industry_context = industry.isna().any()
                if needs_target:
                    decision_skip_reason = (
                        "portfolio_benchmark_snapshot_missing"
                        if benchmark_weights.empty
                        else "portfolio_industry_context_missing"
                        if missing_industry_context
                        else None
                    )
                if needs_target and decision_skip_reason is None:
                    enhanced_decision = _enhanced_decision(
                        benchmark_weights=benchmark_weights,
                        alpha=alpha,
                        factor_risk=factor_risk,
                        current_cross=previous_exposures,
                        calibration_rows=calibration_rows,
                        calibration_intervals=calibration_intervals,
                        trading_sessions=trading_sessions,
                        as_of=decision_timestamp,
                        frozen_calibration=frozen_calibration,
                    )
                    if maintenance:
                        target_override = benchmark_weights.copy(deep=True)
                        decision_reason = PORTFOLIO_DECISION_REASON_MAINTENANCE
                else:
                    target_override = (
                        holdings.div(nav).copy(deep=True)
                        if nav > 0
                        else holdings.multiply(0.0)
                    )
                    decision_reason = PORTFOLIO_DECISION_REASON_BENCHMARK
                    if membership_changed:
                        # The proved membership snapshot authorizes independent
                        # removals even when there is no forecast or risk target.
                        target_override = benchmark_weights.copy(deep=True)
                        decision_reason = PORTFOLIO_DECISION_REASON_MAINTENANCE
            decision_receivable = sum(entitlements.receivables.values())
            if decision_columns is not None and explicit_decisions and dataset.empty:
                decision_skip_reason = "policy_no_decision_rows"
            if decision_columns is not None and (
                not explicit_decisions or not dataset.empty
            ):
                current = dataset.set_index("asset_id")
                marks = view.valuation.set_index("asset_id").close
                decision_assets = current.index.union(held_units.index)
                target_override = yield PolicyDecision(
                    pd.Timestamp(dataset.decision_at.iloc[0])
                    if explicit_decisions
                    else decision_timestamp,
                    current.reindex(decision_assets)
                    .loc[:, list(decision_columns)]
                    .copy(),
                    held_units.reindex(decision_assets, fill_value=0.0),
                    (
                        held_units
                        - pd.Series(
                            entitlements.locked_quantities, dtype="float64"
                        ).reindex(held_units.index, fill_value=0.0)
                    )
                    .clip(lower=0.0)
                    .reindex(decision_assets, fill_value=0.0),
                    marks.reindex(decision_assets),
                    cash,
                    nav,
                )
                target_override = policy_target(
                    target_override, decision_assets, weights=True
                )
                decision_reason = PORTFOLIO_DECISION_REASON_POLICY
            if not emitted:
                # Every account starts in cash. Pre-inception record dates cannot
                # create entitlements for shares first acquired by actual fills.
                for record_at in view.corporate_events.record_at.dropna().unique():
                    if pd.Timestamp(record_at) <= decision_timestamp:
                        entitlements.register(
                            view.corporate_events,
                            pd.Series(dtype="float64"),
                            pd.Series(dtype="datetime64[ns]"),
                            pd.Timestamp(record_at),
                        )
            execution_units, execution_lives, recoveries, adjustments = (
                entitlements.process(
                    next_view.corporate_events,
                    next_view.rights_events,
                    held_units,
                    held_lives,
                    next_timestamp,
                )
            )
            adjustments = tuple(
                event
                for event in adjustments
                if event.event_id not in persisted_endpoint_events
            )
            execution_marks = next_panel.set_index("asset_id")
            active_units = execution_units.loc[execution_units.ne(0)]
            if not active_units.index.isin(execution_marks.index).all():
                raise PortfolioTargetBlocked("portfolio_successor_valuation_missing")
            if (
                not execution_marks.listing_date.reindex(active_units.index)
                .eq(execution_lives.reindex(active_units.index))
                .all()
            ):
                raise PortfolioTargetBlocked("portfolio_held_lifecycle_unproven")
            effective_optimizer_config = optimizer_config
            if enhanced and initializing:
                ordinary_config = optimizer_config or OptimizerConfig()
                effective_optimizer_config = replace(
                    ordinary_config,
                    target_turnover_limit=max(
                        1.0, ordinary_config.target_turnover_limit
                    ),
                )
            account, adv, _optimization, decision_delta, asset_ledger = _build_session(
                candidate=candidate,
                decision_timestamp=decision_timestamp,
                next_timestamp=next_timestamp,
                panel=view.valuation,
                next_panel=next_panel,
                endpoint_panel=endpoint_panel,
                closed_assets=next_view.closed_assets,
                dataset=dataset,
                execution=execution,
                holdings=holdings,
                held_units=execution_units,
                cash=cash,
                recovery_cash=recoveries,
                decision_receivable=decision_receivable,
                receivable=sum(entitlements.receivables.values()),
                locked_quantities=entitlements.locked_quantities,
                nav=nav,
                amount_history=amount_history,
                alpha=alpha,
                enhanced_decision=enhanced_decision,
                benchmark_weights=benchmark_weights
                if candidate.benchmark_id is not None
                else None,
                target_override=target_override,
                decision_reason=decision_reason,
                decision_skip_reason=decision_skip_reason,
                optimizer_config=effective_optimizer_config,
                initializing=enhanced and initializing,
                assumptions=assumptions,
                fee_schedule=fee_schedule,
                fee_identity=fee_identity,
                prepare_trade_cost=prepare_trade_cost,
                market_name_for_type=market_name_for_type,
                unquoted_assets=_unquoted_assets(next_panel),
                external_target=decision_columns is not None,
            )
            if missing_industry_context:
                asset_ledger.attrs["neutrality_exposures"] = {
                    f"{kind}_exposure_{field}": value
                    for kind in ("target", "realized")
                    for field, value in (
                        ("status", "UNDEFINED"),
                        ("reason", "portfolio_industry_context_missing"),
                        ("excluded_weight", None),
                    )
                }
            yielded_value = _ledger_batch(
                account=account,
                asset_ledger=asset_ledger,
                sequence=sequence,
                end_timestamp=endpoint_timestamp,
                decision_timestamp=decision_timestamp,
                held_units=held_units,
                execution_units=execution_units,
                previous_values=previous_values,
                rights_adjustments=adjustments,
                prediction_rank_score=alpha,
                adv20=adv,
                enhanced_decision=enhanced_decision,
                benchmark_weights=benchmark_weights
                if candidate.benchmark_id is not None
                else None,
                optimization_result=_optimization,
                decision_reason=decision_reason,
                decision_delta=decision_delta,
                next_panel=next_panel,
                endpoint_panel=endpoint_panel,
                endpoint_events=endpoint_view.rights_events,
                corporate_events=next_view.corporate_events,
                endpoint_corporate_events=endpoint_view.corporate_events,
                entitlements=entitlements,
                execution_lives=execution_lives,
            )
            capital_allocated |= any(
                trade.quantity > 0 for trade in yielded_value.trades
            )
            persisted_endpoint_events = {
                event.event_id
                for event in yielded_value.rights_adjustments
                if event.phase == "end"
            }
            held_units = pd.Series(
                {
                    row.asset_id: row.quantity
                    for row in yielded_value.valuations
                    if row.phase == "after"
                    and row.asset_id != PORTFOLIO_CASH_ASSET_ID
                    and (row.quantity != 0.0 or row.receivable != 0.0)
                },
                dtype="float64",
            )
            held_lives = pd.Series(
                {
                    row.asset_id: pd.Timestamp(row.lifecycle_id)
                    for row in yielded_value.valuations
                    if row.phase == "after" and row.asset_id in held_units.index
                }
            )
            previous_values = {
                row.asset_id: row
                for row in yielded_value.valuations
                if row.phase == "end"
            }
            holdings = asset_ledger.set_index("asset_id")[
                "post_trade_notional"
            ].reindex(held_units.index)
            cash, nav = (account.post_trade_cash, account.nav_after)
            emitted = True
            yield yielded_value
            sequence += 1
        if not emitted:
            raise PortfolioTargetBlocked("portfolio_execution_window_empty")
        if enhanced and frozen_calibration is None:
            if on_final_calibration is None:
                raise ValueError("portfolio_final_calibration_sink_required")
            calibration = _calibration_fact(
                calibration_rows,
                calibration_intervals,
                as_of=pd.Timestamp(output_window.end),
                trading_sessions=trading_sessions,
            )
            on_final_calibration(
                {
                    "schema_id": "qrp-final-alpha-calibration",
                    "selection_group": candidate.selection_group,
                    "interval_semantics": alpha_interval_semantics(
                        candidate.rebalance_schedule
                    ),
                    "cutoff": pd.Timestamp(output_window.end).date().isoformat(),
                    "inputs": {
                        key: [
                            dict(source) if isinstance(source, Mapping) else str(source)
                            for source in value
                        ]
                        if key == "risk_context"
                        else dict(value)
                        if isinstance(value, Mapping)
                        else str(value)
                        for key, value in inputs.items()
                    },
                    "calibration": asdict(calibration),
                }
            )


def _read_materialized(
    source, columns: tuple[str, ...], *, business_columns=()
) -> pd.DataFrame:
    options = _dataset_input_options(source)
    if options.get("columns") is not None and not set(business_columns) <= set(
        options["columns"]
    ):
        raise ValueError("portfolio_input_business_columns_excluded")
    options["columns"] = columns
    session = open_dataset_session(**options)
    validate_input_market(session.spec, "a_share")
    return materialize_dataset_session(session)


def _iter_input_sessions(
    source,
    columns: tuple[str, ...],
    time_window: TimeWindow,
    *,
    business_columns=(),
    session=None,
) -> Iterator[pd.DataFrame]:
    options = _dataset_input_options(source, time_window=time_window)
    if options.get("columns") is not None and not set(business_columns) <= set(
        options["columns"]
    ):
        raise ValueError("portfolio_input_business_columns_excluded")
    options["columns"] = columns
    session = (
        open_dataset_session(**options)
        if session is None
        else session.filtered_view(time_window)
    )
    validate_input_market(session.spec, "a_share")
    with closing_iterator(
        _iter_dataset_session_frames(session, columns)
    ) as delegated_chunks:
        yield from delegated_chunks


def _session_reader(frames: Iterator[pd.DataFrame], columns: tuple[str, ...]):
    """Retain one future frame when a requested session has no observations."""
    pending = None

    def read(timestamp):
        nonlocal pending
        if pending is None:
            pending = next(frames, None)
        while (
            pending is not None and pd.Timestamp(pending.timestamp.iloc[0]) < timestamp
        ):
            pending = next(frames, None)
        if pending is None or pd.Timestamp(pending.timestamp.iloc[0]) > timestamp:
            return pd.DataFrame(columns=columns)
        result = _portfolio_session_frame(pending, columns)
        pending = None
        return result

    return read


def _decision_session_frames(frames, sessions):
    """Bind real prediction instants to the account's prior-session schedule."""
    for frame in frames:
        signal = pd.DatetimeIndex(frame.signal_session)
        positions = sessions.get_indexer(signal)
        if (positions < 0).any() or (positions + 1 >= len(sessions)).any():
            raise ValueError("portfolio_decision_session_invalid")
        execution = sessions.take(positions + 1)
        auction = (
            (execution + pd.Timedelta(hours=9, minutes=25))
            .tz_localize("Asia/Shanghai")
            .tz_convert("UTC")
            .tz_localize(None)
        )
        close = (
            (signal + pd.Timedelta(hours=15, minutes=30))
            .tz_localize("Asia/Shanghai")
            .tz_convert("UTC")
            .tz_localize(None)
        )
        if (
            not pd.DatetimeIndex(frame.execution_session).equals(execution)
            or frame.timestamp.isna().any()
            or frame.order_submit_at.isna().any()
            or (frame.timestamp.to_numpy() < close.to_numpy()).any()
            or frame.order_submit_at.lt(frame.timestamp).any()
            or (frame.order_submit_at.to_numpy() >= auction.to_numpy()).any()
        ):
            raise ValueError("portfolio_decision_clock_not_before_execution")
        yield frame.assign(decision_at=frame.timestamp, timestamp=signal)


def _risk_context_input_sessions(
    *,
    roots: tuple[object, ...],
    style_columns: tuple[str, ...],
    time_window: TimeWindow,
    retained: ExitStack,
):
    required = set(style_columns)
    available: set[str] = set()
    inputs = {}
    for position, root in enumerate(roots):
        session = open_dataset_session(
            **_dataset_input_options(root, time_window=time_window)
        )
        columns = tuple(
            column for column in style_columns if column in session.logical_columns
        )
        if available.intersection(columns):
            raise ValueError("portfolio_risk_context_owner_duplicate")
        available.update(columns)
        if not columns:
            continue
        validate_input_market(session.spec, "a_share")
        projection = (*_SESSION_KEY, *columns)
        inputs[position] = (
            projection,
            retained.enter_context(
                closing_iterator(_iter_dataset_session_frames(session, projection))
            ),
        )
    if available != required:
        raise ValueError("portfolio_risk_context_schema_invalid")
    return inputs


def _enhanced_risk_cross(
    *,
    panel: pd.DataFrame,
    risk_frames: list[pd.DataFrame],
    style_columns: tuple[str, ...],
    weight_column: str = "total_market_cap",
) -> pd.DataFrame:
    result = _risk_exposure_cross(
        panel=panel,
        risk_frames=risk_frames,
        style_columns=style_columns,
        weight_column=weight_column,
    )
    result["asset_return"] = result.asset_id.map(
        pd.to_numeric(panel.set_index("asset_id")["period_return"], errors="raise")
    ).astype("float64")
    return result


def _risk_exposure_cross(
    *,
    panel: pd.DataFrame,
    risk_frames: list[pd.DataFrame],
    style_columns: tuple[str, ...],
    weight_column: str = "total_market_cap",
) -> pd.DataFrame:
    panel_by_asset = panel.set_index("asset_id")
    assets = pd.Index(sorted(set(panel_by_asset.index)), dtype="string")
    result = pd.DataFrame(index=assets)
    result["timestamp"] = panel["timestamp"].iloc[0] if not panel.empty else pd.NaT
    result["listing_date"] = panel_by_asset["listing_date"].reindex(assets)
    result["is_trading"] = panel_by_asset["is_trading"].reindex(assets)
    result[weight_column] = (
        pd.to_numeric(panel_by_asset[weight_column], errors="raise")
        .reindex(assets)
        .astype("float64")
    )
    result["industry_standard_id"] = (
        panel_by_asset["industry_standard_id"].astype("string").reindex(assets)
    )
    result["industry_level1_code"] = (
        panel_by_asset["industry_level1_code"].astype("string").reindex(assets)
    )
    result["eligible"] = a_share_eligibility(panel_by_asset, "a_share_all").reindex(
        assets
    )
    for frame in risk_frames:
        context = frame.set_index("asset_id")
        for column in style_columns:
            if column in context:
                result[column] = (
                    pd.to_numeric(context[column], errors="raise")
                    .reindex(assets)
                    .astype("float64")
                )
    if panel.empty:
        for column in style_columns:
            result[column] = pd.Series(dtype="float64")
    return result.reset_index(names="asset_id")


def _factor_return_cross(
    *, exposures: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    if panel.empty:
        return exposures.iloc[:0].copy()
    previous = exposures.set_index("asset_id")
    current = panel.set_index("asset_id")
    assets = previous.index.intersection(current.index)
    result = previous.loc[assets].copy()
    result["timestamp"] = pd.Timestamp(panel["timestamp"].iloc[0])
    same_life = (
        previous.loc[assets, "listing_date"].eq(current.loc[assets, "listing_date"])
        & current.loc[assets, "listing_date"].notna()
    )
    result["listing_date"] = current.loc[assets, "listing_date"]
    result["is_trading"] = current.loc[assets, "is_trading"]
    result["asset_return"] = (
        pd.to_numeric(current.loc[assets, "period_return"], errors="raise")
        .astype("float64")
        .where(same_life)
    )
    return result.reset_index()


def _completed_calibration_rows(
    pending: tuple[pd.Series, pd.Series, pd.Timestamp], panel: pd.DataFrame
) -> list[dict[str, float]]:
    alpha, previous_open, timestamp = pending
    current_open = (
        pd.to_numeric(
            panel.set_index(["asset_id", "listing_date"])["adj_open"], errors="raise"
        )
        .astype("float64")
        .reindex(alpha.index)
    )
    returns = current_open.div(previous_open.reindex(alpha.index)) - 1.0
    valid = alpha.notna() & returns.notna() & np.isfinite(alpha) & np.isfinite(returns)
    return [
        {
            "timestamp": timestamp,
            "prediction_rank_score": float(alpha.loc[asset]),
            "holding_return": float(returns.loc[asset]),
        }
        for asset in alpha.index[valid]
    ]


def _iter_benchmark_weights(root, benchmark_id, sessions):
    """Align sparse snapshots without consuming a future session prematurely."""
    with closing_iterator(
        _iter_input_sessions(
            root,
            _BENCHMARK_COLUMNS,
            TimeWindow(sessions[0], sessions[-1]),
            business_columns=("weight",),
        )
    ) as frames:
        pending = next(frames, None)
        for timestamp in sessions:
            while pending is not None and pending["timestamp"].iloc[0] < timestamp:
                pending = next(frames, None)
            if pending is None or pending["timestamp"].iloc[0] > timestamp:
                yielded_value = pd.Series(dtype="float64")
                yield yielded_value
            else:
                yielded_value = _benchmark_weights(pending, benchmark_id, timestamp)
                yield yielded_value


def _benchmark_weights(
    frame: pd.DataFrame, benchmark_id: str, timestamp: pd.Timestamp
) -> pd.Series:
    if type(frame) is not pd.DataFrame:
        raise ValueError("portfolio_input_reader_not_materialized")
    if not set(_BENCHMARK_COLUMNS).issubset(frame.columns):
        raise ValueError("portfolio_input_schema_invalid")
    frame = frame.loc[:, list(_BENCHMARK_COLUMNS)].copy()
    frame["asset_id"] = frame["asset_id"].astype("string")
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="raise"
    ).dt.normalize()
    if frame["asset_id"].isna().any() or frame["timestamp"].isna().any():
        raise ValueError("portfolio_input_key_invalid")
    if not bool(frame["timestamp"].eq(timestamp).all()):
        raise ValueError("portfolio_input_session_window_invalid")
    frame = frame.loc[frame["benchmark_id"].eq(benchmark_id)]
    if frame.empty:
        return pd.Series(dtype="float64")
    weights = pd.to_numeric(frame["weight"], errors="raise").astype("float64")
    if (
        frame["asset_id"].duplicated().any()
        or weights.isna().any()
        or weights.le(0.0).any()
        or (not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-08))
    ):
        raise ValueError("portfolio_benchmark_weights_invalid")
    return pd.Series(
        weights.to_numpy(dtype="float64"),
        index=pd.Index(frame["asset_id"].astype(str), dtype="string"),
        name="benchmark_weight",
    )


def _calibration_fact(rows, intervals, *, as_of, trading_sessions):
    return calibrate_expected_alpha(
        pd.DataFrame(
            rows, columns=["timestamp", "prediction_rank_score", "holding_return"]
        ),
        as_of=as_of,
        scheduled_intervals=intervals,
        trading_sessions=trading_sessions,
    )


def _enhanced_decision(
    *,
    benchmark_weights: pd.Series,
    alpha: pd.Series,
    factor_risk: FactorRiskState,
    current_cross: pd.DataFrame | FactorExposureSession,
    calibration_rows: list[dict[str, object]],
    calibration_intervals: pd.DataFrame,
    trading_sessions: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    frozen_calibration: AlphaCalibration | None = None,
) -> _EnhancedDecision:
    calibration = frozen_calibration or _calibration_fact(
        calibration_rows,
        calibration_intervals,
        as_of=as_of,
        trading_sessions=trading_sessions,
    )
    expected = pd.Series(np.nan, index=alpha.index, dtype="float64")
    if not calibration.benchmark_only:
        expected.loc[alpha.notna()] = calibration.slope * alpha.loc[alpha.notna()]
    else:
        expected.loc[alpha.notna()] = 0.0
    exposures = factor_risk.current_exposures(current_cross)
    try:
        risk = factor_risk.estimate()
    except ValueError as exc:
        if str(exc) != "factor_risk_factor_history_insufficient":
            raise
        risk = None
    return _EnhancedDecision(
        benchmark_weights=benchmark_weights,
        expected_alpha=expected,
        exposures=exposures,
        risk=risk,
        calibration=calibration,
    )


def _optimizer_assets(
    *,
    benchmark_weights: pd.Series,
    current_weights: pd.Series,
    cash: float,
    nav: float,
    expected_alpha: pd.Series,
    exposures: pd.DataFrame,
    specific_variance: pd.Series,
    adv: pd.Series,
    eligible: pd.Series,
    receivable: float = 0.0,
) -> pd.DataFrame:
    assets = pd.Index(
        sorted(
            set(benchmark_weights.index[benchmark_weights.ne(0.0)])
            | set(current_weights.index[current_weights.ne(0.0)])
        ),
        dtype="string",
    )
    exposure_columns = tuple(
        column
        for column in exposures.columns
        if column not in {"asset_id", "industry_standard_id", "industry_level1_code"}
    )
    exposures_by_asset = exposures.set_index("asset_id")
    result = pd.DataFrame(
        {
            "asset_id": assets,
            "expected_alpha": expected_alpha.reindex(assets).to_numpy(),
            "current_weight": current_weights.reindex(assets, fill_value=0.0).to_numpy(
                dtype="float64"
            ),
            "benchmark_weight": benchmark_weights.reindex(
                assets, fill_value=0.0
            ).to_numpy(dtype="float64"),
            "specific_variance": specific_variance.reindex(assets).to_numpy(),
            "adv": adv.reindex(assets).to_numpy(dtype="float64"),
            "eligible": eligible.reindex(assets, fill_value=False).to_numpy(),
            "is_cash": [False] * len(assets),
        }
    )
    for column in exposure_columns:
        result[column] = (
            exposures_by_asset[column].reindex(assets).to_numpy(dtype="float64")
        )
    cash_weight = float(cash / nav)
    if cash_weight < -1e-12:
        raise ValueError("optimizer_current_state_invalid")
    cash_row = {
        "asset_id": PORTFOLIO_CASH_ASSET_ID,
        "expected_alpha": 0.0,
        "current_weight": cash_weight,
        "benchmark_weight": 0.0,
        "specific_variance": 0.0,
        "adv": np.inf,
        "eligible": True,
        "is_cash": True,
        **{column: 0.0 for column in exposure_columns},
    }
    result = pd.concat([result, pd.DataFrame([cash_row])], ignore_index=True)
    if not np.isclose(float(result["current_weight"].sum()) + receivable / nav, 1.0):
        raise ValueError("optimizer_current_state_invalid")
    return result


def _build_session(
    *,
    candidate: PortfolioCandidateSpec,
    decision_timestamp: pd.Timestamp,
    next_timestamp: pd.Timestamp,
    panel: pd.DataFrame,
    next_panel: pd.DataFrame,
    endpoint_panel: pd.DataFrame,
    closed_assets: frozenset[str],
    dataset: pd.DataFrame,
    execution: pd.DataFrame,
    holdings: pd.Series,
    held_units: pd.Series,
    cash: float,
    recovery_cash: float,
    decision_receivable: float,
    receivable: float,
    locked_quantities: Mapping,
    nav: float,
    amount_history: Mapping[tuple[str, pd.Timestamp], deque[float]],
    alpha: pd.Series,
    enhanced_decision: _EnhancedDecision | None,
    target_override: pd.Series | None,
    decision_reason: str,
    decision_skip_reason: str | None = None,
    optimizer_config: OptimizerConfig | None = None,
    assumptions: ExecutionAssumptions,
    fee_schedule: object,
    fee_identity: str,
    prepare_trade_cost: Callable[..., object],
    market_name_for_type: Callable[[object], str],
    unquoted_assets: frozenset[str] = frozenset(),
    initializing: bool = False,
    benchmark_weights: pd.Series | None = None,
    external_target: bool = False,
) -> tuple[
    ExecutionAccounting,
    pd.Series,
    OptimizationResult | None,
    pd.Series,
    pd.DataFrame,
]:
    dataset = dataset.set_index("asset_id", drop=False)
    if not np.isclose(
        float(holdings.sum()) + cash + decision_receivable,
        nav,
        rtol=0.0,
        atol=portfolio_notional_tolerance(nav),
    ):
        raise ValueError("portfolio_state_invalid")
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    industry = (
        _industry_group_codes(
            panel.set_index("asset_id").reindex(
                columns=["industry_standard_id", "industry_level1_code"]
            ),
            "industry_level1_code",
        )
        if candidate.neutrality in {"industry", "industry_size"}
        else None
    )
    domain_diagnostics = {}
    if candidate.neutrality != "none" and candidate.benchmark_id is None:
        original_eligible = eligible
        eligible, capitals = _research_neutrality_domain(
            original_eligible,
            dataset.get("float_market_cap", pd.Series(np.nan, index=dataset.index)),
            dataset.get(candidate.size_cap_column),
            candidate.neutrality,
        )
        known_cap = original_eligible & capitals.gt(0) & np.isfinite(capitals)
        reference_cap = float(capitals.loc[known_cap].sum())
        domain_diagnostics = {
            "neutrality_reference_domain": _NEUTRALITY_REFERENCE_DOMAIN,
            "neutrality_reference_eligible_count": float(original_eligible.sum()),
            "neutrality_reference_calculable_count": float(eligible.sum()),
            "neutrality_reference_excluded_known_float_cap_weight": (
                float(capitals.loc[known_cap & ~eligible].sum()) / reference_cap
                if reference_cap > 0
                else None
            ),
        }
    alpha = alpha.reindex(dataset.index)
    prediction_eligible = eligible & alpha.notna()
    # Research exits and proved index-removal maintenance are independent of
    # forecast availability and the ordinary rebalance schedule.
    exit_assets = eligible.index[~eligible]
    if (
        decision_reason == PORTFOLIO_DECISION_REASON_MAINTENANCE
        and target_override is not None
        and not target_override.empty
    ):
        exit_assets = exit_assets.union(
            holdings.index.difference(target_override.index[target_override.ne(0.0)])
        )
    missing_scores = (
        eligible
        & alpha.isna()
        & ~eligible.index.isin(exit_assets)
        & (not external_target)
    )
    missing_assets = missing_scores.index[missing_scores]
    decision_skip_reason = (
        "all_predictions_missing"
        if missing_scores.any() and not prediction_eligible.any()
        else decision_skip_reason
    )
    neutrality = None
    neutrality_reason = "nonpositive_nav"
    scheduled = not decision_skip_reason and (
        initializing
        or is_rebalance_session(
            decision_timestamp, next_timestamp, candidate.rebalance_schedule
        )
    )
    if candidate.neutrality != "none" and nav > 0:
        if candidate.benchmark_id is not None and benchmark_weights is None:
            raise ValueError("portfolio_neutrality_benchmark_required")
        try:
            neutrality = _neutrality_context(
                eligible,
                dataset.get("float_market_cap", pd.Series(np.nan, index=dataset.index)),
                industry,
                candidate.neutrality,
                size_market_cap=(
                    panel.set_index("asset_id").get(candidate.size_cap_column)
                    if candidate.benchmark_id is not None
                    else dataset.get(candidate.size_cap_column)
                ),
                benchmark_weights=benchmark_weights,
            )
        except PortfolioTargetBlocked as exc:
            if str(exc) not in {
                "target_neutrality_benchmark_empty",
                "target_neutrality_size_missing",
                "target_neutrality_size_undefined",
            }:
                raise
            neutrality_reason = str(exc)
            if scheduled and prediction_eligible.any():
                decision_skip_reason = neutrality_reason
                scheduled = False
        if neutrality is not None:
            prediction_eligible &= prediction_eligible.index.isin(neutrality.index)
    optimization: OptimizationResult | None = None
    risk_repair_reason = None
    if decision_skip_reason:
        target = holdings.div(nav) if nav > 0 else holdings.multiply(0.0)
    elif nav == 0:
        target = pd.Series(0.0, index=holdings.index, dtype="float64")
    elif target_override is not None:
        target = target_override.copy(deep=True)
    elif enhanced_decision is not None:
        target = enhanced_decision.benchmark_weights.copy(deep=True)
    elif scheduled:
        if not prediction_eligible.any():
            if neutrality is not None:
                raise PortfolioTargetBlocked("target_selection_unavailable")
            target = pd.Series(0.0, index=dataset.index, dtype="float64")
        else:
            try:
                target = build_simple_target_weights(
                    alpha,
                    eligible=prediction_eligible,
                    method="float_cap_weight"
                    if candidate.construction == "quantile_float_cap"
                    else "equal_weight",
                    top_quantile=float(candidate.top_quantile),
                    bottom_quantile=float(candidate.top_quantile),
                    float_market_cap=dataset["float_market_cap"]
                    if candidate.construction == "quantile_float_cap"
                    else None,
                )
            except PortfolioTargetBlocked as error:
                if str(error) != "target_float_market_cap_missing":
                    raise
                decision_skip_reason = str(error)
                target = holdings.div(nav).copy(deep=True)
    else:
        target = holdings.div(nav).copy(deep=True)
    if missing_scores.any() and not decision_skip_reason and nav > 0:
        frozen = holdings.div(nav).reindex(missing_assets, fill_value=0.0)
        target = target.drop(missing_assets, errors="ignore")
        if target.sum() > 0:
            target *= max(
                0.0, 1.0 - decision_receivable / nav - float(frozen.sum())
            ) / float(target.sum())
        target = pd.concat((target, frozen))
    target = target.where(~target.index.isin(exit_assets), 0.0)
    if nav > 0.0 and not decision_skip_reason:
        stock_budget = max(0.0, 1.0 - decision_receivable / nav)
        if target.sum() > stock_budget:
            target *= stock_budget / float(target.sum())
        if (
            candidate.construction == "quantile_equal_weight_half_step"
            and scheduled
            and target_override is None
            and enhanced_decision is None
        ):
            # Budget the new stock leg before mixing; missing-score holdings
            # occur unchanged on both sides and never absorb a half-step gap.
            retained = holdings.div(nav).reindex(target.index, fill_value=0.0)
            retained = retained.where(
                eligible.reindex(target.index, fill_value=False), 0.0
            )
            target = (target + retained) / 2.0
    raw_target = target.copy(deep=True)
    stock_gross = float(target.sum())
    if neutrality is not None and scheduled and stock_gross > 0.0:
        try:
            target = (
                _project_neutral_target(
                    target / stock_gross,
                    neutrality,
                    industry_limit=candidate.industry_neutrality_limit,
                    size_limit=candidate.size_neutrality_limit,
                    fixed_weights=target.reindex(missing_assets, fill_value=0.0)
                    / stock_gross,
                )
                * stock_gross
            )
        except PortfolioTargetBlocked as error:
            if str(error) != "target_neutrality_infeasible":
                raise
            risk_repair_reason = str(error)
    target = target.loc[target.ne(0.0)]
    current = holdings.copy(deep=True)
    assets = pd.Index(
        sorted(
            set(current.index)
            | set(target.index)
            | set(held_units.index)
            | set(raw_target.index[raw_target.ne(0)])
            | (
                set(neutrality.index).intersection(next_panel.asset_id)
                if neutrality is not None
                else set()
            )
        ),
        dtype="string",
    )
    unquoted_assets = unquoted_assets.intersection(assets)
    if PORTFOLIO_CASH_ASSET_ID in assets:
        raise ValueError("portfolio_asset_universe_invalid")
    execution_panel = next_panel.set_index("asset_id")
    missing_panel = [asset for asset in assets if asset not in execution_panel.index]
    if missing_panel:
        raise PortfolioTargetBlocked("portfolio_held_asset_missing_from_panel")
    decision_panel = panel.set_index("asset_id")
    decision_markets, market_by_asset = (
        {
            str(asset): market_name_for_type(market_type)
            for asset, market_type in frame.market_type.astype("Int64")
            .reindex(assets)
            .items()
            if pd.notna(market_type)
        }
        for frame in (decision_panel, execution_panel)
    )
    decision_adv, adv = (
        pd.Series(
            {
                asset: float(mean(history))
                if len(history := amount_history.get((asset, listing), ())) == 20
                else np.nan
                for asset, listing in frame.reindex(assets).listing_date.items()
            },
            dtype="float64",
        )
        for frame in (decision_panel, execution_panel)
    )
    missing_execution_quotes = frozenset(
        execution_panel.loc[assets].index[
            execution_panel.loc[assets, "open"].isna()
            | execution_panel.loc[assets, "open"].eq(0)
        ]
    )
    unquoted_assets = unquoted_assets | missing_execution_quotes
    masks = _mask_values(
        execution, assets, next_timestamp, closed_assets, unquoted_assets
    )
    for column in masks:
        masks[column].loc[masks[column].index.intersection(missing_assets)] = False
    prepared = {
        (market, side): prepare_trade_cost(
            next_timestamp.date(), market, side, fee_schedule, assumptions
        )
        for market in set(market_by_asset.values()) | set(decision_markets.values())
        for side in ("buy", "sell")
    }

    execution_cash = cash + recovery_cash
    entry = execution_panel.reindex(assets)
    required = held_units.reindex(assets, fill_value=0.0).ne(0)
    if entry.loc[required, "open"].isna().any():
        raise PortfolioTargetBlocked("portfolio_held_return_missing")
    execution_holdings = _unit_values(
        held_units.reindex(assets, fill_value=0.0), entry.open.fillna(0.0)
    )
    execution_nav = float(execution_holdings.sum() + execution_cash + receivable)
    execution_weights = (
        execution_holdings.div(execution_nav)
        if execution_nav > 0
        else execution_holdings.multiply(0.0)
    )
    independent_exits = assets.isin(exit_assets)
    hold_target = execution_weights.where(~independent_exits, 0.0)
    assets_frame = None
    if nav > 0 and enhanced_decision is not None:
        risk = enhanced_decision.risk
        assets_frame = _optimizer_assets(
            benchmark_weights=enhanced_decision.benchmark_weights,
            current_weights=holdings.div(nav),
            cash=cash,
            nav=nav,
            receivable=decision_receivable,
            expected_alpha=enhanced_decision.expected_alpha,
            exposures=enhanced_decision.exposures,
            specific_variance=(
                risk.specific_variance
                if risk is not None
                else pd.Series(dtype="float64")
            ),
            adv=decision_adv,
            eligible=eligible,
        )
    if (
        nav > 0
        and not decision_skip_reason
        and target_override is None
        and enhanced_decision is not None
        and (enhanced_decision.risk is not None)
        and (not enhanced_decision.calibration.benchmark_only)
    ):
        assets_frame["is_financial"] = False
        stock = ~assets_frame.is_cash
        assets_frame.loc[stock, "is_financial"] = ~_bool_series(
            decision_panel.is_nonfinancial_industry.reindex(
                assets_frame.loc[stock, "asset_id"]
            ),
            "portfolio_financial_context_invalid",
        ).to_numpy()
        unavailable = assets_frame.asset_id.isin(missing_assets)
        assets_frame["can_enter"] = ~unavailable
        assets_frame["can_exit"] = ~unavailable
        assets_frame.loc[unavailable, "expected_alpha"] = 0.0
        adv_values = assets_frame["adv"].to_numpy(dtype="float64")
        market_masks = {
            market: assets_frame["asset_id"].map(decision_markets).eq(market).to_numpy()
            for market in set(decision_markets.values())
        }

        cost_prices = assets_frame.asset_id.map(decision_panel.close).to_numpy()
        cost_face_values = (
            assets_frame.asset_id.map(decision_panel.face_value_per_share).to_numpy()
            if "face_value_per_share" in decision_panel
            else None
        )

        def expected_cost(buys, sells):
            total = 0.0
            gradients = []
            for side, notionals in (("buy", buys), ("sell", sells)):
                gradient = np.zeros(len(notionals), dtype="float64")
                for market, market_mask in market_masks.items():
                    active = market_mask & np.isfinite(adv_values) & (adv_values > 0)
                    if active.any():
                        costs, gradient[active] = prepared[
                            market, side
                        ].totals_and_gradients(
                            notionals[active],
                            adv_values[active],
                            prices=cost_prices[active],
                            face_value_per_share=(
                                cost_face_values[active]
                                if cost_face_values is not None
                                else None
                            ),
                        )
                        total += float(costs.sum())
                gradients.append(gradient)
            return total, *gradients

        try:
            optimization = optimize_portfolio(
                assets_frame,
                factor_covariance=risk.factor_covariance,
                industry_columns=tuple(
                    column
                    for column in risk.factor_covariance.index
                    if column.startswith("industry_")
                ),
                style_columns=candidate.risk_style_columns,
                expected_transaction_cost=expected_cost,
                config=optimizer_config,
                nav=float(nav),
                assumptions=assumptions,
                eligibility_column="eligible",
                financial_column="is_financial",
                cash_column="is_cash",
                receivable_weight=float(decision_receivable / nav),
            )
        except ValueError as error:
            if str(error) not in {
                "optimizer_frozen_constraints_infeasible",
                "optimizer_trade_adv_unavailable",
                "optimizer_budget_bounds_infeasible",
                "optimizer_capacity_bounds_infeasible",
                "optimizer_turnover_bounds_infeasible",
                "optimizer_stock_exposure_unavailable",
                "optimizer_held_risk_unavailable",
            }:
                raise
            if str(error) in {
                "optimizer_frozen_constraints_infeasible",
                "optimizer_budget_bounds_infeasible",
                "optimizer_capacity_bounds_infeasible",
                "optimizer_turnover_bounds_infeasible",
            }:
                risk_repair_reason = str(error)
                target = enhanced_decision.benchmark_weights.reindex(
                    assets, fill_value=0.0
                )
            else:
                decision_skip_reason = str(error)
        if optimization is not None:
            target = optimization.weights.loc[
                optimization.weights.index != PORTFOLIO_CASH_ASSET_ID
            ]
    orders_requested = (
        scheduled or decision_reason == PORTFOLIO_DECISION_REASON_MAINTENANCE
    )
    decision_delta = (
        target.reindex(assets, fill_value=0.0)
        - current.div(nav).reindex(assets, fill_value=0.0)
        if orders_requested and nav > 0
        else pd.Series(0.0, index=assets)
    )
    execution_target = target if orders_requested else execution_weights

    def account_for(weights):
        marks = entry.open.fillna(0.0)
        quantities = held_units.reindex(assets, fill_value=0.0)
        target_quantities = pd.Series(
            {
                asset: quantities[asset]
                if weight == execution_weights.get(asset, 0.0)
                else weight * execution_nav / marks[asset]
                if marks[asset] > 0
                else 0.0
                for asset, weight in weights.reindex(assets, fill_value=0.0).items()
            },
            dtype="float64",
        )
        capacity = adv * assumptions.adv_participation / marks.where(marks > 0)
        orders = pd.DataFrame(
            {
                "quantity": quantities,
                "target_quantity": target_quantities,
                "sellable_quantity": quantities
                - pd.Series(locked_quantities, dtype="float64").reindex(
                    assets, fill_value=0.0
                ),
                "market_type": execution_panel.market_type.reindex(assets),
                "mark_price": marks,
                "buy_price": marks,
                "sell_price": marks,
                "buy_capacity": capacity,
                "sell_capacity": capacity,
                "adv": adv,
                "can_enter": masks["can_enter"],
                "can_exit": masks["can_exit"],
            }
        )
        if "face_value_per_share" in execution_panel:
            orders["face_value_per_share"] = (
                execution_panel.face_value_per_share.reindex(assets)
            )
        return execute_stock_target(
            orders,
            cash=execution_cash,
            execution_date=next_timestamp.date(),
            fee_schedule=fee_schedule,
            assumptions=assumptions,
            unquoted_assets=unquoted_assets,
            receivable=receivable,
        )

    if decision_skip_reason:
        execution_target = hold_target
        decision_delta = pd.Series(0.0, index=assets)
    execution_target = execution_target.where(
        ~execution_target.index.isin(exit_assets), 0.0
    )
    if nav > 0:
        decision_delta.loc[independent_exits] = (
            -current.div(nav).reindex(assets, fill_value=0.0).loc[independent_exits]
        ).astype("float64")
    if assets_frame is not None or neutrality is not None:
        config = optimizer_config or OptimizerConfig()
        if assets_frame is not None:
            risk = enhanced_decision.risk
            benchmark = assets_frame.benchmark_weight.to_numpy(dtype="float64")
            columns = (
                tuple(
                    column
                    for column in enhanced_decision.exposures
                    if column
                    not in {"asset_id", "industry_standard_id", "industry_level1_code"}
                )
                if risk is None
                else tuple(risk.factor_covariance.index)
            )
            exposure = assets_frame.loc[:, columns].to_numpy(dtype="float64")
            specific = assets_frame.specific_variance.to_numpy(dtype="float64")
            covariance = (
                None
                if risk is None
                else risk.factor_covariance.to_numpy(dtype="float64")
            )
            cash_mask = assets_frame.is_cash.to_numpy()
        else:
            bands = _neutrality_bands(
                neutrality,
                industry_limit=candidate.industry_neutrality_limit,
                size_limit=candidate.size_neutrality_limit,
            )
        requested = execution_target.reindex(assets, fill_value=0.0)
        exit_notional = execution_holdings.loc[independent_exits].sum()
        turnover_cap = (
            (
                1.0
                if initializing or assets_frame is None
                else max(config.target_turnover_limit, exit_notional / execution_nav)
            )
            if execution_nav > 0
            else 0.0
        )

        def evaluate(weights):
            result = account_for(weights.where(~weights.index.isin(exit_assets), 0.0))
            if result.nav_after <= 0:
                return (
                    result,
                    {
                        "realized_constraints_status": "UNDEFINED",
                        "realized_constraints_reason": "nonpositive_nav",
                    },
                    (np.inf, np.inf, np.inf),
                )
            notionals = result.asset_ledger.set_index("asset_id").post_trade_notional
            notionals.loc[PORTFOLIO_CASH_ASSET_ID] = result.post_trade_cash
            if assets_frame is not None:
                facts, violation = _portfolio_risk_limits(
                    notionals.reindex(assets_frame.asset_id, fill_value=0.0).to_numpy()
                    / result.nav_after,
                    benchmark,
                    exposure,
                    specific,
                    covariance,
                    cash_mask,
                    columns,
                    config,
                )
            else:
                facts, violation = _neutrality_risk_limits(
                    notionals.reindex(assets, fill_value=0.0), neutrality, bands
                )
            fills = result.asset_ledger.executed_delta_notional
            turnover = (
                max(fills.clip(lower=0).sum(), -fills.clip(upper=0).sum())
                / result.nav_before
            )
            if turnover > turnover_cap + 1e-8:
                violation = (np.inf, np.inf)
            distance = float(
                np.square(
                    notionals.reindex(assets, fill_value=0.0) / result.nav_before
                    - requested
                ).sum()
            )
            return result, facts, (*violation, distance)

        repair = (
            orders_requested
            and execution_nav > 0
            and not decision_skip_reason
            and (neutrality is not None or enhanced_decision.risk is not None)
        )
        if repair:
            minimum, increment = (pd.Series(0.0, index=assets) for _ in range(2))
            known = entry.market_type.isin([1, 4, 16, 32]) & entry.open.gt(0)
            _, lot, step, _ = stock_order_limits(
                entry.loc[known, "market_type"], next_timestamp.date()
            )
            minimum.loc[known] = lot * entry.loc[known, "open"] / execution_nav
            increment.loc[known] = step * entry.loc[known, "open"] / execution_nav
            baseline = evaluate(hold_target)
            benchmark_target = (
                (
                    enhanced_decision.benchmark_weights
                    if assets_frame is not None
                    else neutrality.benchmark_weight * requested.sum()
                )
                .reindex(assets, fill_value=0.0)
                .where(~independent_exits, 0.0)
            )
            corrective_delta = benchmark_target - hold_target
            buys = corrective_delta.clip(lower=0).sum()
            sells = -corrective_delta.clip(upper=0).sum()
            scale = min(
                1.0,
                turnover_cap / buys if buys > 0 else 1.0,
                max(0.0, turnover_cap - exit_notional / execution_nav) / sells
                if sells > 0
                else 1.0,
            )
            benchmark_target = hold_target + scale * corrective_delta
            selected = _repair_stock_allocation(
                requested,
                hold_target,
                benchmark_target,
                minimum,
                increment,
                evaluate,
            )
            selected[1].update(
                execution_constraints_status="UNDEFINED"
                if selected[1]["realized_constraints_status"] == "UNDEFINED"
                or not np.isfinite(selected[2][0])
                else "SATISFIED"
                if selected[2][0] == 0.0
                else "IMPROVED"
                if selected[2][:2] < baseline[2][:2]
                else "UNCHANGED",
                execution_constraints_reason=selected[1].get(
                    "realized_constraints_reason"
                ),
                execution_target_repair_reason=risk_repair_reason,
            )
        else:
            selected = evaluate(execution_target)
        account, facts, _ = selected
        asset_ledger = account.asset_ledger
        asset_ledger.attrs["realized_constraints"] = facts
    else:
        account = account_for(execution_target)
        asset_ledger = account.asset_ledger
    asset_ledger.attrs["initial_allocation_attempt"] = float(initializing)
    if decision_skip_reason:
        target = hold_target
        raw_target = hold_target.copy(deep=True)
        asset_ledger.attrs.setdefault(
            "realized_constraints",
            {
                "realized_constraints_status": "UNDEFINED",
                "realized_constraints_reason": decision_skip_reason,
            },
        )
    if enhanced_decision is not None and enhanced_decision.risk is None:
        asset_ledger.attrs.setdefault(
            "realized_constraints",
            {
                "realized_constraints_status": "UNDEFINED",
                "realized_constraints_reason": (
                    "factor_risk_factor_history_insufficient"
                ),
                "realized_tracking_error_status": "UNDEFINED",
            },
        )
    asset_ledger.attrs["decision_availability"] = {
        "decision_status": "SKIPPED"
        if decision_skip_reason
        else "PARTIAL"
        if missing_scores.any() or risk_repair_reason is not None
        else "AVAILABLE",
        "decision_skip_reason": decision_skip_reason,
        "missing_prediction_count": float(missing_scores.sum()),
    }
    missing_orders = asset_ledger.loc[
        asset_ledger.reason.eq("adv_unavailable"), "asset_id"
    ]
    if not missing_orders.empty:
        for asset in missing_orders:
            history = amount_history.get(
                (asset, execution_panel.loc[asset, "listing_date"]), ()
            )
            selected = asset_ledger.asset_id.eq(asset)
            asset_ledger.loc[selected, "adv20_observation_count"] = len(history)
            asset_ledger.loc[selected, "adv20_missing_observations"] = sum(
                pd.isna(value) for value in history
            )
    if neutrality is not None:
        realized = asset_ledger.set_index("asset_id").post_trade_notional
        exposure_diagnostics = {}
        exposure_weights = {"benchmark": neutrality.benchmark_weight}
        if target.sum() > 0:
            exposure_weights["target"] = target / target.sum()
        if realized.sum() > 0:
            exposure_weights["realized"] = realized / realized.sum()
        if industry is not None:
            for kind, weights in exposure_weights.items():
                exposure_diagnostics[f"{kind}_unknown_industry_weight"] = float(
                    weights.loc[industry.reindex(weights.index).isna()].sum()
                )
        for kind in ("target", "realized"):
            weights = exposure_weights.get(kind)
            outside = (
                weights.loc[~weights.index.isin(neutrality.index)].abs().sum()
                if weights is not None
                else None
            )
            complete = weights is not None and outside == 0.0
            exposure_diagnostics[f"{kind}_exposure_status"] = (
                "DEFINED" if complete else "UNDEFINED"
            )
            exposure_diagnostics[f"{kind}_exposure_reason"] = (
                None
                if complete
                else "nonpositive_stock_value"
                if weights is None
                else "outside_benchmark_domain"
            )
            exposure_diagnostics[f"{kind}_exposure_excluded_weight"] = outside
            if not complete:
                exposure_weights.pop(kind, None)
        known = neutrality.attrs.get("industry_known")
        if known is not None:
            exposure_diagnostics["industry_neutrality_domain"] = (
                _INDUSTRY_NEUTRALITY_DOMAIN
            )
        for kind, weights in exposure_weights.items():
            weights = weights.reindex(neutrality.index, fill_value=0.0)
            known_weight = float(weights.loc[known].sum()) if known is not None else 0.0
            if known is not None:
                exposure_diagnostics[f"{kind}_industry_exposure_status"] = (
                    "DEFINED" if known_weight > 0 else "UNDEFINED"
                )
                exposure_diagnostics[f"{kind}_industry_exposure_reason"] = (
                    None if known_weight > 0 else "no_known_industry_weight"
                )
            for factor in neutrality.columns.drop("benchmark_weight"):
                if factor.startswith("industry:") and known_weight <= 0:
                    continue
                exposure_diagnostics[f"{kind}_exposure:{factor}"] = float(
                    neutrality[factor] @ weights
                ) / (known_weight if factor.startswith("industry:") else 1.0)
        asset_ledger.attrs["neutrality_exposures"] = exposure_diagnostics
        for name, weights in (
            ("raw_target_weight", raw_target),
            ("target_weight", target),
            ("benchmark_weight", neutrality.benchmark_weight),
            (
                "realized_weight",
                realized / account.nav_after
                if account.nav_after > 0
                else realized * np.nan,
            ),
        ):
            asset_ledger[name] = (
                asset_ledger.asset_id.map(weights).fillna(0.0)
                if name != "realized_weight"
                else asset_ledger.asset_id.map(weights)
            )
    elif candidate.neutrality != "none":
        asset_ledger.attrs["neutrality_exposures"] = {
            f"{kind}_exposure_{field}": value
            for kind in ("target", "realized")
            for field, value in (
                ("status", "UNDEFINED"),
                ("reason", neutrality_reason),
                ("excluded_weight", None),
            )
        }
        for name, weights in (
            ("raw_target_weight", raw_target),
            ("target_weight", target),
            (
                "realized_weight",
                asset_ledger.set_index("asset_id").post_trade_notional
                / account.nav_after
                if account.nav_after > 0
                else pd.Series(dtype="float64"),
            ),
        ):
            asset_ledger[name] = (
                asset_ledger.asset_id.map(weights) if nav > 0 else np.nan
            )
    if domain_diagnostics:
        asset_ledger.attrs["neutrality_exposures"].update(domain_diagnostics)
    return (account, adv, optimization, decision_delta, asset_ledger)


def _prediction_alpha(
    dataset: pd.DataFrame,
    prediction_lookup: object,
) -> pd.Series:
    dataset = dataset.set_index("asset_id", drop=False)
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    # Match supplied keys against context even when current eligibility forbids
    # trading. The persisted PRED owns its score domain; historical X is irrelevant.
    required_rows = dataset.loc[:, ["timestamp", "asset_id"]]
    if "decision_at" in dataset:
        required_rows = required_rows.assign(timestamp=dataset.decision_at)
    alpha = pd.Series(np.nan, index=dataset.index, dtype="float64")
    if required_rows.empty:
        return alpha
    values_for_frame = getattr(prediction_lookup, "values_for_frame", None)
    if not callable(values_for_frame):
        raise ValueError("portfolio_prediction_lookup_invalid")
    values, found = values_for_frame(required_rows, mark_matched=True)
    supplied = np.asarray(found, dtype=bool)
    supplied_values = np.asarray(values, dtype="float64")[supplied]
    if not np.isfinite(supplied_values).all():
        raise ValueError("portfolio_prediction_values_invalid")
    alpha.loc[required_rows.loc[supplied, "asset_id"]] = supplied_values
    return alpha.where(eligible)


def _bool_series(values: pd.Series, error: str) -> pd.Series:
    if not isinstance(values, pd.Series) or not pd.api.types.is_bool_dtype(values):
        raise ValueError(error)
    if values.isna().any():
        raise ValueError(error)
    return values.astype(bool)


def _validate_input_keys(dataset: pd.DataFrame, panel: pd.DataFrame) -> None:
    dataset_keys = _session_keys(dataset)
    panel_keys = _session_keys(panel)
    if dataset_keys != panel_keys:
        if dataset_keys.difference(panel_keys):
            raise PortfolioTargetBlocked("portfolio_session_key_missing")
        raise ValueError("portfolio_input_key_alignment_invalid")


def _session_keys(frame: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    return {
        (pd.Timestamp(timestamp), str(asset))
        for timestamp, asset in frame.loc[:, list(_SESSION_KEY)].itertuples(
            index=False, name=None
        )
    }


def _update_amount_history(
    panel: pd.DataFrame,
    history: dict[tuple[str, pd.Timestamp], deque[float]],
    closed_assets: frozenset[str],
    unknown_market_assets: frozenset[str],
) -> None:
    current = dict(
        zip(panel["asset_id"].astype(str), panel["listing_date"], strict=True)
    )
    amounts = strict_real_numeric_series(
        panel["amount"], error="portfolio_amount_invalid", errors="coerce"
    )
    amounts = amounts.where(np.isfinite(amounts) & amounts.ge(0))
    for key in tuple(history):
        asset, listing = key
        if asset in unknown_market_assets or (
            asset in current and current[asset] != listing
        ):
            del history[key]
        elif asset not in current and asset not in closed_assets:
            # A missing own-market open row consumes an unknown observation.
            history[key].append(float("nan"))
    for asset, listing, amount in zip(
        panel["asset_id"].astype(str), panel["listing_date"], amounts, strict=True
    ):
        if asset not in closed_assets | unknown_market_assets:
            history.setdefault((asset, listing), deque(maxlen=20)).append(float(amount))


def _mask_values(
    execution: pd.DataFrame,
    assets: pd.Index,
    execution_at: pd.Timestamp,
    closed_assets: frozenset[str],
    unquoted_assets: frozenset[str] = frozenset(),
) -> dict[str, pd.Series]:
    masks = {
        column: pd.Series(False, index=assets, dtype="boolean")
        for column in ("can_enter", "can_exit")
    }
    for values in masks.values():
        values.loc[values.index.isin(unquoted_assets)] = pd.NA
    open_assets = assets[~assets.isin(closed_assets | unquoted_assets)]
    if open_assets.empty:
        return masks
    if not pd.api.types.is_bool_dtype(
        execution["can_enter"]
    ) or not pd.api.types.is_bool_dtype(execution["can_exit"]):
        raise ValueError("portfolio_execution_mask_invalid")
    indexed = execution.set_index("asset_id")
    current = indexed.reindex(open_assets)
    known_date = current.execution_at.notna()
    if not current.loc[known_date, "execution_at"].eq(execution_at).all():
        raise PortfolioTargetBlocked("portfolio_execution_date_misaligned")
    for column in masks:
        masks[column].loc[open_assets] = (
            current[column].where(known_date).astype("boolean")
        )
    return masks


def _ledger_batch(
    *,
    account: ExecutionAccounting,
    asset_ledger: pd.DataFrame,
    sequence: int,
    end_timestamp: pd.Timestamp,
    decision_timestamp: pd.Timestamp,
    held_units: pd.Series,
    execution_units: pd.Series,
    previous_values: dict,
    rights_adjustments: tuple[PortfolioRightsAdjustment, ...],
    prediction_rank_score: pd.Series,
    adv20: pd.Series,
    enhanced_decision: _EnhancedDecision | None,
    benchmark_weights: pd.Series | None,
    optimization_result: OptimizationResult | None,
    decision_reason: str,
    decision_delta: pd.Series,
    next_panel: pd.DataFrame,
    endpoint_panel: pd.DataFrame,
    endpoint_events: pd.DataFrame,
    corporate_events: pd.DataFrame,
    endpoint_corporate_events: pd.DataFrame,
    entitlements: EquityEntitlements,
    execution_lives: pd.Series,
) -> PortfolioLedgerBatch:
    """Build economic facts directly from one isolated accounting snapshot."""
    trades, cashflows, valuations, diagnostics = [], [], [], []
    timestamp = int(pd.Timestamp(account.execution_date).value)
    mark_frames = {
        "entry": next_panel.set_index("asset_id"),
        "end": endpoint_panel.set_index("asset_id"),
    }

    # Derive quantities from actual fills, retaining zero-priced unquoted claims.
    after_units = execution_units.astype("float64").copy()
    after_lives = execution_lives.copy()
    fill_quantities = {}
    for row in asset_ledger.itertuples(index=False):
        delta = float(row.executed_delta_notional)
        if delta:
            quantity = float(row.executed_quantity)
            fill_quantities[row.asset_id] = quantity
            after_units.loc[row.asset_id] = (
                float(after_units.get(row.asset_id, 0)) + quantity
            )
            after_lives.loc[row.asset_id] = mark_frames["entry"].loc[
                row.asset_id, "listing_date"
            ]
    entry_receivables = dict(entitlements.receivables)
    entry_locks = dict(entitlements.locked_quantities)
    registration = entitlements.register(
        corporate_events,
        after_units,
        after_lives,
        pd.Timestamp(account.execution_date),
        phase="end",
    )
    # Value the next boundary with the same accounting owner. Keep the actual
    # decision-close snapshot unchanged; the next entry suppresses these already
    # persisted transition IDs when it advances that snapshot to the same open.
    endpoint_state = deepcopy(entitlements)
    end_units, end_lives, end_cash, endpoint_adjustments = endpoint_state.process(
        endpoint_corporate_events,
        endpoint_events,
        after_units,
        after_lives,
        end_timestamp,
        phase="end",
    )
    rights_adjustments = (*rights_adjustments, *registration, *endpoint_adjustments)

    def emit(asset, before, after, fill_id, values):
        if asset == PORTFOLIO_CASH_ASSET_ID:
            entry_mark = end_mark = 1.0
            before = (
                previous_values[asset].value
                if asset in previous_values
                else before
                - sum(
                    event.cash_amount
                    for event in rights_adjustments
                    if event.phase == "after"
                )
            )
            before_quantity, after_quantity = before, after
            lifecycle = None
            before_lifecycle = None
            before_mark = 1.0
        else:
            if asset in mark_frames["entry"].index:
                lifecycle = (
                    pd.Timestamp(mark_frames["entry"].loc[asset, "listing_date"])
                    .date()
                    .isoformat()
                )
                entry_mark = mark_frames["entry"].loc[asset, "open"]
            elif asset in end_lives and not execution_units.get(asset, 0):
                lifecycle = pd.Timestamp(end_lives[asset]).date().isoformat()
                entry_mark = 0.0
            else:
                raise PortfolioTargetBlocked("portfolio_held_lifecycle_unproven")
            after_quantity = float(after_units.get(asset, 0))
            if not after_quantity and asset in previous_values:
                lifecycle = previous_values[asset].lifecycle_id
            end_quantity = float(end_units.get(asset, 0))
            end_lifecycle = (
                pd.Timestamp(end_lives[asset]).date().isoformat()
                if asset in end_lives
                else lifecycle
            )
            if end_quantity:
                if asset not in mark_frames["end"].index:
                    raise PortfolioTargetBlocked(
                        "portfolio_successor_valuation_missing"
                    )
                if (
                    pd.Timestamp(end_lifecycle)
                    != mark_frames["end"].loc[asset, "listing_date"]
                ):
                    raise PortfolioTargetBlocked("portfolio_held_lifecycle_unproven")
                end_mark = mark_frames["end"].loc[asset, "open"]
            else:
                end_mark = (
                    mark_frames["end"].loc[asset, "open"]
                    if asset in mark_frames["end"].index
                    and pd.Timestamp(end_lifecycle)
                    == mark_frames["end"].loc[asset, "listing_date"]
                    else 0.0
                )
            if pd.isna(entry_mark):
                if (
                    held_units.get(asset, 0)
                    or execution_units.get(asset, 0)
                    or fill_id is not None
                ):
                    raise PortfolioTargetBlocked("portfolio_held_return_missing")
                entry_mark = 0.0
            before_quantity = float(held_units.get(asset, 0))
            before_lifecycle = lifecycle
            before_mark = entry_mark
            if asset in previous_values:
                prior = previous_values[asset]
                before_quantity, before, before_mark = (
                    prior.quantity,
                    prior.value,
                    prior.mark_price,
                )
                before_lifecycle = prior.lifecycle_id
            else:
                before = float(before_quantity * float(entry_mark))
            if entry_mark == 0 and fill_id is not None:
                raise ValueError("portfolio_zero_mark_trade_invalid")
            after = float(after_quantity * float(entry_mark))
            if pd.isna(end_mark):
                if end_quantity:
                    raise PortfolioTargetBlocked("portfolio_held_return_missing")
                end_mark = 0.0
        valuations.extend(
            (
                PortfolioValuation(
                    timestamp,
                    asset,
                    "before",
                    before_quantity,
                    before,
                    before_mark,
                    lifecycle_id=before_lifecycle,
                    receivable=previous_values[asset].receivable
                    if asset in previous_values
                    else 0.0,
                    locked_quantity=previous_values[asset].locked_quantity
                    if asset in previous_values
                    else 0.0,
                ),
                PortfolioValuation(
                    timestamp,
                    asset,
                    "after",
                    after_quantity,
                    after + entry_receivables.get(asset, 0.0),
                    entry_mark,
                    lifecycle_id=lifecycle,
                    receivable=entry_receivables.get(asset, 0.0),
                    locked_quantity=entry_locks.get(asset, 0.0),
                ),
                PortfolioValuation(
                    int(end_timestamp.value),
                    asset,
                    "end",
                    after_quantity + end_cash
                    if asset == PORTFOLIO_CASH_ASSET_ID
                    else end_quantity,
                    (
                        after_quantity + end_cash
                        if asset == PORTFOLIO_CASH_ASSET_ID
                        else float(end_quantity * float(end_mark))
                        + endpoint_state.receivables.get(asset, 0.0)
                    ),
                    end_mark,
                    lifecycle_id=lifecycle
                    if asset == PORTFOLIO_CASH_ASSET_ID
                    else end_lifecycle,
                    receivable=endpoint_state.receivables.get(asset, 0.0),
                    locked_quantity=endpoint_state.locked_quantities.get(asset, 0.0),
                ),
            )
        )
        diagnostics.append(
            PortfolioDiagnostic(
                asset,
                fill_id,
                "decision_timestamp",
                text=decision_timestamp.isoformat(),
            )
        )
        for name, value in values.items():
            if value is None or pd.isna(value):
                continue
            diagnostics.append(
                PortfolioDiagnostic(
                    asset,
                    fill_id,
                    name,
                    number=strict_finite_float(
                        value, error="portfolio_ledger_numeric_invalid"
                    )
                    if not isinstance(value, str)
                    else None,
                    text=value if isinstance(value, str) else None,
                )
            )

    for row in asset_ledger.itertuples(index=False):
        asset = str(row.asset_id)
        delta = float(row.executed_delta_notional)
        fill_id = f"{sequence}:{asset}" if delta else None
        if fill_id is not None:
            trades.append(
                ExecutedTrade(
                    fill_id,
                    fill_id,
                    timestamp,
                    asset,
                    fill_quantities[asset],
                    delta,
                )
            )
            cashflows.append(
                PortfolioCashFlow(
                    f"{fill_id}:principal",
                    timestamp,
                    asset,
                    fill_id,
                    "principal",
                    -delta,
                )
            )
        for cost in PORTFOLIO_LEDGER_COST_COLUMNS:
            amount = float(getattr(row, cost))
            if amount:
                cashflows.append(
                    PortfolioCashFlow(
                        f"{fill_id}:{cost}",
                        timestamp,
                        asset,
                        fill_id,
                        cost,
                        -amount,
                    )
                )
        emit(
            asset,
            float(row.pre_trade_notional),
            float(row.post_trade_notional),
            fill_id,
            {
                "target_notional": float(row.target_notional),
                **{
                    name: getattr(row, name)
                    for name in (
                        "raw_target_weight",
                        "target_weight",
                        "realized_weight",
                    )
                    if hasattr(row, name)
                },
                "block_reason": row.reason,
                "adv20": adv20.get(asset),
                "adv20_observation_count": getattr(
                    row, "adv20_observation_count", None
                ),
                "adv20_missing_observations": getattr(
                    row, "adv20_missing_observations", None
                ),
                "prediction_rank_score": prediction_rank_score.get(asset),
                "decision_delta_weight": float(decision_delta.loc[asset]),
                "expected_alpha": enhanced_decision.expected_alpha.get(asset)
                if enhanced_decision is not None
                else None,
                "benchmark_weight": benchmark_weights.get(asset, 0.0)
                if benchmark_weights is not None and not benchmark_weights.empty
                else getattr(row, "benchmark_weight", None),
                "asset_specific_variance": enhanced_decision.risk.specific_variance.get(
                    asset
                )
                if enhanced_decision is not None and enhanced_decision.risk is not None
                else None,
            },
        )
        for phase, marks in mark_frames.items():
            if phase == "end" and valuations[-1].quantity == 0:
                continue
            if asset not in marks.index or pd.isna(
                mark_at := marks.loc[asset, "research_open_mark_at"]
            ):
                continue
            at = pd.Timestamp(marks.loc[asset, "timestamp"])
            if "research_open_evidence_identity" in marks and pd.notna(
                marks.loc[asset, "research_open_evidence_identity"]
            ):
                diagnostics.append(
                    PortfolioDiagnostic(
                        asset,
                        None,
                        f"research_{phase}_evidence_identity",
                        text=marks.loc[asset, "research_open_evidence_identity"],
                    )
                )
            diagnostics.extend(
                (
                    PortfolioDiagnostic(
                        asset,
                        None,
                        f"research_{phase}_mark_at",
                        text=mark_at.isoformat(),
                    ),
                    PortfolioDiagnostic(
                        asset,
                        None,
                        f"research_{phase}_mark_age_days",
                        number=float((at - mark_at).days),
                    ),
                    PortfolioDiagnostic(
                        asset,
                        None,
                        f"research_{phase}_mark_kind",
                        text=str(marks.loc[asset, "research_open_mark_kind"]),
                    ),
                    PortfolioDiagnostic(
                        asset,
                        None,
                        f"research_{phase}_listing_date",
                        text=marks.loc[asset, "listing_date"].isoformat(),
                    ),
                )
            )
    for asset in sorted(set(end_units.index) - set(asset_ledger.asset_id)):
        if end_units.get(asset, 0) or endpoint_state.receivables.get(asset, 0):
            emit(
                asset,
                0.0,
                0.0,
                None,
                {"target_notional": 0.0, "decision_delta_weight": 0.0},
            )
    emit(
        PORTFOLIO_CASH_ASSET_ID,
        account.pre_trade_cash,
        account.post_trade_cash,
        None,
        {
            "target_notional": max(
                0.0,
                account.nav_before
                - account.receivable
                - float(asset_ledger.target_notional.sum()),
            ),
            "decision_delta_weight": -float(decision_delta.sum()),
            "decision_reason": PORTFOLIO_DECISION_REASON_ENHANCED
            if optimization_result is not None
            and decision_reason == PORTFOLIO_DECISION_REASON_BENCHMARK
            else decision_reason,
            "research_valuation_policy": RESEARCH_VALUATION_POLICY,
            "initial_allocation_attempt": asset_ledger.attrs.get(
                "initial_allocation_attempt", 0.0
            ),
            **asset_ledger.attrs.get("decision_availability", {}),
            **asset_ledger.attrs.get("neutrality_exposures", {}),
            **asset_ledger.attrs.get("realized_constraints", {}),
            **(
                {
                    name: getattr(enhanced_decision.calibration, field)
                    for name, field in _ALPHA_DIAGNOSTICS.items()
                }
                if enhanced_decision
                else {}
            ),
            "predicted_tracking_error": optimization_result.predicted_tracking_error
            if optimization_result
            else None,
            "portfolio_factor_variance": optimization_result.factor_variance
            if optimization_result
            else None,
            "portfolio_specific_variance": optimization_result.specific_variance
            if optimization_result
            else None,
            "expected_transaction_cost": optimization_result.expected_transaction_cost
            if optimization_result
            else None,
        },
    )
    for event in rights_adjustments:
        if event.cash_amount:
            cashflows.append(
                PortfolioCashFlow(
                    f"{event.kind}:{event.event_id}",
                    event.timestamp_ns,
                    event.asset_id,
                    None,
                    event.kind,
                    event.cash_amount,
                )
            )
    unquoted = _unquoted_assets(endpoint_panel)
    end_values = [row for row in valuations if row.phase == "end"]
    nav_end = sum(row.value for row in end_values)
    diagnostics.extend(
        PortfolioDiagnostic(
            PORTFOLIO_CASH_ASSET_ID,
            None,
            f"research_valuation_nav_{percent}pct",
            number=nav_end
            if percent == 100
            else sum(
                row.receivable + percent / 100 * row.share_value
                if row.asset_id in unquoted
                else row.value
                for row in end_values
            ),
        )
        for percent in (0, 50, 100)
    )
    return PortfolioLedgerBatch(
        sequence,
        tuple(trades),
        tuple(cashflows),
        tuple(valuations),
        tuple(diagnostics),
        rights_adjustments,
    )


def _validate_windows(
    output_window: TimeWindow,
    input_window: TimeWindow,
    formal_dataset_window: TimeWindow,
) -> None:
    if not all(
        isinstance(value, TimeWindow)
        for value in (output_window, input_window, formal_dataset_window)
    ):
        raise ValueError("portfolio_windows_invalid")
    bounds = []
    for window in (output_window, input_window, formal_dataset_window):
        start, end, closed = time_window_timestamp_interval(window)
        bounds.append(
            (pd.Timestamp(start).value, pd.Timestamp(end).value - (0 if closed else 1))
        )
    output, inputs, formal = bounds
    if output[0] < formal[0] or output[1] > inputs[1]:
        raise ValueError("portfolio_windows_invalid")
