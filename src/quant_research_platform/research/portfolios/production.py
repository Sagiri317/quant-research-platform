"""Bounded production portfolio producer for the active target DAG."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_DECISION_REASON_BENCHMARK,
    PORTFOLIO_DECISION_REASON_ENHANCED,
    PORTFOLIO_DECISION_REASON_MAINTENANCE,
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutedTrade,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
    PortfolioValuation,
    _portfolio_session_frame,
    portfolio_notional_tolerance,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    _industry_group_codes,
    model_input_validity_columns,
)
from quant_research_platform.research.filtering import (
    A_SHARE_ELIGIBILITY_FACTS,
    a_share_eligibility,
)
from quant_research_platform.research.portfolios.accounting import (
    AlphaCalibration,
    ExecutionAccounting,
    alpha_calibration_intervals,
    calibrate_expected_alpha,
    drift_continuous_holdings,
    execute_continuous_target,
)
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    optimize_portfolio,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskEstimate,
    FactorRiskState,
    preprocess_risk_styles,
)
from quant_research_platform.research.portfolios.schedules import (
    holding_intervals,
    is_rebalance_session,
)
from quant_research_platform.runtime.execution_metrics import record_run_observations
from quant_research_platform.storage.datasets import DatasetReadMode
from quant_research_platform.storage.datasets.reading import (
    _iter_dataset_session_frames,
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.identity import filesystem_exists
from quant_research_platform.storage.layout import StorageLayout

_CALENDAR_COLUMNS = ("calendar_date", "source_market_type", "is_trading_day")
_DATASET_COLUMNS = ("asset_id", "timestamp", "eligible", "float_market_cap")
_PANEL_COLUMNS = (
    "asset_id",
    "listing_date",
    "timestamp",
    "adj_open",
    "adj_close",
    "amount",
    "market_type",
)
_ENHANCED_PANEL_COLUMNS = (
    "asset_id",
    "listing_date",
    "timestamp",
    "adj_open",
    "adj_close",
    "amount",
    "market_type",
    "period_return",
    "float_market_cap",
    "industry_standard_id",
    "industry_level1_code",
    "is_nonfinancial_industry",
    *tuple(
        column
        for column in A_SHARE_ELIGIBILITY_FACTS
        if column not in {"adj_close", "amount"}
    ),
)
_BENCHMARK_COLUMNS = ("benchmark_id", "asset_id", "timestamp", "weight")
_SESSION_KEY = ("timestamp", "asset_id")
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


@dataclass(frozen=True, slots=True)
class _EnhancedDecision:
    benchmark_weights: pd.Series
    expected_alpha: pd.Series
    exposures: pd.DataFrame
    risk: FactorRiskEstimate | None
    calibration: AlphaCalibration


def _price_returns(start, end, assets, *, start_column, end_column):
    """Value held assets only across endpoints of the same proved listing life."""
    start, end = start.reindex(assets), end.reindex(assets)
    prices = []
    for source, column in ((start, start_column), (end, end_column)):
        values = source[column]
        if values.isna().any():
            raise PortfolioTargetBlocked("portfolio_held_return_missing")
        values = pd.to_numeric(values, errors="raise").astype("float64")
        if not np.isfinite(values).all() or values.le(0).any():
            raise ValueError("portfolio_return_invalid")
        prices.append(values)
    if (
        start.listing_date.isna().any()
        or end.listing_date.isna().any()
        or not start.listing_date.eq(end.listing_date).all()
    ):
        raise PortfolioTargetBlocked("portfolio_held_lifecycle_unproven")
    with np.errstate(over="ignore"):
        result = prices[1] / prices[0] - 1.0
    if not np.isfinite(result).all():
        raise ValueError("portfolio_return_invalid")
    return result


def build_portfolio_ledger(
    *,
    candidate: PortfolioCandidateSpec,
    layout: StorageLayout,
    upstream_identities: Mapping[str, str],
    output_window: TimeWindow,
    input_window: TimeWindow,
    formal_dataset_window: TimeWindow,
    prediction_window: TimeWindow | None = None,
    risk_config: FactorRiskConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    frozen_calibration: AlphaCalibration | None = None,
    on_final_calibration=None,
    shared_alpha_facts: dict | None = None,
    shared_risk_fits: dict | None = None,
    prediction_lookup: object,
    persisted_model: object,
    access_window: str,
    assumptions: ExecutionAssumptions,
    fee_schedule_from_frame: Callable[..., object],
    require_fee_coverage: Callable[..., None],
    prepare_trade_cost: Callable[..., object],
    market_name_for_type: Callable[[object], str],
    unified_trading_sessions: Callable[[pd.DataFrame], pd.DatetimeIndex],
    iter_portfolio_sessions: Callable[..., Iterator],
) -> Iterator[PortfolioLedgerBatch]:
    """Build formal candidate sessions from bounded, exact-date reads."""
    if not isinstance(candidate, PortfolioCandidateSpec):
        raise ValueError("portfolio_candidate_invalid")
    if not isinstance(assumptions, ExecutionAssumptions):
        raise ValueError("portfolio_assumptions_invalid")
    _validate_windows(output_window, input_window, formal_dataset_window)
    enhanced = candidate.construction == "benchmark_enhanced"
    risk_style_columns = candidate.risk_style_columns
    calendar_identity = _required_identity(
        upstream_identities, candidate.trading_calendar_target_id
    )
    calendar_root = _required_target_root(
        layout,
        candidate.trading_calendar_target_id,
        expected_identity=calendar_identity,
    )
    calendar = _read_materialized(calendar_root, _CALENDAR_COLUMNS, time_window=None)
    try:
        sessions = pd.DatetimeIndex(unified_trading_sessions(calendar))
    except (TypeError, ValueError) as exc:
        raise ValueError("portfolio_trading_calendar_invalid") from exc
    sessions = sessions.normalize().drop_duplicates().sort_values()
    session_start = pd.Timestamp(input_window.start).normalize()
    session_end = min(
        pd.Timestamp(input_window.end).normalize(),
        pd.Timestamp(output_window.end).normalize(),
    )
    output_start = pd.Timestamp(output_window.start).normalize()
    output_positions = np.flatnonzero(sessions >= output_start)
    if len(output_positions):
        config = risk_config or FactorRiskConfig()
        warmup_sessions = (
            max(19, config.factor_window, config.specific_window) + 1
            if enhanced
            else 19
        )
        warmup_start = max(0, int(output_positions[0]) - warmup_sessions)
        session_start = max(session_start, sessions[warmup_start])
    calibration_intervals = alpha_calibration_intervals(
        sessions, candidate.rebalance_schedule
    )
    sessions = sessions[(sessions >= session_start) & (sessions <= session_end)]
    output_intervals = holding_intervals(sessions, output_window).set_index(
        "decision_at"
    )

    if sessions.empty:
        raise PortfolioTargetBlocked("portfolio_execution_window_empty")
    prediction_window = prediction_window or TimeWindow(
        max(
            pd.Timestamp(formal_dataset_window.start),
            output_start - pd.DateOffset(months=12),
        ).date(),
        pd.Timestamp(output_window.end).date(),
    )
    validity_by_quarter = {
        item["quarter_id"]: model_input_validity_columns(
            item["model_metadata"]["input_columns"],
            item["model_metadata"]["model_algorithm"],
        )
        for item in persisted_model.prediction_models(
            prediction_window,
            selection_group=candidate.selection_group,
            access_window=access_window,
        )
        if item["family_id"]
        == (
            persisted_model.selection(
                candidate.selection_group,
                item["quarter_id"],
                access_window=access_window,
            )["selected_family_id"]
            if candidate.prediction_selector == "champion"
            else candidate.prediction_selector
        )
    }
    dataset_columns = tuple(
        dict.fromkeys(
            (
                *_DATASET_COLUMNS,
                *(
                    column
                    for columns in validity_by_quarter.values()
                    for column in columns
                ),
            )
        )
    )
    alpha_start = max(session_start, pd.Timestamp(prediction_window.start))
    calibration_intervals = calibration_intervals.loc[
        calibration_intervals.timestamp.ge(alpha_start)
    ].copy()
    fee_identity = _required_identity(
        upstream_identities, candidate.fee_schedule_target_id
    )
    fee_root = _required_target_root(
        layout, candidate.fee_schedule_target_id, expected_identity=fee_identity
    )
    fee_frame = _read_materialized(fee_root, None, time_window=None)
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
    dataset_root = _required_input_root(
        layout, upstream_identities, candidate.research_dataset_id
    )
    panel_root = _required_input_root(
        layout, upstream_identities, candidate.market_panel_target_id
    )
    execution_root = _required_input_root(
        layout, upstream_identities, candidate.execution_mask_target_id
    )
    for target_id in ("market_bars_1d", "stock_master"):
        _required_input_root(layout, upstream_identities, target_id)
    benchmark_root = None
    if enhanced:
        benchmark_root = _required_input_root(
            layout, upstream_identities, candidate.benchmark_weights_target_id or ""
        )
    holdings = pd.Series(dtype="float64")
    cash = float(assumptions.initial_nav)
    nav = cash
    amount_history: dict[tuple[str, pd.Timestamp], deque[float]] = {}
    factor_risk = (
        FactorRiskState(style_columns=risk_style_columns, config=risk_config)
        if enhanced
        else None
    )
    previous_risk_cross: pd.DataFrame | None = None
    market_facts = tuple(
        (target_id, upstream_identities[target_id])
        for target_id in (
            candidate.market_panel_target_id,
            candidate.trading_calendar_target_id,
            "market_bars_1d",
            "stock_master",
        )
    )
    alpha_fact_key = (
        market_facts,
        upstream_identities[candidate.prediction_target_id],
        upstream_identities[candidate.research_dataset_id],
        candidate.prediction_selector,
        candidate.selection_group,
        candidate.rebalance_schedule,
        access_window,
        alpha_start,
        tuple(sorted(validity_by_quarter.items())),
    )
    risk_fit_key = (
        market_facts,
        tuple(
            (target_id, upstream_identities[target_id])
            for target_id in candidate.risk_context_target_ids
        ),
        risk_style_columns,
        risk_config or FactorRiskConfig(),
        session_start,
    )
    calibration_rows: list[dict[str, object]] = []
    pending_calibration: tuple[pd.Series, pd.Series, pd.Timestamp] | None = None
    waiting_alpha: tuple[pd.Series, pd.Timestamp] | None = None
    calibration_dates = calibration_intervals.set_index("timestamp")
    previous_benchmark_assets: frozenset[str] | None = None
    emitted = False
    sequence = 0
    with ExitStack() as input_channels:
        history = TimeWindow(sessions[0], sessions[-1])
        panel_columns = _ENHANCED_PANEL_COLUMNS if enhanced else _PANEL_COLUMNS
        panel_frames = input_channels.enter_context(
            closing_iterator(
                iter_portfolio_sessions(
                    layout=layout,
                    panel_root=panel_root,
                    panel_columns=panel_columns,
                    calendar=calendar,
                    time_window=history,
                    execution_root=execution_root,
                )
            )
        )
        dataset_frames = input_channels.enter_context(
            closing_iterator(
                _iter_input_sessions(dataset_root, dataset_columns, history)
            )
        )
        risk_inputs = (
            _risk_context_input_sessions(
                roots=tuple(
                    _required_input_root(layout, upstream_identities, target_id)
                    for target_id in candidate.risk_context_target_ids
                ),
                style_columns=risk_style_columns,
                time_window=history,
                retained=input_channels,
            )
            if enhanced
            else {}
        )
        panel_window = {}

        def release_panels(before):
            for timestamp in tuple(panel_window):
                if timestamp < before:
                    del panel_window[timestamp]

        def panel_at(timestamp):
            if timestamp not in panel_window:
                for view in panel_frames:
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
            if enhanced
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
            _update_amount_history(panel, amount_history, holdings, view.closed_assets)
            benchmark_weights = pd.Series(dtype="float64")
            alpha: pd.Series | None = None
            current_cross: pd.DataFrame | None = None
            if enhanced:
                assert factor_risk is not None
                risk_frames = []
                for columns, context_frames in (
                    risk_inputs.values() if not panel.empty else ()
                ):
                    context = _read_next_session(
                        context_frames, columns, decision_timestamp
                    )
                    _validate_input_keys(panel, context)
                    risk_frames.append(context)
                current_cross = _enhanced_risk_cross(
                    panel=panel,
                    risk_frames=risk_frames,
                    style_columns=risk_style_columns,
                )
                risk_frames.clear()

                if previous_risk_cross is not None and not panel.empty:
                    risk_cross = _factor_return_cross(
                        exposures=previous_risk_cross, panel=panel
                    )
                    if shared_risk_fits is None:
                        factor_risk.observe_return_session(risk_cross)
                    else:
                        factor_risk.observe_return_session(
                            risk_cross,
                            shared_fits=shared_risk_fits,
                            fit_key=(risk_fit_key, decision_timestamp),
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
                if calibration_rows and calibration_rows[0][
                    "timestamp"
                ] < decision_timestamp - pd.DateOffset(months=12):
                    calibration_rows[:] = [
                        row
                        for row in calibration_rows
                        if row["timestamp"]
                        >= decision_timestamp - pd.DateOffset(months=12)
                    ]
                if (
                    pending_calibration is not None
                    and decision_timestamp
                    == calibration_dates.loc[pending_calibration[2], "available_at"]
                ):
                    calibration_rows.extend(
                        _completed_calibration_rows(pending_calibration, view.valuation)
                    )
                    pending_calibration = None
                if (
                    waiting_alpha is not None
                    and decision_timestamp
                    == calibration_dates.loc[waiting_alpha[1], "entry_at"]
                ):
                    pending_calibration = (
                        waiting_alpha[0],
                        view.valuation.set_index(["asset_id", "listing_date"])[
                            "adj_open"
                        ].astype("float64"),
                        waiting_alpha[1],
                    )
                    waiting_alpha = None
                if decision_timestamp >= alpha_start:
                    dataset = (
                        _read_next_session(
                            dataset_frames, dataset_columns, decision_timestamp
                        )
                        if not panel.empty
                        else pd.DataFrame(columns=dataset_columns).astype(
                            {"eligible": "bool"}
                        )
                    )
                    if dataset.empty and not panel.empty:
                        raise PortfolioTargetBlocked(
                            "portfolio_research_dataset_session_missing"
                        )
                    _validate_input_keys(dataset, panel)
                    alpha = _prediction_alpha(
                        dataset,
                        prediction_lookup,
                        validity_by_quarter[str(decision_timestamp.to_period("Q"))],
                    )
                    has_next_session = position + 1 < len(sessions)
                    if (
                        alpha is not None
                        and frozen_calibration is None
                        and has_next_session
                        and (decision_timestamp in calibration_dates.index)
                        and is_rebalance_session(
                            decision_timestamp,
                            next_timestamp,
                            candidate.rebalance_schedule,
                        )
                    ):
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
                assert benchmark_frames is not None
                benchmark_weights = next(benchmark_frames)
            if decision_timestamp < output_start:
                continue
            if not enhanced:
                dataset = (
                    _read_next_session(
                        dataset_frames, dataset_columns, decision_timestamp
                    )
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
            if not enhanced:
                alpha = _prediction_alpha(
                    dataset,
                    prediction_lookup,
                    validity_by_quarter[str(decision_timestamp.to_period("Q"))],
                )
            if decision_timestamp not in output_intervals.index:
                continue
            endpoint_timestamp = output_intervals.loc[decision_timestamp, "end_at"]
            next_view = panel_at(next_timestamp)
            next_panel = next_view.valuation
            endpoint_panel = panel_at(endpoint_timestamp).valuation
            enhanced_decision: _EnhancedDecision | None = None
            target_override: pd.Series | None = None
            decision_reason = PORTFOLIO_DECISION_REASON_SIMPLE
            if not holdings.empty:
                prices = view.valuation.set_index("asset_id")
                holdings, cash, nav = drift_continuous_holdings(
                    holdings,
                    cash,
                    _price_returns(
                        prices,
                        prices,
                        holdings.index,
                        start_column="adj_open",
                        end_column="adj_close",
                    ),
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
                initializing = not emitted
                maintenance = membership_changed and (not scheduled)
                if initializing:
                    if benchmark_weights.empty:
                        raise PortfolioTargetBlocked(
                            "portfolio_benchmark_snapshot_missing"
                        )
                    if set(benchmark_weights.index) & next_view.closed_assets:
                        raise PortfolioTargetBlocked(
                            "portfolio_initial_holding_market_closed"
                        )
                    holdings = benchmark_weights.multiply(nav)
                    cash = 0.0
                if scheduled or maintenance:
                    if benchmark_weights.empty:
                        raise PortfolioTargetBlocked(
                            "portfolio_benchmark_snapshot_missing"
                        )
                    enhanced_decision = _enhanced_decision(
                        benchmark_weights=benchmark_weights,
                        alpha=alpha,
                        factor_risk=factor_risk,
                        current_cross=current_cross,
                        calibration_rows=calibration_rows,
                        calibration_intervals=calibration_intervals,
                        as_of=decision_timestamp,
                        risk_style_columns=risk_style_columns,
                        frozen_calibration=frozen_calibration,
                        shared_alpha_facts=shared_alpha_facts,
                        alpha_fact_key=alpha_fact_key,
                    )
                    if maintenance:
                        target_override = benchmark_weights.copy(deep=True)
                        decision_reason = PORTFOLIO_DECISION_REASON_MAINTENANCE
                else:
                    target_override = holdings.div(nav).copy(deep=True)
                    decision_reason = PORTFOLIO_DECISION_REASON_BENCHMARK
            if enhanced:
                required_assets = benchmark_weights.index.union(holdings.index)
                industry = _industry_group_codes(
                    view.valuation.set_index("asset_id"), "industry_level1_code"
                ).reindex(required_assets)
                if industry.isna().any():
                    raise PortfolioTargetBlocked("portfolio_industry_context_missing")
            account, returns, adv, _optimization, decision_delta, asset_ledger = (
                _build_session(
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
                    cash=cash,
                    nav=nav,
                    initialize_at_execution=enhanced and (not emitted),
                    amount_history=amount_history,
                    alpha=alpha,
                    enhanced_decision=enhanced_decision,
                    target_override=target_override,
                    decision_reason=decision_reason,
                    optimizer_config=optimizer_config,
                    assumptions=assumptions,
                    fee_schedule=fee_schedule,
                    fee_identity=fee_identity,
                    prepare_trade_cost=prepare_trade_cost,
                    market_name_for_type=market_name_for_type,
                )
            )
            yielded_value = _ledger_batch(
                account=account,
                asset_ledger=asset_ledger,
                sequence=sequence,
                end_timestamp=endpoint_timestamp,
                decision_timestamp=decision_timestamp,
                returns=returns,
                alpha_score=alpha,
                adv20=adv,
                enhanced_decision=enhanced_decision,
                benchmark_weights=benchmark_weights if enhanced else None,
                optimization_result=_optimization,
                decision_reason=decision_reason,
                decision_delta=decision_delta,
            )
            holdings = asset_ledger.set_index("asset_id")["post_trade_notional"]
            holdings = holdings.loc[holdings.ne(0.0)]
            cash, nav = (account.post_trade_cash, account.nav_after)
            emitted = True
            try:
                yield yielded_value
            except GeneratorExit:
                return
            sequence += 1
        missing, extra = prediction_lookup.coverage_counts()
        if missing or extra:
            raise ValueError("misaligned_persisted_prediction_keys")
        if not emitted:
            raise PortfolioTargetBlocked("portfolio_execution_window_empty")
        if enhanced and access_window == "development":
            if on_final_calibration is None:
                raise ValueError("portfolio_final_calibration_sink_required")
            observation = {}
            with record_run_observations(observation):
                calibration = _calibration_fact(
                    calibration_rows,
                    calibration_intervals,
                    as_of=pd.Timestamp(output_window.end),
                    shared_alpha_facts=shared_alpha_facts,
                    alpha_fact_key=alpha_fact_key,
                )
            on_final_calibration(
                {
                    "schema_id": "qrp-final-alpha-calibration",
                    "selection_group": candidate.selection_group,
                    "cutoff": pd.Timestamp(output_window.end).date().isoformat(),
                    "upstream_identities": dict(upstream_identities),
                    "calibration": asdict(calibration),
                    "observations": observation,
                }
            )


def _read_materialized(
    root: Path, columns: tuple[str, ...] | None, *, time_window: TimeWindow | None
) -> pd.DataFrame:
    frame = read_dataset(
        root,
        columns=columns,
        time_window=time_window,
        mode=DatasetReadMode.MATERIALIZED,
    )
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("portfolio_input_reader_not_materialized")
    return frame


def _iter_input_sessions(
    root: Path, columns: tuple[str, ...], time_window: TimeWindow
) -> Iterator[pd.DataFrame]:
    session = open_dataset_session(root, columns=columns, time_window=time_window)
    with closing_iterator(
        _iter_dataset_session_frames(session, columns)
    ) as delegated_chunks:
        for yielded_value in delegated_chunks:
            try:
                yield yielded_value
            except GeneratorExit:
                return


def _read_next_session(
    frames: Iterator[pd.DataFrame], columns: tuple[str, ...], timestamp: pd.Timestamp
) -> pd.DataFrame:
    for frame in frames:
        actual = pd.Timestamp(frame["timestamp"].iloc[0])
        if actual < timestamp:
            continue
        if actual > timestamp:
            break
        normalized = _portfolio_session_frame(frame, columns)
        return normalized
    return pd.DataFrame(columns=columns)


def _risk_context_input_sessions(
    *,
    roots: tuple[Path, ...],
    style_columns: tuple[str, ...],
    time_window: TimeWindow,
    retained: ExitStack,
):
    required = set(style_columns)
    available: set[str] = set()
    inputs = {}
    for root in roots:
        session = open_dataset_session(root, columns=None, time_window=time_window)
        columns = tuple(
            column for column in style_columns if column in session.arrow_schema.names
        )
        if available.intersection(columns):
            raise ValueError("portfolio_risk_context_owner_duplicate")
        available.update(columns)
        projection = (*_SESSION_KEY, *columns)
        inputs[root] = (
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
) -> pd.DataFrame:
    panel_by_asset = panel.set_index("asset_id")
    assets = pd.Index(sorted(set(panel_by_asset.index)), dtype="string")
    result = pd.DataFrame(index=assets)
    result["timestamp"] = panel["timestamp"].iloc[0] if not panel.empty else pd.NaT
    result["listing_date"] = panel_by_asset["listing_date"].reindex(assets)
    result["asset_return"] = (
        pd.to_numeric(panel_by_asset["period_return"], errors="raise")
        .reindex(assets)
        .astype("float64")
    )
    result["float_market_cap"] = (
        pd.to_numeric(panel_by_asset["float_market_cap"], errors="raise")
        .reindex(assets)
        .astype("float64")
    )
    result["industry_standard_id"] = (
        panel_by_asset["industry_standard_id"].astype("string").reindex(assets)
    )
    result["industry_level1_code"] = (
        panel_by_asset["industry_level1_code"].astype("string").reindex(assets)
    )
    result["eligible"] = a_share_eligibility(panel_by_asset, "pv_expanded").reindex(
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
            "alpha_score": float(alpha.loc[asset]),
            "holding_return": float(returns.loc[asset]),
        }
        for asset in alpha.index[valid]
    ]


def _iter_benchmark_weights(root, benchmark_id, sessions):
    """Align sparse snapshots without consuming a future session prematurely."""
    with closing_iterator(
        _iter_input_sessions(
            root, _BENCHMARK_COLUMNS, TimeWindow(sessions[0], sessions[-1])
        )
    ) as frames:
        pending = next(frames, None)
        for timestamp in sessions:
            while pending is not None and pending["timestamp"].iloc[0] < timestamp:
                pending = next(frames, None)
            if pending is None or pending["timestamp"].iloc[0] > timestamp:
                yielded_value = pd.Series(dtype="float64")
                try:
                    yield yielded_value
                except GeneratorExit:
                    return
            else:
                yielded_value = _benchmark_weights(pending, benchmark_id, timestamp)
                try:
                    yield yielded_value
                except GeneratorExit:
                    return


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


def _calibration_fact(rows, intervals, *, as_of, shared_alpha_facts, alpha_fact_key):
    key = (alpha_fact_key, as_of)
    if shared_alpha_facts is not None and key in shared_alpha_facts:
        return shared_alpha_facts[key]
    calibration = calibrate_expected_alpha(
        pd.DataFrame(rows, columns=["timestamp", "alpha_score", "holding_return"]),
        as_of=as_of,
        scheduled_intervals=intervals,
    )
    if shared_alpha_facts is not None:
        shared_alpha_facts[key] = calibration
    return calibration


def _enhanced_decision(
    *,
    benchmark_weights: pd.Series,
    alpha: pd.Series,
    factor_risk: FactorRiskState,
    current_cross: pd.DataFrame,
    calibration_rows: list[dict[str, object]],
    calibration_intervals: pd.DataFrame,
    as_of: pd.Timestamp,
    risk_style_columns: tuple[str, ...],
    frozen_calibration: AlphaCalibration | None = None,
    shared_alpha_facts: dict | None = None,
    alpha_fact_key=None,
) -> _EnhancedDecision:
    calibration = frozen_calibration or _calibration_fact(
        calibration_rows,
        calibration_intervals,
        as_of=as_of,
        shared_alpha_facts=shared_alpha_facts,
        alpha_fact_key=alpha_fact_key,
    )
    expected = pd.Series(np.nan, index=alpha.index, dtype="float64")
    if not calibration.benchmark_only:
        expected.loc[alpha.notna()] = calibration.slope * alpha.loc[alpha.notna()]
    else:
        expected.loc[alpha.notna()] = 0.0
    complete = current_cross.loc[
        current_cross.loc[:, risk_style_columns].notna().all(axis=1)
    ]
    complete = complete.assign(
        eligible=complete.asset_id.isin(
            factor_risk.prepare_exposures(current_cross).asset_id
        )
    )
    processed = (
        preprocess_risk_styles(complete, style_columns=risk_style_columns)
        if not complete.empty
        else complete
    )
    exposures = factor_risk.current_exposures(processed)
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
    panel: pd.DataFrame,
) -> pd.DataFrame:
    assets = pd.Index(
        sorted(set(benchmark_weights.index) | set(current_weights.index)),
        dtype="string",
    )
    panel_by_asset = panel.set_index("asset_id")
    exposure_columns = tuple(
        column
        for column in exposures.columns
        if column not in {"asset_id", "industry_standard_id", "industry_level1_code"}
    )
    exposures_by_asset = exposures.set_index("asset_id")
    nonfinancial = _bool_series(
        panel_by_asset["is_nonfinancial_industry"].reindex(assets),
        "portfolio_financial_context_invalid",
    )
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
            "is_financial": (~nonfinancial).to_numpy(),
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
    if cash_weight > 1e-12:
        cash_row = {
            "asset_id": PORTFOLIO_CASH_ASSET_ID,
            "expected_alpha": 0.0,
            "current_weight": cash_weight,
            "benchmark_weight": 0.0,
            "specific_variance": 0.0,
            "adv": np.inf,
            "eligible": True,
            "is_financial": False,
            "is_cash": True,
            **{column: 0.0 for column in exposure_columns},
        }
        result = pd.concat([result, pd.DataFrame([cash_row])], ignore_index=True)
    if not np.isclose(float(result["current_weight"].sum()), 1.0):
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
    cash: float,
    nav: float,
    initialize_at_execution: bool,
    amount_history: Mapping[tuple[str, pd.Timestamp], deque[float]],
    alpha: pd.Series,
    enhanced_decision: _EnhancedDecision | None,
    target_override: pd.Series | None,
    decision_reason: str,
    optimizer_config: OptimizerConfig | None = None,
    assumptions: ExecutionAssumptions,
    fee_schedule: object,
    fee_identity: str,
    prepare_trade_cost: Callable[..., object],
    market_name_for_type: Callable[[object], str],
) -> tuple[
    ExecutionAccounting,
    pd.Series,
    pd.Series,
    OptimizationResult | None,
    pd.Series,
    pd.DataFrame,
]:
    dataset = dataset.set_index("asset_id", drop=False)
    if not np.isclose(
        float(holdings.sum()) + cash,
        nav,
        rtol=0.0,
        atol=portfolio_notional_tolerance(nav),
    ):
        raise ValueError("portfolio_state_invalid")
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    alpha = alpha.reindex(dataset.index)
    prediction_eligible = eligible & alpha.notna()
    optimization: OptimizationResult | None = None
    if target_override is not None:
        target = target_override.copy(deep=True)
    elif enhanced_decision is not None:
        target = enhanced_decision.benchmark_weights.copy(deep=True)
    elif is_rebalance_session(
        decision_timestamp, next_timestamp, candidate.rebalance_schedule
    ):
        if not prediction_eligible.any():
            target = pd.Series(0.0, index=dataset.index, dtype="float64")
        else:
            target = build_simple_target_weights(
                alpha,
                eligible=prediction_eligible,
                method="float_cap_weight"
                if candidate.construction == "top20_float_cap"
                else "equal_weight",
                top_quantile=float(candidate.top_quantile),
                bottom_quantile=float(candidate.top_quantile),
                float_market_cap=dataset["float_market_cap"]
                if candidate.construction == "top20_float_cap"
                else None,
            )
            if candidate.construction == "top20_equal_weight_half_step":
                retained = holdings.div(nav).reindex(target.index, fill_value=0.0)
                retained = retained.where(prediction_eligible, 0.0)
                target = (target + retained) / 2.0
    else:
        target = holdings.div(nav).copy(deep=True)
    target = target.loc[target.ne(0.0)]
    current = holdings.loc[holdings.ne(0.0)].copy(deep=True)
    assets = pd.Index(sorted(set(current.index) | set(target.index)), dtype="string")
    if PORTFOLIO_CASH_ASSET_ID in assets:
        raise ValueError("portfolio_asset_universe_invalid")
    panel_by_asset = panel.set_index("asset_id")
    missing_panel = [asset for asset in assets if asset not in panel_by_asset.index]
    if missing_panel:
        raise PortfolioTargetBlocked("portfolio_held_asset_missing_from_panel")
    market_by_asset = {
        str(asset): market_name_for_type(panel_by_asset.loc[asset, "market_type"])
        for asset in assets
    }
    adv = pd.Series(
        {
            asset: float(mean(history))
            if len(history := amount_history.get((asset, listing), ())) == 20
            else np.nan
            for asset, listing in panel_by_asset.loc[assets, "listing_date"].items()
        },
        dtype="float64",
    )
    masks = _mask_values(execution, assets, next_timestamp, closed_assets)
    prepared = {
        (market, side): prepare_trade_cost(
            next_timestamp.date(), market, side, fee_schedule, assumptions
        )
        for market in set(market_by_asset.values())
        for side in ("buy", "sell")
    }

    def cost_for_trade(
        execution_date, asset, side, notional, adv20, called_assumptions
    ):
        return prepared[market_by_asset[asset], side].trade(notional, adv20)

    def cost_totals_for_buys(buy_assets, notionals, adv_values):
        totals = np.zeros(len(buy_assets), dtype="float64")
        for market in set(market_by_asset.values()):
            mask = np.asarray(
                [market_by_asset[asset] == market for asset in buy_assets]
            ) & (notionals > 0.0)
            if mask.any():
                totals[mask] = prepared[market, "buy"].totals(
                    notionals[mask], adv_values[mask]
                )
        return totals

    if (
        target_override is None
        and enhanced_decision is not None
        and (enhanced_decision.risk is not None)
        and (not enhanced_decision.calibration.benchmark_only)
    ):
        risk = enhanced_decision.risk
        assets_frame = _optimizer_assets(
            benchmark_weights=enhanced_decision.benchmark_weights,
            current_weights=holdings.div(nav),
            cash=cash,
            nav=nav,
            expected_alpha=enhanced_decision.expected_alpha,
            exposures=enhanced_decision.exposures,
            specific_variance=risk.specific_variance,
            adv=adv,
            eligible=eligible,
            panel=panel,
        )
        current_values = assets_frame["current_weight"].to_numpy(dtype="float64")
        adv_values = assets_frame["adv"].to_numpy(dtype="float64")
        market_masks = {
            market: assets_frame["asset_id"].map(market_by_asset).eq(market).to_numpy()
            for market in set(market_by_asset.values())
        }

        def expected_cost(weights: np.ndarray) -> float:
            delta = weights - current_values
            notionals = np.abs(delta) * nav
            totals = np.zeros(len(delta), dtype="float64")
            for (market, side), evaluator in prepared.items():
                active = market_masks[market] & (
                    delta > 0 if side == "buy" else delta < 0
                )
                if active.any():
                    totals[active] = evaluator.totals(
                        notionals[active], adv_values[active]
                    )
            return sum(map(float, totals), 0.0)

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
        )
        target = optimization.weights.loc[
            optimization.weights.index != PORTFOLIO_CASH_ASSET_ID
        ]
    orders_requested = (
        is_rebalance_session(
            decision_timestamp, next_timestamp, candidate.rebalance_schedule
        )
        or decision_reason == PORTFOLIO_DECISION_REASON_MAINTENANCE
    )
    decision_delta = (
        target.reindex(assets, fill_value=0.0)
        - current.div(nav).reindex(assets, fill_value=0.0)
        if orders_requested
        else pd.Series(0.0, index=assets)
    )
    execution_holdings, execution_cash, execution_nav = (current, cash, nav)
    if not initialize_at_execution and (not current.empty):
        overnight = _price_returns(
            panel_by_asset,
            next_panel.set_index("asset_id"),
            current.index,
            start_column="adj_close",
            end_column="adj_open",
        )
        execution_holdings, execution_cash, execution_nav = drift_continuous_holdings(
            current, cash, overnight
        )
    execution_weights = execution_holdings.div(execution_nav)
    execution_target = target if orders_requested else execution_weights
    try:
        account = execute_continuous_target(
            execution_weights,
            execution_target,
            nav=execution_nav,
            average_daily_volume=adv,
            can_enter=masks["can_enter"],
            can_exit=masks["can_exit"],
            execution_date=next_timestamp.date(),
            cost_for_trade=cost_for_trade,
            cost_totals_for_buys=cost_totals_for_buys,
            assumptions=assumptions,
            cost_source_identity=fee_identity,
        )
    except ValueError as error:
        if str(error) != "execution_adv_invalid:missing_required_assets":
            raise
        raise PortfolioTargetBlocked("portfolio_adv_warmup_insufficient") from error
    asset_ledger = account.asset_ledger
    returns = _holding_returns(asset_ledger, next_panel, endpoint_panel)
    return (account, returns, adv, optimization, decision_delta, asset_ledger)


def _prediction_alpha(
    dataset: pd.DataFrame, prediction_lookup: object, validity_columns: tuple[str, ...]
) -> pd.Series:
    dataset = dataset.set_index("asset_id", drop=False)
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    prediction_eligible = eligible & dataset.loc[:, list(validity_columns)].notna().all(
        axis=1
    )
    required_rows = dataset.loc[prediction_eligible, ["timestamp", "asset_id"]]
    alpha = pd.Series(np.nan, index=dataset.index, dtype="float64")
    if required_rows.empty:
        return alpha
    values_for_frame = getattr(prediction_lookup, "values_for_frame", None)
    if not callable(values_for_frame):
        raise ValueError("portfolio_prediction_lookup_invalid")
    values, found = values_for_frame(required_rows, mark_matched=True)
    if not all(found):
        raise ValueError("missing_required_prediction_keys")
    alpha.loc[required_rows["asset_id"]] = np.asarray(values, dtype="float64")
    if not np.isfinite(alpha.loc[prediction_eligible].to_numpy(dtype="float64")).all():
        raise ValueError("portfolio_prediction_values_invalid")
    return alpha


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
    holdings: pd.Series,
    closed_assets: frozenset[str],
) -> None:
    current = dict(
        zip(panel["asset_id"].astype(str), panel["listing_date"], strict=True)
    )
    for key in tuple(history):
        asset, listing = key
        if asset in current and current[asset] != listing:
            if float(holdings.get(asset, 0.0)) != 0.0:
                raise PortfolioTargetBlocked("portfolio_held_lifecycle_unproven")
            del history[key]
        elif asset not in current and asset not in closed_assets:
            if float(holdings.get(asset, 0.0)) != 0.0:
                raise PortfolioTargetBlocked("portfolio_held_asset_missing_from_panel")
            del history[key]
    amounts = pd.to_numeric(panel["amount"], errors="raise").astype("float64")
    if not np.isfinite(amounts.dropna().to_numpy()).all() or amounts.lt(0).any():
        raise ValueError("portfolio_amount_invalid")
    for asset, listing, amount in zip(
        panel["asset_id"].astype(str), panel["listing_date"], amounts, strict=True
    ):
        history.setdefault((asset, listing), deque(maxlen=20)).append(float(amount))


def _mask_values(
    execution: pd.DataFrame,
    assets: pd.Index,
    execution_at: pd.Timestamp,
    closed_assets: frozenset[str],
) -> dict[str, pd.Series]:
    masks = {
        column: pd.Series(False, index=assets, dtype="bool")
        for column in ("can_enter", "can_exit")
    }
    open_assets = assets[~assets.isin(closed_assets)]
    if open_assets.empty:
        return masks
    if not pd.api.types.is_bool_dtype(
        execution["can_enter"]
    ) or not pd.api.types.is_bool_dtype(execution["can_exit"]):
        raise ValueError("portfolio_execution_mask_invalid")
    indexed = execution.set_index("asset_id")
    if not open_assets.isin(indexed.index).all():
        raise PortfolioTargetBlocked("portfolio_execution_mask_missing")
    current = indexed.loc[open_assets]
    if current[["can_enter", "can_exit", "execution_at"]].isna().any().any():
        raise PortfolioTargetBlocked("portfolio_execution_fact_unknown")
    if not current.execution_at.eq(execution_at).all():
        raise PortfolioTargetBlocked("portfolio_execution_date_misaligned")
    for column in masks:
        masks[column].loc[open_assets] = current[column].astype(bool)
    return masks


def _holding_returns(
    asset_ledger: pd.DataFrame, next_panel: pd.DataFrame, endpoint_panel: pd.DataFrame
) -> pd.Series:
    assets = asset_ledger["asset_id"].astype(str)
    returns = pd.Series(0.0, index=assets, dtype="float64")
    selected = assets.loc[asset_ledger["post_trade_notional"].to_numpy() > 0.0]
    if selected.empty:
        return returns
    returns.loc[selected] = _price_returns(
        next_panel.set_index("asset_id"),
        endpoint_panel.set_index("asset_id"),
        selected,
        start_column="adj_open",
        end_column="adj_open",
    )
    return returns


def _ledger_batch(
    *,
    account: ExecutionAccounting,
    asset_ledger: pd.DataFrame,
    sequence: int,
    end_timestamp: pd.Timestamp,
    decision_timestamp: pd.Timestamp,
    returns: pd.Series,
    alpha_score: pd.Series,
    adv20: pd.Series,
    enhanced_decision: _EnhancedDecision | None,
    benchmark_weights: pd.Series | None,
    optimization_result: OptimizationResult | None,
    decision_reason: str,
    decision_delta: pd.Series,
) -> PortfolioLedgerBatch:
    """Build economic facts directly from one isolated accounting snapshot."""
    trades, cashflows, valuations, diagnostics = [], [], [], []
    timestamp = int(pd.Timestamp(account.execution_date).value)

    def emit(asset, before, after, holding_return, fill_id, values):
        valuations.extend(
            (
                PortfolioValuation(timestamp, asset, "before", before, before),
                PortfolioValuation(timestamp, asset, "after", after, after),
                PortfolioValuation(
                    int(end_timestamp.value),
                    asset,
                    "end",
                    after,
                    after * (1.0 + holding_return),
                    1.0 + holding_return,
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
                ExecutedTrade(fill_id, fill_id, timestamp, asset, delta, delta)
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
            float(returns.get(asset, 0.0)),
            fill_id,
            {
                "target_notional": float(row.target_notional),
                "block_reason": row.reason,
                "adv20": adv20.get(asset),
                "alpha_score": alpha_score.get(asset),
                "decision_delta_weight": float(decision_delta.loc[asset]),
                "expected_alpha": enhanced_decision.expected_alpha.get(asset)
                if enhanced_decision is not None
                else None,
                "benchmark_weight": benchmark_weights.get(asset)
                if benchmark_weights is not None
                else None,
                "asset_specific_variance": enhanced_decision.risk.specific_variance.get(
                    asset
                )
                if enhanced_decision is not None and enhanced_decision.risk is not None
                else None,
            },
        )
    emit(
        PORTFOLIO_CASH_ASSET_ID,
        account.pre_trade_cash,
        account.post_trade_cash,
        0.0,
        None,
        {
            "target_notional": max(
                0.0, account.nav_before - float(asset_ledger.target_notional.sum())
            ),
            "decision_delta_weight": -float(decision_delta.sum()),
            "decision_reason": PORTFOLIO_DECISION_REASON_ENHANCED
            if optimization_result is not None
            and decision_reason == PORTFOLIO_DECISION_REASON_BENCHMARK
            else decision_reason,
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
    return PortfolioLedgerBatch(
        sequence, tuple(trades), tuple(cashflows), tuple(valuations), tuple(diagnostics)
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
    if (
        input_window.start > input_window.end
        or formal_dataset_window.start > formal_dataset_window.end
        or output_window.start > output_window.end
        or (output_window.start < formal_dataset_window.start)
        or (output_window.end > input_window.end)
    ):
        raise ValueError("portfolio_windows_invalid")


def _required_input_root(
    layout: StorageLayout, upstream_identities: Mapping[str, str], target_id: str
) -> Path:
    return _required_target_root(
        layout,
        target_id,
        expected_identity=_required_identity(upstream_identities, target_id),
    )


def _required_target_root(
    layout: StorageLayout, target_id: str, *, expected_identity: str
) -> Path:
    marker = Path("__missing_portfolio_target__")
    root = layout.target_root(target_id, marker)
    if root == marker or not filesystem_exists(root) or root.name != expected_identity:
        raise ValueError(f"portfolio_input_missing: {target_id}")
    return root


def _required_identity(upstream_identities: Mapping[str, str], target_id: str) -> str:
    identity = upstream_identities.get(target_id)
    if (
        type(identity) is not str
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError(f"portfolio_input_identity_invalid: {target_id}")
    return identity
