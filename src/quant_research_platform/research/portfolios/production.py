"""Bounded production portfolio producer for the active target DAG."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import (
    CONTINUOUS_EXECUTION_COLUMNS,
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_DECISION_REASON_BENCHMARK,
    PORTFOLIO_DECISION_REASON_ENHANCED,
    PORTFOLIO_DECISION_REASON_MAINTENANCE,
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import (
    A_SHARE_ELIGIBILITY_FACTS,
    a_share_eligibility,
)
from quant_research_platform.research.portfolios.accounting import (
    AlphaCalibration,
    ExecutionAccounting,
    alpha_calibration_intervals,
    calibrate_expected_alpha,
    continuous_ledger_batch,
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
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import DatasetReadMode
from quant_research_platform.storage.datasets.reading import (
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
)
from quant_research_platform.storage.layout import StorageLayout

_CALENDAR_COLUMNS = ("calendar_date", "source_market_type", "is_trading_day")
_DATASET_COLUMNS = ("asset_id", "timestamp", "eligible", "float_market_cap")
_PANEL_COLUMNS = (
    "asset_id",
    "timestamp",
    "adj_open",
    "adj_close",
    "amount",
    "market_type",
)
_ENHANCED_PANEL_COLUMNS = (
    "asset_id",
    "timestamp",
    "adj_open",
    "adj_close",
    "amount",
    "market_type",
    "period_return",
    "float_market_cap",
    "industry_level1_code",
    "is_nonfinancial_industry",
    *tuple(
        column
        for column in A_SHARE_ELIGIBILITY_FACTS
        if column not in {"adj_close", "amount"}
    ),
)
_EXECUTION_COLUMNS = ("asset_id", "timestamp", "can_enter", "can_exit")
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


def frozen_alpha_calibration(
    batches, *, cutoff: pd.Timestamp
) -> tuple[AlphaCalibration, str]:
    """Read the last published, sufficiently observed development calibration."""
    latest = None
    with closing_iterator(batches):
        for batch in batches:
            observed_at = pd.Timestamp(
                min(row.timestamp_ns for row in batch.valuations)
            )
            if observed_at > cutoff:
                raise ValueError("holdout_alpha_source_after_cutoff")
            values = {
                _ALPHA_DIAGNOSTICS[row.name]: (
                    row.text if row.text is not None else row.number
                )
                for row in batch.diagnostics
                if row.asset_id == PORTFOLIO_CASH_ASSET_ID
                and row.name in _ALPHA_DIAGNOSTICS
            }
            if values and "status" not in values:
                raise PortfolioTargetBlocked("holdout_alpha_calibration_missing")
            if "status" in values:
                latest = (values, observed_at.date().isoformat())
    if latest is None:
        raise PortfolioTargetBlocked("holdout_alpha_calibration_missing")
    values, observed_at = latest
    if values["status"] == "UNDEFINED":
        raise PortfolioTargetBlocked("holdout_alpha_calibration_undefined")
    if values.pop("status") != "DEFINED":
        raise ValueError("holdout_alpha_calibration_status_invalid")
    diagnostic_names = {
        "inference_status",
        "inference_reason",
        "r2_status",
        "r2_reason",
    }
    if not diagnostic_names.issubset(values):
        raise ValueError("holdout_alpha_calibration_statistics_invalid")
    diagnostics = {name: values.pop(name) for name in diagnostic_names}
    missing = set()
    if diagnostics["inference_status"] == "UNDEFINED":
        if values.get("slope_standard_error") != 0.0:
            raise ValueError("holdout_alpha_calibration_statistics_invalid")
        missing.update(
            {"slope_t_statistic", "slope_pvalue", "slope_ci_low", "slope_ci_high"}
        )
    if diagnostics["r2_status"] == "UNDEFINED":
        missing.add("weighted_r2")
    expected = (
        set(_ALPHA_DIAGNOSTICS.values())
        - {"status", "undefined_reason"}
        - diagnostic_names
        - missing
    )
    if set(values) != expected:
        raise ValueError("holdout_alpha_calibration_statistics_invalid")
    values.update(dict.fromkeys(missing))
    if values["valid_rebalance_dates"] < 6 or values["coverage"] < 0.8:
        raise PortfolioTargetBlocked("holdout_alpha_calibration_insufficient")
    for name in ("valid_rebalance_dates", "scheduled_rebalance_dates"):
        values[name] = int(values[name])
    calibration = AlphaCalibration(**values, undefined_reason=None)
    if any(getattr(calibration, name) != value for name, value in diagnostics.items()):
        raise ValueError("holdout_alpha_calibration_statistics_invalid")
    return calibration, observed_at


@dataclass(frozen=True, slots=True)
class _EnhancedDecision:
    benchmark_weights: pd.Series
    expected_alpha: pd.Series
    exposures: pd.DataFrame
    risk: FactorRiskEstimate | None
    calibration: AlphaCalibration


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
    prediction_lookup: object,
    assumptions: ExecutionAssumptions,
    resource_policy: RuntimeResourcePolicy | None,
    fee_schedule_from_frame: Callable[..., object],
    require_fee_coverage: Callable[..., None],
    prepare_trade_cost: Callable[..., object],
    market_name_for_type: Callable[[object], str],
    unified_trading_sessions: Callable[[pd.DataFrame], pd.DatetimeIndex],
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
        upstream_identities,
        candidate.trading_calendar_target_id,
    )
    calendar_root = _required_target_root(
        layout,
        candidate.trading_calendar_target_id,
        expected_identity=calendar_identity,
    )
    calendar = _read_materialized(
        calendar_root,
        _CALENDAR_COLUMNS,
        time_window=None,
        resource_policy=resource_policy,
    )
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
    del calendar
    if sessions.empty:
        raise PortfolioTargetBlocked("portfolio_execution_window_empty")
    prediction_window = prediction_window or TimeWindow(
        max(
            pd.Timestamp(formal_dataset_window.start),
            output_start - pd.DateOffset(months=12),
        ).date(),
        pd.Timestamp(output_window.end).date(),
    )
    alpha_start = max(session_start, pd.Timestamp(prediction_window.start))
    calibration_intervals = calibration_intervals.loc[
        calibration_intervals.timestamp.ge(alpha_start)
    ].copy()

    fee_identity = _required_identity(
        upstream_identities,
        candidate.fee_schedule_target_id,
    )
    fee_root = _required_target_root(
        layout,
        candidate.fee_schedule_target_id,
        expected_identity=fee_identity,
    )
    fee_frame = _read_materialized(
        fee_root,
        None,
        time_window=None,
        resource_policy=resource_policy,
    )
    fee_schedule = fee_schedule_from_frame(
        fee_frame,
        content_identity=fee_identity,
    )
    del fee_frame
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
        layout,
        upstream_identities,
        candidate.research_dataset_id,
    )
    panel_root = _required_input_root(
        layout,
        upstream_identities,
        candidate.market_panel_target_id,
    )
    execution_root = _required_input_root(
        layout,
        upstream_identities,
        candidate.execution_mask_target_id,
    )
    benchmark_root = None
    if enhanced:
        benchmark_root = _required_input_root(
            layout,
            upstream_identities,
            candidate.benchmark_weights_target_id or "",
        )

    risk_inputs: dict[Path, tuple[str, ...]] = {}
    if enhanced:
        risk_inputs = _risk_context_input_columns(
            roots=tuple(
                _required_input_root(layout, upstream_identities, target_id)
                for target_id in candidate.risk_context_target_ids
            ),
            style_columns=risk_style_columns,
            timestamp=sessions[0],
            resource_policy=resource_policy,
        )

    holdings = pd.Series(dtype="float64")
    cash = float(assumptions.initial_nav)
    nav = cash
    amount_history: dict[str, deque[float]] = {}
    risk_assets: set[str] = set()
    factor_risk = (
        FactorRiskState(style_columns=risk_style_columns, config=risk_config)
        if enhanced
        else None
    )
    previous_risk_cross: pd.DataFrame | None = None
    calibration_rows: list[dict[str, object]] = []
    pending_calibration: tuple[pd.Series, pd.Series, pd.Timestamp] | None = None
    waiting_alpha: tuple[pd.Series, pd.Timestamp] | None = None
    calibration_dates = calibration_intervals.set_index("timestamp")
    previous_benchmark_assets: frozenset[str] | None = None
    emitted = False
    sequence = 0
    state_lease: MemoryLease | None = None
    if resource_policy is not None:
        state_lease = resource_policy.lease(
            "research.portfolio.state",
            1,
        )

    def resize_state(*, extra_assets=0, extra_bytes=0):
        if state_lease is None:
            return
        retained_bytes = (
            4096
            + (len(amount_history) + len(holdings) + extra_assets) * 672
            + len(calibration_rows) * 640
        )
        if enhanced:
            retained_bytes += max(65536, (len(risk_assets) + extra_assets) * 16384)
        retained_bytes += pandas_frame_payload_bytes(
            calibration_intervals
        ) + pandas_frame_payload_bytes(calibration_dates)
        if previous_risk_cross is not None:
            retained_bytes += pandas_frame_payload_bytes(previous_risk_cross)
        if waiting_alpha is not None:
            retained_bytes += int(waiting_alpha[0].memory_usage(deep=True))
        if pending_calibration is not None:
            retained_bytes += sum(
                int(value.memory_usage(deep=True)) for value in pending_calibration[:2]
            )
        retained_bytes += extra_bytes
        if retained_bytes > state_lease.reserved_bytes:
            state_lease.grow(retained_bytes - state_lease.reserved_bytes)
        else:
            state_lease.shrink_to(retained_bytes)

    try:
        with ExitStack() as input_channels:
            history = TimeWindow(sessions[0], sessions[-1])
            panel_columns = _ENHANCED_PANEL_COLUMNS if enhanced else _PANEL_COLUMNS
            panel_frames = input_channels.enter_context(
                closing_iterator(
                    _iter_input_sessions(
                        panel_root, panel_columns, history, resource_policy
                    )
                )
            )
            dataset_frames = input_channels.enter_context(
                closing_iterator(
                    _iter_input_sessions(
                        dataset_root, _DATASET_COLUMNS, history, resource_policy
                    )
                )
            )
            context_frames = {
                root: input_channels.enter_context(
                    closing_iterator(
                        _iter_input_sessions(root, columns, history, resource_policy)
                    )
                )
                for root, columns in risk_inputs.items()
            }
            next_panel_frames = input_channels.enter_context(
                closing_iterator(
                    _iter_input_sessions(
                        panel_root, panel_columns, history, resource_policy
                    )
                )
            )
            endpoint_panel_frames = input_channels.enter_context(
                closing_iterator(
                    _iter_input_sessions(
                        panel_root, panel_columns, history, resource_policy
                    )
                )
            )
            execution_frames = input_channels.enter_context(
                closing_iterator(
                    _iter_input_sessions(
                        execution_root, _EXECUTION_COLUMNS, history, resource_policy
                    )
                )
            )
            benchmark_frames = (
                input_channels.enter_context(
                    closing_iterator(
                        _iter_benchmark_weights(
                            benchmark_root,
                            candidate.benchmark_id,
                            sessions,
                            resource_policy,
                        )
                    )
                )
                if enhanced
                else None
            )
            for position, decision_timestamp in enumerate(sessions):
                with ExitStack() as retained:
                    next_timestamp = (
                        sessions[position + 1]
                        if position + 1 < len(sessions)
                        else decision_timestamp
                    )
                    panel = _read_next_session(
                        panel_frames,
                        panel_columns,
                        decision_timestamp,
                        retained,
                    )
                    if panel.empty:
                        raise PortfolioTargetBlocked("portfolio_panel_session_missing")
                    resize_state(extra_assets=len(panel))
                    if enhanced:
                        risk_assets.update(panel.asset_id.astype(str))
                    _update_amount_history(panel, amount_history, holdings)
                    resize_state()

                    benchmark_weights = pd.Series(dtype="float64")
                    alpha: pd.Series | None = None
                    current_cross: pd.DataFrame | None = None
                    if enhanced:
                        assert factor_risk is not None
                        dataset = _read_next_session(
                            dataset_frames,
                            _DATASET_COLUMNS,
                            decision_timestamp,
                            retained,
                        )
                        if dataset.empty:
                            raise PortfolioTargetBlocked(
                                "portfolio_research_dataset_session_missing"
                            )
                        _validate_input_keys(dataset, panel)
                        risk_frames = []
                        for root, columns in risk_inputs.items():
                            context = _read_next_session(
                                context_frames[root],
                                columns,
                                decision_timestamp,
                                retained,
                            )
                            _validate_input_keys(panel, context)
                            risk_frames.append(context)
                        current_cross = _enhanced_risk_cross(
                            panel=panel,
                            risk_frames=risk_frames,
                            style_columns=risk_style_columns,
                        )
                        risk_frames.clear()
                        context = None
                        retained.enter_context(
                            _retain_frame(
                                current_cross,
                                resource_policy,
                                "research.portfolio.risk_cross",
                            )
                        )
                        if previous_risk_cross is not None:
                            factor_risk.observe_return_session(
                                _factor_return_cross(
                                    exposures=previous_risk_cross,
                                    panel=panel,
                                )
                            )
                        resize_state(
                            extra_bytes=pandas_frame_payload_bytes(current_cross)
                        )
                        previous_risk_cross = current_cross.copy(deep=True)
                        resize_state()

                        if calibration_rows and calibration_rows[0][
                            "timestamp"
                        ] < decision_timestamp - pd.DateOffset(months=12):
                            resize_state(extra_bytes=8 * len(calibration_rows))
                            calibration_rows[:] = [
                                row
                                for row in calibration_rows
                                if row["timestamp"]
                                >= decision_timestamp - pd.DateOffset(months=12)
                            ]
                            resize_state()
                        if (
                            pending_calibration is not None
                            and decision_timestamp
                            == calibration_dates.loc[
                                pending_calibration[2], "available_at"
                            ]
                        ):
                            resize_state(extra_bytes=1536 * len(pending_calibration[0]))
                            calibration_rows.extend(
                                _completed_calibration_rows(pending_calibration, panel)
                            )
                            pending_calibration = None
                            resize_state()
                        if (
                            waiting_alpha is not None
                            and decision_timestamp
                            == calibration_dates.loc[waiting_alpha[1], "entry_at"]
                        ):
                            resize_state(
                                extra_bytes=2 * pandas_frame_payload_bytes(panel)
                            )
                            pending_calibration = (
                                waiting_alpha[0],
                                panel.set_index("asset_id")["adj_open"].astype(
                                    "float64"
                                ),
                                waiting_alpha[1],
                            )
                            waiting_alpha = None
                            resize_state()
                        if decision_timestamp >= alpha_start:
                            alpha = _prediction_alpha(
                                dataset,
                                prediction_lookup,
                            )
                            if alpha is not None:
                                retained.enter_context(
                                    _retain_payload(
                                        alpha,
                                        resource_policy,
                                        "research.portfolio.alpha",
                                    )
                                )
                            has_next_session = position + 1 < len(sessions)
                            if (
                                alpha is not None
                                and frozen_calibration is None
                                and has_next_session
                                and decision_timestamp in calibration_dates.index
                                and is_rebalance_session(
                                    decision_timestamp,
                                    next_timestamp,
                                    candidate.rebalance_schedule,
                                )
                            ):
                                resize_state(
                                    extra_bytes=int(alpha.memory_usage(deep=True))
                                )
                                waiting_alpha = (
                                    alpha.copy(deep=True),
                                    decision_timestamp,
                                )
                                resize_state()

                        assert benchmark_frames is not None
                        benchmark_weights = next(benchmark_frames)

                    if decision_timestamp < output_start:
                        continue
                    if not enhanced:
                        dataset = _read_next_session(
                            dataset_frames,
                            _DATASET_COLUMNS,
                            decision_timestamp,
                            retained,
                        )
                        if dataset.empty:
                            raise PortfolioTargetBlocked(
                                "portfolio_research_dataset_session_missing"
                            )
                    assert dataset is not None
                    execution = _read_next_session(
                        execution_frames,
                        _EXECUTION_COLUMNS,
                        decision_timestamp,
                        retained,
                    )
                    if execution.empty:
                        raise PortfolioTargetBlocked("portfolio_execution_mask_missing")
                    with _retain_session_workspace(
                        len(dataset),
                        len(holdings),
                        resource_policy,
                    ):
                        _validate_input_keys(dataset, panel)
                        _validate_input_keys(dataset, execution)
                        if not enhanced:
                            alpha = _prediction_alpha(dataset, prediction_lookup)
                            retained.enter_context(
                                _retain_payload(
                                    alpha,
                                    resource_policy,
                                    "research.portfolio.alpha",
                                )
                            )
                        if decision_timestamp not in output_intervals.index:
                            continue
                        endpoint_timestamp = output_intervals.loc[
                            decision_timestamp, "end_at"
                        ]
                        next_panel = _read_next_session(
                            next_panel_frames,
                            panel_columns,
                            next_timestamp,
                            retained,
                        )
                        if next_panel.empty:
                            raise PortfolioTargetBlocked(
                                "portfolio_market_session_missing"
                            )
                        endpoint_panel = _read_next_session(
                            endpoint_panel_frames,
                            panel_columns,
                            endpoint_timestamp,
                            retained,
                        )
                        if endpoint_panel.empty:
                            raise PortfolioTargetBlocked(
                                "portfolio_market_session_missing"
                            )

                        enhanced_decision: _EnhancedDecision | None = None
                        target_override: pd.Series | None = None
                        decision_reason = PORTFOLIO_DECISION_REASON_SIMPLE
                        # Retained state is the last post-trade execution open. Mark
                        # only to today's close before selecting the next target.
                        if not holdings.empty:
                            prices = panel.set_index("asset_id")
                            holdings, cash, nav = drift_continuous_holdings(
                                holdings,
                                cash,
                                prices.adj_close.reindex(holdings.index).div(
                                    prices.adj_open.reindex(holdings.index)
                                )
                                - 1.0,
                            )
                        if enhanced:
                            assert alpha is not None and current_cross is not None
                            decision_reason = PORTFOLIO_DECISION_REASON_BENCHMARK
                            scheduled = is_rebalance_session(
                                decision_timestamp,
                                next_timestamp,
                                candidate.rebalance_schedule,
                            )
                            membership_changed = (
                                not benchmark_weights.empty
                                and previous_benchmark_assets is not None
                                and frozenset(benchmark_weights.index)
                                != previous_benchmark_assets
                            )
                            if not benchmark_weights.empty:
                                previous_benchmark_assets = frozenset(
                                    benchmark_weights.index
                                )
                            initializing = not emitted
                            maintenance = membership_changed and not scheduled
                            if initializing:
                                if benchmark_weights.empty:
                                    raise PortfolioTargetBlocked(
                                        "portfolio_benchmark_snapshot_missing"
                                    )
                                holdings = benchmark_weights.multiply(nav)
                                cash = 0.0
                            if scheduled or maintenance:
                                if benchmark_weights.empty:
                                    raise PortfolioTargetBlocked(
                                        "portfolio_benchmark_snapshot_missing"
                                    )
                                resize_state(extra_bytes=512 * len(calibration_rows))
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
                                )
                                resize_state()
                                if maintenance:
                                    target_override = benchmark_weights.copy(deep=True)
                                    decision_reason = (
                                        PORTFOLIO_DECISION_REASON_MAINTENANCE
                                    )
                            else:
                                target_override = holdings.div(nav).copy(deep=True)
                                decision_reason = PORTFOLIO_DECISION_REASON_BENCHMARK

                        account, returns, adv, _optimization, decision_delta = (
                            _build_session(
                                candidate=candidate,
                                decision_timestamp=decision_timestamp,
                                next_timestamp=next_timestamp,
                                panel=panel,
                                next_panel=next_panel,
                                endpoint_panel=endpoint_panel,
                                dataset=dataset,
                                execution=execution,
                                holdings=holdings,
                                cash=cash,
                                nav=nav,
                                initialize_at_execution=enhanced and not emitted,
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
                        output = _ledger_frame(
                            account=account,
                            decision_timestamp=decision_timestamp,
                            returns=returns,
                            alpha_score=alpha,
                            adv20=adv,
                            enhanced_decision=enhanced_decision,
                            benchmark_weights=(benchmark_weights if enhanced else None),
                            optimization_result=_optimization,
                            decision_reason=decision_reason,
                            decision_delta=decision_delta,
                        )
                        holdings = account.asset_ledger.set_index("asset_id")[
                            "post_trade_notional"
                        ]
                        holdings = holdings.loc[holdings.ne(0.0)]
                        cash, nav = account.post_trade_cash, account.nav_after
                        resize_state()
                        retained.enter_context(
                            _retain_frame(
                                output,
                                resource_policy,
                                "research.portfolio.output",
                            )
                        )
                        emitted = True
                        yield continuous_ledger_batch(
                            output, sequence, end_timestamp=endpoint_timestamp
                        )
                        sequence += 1

            missing, extra = prediction_lookup.coverage_counts()
            if missing or extra:
                raise ValueError("misaligned_persisted_prediction_keys")
            if not emitted:
                raise PortfolioTargetBlocked("portfolio_execution_window_empty")
    finally:
        primary = sys.exception()
        if primary is not None:
            clear_exception_frames(primary)
        calibration_rows.clear()
        amount_history.clear()
        risk_assets.clear()
        previous_risk_cross = pending_calibration = waiting_alpha = holdings = None
        risk_frames = context = None
        factor_risk = None
        account = output = current_cross = dataset = panel = next_panel = (
            endpoint_panel
        ) = None
        returns = adv = _optimization = decision_delta = enhanced_decision = None
        calibration_intervals = calibration_dates = None
        if state_lease is not None:
            _close_lease_preserving_primary(
                state_lease,
                owner="research.portfolio.state",
            )


def _read_materialized(
    root: Path,
    columns: tuple[str, ...] | None,
    *,
    time_window: TimeWindow | None,
    resource_policy: RuntimeResourcePolicy | None,
) -> pd.DataFrame:
    frame = read_dataset(
        root,
        columns=columns,
        time_window=time_window,
        mode=DatasetReadMode.MATERIALIZED,
        resource_policy=resource_policy,
    )
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("portfolio_input_reader_not_materialized")
    return frame


def _iter_input_sessions(
    root: Path,
    columns: tuple[str, ...],
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy | None,
) -> Iterator[pd.DataFrame]:
    from quant_research_platform.runtime.dataframes import iter_complete_session_frames

    with runtime_resource_policy_scope(resource_policy) as policy:
        source = read_dataset(
            root,
            columns=columns,
            time_window=time_window,
            mode=DatasetReadMode.CHUNKED,
            resource_policy=policy,
        )
        with closing_iterator(
            iter_complete_session_frames(
                source, timestamp_column="timestamp", resource_policy=policy
            )
        ) as frames:
            for frame in frames:
                if not frame.empty:
                    yield frame


def _read_next_session(
    frames: Iterator[pd.DataFrame],
    columns: tuple[str, ...],
    timestamp: pd.Timestamp,
    retained: ExitStack,
) -> pd.DataFrame:
    for frame in frames:
        actual = pd.Timestamp(frame["timestamp"].iloc[0])
        if actual < timestamp:
            continue
        if actual > timestamp:
            break
        policy = current_runtime_resource_policy()
        assert policy is not None
        lease = retained.enter_context(
            policy.lease(
                "research.portfolio.input_session",
                2 * pandas_frame_payload_bytes(frame) + 4096,
            )
        )
        normalized = _session_frame(frame, columns)
        lease.shrink_to(pandas_frame_payload_bytes(normalized))
        return normalized
    return pd.DataFrame(columns=columns)


@contextmanager
def _retain_frame(
    frame: pd.DataFrame,
    resource_policy: RuntimeResourcePolicy | None,
    owner: str,
) -> Iterator[pd.DataFrame]:
    if resource_policy is None:
        yield frame
        return
    with resource_policy.lease(owner, max(1, pandas_frame_payload_bytes(frame))):
        yield frame


@contextmanager
def _retain_payload(
    payload: pd.Series,
    resource_policy: RuntimeResourcePolicy | None,
    owner: str,
) -> Iterator[pd.Series]:
    if resource_policy is None:
        yield payload
        return
    with resource_policy.lease(owner, max(1, int(payload.memory_usage(deep=True)))):
        yield payload


@contextmanager
def _retain_session_workspace(
    input_rows: int,
    holding_count: int,
    resource_policy: RuntimeResourcePolicy | None,
) -> Iterator[None]:
    if resource_policy is None:
        yield
        return
    if (
        isinstance(input_rows, bool)
        or input_rows < 0
        or isinstance(holding_count, bool)
        or holding_count < 0
    ):
        raise ValueError("portfolio_session_workspace_invalid")
    # Covers the transient execution frame and typed rows through the consumer yield.
    reserved = max(
        1,
        (input_rows + holding_count) * len(CONTINUOUS_EXECUTION_COLUMNS) * 256
        + 16 * 1024,
    )
    with resource_policy.lease("research.portfolio.session_workspace", reserved):
        yield


def _risk_context_input_columns(
    *,
    roots: tuple[Path, ...],
    style_columns: tuple[str, ...],
    timestamp: pd.Timestamp,
    resource_policy: RuntimeResourcePolicy | None,
) -> dict[Path, tuple[str, ...]]:
    required = set(style_columns)
    available: set[str] = set()
    inputs = {}
    for root in roots:
        with open_dataset_session(
            root,
            columns=None,
            time_window=TimeWindow(timestamp, timestamp),
            resource_policy=resource_policy,
        ) as session:
            columns = tuple(
                column
                for column in style_columns
                if column in session.arrow_schema.names
            )
        if available.intersection(columns):
            raise ValueError("portfolio_risk_context_owner_duplicate")
        available.update(columns)
        inputs[root] = (*_SESSION_KEY, *columns)
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
    result["timestamp"] = pd.Timestamp(panel["timestamp"].iloc[0])
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
    return result.reset_index(names="asset_id")


def _factor_return_cross(
    *,
    exposures: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    previous = exposures.set_index("asset_id")
    current = panel.set_index("asset_id")
    assets = previous.index.intersection(current.index)
    result = previous.loc[assets].copy()
    result["timestamp"] = pd.Timestamp(panel["timestamp"].iloc[0])
    result["asset_return"] = (
        pd.to_numeric(current.loc[assets, "period_return"], errors="raise")
        .astype("float64")
        .to_numpy()
    )
    return result.reset_index()


def _completed_calibration_rows(
    pending: tuple[pd.Series, pd.Series, pd.Timestamp],
    panel: pd.DataFrame,
) -> list[dict[str, float]]:
    alpha, previous_open, timestamp = pending
    current_open = (
        pd.to_numeric(panel.set_index("asset_id")["adj_open"], errors="raise")
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


def _iter_benchmark_weights(root, benchmark_id, sessions, resource_policy):
    """Align sparse snapshots without consuming a future session prematurely."""
    with closing_iterator(
        _iter_input_sessions(
            root,
            _BENCHMARK_COLUMNS,
            TimeWindow(sessions[0], sessions[-1]),
            resource_policy,
        )
    ) as frames:
        pending = next(frames, None)
        try:
            for timestamp in sessions:
                while pending is not None and pending["timestamp"].iloc[0] < timestamp:
                    pending = next(frames, None)
                if pending is None or pending["timestamp"].iloc[0] > timestamp:
                    yield pd.Series(dtype="float64")
                else:
                    with _retain_session_workspace(len(pending), 0, resource_policy):
                        yield _benchmark_weights(pending, benchmark_id, timestamp)
        finally:
            pending = None


def _benchmark_weights(
    frame: pd.DataFrame,
    benchmark_id: str,
    timestamp: pd.Timestamp,
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
        or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("portfolio_benchmark_weights_invalid")
    return pd.Series(
        weights.to_numpy(dtype="float64"),
        index=pd.Index(frame["asset_id"].astype(str), dtype="string"),
        name="benchmark_weight",
    )


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
) -> _EnhancedDecision:
    calibration = frozen_calibration or calibrate_expected_alpha(
        pd.DataFrame(
            calibration_rows,
            columns=["timestamp", "alpha_score", "holding_return"],
        ),
        as_of=as_of,
        scheduled_intervals=calibration_intervals,
    )
    expected = pd.Series(np.nan, index=alpha.index, dtype="float64")
    if not calibration.benchmark_only:
        expected.loc[alpha.notna()] = calibration.slope * alpha.loc[alpha.notna()]
    else:
        expected.loc[alpha.notna()] = 0.0
    complete = current_cross.loc[
        current_cross.loc[:, risk_style_columns].notna().all(axis=1)
    ]
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
        if column not in {"asset_id", "industry_level1_code"}
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
        result = pd.concat(
            [result, pd.DataFrame([cash_row])],
            ignore_index=True,
        )
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
    dataset: pd.DataFrame,
    execution: pd.DataFrame,
    holdings: pd.Series,
    cash: float,
    nav: float,
    initialize_at_execution: bool,
    amount_history: Mapping[str, deque[float]],
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
    ExecutionAccounting, pd.Series, pd.Series, OptimizationResult | None, pd.Series
]:
    panel_columns = _ENHANCED_PANEL_COLUMNS if enhanced_decision else _PANEL_COLUMNS
    panel = _session_frame(panel, panel_columns)
    dataset = _session_frame(dataset, _DATASET_COLUMNS).set_index(
        "asset_id",
        drop=False,
    )
    execution = _session_frame(execution, _EXECUTION_COLUMNS)
    if not np.isclose(
        float(holdings.sum()) + cash,
        nav,
        rtol=0.0,
        atol=max(np.finfo("float64").eps * max(abs(nav), 1.0) * 512.0, 1e-12),
    ):
        raise ValueError("portfolio_state_invalid")
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    alpha = alpha.reindex(dataset.index)
    if (
        alpha.loc[eligible].isna().any()
        or not np.isfinite(alpha.loc[eligible].to_numpy(dtype="float64")).all()
    ):
        raise ValueError("portfolio_prediction_values_invalid")
    optimization: OptimizationResult | None = None
    if target_override is not None:
        target = target_override.copy(deep=True)
    elif enhanced_decision is not None:
        target = enhanced_decision.benchmark_weights.copy(deep=True)
    elif is_rebalance_session(
        decision_timestamp,
        next_timestamp,
        candidate.rebalance_schedule,
    ):
        target = build_simple_target_weights(
            alpha,
            eligible=eligible,
            method=(
                "float_cap_weight"
                if candidate.construction == "top20_float_cap"
                else "equal_weight"
            ),
            top_quantile=float(candidate.top_quantile),
            bottom_quantile=float(candidate.top_quantile),
            float_market_cap=(
                dataset["float_market_cap"]
                if candidate.construction == "top20_float_cap"
                else None
            ),
        )
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
            asset: (
                float(np.mean(amount_history[asset]))
                if asset in amount_history and len(amount_history[asset]) == 20
                else np.nan
            )
            for asset in assets
        },
        dtype="float64",
    )
    masks = _mask_values(execution, assets)

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
        and enhanced_decision.risk is not None
        and not enhanced_decision.calibration.benchmark_only
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
    execution_holdings, execution_cash, execution_nav = current, cash, nav
    # Seed enhanced capital at its first execution open; later executions carry
    # the existing exposure through the overnight adjusted-price move.
    if not initialize_at_execution and not current.empty:
        overnight = (
            next_panel.set_index("asset_id")
            .adj_open.reindex(current.index)
            .div(panel_by_asset.adj_close.reindex(current.index))
            - 1.0
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
    returns = _holding_returns(account, next_panel, endpoint_panel)
    return account, returns, adv, optimization, decision_delta


def _prediction_alpha(
    dataset: pd.DataFrame,
    prediction_lookup: object,
) -> pd.Series:
    dataset = _session_frame(dataset, _DATASET_COLUMNS).set_index(
        "asset_id",
        drop=False,
    )
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    required_rows = dataset.loc[eligible, ["timestamp", "asset_id"]]
    alpha = pd.Series(np.nan, index=dataset.index, dtype="float64")
    if required_rows.empty:
        return alpha
    values_for_frame = getattr(prediction_lookup, "values_for_frame", None)
    if not callable(values_for_frame):
        raise ValueError("portfolio_prediction_lookup_invalid")
    with values_for_frame(
        required_rows,
        mark_matched=True,
    ) as (values, found):
        if not all(found):
            raise ValueError("missing_required_prediction_keys")
        alpha.loc[required_rows["asset_id"]] = np.asarray(values, dtype="float64")
    if (
        alpha.loc[eligible].isna().any()
        or not np.isfinite(alpha.loc[eligible].to_numpy(dtype="float64")).all()
    ):
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
            index=False,
            name=None,
        )
    }


def _session_frame(
    frame: pd.DataFrame | None,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=required_columns)
    if type(frame) is not pd.DataFrame or not set(required_columns).issubset(
        frame.columns
    ):
        raise ValueError("portfolio_input_schema_invalid")
    result = frame.loc[:, list(required_columns)].copy()
    if "asset_id" in result:
        result["asset_id"] = result["asset_id"].astype("string")
    if "timestamp" in result:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="raise",
        ).dt.normalize()
    if "asset_id" in result and result["asset_id"].isna().any():
        raise ValueError("portfolio_input_key_invalid")
    if "timestamp" in result and result["timestamp"].isna().any():
        raise ValueError("portfolio_input_key_invalid")
    if {"timestamp", "asset_id"}.issubset(result.columns) and result.duplicated(
        list(_SESSION_KEY)
    ).any():
        raise ValueError("portfolio_input_key_invalid")
    return result.sort_values(
        [column for column in ("timestamp", "asset_id") if column in result],
        kind="mergesort",
    ).reset_index(drop=True)


def _update_amount_history(
    panel: pd.DataFrame,
    history: dict[str, deque[float]],
    holdings: pd.Series,
) -> None:
    current_assets = set(panel["asset_id"].astype(str))
    for asset in tuple(history):
        if asset not in current_assets:
            if float(holdings.get(asset, 0.0)) != 0.0:
                raise PortfolioTargetBlocked("portfolio_held_asset_missing_from_panel")
            del history[asset]
    amounts = pd.to_numeric(panel["amount"], errors="raise").astype("float64")
    if (
        amounts.isna().any()
        or not np.isfinite(amounts.to_numpy()).all()
        or amounts.lt(0).any()
    ):
        raise ValueError("portfolio_amount_invalid")
    for asset, amount in zip(panel["asset_id"].astype(str), amounts, strict=True):
        history.setdefault(asset, deque(maxlen=20)).append(float(amount))


def _mask_values(execution: pd.DataFrame, assets: pd.Index) -> dict[str, pd.Series]:
    if assets.empty:
        empty = pd.Series(dtype="bool")
        return {"can_enter": empty, "can_exit": empty}
    if not pd.api.types.is_bool_dtype(
        execution["can_enter"]
    ) or not pd.api.types.is_bool_dtype(execution["can_exit"]):
        raise ValueError("portfolio_execution_mask_invalid")
    indexed = execution.set_index("asset_id")
    missing = [asset for asset in assets if asset not in indexed.index]
    if missing:
        raise PortfolioTargetBlocked("portfolio_execution_mask_missing")
    if indexed.loc[list(assets), ["can_enter", "can_exit"]].isna().any().any():
        raise PortfolioTargetBlocked("portfolio_execution_fact_unknown")
    return {
        column: indexed.loc[list(assets), column].astype(bool)
        for column in ("can_enter", "can_exit")
    }


def _holding_returns(
    account: ExecutionAccounting,
    next_panel: pd.DataFrame,
    endpoint_panel: pd.DataFrame,
) -> pd.Series:
    assets = account.asset_ledger["asset_id"].astype(str)
    returns = pd.Series(0.0, index=assets, dtype="float64")
    selected = assets.loc[account.asset_ledger["post_trade_notional"].to_numpy() > 0.0]
    if selected.empty:
        return returns
    prices = []
    for frame in (next_panel, endpoint_panel):
        values = frame.set_index("asset_id")["adj_open"].reindex(selected)
        if values.isna().any():
            raise PortfolioTargetBlocked("portfolio_held_return_missing")
        values = pd.to_numeric(values, errors="raise").astype("float64")
        if not np.isfinite(values).all() or values.le(0).any():
            raise ValueError("portfolio_return_invalid")
        prices.append(values)
    with np.errstate(over="ignore"):
        values = prices[1] / prices[0] - 1.0
    if not np.isfinite(values).all():
        raise ValueError("portfolio_return_invalid")
    returns.loc[selected] = values
    return returns


def _ledger_frame(
    *,
    account: ExecutionAccounting,
    decision_timestamp: pd.Timestamp,
    returns: pd.Series,
    alpha_score: pd.Series,
    adv20: pd.Series,
    enhanced_decision: _EnhancedDecision | None,
    benchmark_weights: pd.Series | None,
    optimization_result: OptimizationResult | None,
    decision_reason: str,
    decision_delta: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in account.asset_ledger.itertuples(index=False):
        asset = str(row.asset_id)
        values: dict[str, object] = {
            "decision_timestamp": decision_timestamp,
            "execution_date": pd.Timestamp(account.execution_date),
            "asset_id": asset,
            "pre_trade_notional": float(row.pre_trade_notional),
            "target_notional": float(row.target_notional),
            "decision_delta_weight": float(decision_delta.loc[asset]),
            "executed_delta_notional": float(row.executed_delta_notional),
            "post_trade_notional": float(row.post_trade_notional),
            "holding_return": float(returns.get(asset, 0.0)),
            "end_notional": float(row.post_trade_notional)
            * (1.0 + float(returns.get(asset, 0.0))),
            "block_reason": row.reason,
            "adv20": float(adv20[asset]) if pd.notna(adv20.get(asset)) else np.nan,
            "alpha_score": (
                float(alpha_score.get(asset))
                if pd.notna(alpha_score.get(asset))
                else np.nan
            ),
            "expected_alpha": (
                float(enhanced_decision.expected_alpha.get(asset))
                if enhanced_decision is not None
                and pd.notna(enhanced_decision.expected_alpha.get(asset))
                else np.nan
            ),
            "benchmark_weight": (
                float(benchmark_weights.get(asset))
                if benchmark_weights is not None
                and pd.notna(benchmark_weights.get(asset))
                else np.nan
            ),
            "asset_specific_variance": (
                float(enhanced_decision.risk.specific_variance.get(asset))
                if enhanced_decision is not None
                and enhanced_decision.risk is not None
                and pd.notna(enhanced_decision.risk.specific_variance.get(asset))
                else np.nan
            ),
            "decision_reason": None,
        }
        values.update(
            {
                column: float(getattr(row, column))
                for column in PORTFOLIO_LEDGER_COST_COLUMNS
            }
        )
        values.update(
            {
                column: np.nan
                for column in CONTINUOUS_EXECUTION_COLUMNS
                if column not in values
            }
        )
        rows.append(values)

    asset_target_total = float(account.asset_ledger["target_notional"].sum())
    cash_row: dict[str, object] = {
        "decision_timestamp": decision_timestamp,
        "execution_date": pd.Timestamp(account.execution_date),
        "asset_id": PORTFOLIO_CASH_ASSET_ID,
        "pre_trade_notional": account.pre_trade_cash,
        "target_notional": max(0.0, account.nav_before - asset_target_total),
        "executed_delta_notional": account.post_trade_cash - account.pre_trade_cash,
        "post_trade_notional": account.post_trade_cash,
        "holding_return": 0.0,
        "end_notional": account.post_trade_cash,
        "block_reason": None,
        "adv20": np.nan,
        "alpha_score": np.nan,
        "expected_alpha": np.nan,
        "decision_delta_weight": -float(decision_delta.sum()),
        "benchmark_weight": np.nan,
        "asset_specific_variance": np.nan,
        **{
            name: getattr(enhanced_decision.calibration, field)
            if enhanced_decision
            else np.nan
            for name, field in _ALPHA_DIAGNOSTICS.items()
        },
        "predicted_tracking_error": (
            optimization_result.predicted_tracking_error
            if optimization_result
            else np.nan
        ),
        "portfolio_factor_variance": (
            optimization_result.factor_variance if optimization_result else np.nan
        ),
        "portfolio_specific_variance": (
            optimization_result.specific_variance if optimization_result else np.nan
        ),
        "expected_transaction_cost": (
            optimization_result.expected_transaction_cost
            if optimization_result
            else np.nan
        ),
        "decision_reason": (
            PORTFOLIO_DECISION_REASON_ENHANCED
            if optimization_result is not None
            and decision_reason == PORTFOLIO_DECISION_REASON_BENCHMARK
            else decision_reason
        ),
    }
    cash_row.update({column: 0.0 for column in PORTFOLIO_LEDGER_COST_COLUMNS})
    cash_row.update(
        {
            column: np.nan
            for column in CONTINUOUS_EXECUTION_COLUMNS
            if column not in cash_row
        }
    )
    rows.append(cash_row)
    return pd.DataFrame.from_records(rows, columns=CONTINUOUS_EXECUTION_COLUMNS)


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
        or output_window.start < formal_dataset_window.start
        or output_window.end > input_window.end
    ):
        raise ValueError("portfolio_windows_invalid")


def _required_input_root(
    layout: StorageLayout,
    upstream_identities: Mapping[str, str],
    target_id: str,
) -> Path:
    return _required_target_root(
        layout,
        target_id,
        expected_identity=_required_identity(upstream_identities, target_id),
    )


def _required_target_root(
    layout: StorageLayout,
    target_id: str,
    *,
    expected_identity: str,
) -> Path:
    marker = Path("__missing_portfolio_target__")
    root = layout.target_root(target_id, marker)
    if root == marker or not filesystem_exists(root) or root.name != expected_identity:
        raise ValueError(f"portfolio_input_missing: {target_id}")
    return root


def _required_identity(
    upstream_identities: Mapping[str, str],
    target_id: str,
) -> str:
    identity = upstream_identities.get(target_id)
    if (
        type(identity) is not str
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError(f"portfolio_input_identity_invalid: {target_id}")
    return identity
