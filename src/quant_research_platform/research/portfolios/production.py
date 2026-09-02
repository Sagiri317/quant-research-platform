"""Bounded production portfolio producer for the active target DAG."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PORTFOLIO_LEDGER_COLUMNS,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioTargetBlocked,
    TradeCostComponents,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.portfolios.accounting import (
    ExecutionAccounting,
    drift_continuous_holdings,
    execute_continuous_target,
)
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.schedules import is_rebalance_session
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage.datasets import DatasetReadMode
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.layout import StorageLayout

_CALENDAR_COLUMNS = ("calendar_date", "source_market_type", "is_trading_day")
_DATASET_COLUMNS = ("asset_id", "timestamp", "eligible", "float_market_cap")
_PANEL_COLUMNS = ("asset_id", "timestamp", "adj_open", "amount", "market_type")
_EXECUTION_COLUMNS = ("asset_id", "timestamp", "can_enter", "can_exit")
_SESSION_KEY = ("timestamp", "asset_id")


def build_portfolio_ledger(
    *,
    candidate: PortfolioCandidateSpec,
    layout: StorageLayout,
    upstream_identities: Mapping[str, str],
    output_window: TimeWindow,
    input_window: TimeWindow,
    formal_dataset_window: TimeWindow,
    prediction_lookup: object,
    assumptions: ExecutionAssumptions,
    resource_policy: RuntimeResourcePolicy | None,
    fee_schedule_from_frame: Callable[..., object],
    require_fee_coverage: Callable[..., None],
    trade_cost: Callable[..., TradeCostComponents],
    market_name_for_type: Callable[[object], str],
    unified_trading_sessions: Callable[[pd.DataFrame], pd.DatetimeIndex],
) -> Iterator[pd.DataFrame]:
    """Build simple candidate sessions from bounded, exact-date reads."""

    if not isinstance(candidate, PortfolioCandidateSpec):
        raise ValueError("portfolio_candidate_invalid")
    if candidate.construction == "benchmark_enhanced":
        raise PortfolioTargetBlocked("portfolio_enhanced_producer_unbound")
    if not isinstance(assumptions, ExecutionAssumptions):
        raise ValueError("portfolio_assumptions_invalid")
    _validate_windows(output_window, input_window, formal_dataset_window)

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
        warmup_start = max(0, int(output_positions[0]) - 19)
        session_start = max(session_start, sessions[warmup_start])
    sessions = sessions[(sessions >= session_start) & (sessions <= session_end)]
    del calendar
    if sessions.empty:
        raise PortfolioTargetBlocked("portfolio_execution_window_empty")

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
    holdings = pd.Series(dtype="float64")
    cash = float(assumptions.initial_nav)
    nav = cash
    amount_history: dict[str, deque[float]] = {}
    output_end = pd.Timestamp(output_window.end).normalize()
    emitted = False
    amount_state_lease: MemoryLease | None = None
    if resource_policy is not None:
        amount_state_lease = resource_policy.lease(
            "research.portfolio.amount_state",
            1,
        )

    try:
        for position, decision_timestamp in enumerate(sessions):
            with ExitStack() as retained:
                panel = _read_session(
                    panel_root,
                    _PANEL_COLUMNS,
                    decision_timestamp,
                    resource_policy,
                )
                retained.enter_context(
                    _retain_frame(panel, resource_policy, "research.portfolio.panel")
                )
                if panel.empty:
                    raise PortfolioTargetBlocked("portfolio_panel_session_missing")
                _resize_amount_state_lease(
                    amount_state_lease,
                    len(amount_history) + len(panel) + len(holdings),
                )
                _update_amount_history(panel, amount_history, holdings)
                _resize_amount_state_lease(
                    amount_state_lease,
                    len(amount_history) + len(holdings),
                )
                if decision_timestamp < output_start:
                    continue
                dataset = _read_session(
                    dataset_root,
                    _DATASET_COLUMNS,
                    decision_timestamp,
                    resource_policy,
                )
                retained.enter_context(
                    _retain_frame(
                        dataset,
                        resource_policy,
                        "research.portfolio.dataset",
                    )
                )
                if dataset.empty:
                    raise PortfolioTargetBlocked(
                        "portfolio_research_dataset_session_missing"
                    )
                execution = _read_session(
                    execution_root,
                    _EXECUTION_COLUMNS,
                    decision_timestamp,
                    resource_policy,
                )
                retained.enter_context(
                    _retain_frame(
                        execution,
                        resource_policy,
                        "research.portfolio.execution",
                    )
                )
                if execution.empty:
                    raise PortfolioTargetBlocked("portfolio_execution_mask_missing")
                with _retain_session_workspace(
                    len(dataset),
                    len(holdings),
                    resource_policy,
                ):
                    _validate_session_alignment(dataset, panel, execution)
                    alpha = _prediction_alpha(dataset, prediction_lookup)
                    retained.enter_context(
                        _retain_payload(
                            alpha,
                            resource_policy,
                            "research.portfolio.alpha",
                        )
                    )
                    if position + 2 >= len(sessions):
                        continue
                    next_timestamp = sessions[position + 1]
                    endpoint_timestamp = sessions[position + 2]
                    if endpoint_timestamp > output_end:
                        continue
                    next_panel = _read_session(
                        panel_root,
                        _PANEL_COLUMNS,
                        next_timestamp,
                        resource_policy,
                    )
                    retained.enter_context(
                        _retain_frame(
                            next_panel,
                            resource_policy,
                            "research.portfolio.next_panel",
                        )
                    )
                    if next_panel.empty:
                        raise PortfolioTargetBlocked("portfolio_market_session_missing")
                    endpoint_panel = _read_session(
                        panel_root,
                        _PANEL_COLUMNS,
                        endpoint_timestamp,
                        resource_policy,
                    )
                    retained.enter_context(
                        _retain_frame(
                            endpoint_panel,
                            resource_policy,
                            "research.portfolio.endpoint_panel",
                        )
                    )
                    if endpoint_panel.empty:
                        raise PortfolioTargetBlocked("portfolio_market_session_missing")
                    account, returns, adv = _build_session(
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
                        amount_history=amount_history,
                        alpha=alpha,
                        assumptions=assumptions,
                        fee_schedule=fee_schedule,
                        fee_identity=fee_identity,
                        trade_cost=trade_cost,
                        market_name_for_type=market_name_for_type,
                    )
                    output = _ledger_frame(
                        account=account,
                        decision_timestamp=decision_timestamp,
                        returns=returns,
                        alpha_score=alpha,
                        adv20=adv,
                    )
                    holdings, cash, nav = _state_after_holding(
                        account,
                        returns,
                    )
                    _resize_amount_state_lease(
                        amount_state_lease,
                        len(amount_history) + len(holdings),
                    )
                    retained.enter_context(
                        _retain_frame(
                            output,
                            resource_policy,
                            "research.portfolio.output",
                        )
                    )
                    emitted = True
                    yield output

        missing, extra = prediction_lookup.coverage_counts()
        if missing or extra:
            raise ValueError("misaligned_persisted_prediction_keys")
        if not emitted:
            raise PortfolioTargetBlocked("portfolio_execution_window_empty")
    finally:
        if amount_state_lease is not None:
            _close_lease_preserving_primary(
                amount_state_lease,
                owner="research.portfolio.amount_state",
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


def _read_session(
    root: Path,
    columns: tuple[str, ...],
    timestamp: pd.Timestamp,
    resource_policy: RuntimeResourcePolicy | None,
) -> pd.DataFrame:
    frame = _read_materialized(
        root,
        columns,
        time_window=TimeWindow(timestamp, timestamp),
        resource_policy=resource_policy,
    )
    if "timestamp" not in frame.columns:
        raise ValueError("portfolio_input_schema_invalid")
    normalized = _session_frame(frame, columns)
    if not bool(normalized["timestamp"].eq(timestamp).all()):
        raise ValueError("portfolio_input_session_window_invalid")
    return normalized


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
    reserved = max(1, (input_rows + holding_count) * 256 + 16 * 1024)
    with resource_policy.lease("research.portfolio.session_workspace", reserved):
        yield


def _resize_amount_state_lease(
    lease: MemoryLease | None,
    asset_count: int,
) -> None:
    if lease is None:
        return
    if isinstance(asset_count, bool) or asset_count < 0:
        raise ValueError("portfolio_amount_state_invalid")
    reserved = max(1, 4 * 1024 + asset_count * (20 * 8 + 512))
    if reserved > lease.reserved_bytes:
        lease.grow(reserved - lease.reserved_bytes)
    elif reserved < lease.reserved_bytes:
        lease.shrink_to(reserved)


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
    amount_history: Mapping[str, deque[float]],
    alpha: pd.Series,
    assumptions: ExecutionAssumptions,
    fee_schedule: object,
    fee_identity: str,
    trade_cost: Callable[..., TradeCostComponents],
    market_name_for_type: Callable[[object], str],
) -> tuple[ExecutionAccounting, pd.Series, pd.Series]:
    panel = _session_frame(panel, _PANEL_COLUMNS)
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
    if is_rebalance_session(
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
    desired = target.reindex(assets, fill_value=0.0) - current.reindex(
        assets,
        fill_value=0.0,
    )
    for asset in assets:
        wants_trade = (
            desired[asset] > 0.0 and bool(masks["can_enter"].get(asset, False))
        ) or (desired[asset] < 0.0 and bool(masks["can_exit"].get(asset, False)))
        if wants_trade and pd.isna(adv.get(asset)):
            raise PortfolioTargetBlocked("portfolio_adv_warmup_insufficient")

    def cost_for_trade(
        execution_date: date,
        asset: str,
        side: str,
        notional: float,
        adv20: float,
        called_assumptions: ExecutionAssumptions,
    ) -> TradeCostComponents:
        return trade_cost(
            execution_date,
            market_by_asset[asset],
            side,
            notional,
            adv20,
            fee_schedule,
            called_assumptions,
        )

    account = execute_continuous_target(
        current.div(nav),
        target,
        nav=nav,
        average_daily_volume=adv,
        can_enter=masks["can_enter"],
        can_exit=masks["can_exit"],
        execution_date=next_timestamp.date(),
        cost_for_trade=cost_for_trade,
        assumptions=assumptions,
        cost_source_identity=fee_identity,
    )
    returns = _holding_returns(
        account,
        panel,
        next_panel,
        endpoint_panel,
        decision_timestamp=decision_timestamp,
        next_timestamp=next_timestamp,
    )
    return account, returns, adv


def _prediction_alpha(dataset: pd.DataFrame, prediction_lookup: object) -> pd.Series:
    dataset = _session_frame(dataset, _DATASET_COLUMNS).set_index(
        "asset_id",
        drop=False,
    )
    eligible = _bool_series(dataset["eligible"], "portfolio_eligibility_invalid")
    required = dataset.loc[eligible, ["timestamp", "asset_id"]]
    alpha = pd.Series(np.nan, index=dataset.index, dtype="float64")
    if required.empty:
        return alpha
    values_for_frame = getattr(prediction_lookup, "values_for_frame", None)
    if not callable(values_for_frame):
        raise ValueError("portfolio_prediction_lookup_invalid")
    with values_for_frame(required, mark_matched=True) as (values, found):
        if not all(found):
            raise ValueError("missing_required_prediction_keys")
        alpha.loc[required["asset_id"]] = np.asarray(values, dtype="float64")
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


def _validate_session_alignment(
    dataset: pd.DataFrame,
    panel: pd.DataFrame,
    execution: pd.DataFrame,
) -> None:
    keys = {
        tuple(row)
        for row in dataset.loc[:, list(_SESSION_KEY)].itertuples(
            index=False,
            name=None,
        )
    }
    for frame in (panel, execution):
        other = {
            tuple(row)
            for row in frame.loc[:, list(_SESSION_KEY)].itertuples(
                index=False,
                name=None,
            )
        }
        if keys != other:
            if keys.difference(other):
                raise PortfolioTargetBlocked("portfolio_session_key_missing")
            raise ValueError("portfolio_input_key_alignment_invalid")


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
    return {
        column: indexed.loc[list(assets), column].astype(bool)
        for column in ("can_enter", "can_exit")
    }


def _holding_returns(
    account: ExecutionAccounting,
    panel: pd.DataFrame,
    next_panel: pd.DataFrame,
    endpoint_panel: pd.DataFrame,
    *,
    decision_timestamp: pd.Timestamp,
    next_timestamp: pd.Timestamp,
) -> pd.Series:
    assets = account.asset_ledger["asset_id"].astype(str)
    returns = pd.Series(0.0, index=assets, dtype="float64")
    positive = account.asset_ledger["post_trade_notional"].to_numpy() > 0.0
    if not positive.any():
        return returns
    selected = assets.loc[positive]
    frames = []
    endpoint_timestamp = pd.Timestamp(endpoint_panel["timestamp"].iloc[0])
    for frame in (panel, next_panel, endpoint_panel):
        selected_frame = frame.loc[frame["asset_id"].astype(str).isin(set(selected))]
        if not selected_frame.empty:
            frames.append(
                selected_frame.loc[:, ["asset_id", "timestamp", "adj_open"]].copy()
            )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise PortfolioTargetBlocked("portfolio_held_return_missing")
    combined["adj_open"] = pd.to_numeric(combined["adj_open"], errors="raise")
    if (
        combined["adj_open"].isna().any()
        or not np.isfinite(combined["adj_open"].to_numpy()).all()
    ):
        raise ValueError("portfolio_return_invalid")
    labels = add_forward_return_labels(
        combined,
        horizons=(1,),
        sessions=(decision_timestamp, next_timestamp, endpoint_timestamp),
        can_enter_column=None,
    )
    label = labels.loc[labels["timestamp"].eq(decision_timestamp)]
    values = label.set_index("asset_id")["forward_return_1d"]
    missing = [
        asset
        for asset in selected
        if asset not in values.index or pd.isna(values[asset])
    ]
    if missing:
        raise PortfolioTargetBlocked("portfolio_held_return_missing")
    returns.loc[selected] = values.reindex(selected).astype("float64")
    return returns


def _ledger_frame(
    *,
    account: ExecutionAccounting,
    decision_timestamp: pd.Timestamp,
    returns: pd.Series,
    alpha_score: pd.Series,
    adv20: pd.Series,
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
            "expected_alpha": np.nan,
            "benchmark_weight": np.nan,
            "asset_specific_variance": np.nan,
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
                for column in PORTFOLIO_LEDGER_COLUMNS
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
        "target_notional": account.nav_before - asset_target_total,
        "executed_delta_notional": account.post_trade_cash - account.pre_trade_cash,
        "post_trade_notional": account.post_trade_cash,
        "holding_return": 0.0,
        "end_notional": account.post_trade_cash,
        "block_reason": None,
        "adv20": np.nan,
        "alpha_score": np.nan,
        "expected_alpha": np.nan,
        "benchmark_weight": np.nan,
        "asset_specific_variance": np.nan,
        "decision_reason": PORTFOLIO_DECISION_REASON_SIMPLE,
    }
    cash_row.update({column: 0.0 for column in PORTFOLIO_LEDGER_COST_COLUMNS})
    cash_row.update(
        {
            column: np.nan
            for column in PORTFOLIO_LEDGER_COLUMNS
            if column not in cash_row
        }
    )
    rows.append(cash_row)
    return pd.DataFrame.from_records(rows, columns=PORTFOLIO_LEDGER_COLUMNS)


def _state_after_holding(
    account: ExecutionAccounting,
    returns: pd.Series,
) -> tuple[pd.Series, float, float]:
    holdings = account.asset_ledger.set_index("asset_id")["post_trade_notional"]
    next_holdings, next_cash, next_nav = drift_continuous_holdings(
        holdings,
        account.post_trade_cash,
        returns.reindex(holdings.index),
    )
    return next_holdings.loc[next_holdings.ne(0.0)], next_cash, next_nav


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
    if root == marker or not root.exists() or root.name != expected_identity:
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
