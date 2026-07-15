"""Spec-driven portfolio runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXPOSURE_SUMMARY_COLUMNS,
    POSITION_COLUMNS,
    RETURNS_COLUMNS,
)
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.research._memory import iter_complete_time_sized_frames
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.portfolios.engine import (
    compute_cumulative_returns,
    compute_performance_summary,
    construct_portfolio_positions,
    summarize_period_breakdown,
)
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class PortfolioResult:
    spec: PortfolioSpec
    output_chunks: Iterable[Mapping[str, pd.DataFrame]]
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]]


def build_portfolio_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    portfolio_catalog: PortfolioCatalog,
    portfolio_id: str,
    time_window: TimeWindow | None = None,
) -> PortfolioResult:
    spec = portfolio_catalog.get(portfolio_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    read_columns = _portfolio_read_columns(spec)
    source_columns_by_kind = {
        "factor": tuple(dataset_spec.factor_columns),
        "feature": tuple(dataset_spec.feature_columns),
        "prediction": tuple(dataset_spec.context_columns),
    }
    summary_state = _new_portfolio_summary_state()
    output_chunks = _observe_portfolio_summary_chunks(
        _iter_portfolio_output_chunks(
            _portfolio_input_chunks(
                layout=layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_spec.data_id,
                time_window=time_window,
                columns=read_columns,
            ),
            spec,
            source_columns_by_kind=source_columns_by_kind,
        ),
        summary_state,
    )
    return PortfolioResult(
        spec=spec,
        output_chunks=output_chunks,
        summary_factory=lambda: _portfolio_summary_outputs(summary_state),
    )


def _portfolio_read_columns(spec: PortfolioSpec) -> tuple[str, ...]:
    optional_columns = (
        spec.weight_config.risk_column,
        spec.tradability_config.eligibility_column,
        spec.tradability_config.can_enter_column,
        spec.tradability_config.can_exit_column,
        spec.diagnostics_config.capacity_amount_column,
        spec.diagnostics_config.capacity_float_market_cap_column,
        *spec.diagnostics_config.numeric_exposure_columns,
        *spec.diagnostics_config.categorical_exposure_columns,
    )
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(signal.source_column for signal in spec.signals),
                spec.return_source_column,
                *(column for column in optional_columns if column is not None),
            )
        )
    )


def _portfolio_input_chunks(
    *,
    layout: StorageLayout,
    market_id: str,
    dataset_id: str,
    time_window: TimeWindow | None,
    columns: tuple[str, ...],
):
    return iter_complete_time_sized_frames(
        iter_research_dataset_chunks(
            layout,
            market_id=market_id,
            dataset_id=dataset_id,
            time_window=time_window,
            columns=columns,
        ),
        timestamp_column="timestamp",
    )


def _iter_portfolio_output_chunks(
    chunks,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind,
) -> Iterable[dict[str, pd.DataFrame]]:
    context = pd.DataFrame()
    context_periods = max(1, int(spec.execution_lag_steps) + 1)
    saw_rows = False

    for chunk in chunks:
        if chunk.empty:
            continue
        saw_rows = True
        actual_timestamps = _timestamp_set(chunk)
        working = _combine_context_and_chunk(context, chunk)
        result = construct_portfolio_positions(
            working,
            spec,
            source_columns_by_kind=source_columns_by_kind,
        )
        yield {
            "positions": _select_actual_timestamps(
                result.positions,
                actual_timestamps,
            ),
            "turnover": _select_actual_timestamps(
                result.turnover,
                actual_timestamps,
            ),
            "cost_breakdown": _select_actual_timestamps(
                result.cost_breakdown,
                actual_timestamps,
            ),
            "portfolio_returns": _select_actual_timestamps(
                result.portfolio_returns,
                actual_timestamps,
            ),
            "diagnostics": _select_actual_timestamps(
                result.diagnostics,
                actual_timestamps,
            ),
            "capacity_diagnostics": _select_actual_timestamps(
                result.capacity_diagnostics,
                actual_timestamps,
            ),
            "numeric_exposure_diagnostics": _select_actual_timestamps(
                result.numeric_exposure_diagnostics,
                actual_timestamps,
            ),
            "categorical_exposure_diagnostics": _select_actual_timestamps(
                result.categorical_exposure_diagnostics,
                actual_timestamps,
            ),
        }
        context = _trailing_timestamp_context(working, periods=context_periods)

    if not saw_rows:
        raise ValueError("empty_portfolio_input")


def _new_portfolio_summary_state() -> dict[str, object]:
    return {
        "return_records": [],
        "exposure_periods": {},
        "concentration_periods": {},
    }


def _observe_portfolio_summary_chunks(
    output_chunks: Iterable[dict[str, pd.DataFrame]],
    summary_state: dict[str, object],
) -> Iterable[dict[str, pd.DataFrame]]:
    for output_chunk in output_chunks:
        _update_portfolio_summary_state(
            summary_state,
            positions=output_chunk["positions"],
            portfolio_returns=output_chunk["portfolio_returns"],
        )
        yield output_chunk


def _update_portfolio_summary_state(
    summary_state: dict[str, object],
    *,
    positions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
) -> None:
    _portfolio_return_records(summary_state).extend(
        _portfolio_return_summary_records(portfolio_returns)
    )
    _update_position_period_summary(
        _portfolio_period_map(summary_state, "exposure_periods"),
        _portfolio_period_map(summary_state, "concentration_periods"),
        positions,
    )


def _portfolio_return_summary_records(
    portfolio_returns: pd.DataFrame,
) -> list[dict[str, object]]:
    if portfolio_returns.empty:
        return []
    selected = portfolio_returns.loc[:, RETURNS_COLUMNS].copy()
    selected["timestamp"] = strict_timestamp_series(
        selected["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return selected.sort_values(["signal", "timestamp"], kind="mergesort").to_dict(
        "records"
    )


def _update_position_period_summary(
    exposure_periods: dict[str, list[dict[str, object]]],
    concentration_periods: dict[str, list[dict[str, object]]],
    positions: pd.DataFrame,
) -> None:
    if positions.empty:
        return
    frame = positions.loc[:, POSITION_COLUMNS].copy()
    frame["timestamp"] = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    frame["abs_position"] = frame["position"].abs()
    for (signal, _timestamp), group in frame.groupby(
        ["signal", "timestamp"],
        sort=True,
        dropna=False,
    ):
        signal_id = str(signal)
        gross_exposure = float(group["abs_position"].sum())
        net_exposure = float(group["position"].sum())
        position_count = int((group["position"] != 0).sum())
        exposure_periods.setdefault(signal_id, []).append(
            {
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "position_count": position_count,
            }
        )
        if gross_exposure:
            shares = group["abs_position"] / gross_exposure
            hhi = float((shares.pow(2)).sum())
            top_weight = float(shares.max())
        else:
            hhi = 0.0
            top_weight = 0.0
        concentration_periods.setdefault(signal_id, []).append(
            {
                "hhi": hhi,
                "top_weight": top_weight,
                "position_count": position_count,
            }
        )


def _portfolio_summary_outputs(
    summary_state: dict[str, object],
) -> dict[str, pd.DataFrame]:
    portfolio_returns = _portfolio_returns_summary_frame(summary_state)
    cumulative_returns = compute_cumulative_returns(portfolio_returns)
    performance_summary = compute_performance_summary(
        portfolio_returns,
        cumulative_returns,
    )
    exposure_summary = _portfolio_exposure_summary(summary_state, portfolio_returns)
    concentration = _portfolio_concentration_summary(summary_state)
    period_breakdown = summarize_period_breakdown(portfolio_returns)
    return {
        "cumulative_returns": cumulative_returns,
        "performance_summary": performance_summary,
        "exposure_summary": exposure_summary,
        "concentration": concentration,
        "period_breakdown": period_breakdown,
    }


def _portfolio_returns_summary_frame(
    summary_state: dict[str, object],
) -> pd.DataFrame:
    records = _portfolio_return_records(summary_state)
    if not records:
        return _empty_summary_frame(RETURNS_COLUMNS)
    frame = pd.DataFrame.from_records(records, columns=RETURNS_COLUMNS)
    frame["timestamp"] = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return frame.sort_values(["signal", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )


def _portfolio_exposure_summary(
    summary_state: dict[str, object],
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    exposure_periods = _portfolio_period_map(summary_state, "exposure_periods")
    if not exposure_periods:
        return _empty_summary_frame(EXPOSURE_SUMMARY_COLUMNS)
    count_summary = _return_count_summary(portfolio_returns)
    records: list[dict[str, object]] = []
    for signal_id in sorted(exposure_periods):
        periods = exposure_periods[signal_id]
        gross_values = [float(period["gross_exposure"]) for period in periods]
        net_values = [float(period["net_exposure"]) for period in periods]
        position_counts = [float(period["position_count"]) for period in periods]
        count_values = count_summary.get(signal_id, {})
        records.append(
            {
                "signal": signal_id,
                "mean_gross_exposure": _mean(gross_values),
                "mean_net_exposure": _mean(net_values),
                "max_gross_exposure": max(gross_values, default=0.0),
                "max_abs_net_exposure": max(
                    (abs(value) for value in net_values),
                    default=0.0,
                ),
                "mean_long_count": float(count_values.get("mean_long_count", 0.0)),
                "mean_short_count": float(count_values.get("mean_short_count", 0.0)),
                "mean_position_count": _mean(position_counts),
            }
        )
    return pd.DataFrame.from_records(records, columns=EXPOSURE_SUMMARY_COLUMNS)


def _portfolio_concentration_summary(summary_state: dict[str, object]) -> pd.DataFrame:
    concentration_periods = _portfolio_period_map(
        summary_state,
        "concentration_periods",
    )
    columns = [
        "signal",
        "periods",
        "mean_hhi",
        "max_hhi",
        "mean_top_weight",
        "max_top_weight",
        "mean_position_count",
    ]
    if not concentration_periods:
        return _empty_summary_frame(columns)
    records: list[dict[str, object]] = []
    for signal_id in sorted(concentration_periods):
        periods = concentration_periods[signal_id]
        hhi_values = [float(period["hhi"]) for period in periods]
        top_weight_values = [float(period["top_weight"]) for period in periods]
        position_counts = [float(period["position_count"]) for period in periods]
        records.append(
            {
                "signal": signal_id,
                "periods": len(periods),
                "mean_hhi": _mean(hhi_values),
                "max_hhi": max(hhi_values, default=0.0),
                "mean_top_weight": _mean(top_weight_values),
                "max_top_weight": max(top_weight_values, default=0.0),
                "mean_position_count": _mean(position_counts),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _return_count_summary(
    portfolio_returns: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if portfolio_returns.empty:
        return {}
    frame = portfolio_returns.loc[:, ["signal", "long_count", "short_count"]].copy()
    frame["long_count"] = pd.to_numeric(frame["long_count"], errors="coerce")
    frame["short_count"] = pd.to_numeric(frame["short_count"], errors="coerce")
    result: dict[str, dict[str, float]] = {}
    for signal_id, group in frame.groupby("signal", sort=True, dropna=False):
        result[str(signal_id)] = {
            "mean_long_count": float(group["long_count"].mean()),
            "mean_short_count": float(group["short_count"].mean()),
        }
    return result


def _portfolio_return_records(
    summary_state: dict[str, object],
) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], summary_state["return_records"])


def _portfolio_period_map(
    summary_state: dict[str, object],
    state_key: str,
) -> dict[str, list[dict[str, object]]]:
    return cast(dict[str, list[dict[str, object]]], summary_state[state_key])


def _mean(values: Iterable[float]) -> float:
    sequence = tuple(values)
    if not sequence:
        return 0.0
    return float(sum(sequence) / len(sequence))


def _combine_context_and_chunk(
    context: pd.DataFrame,
    chunk: pd.DataFrame,
) -> pd.DataFrame:
    if context.empty:
        return chunk
    # Bounded context carries only recent timestamps for lag/turnover continuity.
    return pd.concat([context, chunk], ignore_index=True)


def _timestamp_set(frame: pd.DataFrame) -> set[pd.Timestamp]:
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return set(pd.Index(timestamps).unique())


def _select_actual_timestamps(
    frame: pd.DataFrame,
    actual_timestamps: set[pd.Timestamp],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return frame.loc[timestamps.isin(actual_timestamps)].copy()


def _trailing_timestamp_context(frame: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    trailing = pd.Index(timestamps.unique()).sort_values().tolist()[-periods:]
    return frame.loc[timestamps.isin(trailing)].copy()


def _empty_summary_frame(columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(columns))
    if "position" in frame.columns:
        frame["position"] = frame["position"].astype("float64")
    return frame


__all__ = ["PortfolioResult", "build_portfolio_result_from_research_dataset"]
