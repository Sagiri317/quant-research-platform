"""Bridge private A-share factor tables into alpha-research-core evaluations.

This module owns only the child-project adaptation layer: finding the processed
Parquet, reading the columns requested by the CLI, adding A-share metadata, and
calling core. Metric definitions, position paths, and period breakdown semantics
remain in alpha-research-core.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from alpha_research_core import (
    EvaluationSpec,
    NeutralizationConfig,
    TradabilityConfig,
    WeightConfig,
    run_evaluation,
)
from runtime.paths import factor_full_path_covering_range

FACTOR_PREFIXES = (
    "mom_",
    "rev_",
    "vol_",
    "avg_amount_",
    "turnover_",
    "amihud_",
    "amplitude_",
)
LABEL_PREFIX = "forward_return_"
PNL_RETURN_COLUMN = "forward_return_1d"
DATE_COLUMN = "trade_date"
DEFAULT_REPRESENTATIVE_FACTORS = ("mom_20d", "vol_20d", "amihud_20d")
LABEL_AVAILABILITY_COLUMNS = (
    "eligible",
    "can_enter",
    "can_exit",
)


def schema_columns(path: Path) -> tuple[str, ...]:
    """Return Parquet schema column names."""

    return tuple(pq.ParquetFile(path).schema.names)


def discover_factor_columns(path: Path) -> tuple[str, ...]:
    """Discover price-volume factor columns in a slim factor file."""

    return tuple(
        sorted(
            column
            for column in schema_columns(path)
            if column.startswith(FACTOR_PREFIXES)
        )
    )


def discover_label_columns(path: Path) -> tuple[str, ...]:
    """Discover forward-return label columns in a slim factor file."""

    return tuple(
        sorted(
            column
            for column in schema_columns(path)
            if column.startswith(LABEL_PREFIX)
        )
    )


def choose_default_factor_columns(factor_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Choose representative factors when the caller does not request columns."""

    selected = [
        column for column in DEFAULT_REPRESENTATIVE_FACTORS if column in factor_columns
    ]
    if selected:
        return tuple(selected)
    return factor_columns[:3]


def parse_columns(value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI column list."""

    return tuple(column.strip() for column in value.split(",") if column.strip())


def summarize_label_availability(
    frame: pd.DataFrame,
    *,
    label_columns: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    """Summarize label non-null rates and filter-column counts."""

    records: list[dict[str, object]] = []
    rows = len(frame)
    for label in label_columns:
        label_series = frame[label]
        records.append(
            {
                "year": year,
                "label": label,
                "rows": rows,
                "non_null_rows": int(label_series.notna().sum()),
                "non_null_rate": float(label_series.notna().mean()) if rows else 0.0,
                "eligible_rows": true_count(frame, "eligible"),
                "can_enter_rows": true_count(frame, "can_enter"),
                "can_exit_rows": true_count(frame, "can_exit"),
            }
        )
    return pd.DataFrame.from_records(records)


def true_count(frame: pd.DataFrame, column: str) -> int:
    """Return count of truthy values in a filter column."""

    if column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def normalize_date_arg(value: str | None) -> pd.Timestamp | None:
    """Normalize a CLI date argument to a pandas timestamp."""

    if not value:
        return None
    return pd.Timestamp(value).normalize()


def date_range_id(start_date: str | None, end_date: str | None) -> str:
    """Return a compact date-range id for filtered evaluation artifacts."""

    start = normalize_date_arg(start_date)
    end = normalize_date_arg(end_date)
    start_text = start.strftime("%Y%m%d") if start is not None else "start"
    end_text = end.strftime("%Y%m%d") if end is not None else "end"
    return f"{start_text}_{end_text}"


def filter_by_trade_date(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filter a frame by inclusive trade_date bounds."""

    start = normalize_date_arg(start_date)
    end = normalize_date_arg(end_date)
    if start is None and end is None:
        return frame
    trade_dates = pd.to_datetime(frame[DATE_COLUMN])
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= trade_dates >= start
    if end is not None:
        mask &= trade_dates <= end
    return frame.loc[mask].copy()


def read_factor_frame(
    path: Path,
    *,
    columns: tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read a factor table and optionally filter by trade_date."""

    read_columns = list(columns) if columns is not None else None
    if (start_date or end_date) and read_columns is not None:
        read_columns = list(dict.fromkeys((*read_columns, DATE_COLUMN)))
    frame = pd.read_parquet(path, columns=read_columns, engine="pyarrow")
    return filter_by_trade_date(frame, start_date=start_date, end_date=end_date)


def write_csv_utf8_sig(path: Path, frame: pd.DataFrame) -> Path:
    """Write CSV with BOM for direct Windows Excel opening."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_label_availability_frame(
    path: Path,
    label_columns: tuple[str, ...],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read only filter and label columns needed for availability summary."""

    columns = tuple(dict.fromkeys((*LABEL_AVAILABILITY_COLUMNS, *label_columns)))
    return read_factor_frame(
        path,
        columns=columns,
        start_date=start_date,
        end_date=end_date,
    )


def read_evaluation_frame(
    path: Path,
    *,
    feature_columns: tuple[str, ...],
    label_column: str,
    pnl_return_column: str = PNL_RETURN_COLUMN,
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read the filtered DataFrame needed for one core evaluation request.

    Only columns referenced by EvaluationSpec are loaded. This avoids pulling
    panel-only fields into the core handoff while preserving optional segment,
    risk, and neutralization columns when the CLI asks for them.
    """

    columns = tuple(
        dict.fromkeys(
            (
                "trade_date",
                "stock_code",
                *feature_columns,
                label_column,
                pnl_return_column,
                *segment_columns,
                *(column for column in (risk_column,) if column),
                *neutralization_group_columns,
                *neutralization_exposure_columns,
                *LABEL_AVAILABILITY_COLUMNS,
            )
        )
    )
    frame = read_factor_frame(
        path,
        columns=columns,
        start_date=start_date,
        end_date=end_date,
    )
    if frame.empty:
        raise ValueError(
            f"no rows after date filter for {path}: "
            f"start_date={start_date} end_date={end_date}"
        )
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def make_a_share_evaluation_spec(
    *,
    panel: pd.DataFrame,
    start_year: int,
    end_year: int,
    label_column: str,
    feature_columns: tuple[str, ...],
    output_dir: Path,
    run_id: str,
    n_groups: int,
    source_file: str,
    start_date: str | None = None,
    end_date: str | None = None,
    pnl_return_column: str = PNL_RETURN_COLUMN,
    cost_bps: float = 0.0,
    portfolio_mode: str = "long_short",
    engine: str = "dataframe",
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_method: str = "none",
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
) -> EvaluationSpec:
    """Build an A-share evaluation-set spec from an already filtered DataFrame.

    The year range describes the requested research window; core still receives
    one continuous panel and produces metrics_by_year by slicing the completed
    path, not by restarting positions at calendar year boundaries.
    """

    metadata = {
        "run_id": run_id,
        "project": "a-share-alpha-lab",
        "kind": "factor",
        "domain": "a_share",
        "data_stage": "approved_offline_real_data",
        "uses_external_data": True,
        "adapter": "a_share_dataframe",
        "adapter_kind": "processed_dataframe_child_project_contract",
        "start_year": start_year,
        "end_year": end_year,
        "source_file": source_file,
        "factor_count": len(feature_columns),
        "label_columns": [label_column],
        "start_date": start_date,
        "end_date": end_date,
        "pnl_return_column": pnl_return_column,
        "cost_bps": cost_bps,
        "portfolio_mode": portfolio_mode,
        "engine": engine,
        "segment_columns": list(segment_columns),
        "risk_column": risk_column,
        "neutralization_method": neutralization_method,
        "neutralization_group_columns": list(neutralization_group_columns),
        "neutralization_exposure_columns": list(neutralization_exposure_columns),
    }
    return EvaluationSpec(
        panel=panel,
        timestamp_column="trade_date",
        asset_column="stock_code",
        feature_columns=feature_columns,
        label_columns=(label_column,),
        metadata=metadata,
        n_groups=n_groups,
        weight_configs=(
            WeightConfig(
                cost_bps=cost_bps,
                portfolio_mode=portfolio_mode,
                risk_column=risk_column,
            ),
        ),
        tradability_config=TradabilityConfig(
            eligibility_column="eligible",
            can_enter_column="can_enter",
            can_exit_column="can_exit",
        ),
        neutralization_config=NeutralizationConfig(
            method=neutralization_method,
            group_columns=neutralization_group_columns,
            exposure_columns=neutralization_exposure_columns,
        ),
        segment_columns=segment_columns,
        engine=engine,
        pnl_return_column=pnl_return_column,
        time_breakdowns=("year",),
        output_dir=output_dir,
        domain_notes={
            "boundary": (
                "Processed DataFrame adapter only; core does not generate "
                "private CSMAR data, download data, connect APIs, or provide "
                "investment advice."
            ),
            "domain_detail": "a_share_processed_dataframe_factor_panel",
        },
    )


def run_label_evaluation(
    *,
    panel: pd.DataFrame,
    source_file: str,
    start_year: int,
    end_year: int,
    label_column: str,
    feature_columns: tuple[str, ...],
    output_root: Path,
    n_groups: int,
    date_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    pnl_return_column: str = PNL_RETURN_COLUMN,
    cost_bps: float = 0.0,
    portfolio_mode: str = "long_short",
    engine: str = "dataframe",
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_method: str = "none",
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
) -> Path:
    """Run one label core evaluation over a filtered full-sample frame.

    Each label gets its own core run directory because core combinations are
    built over one label set, one weight config, and one cost config at a time.
    """

    run_id = f"csmar_{start_year}_{end_year}_{label_column}"
    if date_id:
        run_id = f"{run_id}_{date_id}"
    output_dir = output_root / "core_runs" / run_id
    spec = make_a_share_evaluation_spec(
        panel=panel,
        feature_columns=feature_columns,
        label_column=label_column,
        output_dir=output_dir,
        run_id=run_id,
        n_groups=n_groups,
        start_year=start_year,
        end_year=end_year,
        source_file=source_file,
        start_date=start_date,
        end_date=end_date,
        pnl_return_column=pnl_return_column,
        cost_bps=cost_bps,
        portfolio_mode=portfolio_mode,
        engine=engine,
        segment_columns=segment_columns,
        risk_column=risk_column,
        neutralization_method=neutralization_method,
        neutralization_group_columns=neutralization_group_columns,
        neutralization_exposure_columns=neutralization_exposure_columns,
    )
    run_evaluation(spec)
    return output_dir


def run_processed_factor_evaluation(
    *,
    processed_root: Path,
    report_root: Path,
    start_year: int,
    end_year: int,
    requested_labels: tuple[str, ...],
    requested_factors: tuple[str, ...],
    all_factors: bool,
    n_groups: int,
    start_date: str | None = None,
    end_date: str | None = None,
    pnl_return_column: str = PNL_RETURN_COLUMN,
    cost_bps: float = 0.0,
    portfolio_mode: str = "long_short",
    engine: str = "dataframe",
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_method: str = "none",
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, Path, list[dict[str, object]]]:
    """Run core evaluation over one full-sample slim factor Parquet file.

    Label availability is summarized by calendar year for diagnostics, but the
    evaluation itself reads one continuous date range so path metrics can carry
    positions across year-end when tradability permits.
    """

    label_summaries = []
    core_runs: list[dict[str, object]] = []
    parquet_path = factor_full_path_covering_range(
        processed_root,
        start_year,
        end_year,
    )
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    available_factors = discover_factor_columns(parquet_path)
    available_labels = discover_label_columns(parquet_path)
    labels = requested_labels or available_labels
    missing_labels = sorted(set(labels).difference(available_labels))
    if missing_labels:
        raise ValueError(f"missing labels in {parquet_path}: {missing_labels}")
    if requested_factors:
        feature_columns = requested_factors
    elif all_factors:
        feature_columns = available_factors
    else:
        feature_columns = choose_default_factor_columns(available_factors)
    missing_factors = sorted(set(feature_columns).difference(available_factors))
    if missing_factors:
        raise ValueError(f"missing factors in {parquet_path}: {missing_factors}")

    range_start = start_date or f"{start_year}-01-01"
    range_end = end_date or f"{end_year}-12-31"
    availability_frame = read_label_availability_frame(
        parquet_path,
        available_labels,
        start_date=range_start,
        end_date=range_end,
    )
    availability_frame[DATE_COLUMN] = pd.to_datetime(availability_frame[DATE_COLUMN])
    for year in range(start_year, end_year + 1):
        year_frame = availability_frame[
            availability_frame[DATE_COLUMN].dt.year.eq(year)
        ]
        if year_frame.empty:
            continue
        availability = summarize_label_availability(
            year_frame,
            label_columns=available_labels,
            year=year,
        )
        label_summaries.append(availability)

    date_id = date_range_id(start_date, end_date) if start_date or end_date else None
    for label in labels:
        evaluation_frame = read_evaluation_frame(
            parquet_path,
            feature_columns=feature_columns,
            label_column=label,
            pnl_return_column=pnl_return_column,
            segment_columns=segment_columns,
            risk_column=risk_column,
            neutralization_group_columns=neutralization_group_columns,
            neutralization_exposure_columns=neutralization_exposure_columns,
            start_date=range_start,
            end_date=range_end,
        )
        output_dir = run_label_evaluation(
            panel=evaluation_frame,
            source_file=parquet_path.as_posix(),
            start_year=start_year,
            end_year=end_year,
            label_column=label,
            feature_columns=feature_columns,
            output_root=report_root,
            n_groups=n_groups,
            date_id=date_id,
            start_date=start_date,
            end_date=end_date,
            pnl_return_column=pnl_return_column,
            cost_bps=cost_bps,
            portfolio_mode=portfolio_mode,
            engine=engine,
            segment_columns=segment_columns,
            risk_column=risk_column,
            neutralization_method=neutralization_method,
            neutralization_group_columns=neutralization_group_columns,
            neutralization_exposure_columns=neutralization_exposure_columns,
        )
        run_id = output_dir.name
        core_runs.append(
            {
                "run_id": run_id,
                "start_year": start_year,
                "end_year": end_year,
                "label": label,
                "features": list(feature_columns),
                "start_date": start_date,
                "end_date": end_date,
                "pnl_return_column": pnl_return_column,
                "cost_bps": cost_bps,
                "portfolio_mode": portfolio_mode,
                "engine": engine,
                "segment_columns": list(segment_columns),
                "risk_column": risk_column,
                "neutralization_method": neutralization_method,
                "neutralization_group_columns": list(neutralization_group_columns),
                "neutralization_exposure_columns": list(
                    neutralization_exposure_columns
                ),
                "input_file": parquet_path.as_posix(),
                "input_materialized": False,
                "output_dir": output_dir.as_posix(),
            }
        )

    label_summary = pd.concat(label_summaries, ignore_index=True)
    write_csv_utf8_sig(report_root / "label_availability.csv", label_summary)
    manifest_path = report_root / "evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"core_runs": core_runs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return label_summary, manifest_path, core_runs
