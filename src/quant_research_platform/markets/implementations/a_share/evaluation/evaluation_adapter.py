"""Adapt private A-share factor tables into evaluation artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.artifacts.artifact_writer import (
    write_evaluation_set_artifacts,
)
from quant_research_platform.evaluations import (
    EvaluationSpec,
    NeutralizationConfig,
    TimeSemanticsConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.evaluations.diagnostics import DiagnosticsSpec
from quant_research_platform.evaluations.evaluation_payload import EvaluationRunPayload
from quant_research_platform.evaluations.evaluation_runner import evaluate_to_payload
from quant_research_platform.evaluations.evaluation_spec import PORTFOLIO_MODES
from quant_research_platform.kernels.kernel_registry import (
    normalize_kernel_name,
    supported_kernel_names,
)
from quant_research_platform.runtime.paths import factor_full_path_covering_range

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
DEFAULT_RETURN_SOURCE_COLUMN = "forward_return_1d"
DATE_COLUMN = "trade_date"
DEFAULT_REPRESENTATIVE_FACTORS = ("mom_20d", "vol_20d", "amihud_20d")
LABEL_AVAILABILITY_COLUMNS = (
    "eligible",
    "can_enter",
    "can_exit",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FORWARD_RETURN_LABEL_PATTERN = re.compile(r"^forward_return_\d+d$")
DEFAULT_A_SHARE_SOURCE_ALIAS = "a_share"


def schema_columns(path: Path) -> tuple[str, ...]:
    """Return Parquet schema column names."""

    return tuple(pq.ParquetFile(path).schema.names)


def discover_factor_columns(path: Path) -> tuple[str, ...]:
    """Discover value factor columns in a slim factor file."""

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
            column for column in schema_columns(path) if column.startswith(LABEL_PREFIX)
        )
    )


def choose_default_factor_columns(factor_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Choose representative factors when columns are not specified."""

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
    """Summarize label coverage and filter-column counts."""

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
    """Return the count of truthy values in a filter column."""

    if column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def normalize_date_arg(value: str | None) -> pd.Timestamp | None:
    """Normalize a CLI date parameter to a pandas timestamp."""

    if not value:
        return None
    return pd.Timestamp(value).normalize()


def date_range_id(start_date: str | None, end_date: str | None) -> str:
    """Return a compact date-range id for filtered artifacts."""

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
    """Filter a frame by inclusive trade-date bounds."""

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
    """Read the factor table with an optional trade-date filter."""

    read_columns = list(columns) if columns is not None else None
    if (start_date or end_date) and read_columns is not None:
        read_columns = list(dict.fromkeys((*read_columns, DATE_COLUMN)))
    frame = pd.read_parquet(
        path,
        columns=read_columns,
        engine="pyarrow",
        filters=_trade_date_filters(start_date=start_date, end_date=end_date),
    )
    return filter_by_trade_date(frame, start_date=start_date, end_date=end_date)


def _trade_date_filters(
    *,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str, object]] | None:
    filters: list[tuple[str, str, object]] = []
    if start_date is not None:
        filters.append((DATE_COLUMN, ">=", pd.Timestamp(start_date).to_pydatetime()))
    if end_date is not None:
        filters.append((DATE_COLUMN, "<=", pd.Timestamp(end_date).to_pydatetime()))
    return filters or None


def write_csv_utf8_sig(path: Path, frame: pd.DataFrame) -> Path:
    """Write a UTF-8 BOM CSV for direct Windows Excel opening."""

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
    """Read only columns needed for availability summaries."""

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
    label_columns: tuple[str, ...],
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read the filtered DataFrame needed for one evaluation request."""

    columns = tuple(
        dict.fromkeys(
            (
                "trade_date",
                "stock_code",
                *feature_columns,
                *label_columns,
                return_source_column,
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
    label_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    output_dir: Path,
    run_id: str,
    n_groups: int,
    source_file: str,
    start_date: str | None = None,
    end_date: str | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    cost_bps: float = 0.0,
    portfolio_mode: str = "long_short",
    kernel: str = "dataframe",
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_method: str = "none",
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
    diagnostics_spec: DiagnosticsSpec | None = None,
) -> EvaluationSpec:
    """Build an A-share evaluation-set spec from a filtered DataFrame."""

    for label_column in label_columns:
        validate_forward_return_label(label_column)
    metadata = {
        "run_id": run_id,
        "project": "quant-research-platform",
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
        "label_columns": list(label_columns),
        "start_date": start_date,
        "end_date": end_date,
        "return_source_column": return_source_column,
        "cost_bps": cost_bps,
        "portfolio_mode": portfolio_mode,
        "kernel": normalize_kernel_name(kernel),
        "segment_columns": list(segment_columns),
        "risk_column": risk_column,
        "neutralization_method": neutralization_method,
        "neutralization_group_columns": list(neutralization_group_columns),
        "neutralization_exposure_columns": list(neutralization_exposure_columns),
        "time_semantics": {
            "decision_time_column": "trade_date",
            "decision_time_convention": "A 股日频收盘后可知",
            "label_window_source": "processed forward_return_* columns",
            "label_column_rule": "forward_return_<horizon>d",
            "frequency": "1d",
            "timezone": "Asia/Shanghai",
        },
    }
    return EvaluationSpec(
        panel=panel,
        timestamp_column="trade_date",
        asset_column="stock_code",
        feature_columns=feature_columns,
        label_columns=label_columns,
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
        time_semantics_config=TimeSemanticsConfig(
            decision_time_column="trade_date",
            frequency="1d",
            timezone="Asia/Shanghai",
        ),
        segment_columns=segment_columns,
        kernel=normalize_kernel_name(kernel),
        return_source_column=return_source_column,
        time_breakdowns=("year",),
        output_dir=output_dir,
        diagnostics_spec=diagnostics_spec or DiagnosticsSpec.disabled(),
        domain_notes={
            "boundary": (
                "Processed DataFrame adapter only; evaluation layer does not generate "
                "private CSMAR data, download data, connect APIs, or provide "
                "investment advice."
            ),
            "domain_detail": "a_share_processed_dataframe_factor_panel",
        },
    )


def evaluate_a_share_matrix_payload(
    *,
    panel: pd.DataFrame,
    source_file: str,
    start_year: int,
    end_year: int,
    label_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    output_dir: Path,
    run_id: str,
    n_groups: int,
    start_date: str | None = None,
    end_date: str | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    cost_bps: float = 0.0,
    portfolio_mode: str = "long_short",
    kernel: str = "dataframe",
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_method: str = "none",
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
    diagnostics_spec: DiagnosticsSpec | None = None,
) -> EvaluationRunPayload:
    """Compute one kernel and portfolio-mode payload from a filtered frame."""

    spec = make_a_share_evaluation_spec(
        panel=panel,
        feature_columns=feature_columns,
        label_columns=label_columns,
        output_dir=output_dir,
        run_id=run_id,
        n_groups=n_groups,
        start_year=start_year,
        end_year=end_year,
        source_file=source_file,
        start_date=start_date,
        end_date=end_date,
        return_source_column=return_source_column,
        cost_bps=cost_bps,
        portfolio_mode=portfolio_mode,
        kernel=kernel,
        segment_columns=segment_columns,
        risk_column=risk_column,
        neutralization_method=neutralization_method,
        neutralization_group_columns=neutralization_group_columns,
        neutralization_exposure_columns=neutralization_exposure_columns,
        diagnostics_spec=diagnostics_spec,
    )
    return evaluate_to_payload(spec)


def validate_forward_return_label(label: str) -> None:
    """Require supported A-share forward-return label names."""

    if not FORWARD_RETURN_LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "A-share label columns must use forward_return_<horizon>d naming; "
            f"got {label!r}."
        )


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
    input_file: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    cost_bps: float = 0.0,
    portfolio_modes: tuple[str, ...] = ("long_short",),
    kernels: tuple[str, ...] = ("dataframe",),
    segment_columns: tuple[str, ...] = (),
    risk_column: str | None = None,
    neutralization_method: str = "none",
    neutralization_group_columns: tuple[str, ...] = (),
    neutralization_exposure_columns: tuple[str, ...] = (),
    source_alias: str = DEFAULT_A_SHARE_SOURCE_ALIAS,
    manifest_run_id: str | None = None,
    diagnostics_spec: DiagnosticsSpec | None = None,
) -> tuple[pd.DataFrame, Path, list[dict[str, object]]]:
    """Write one schema 0.7 evaluation_set from a slim factor Parquet file."""

    _validate_run_processed_options(
        start_year=start_year,
        end_year=end_year,
        kernels=kernels,
        portfolio_modes=portfolio_modes,
        requested_labels=requested_labels,
        n_groups=n_groups,
        start_date=start_date,
        end_date=end_date,
        cost_bps=cost_bps,
    )
    diagnostics_spec = diagnostics_spec or DiagnosticsSpec.disabled()
    label_summaries = []
    parquet_path = (
        input_file
        if input_file is not None
        else factor_full_path_covering_range(
            processed_root,
            start_year,
            end_year,
        )
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
    input_reference = _processed_factor_input_reference(parquet_path)

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
    evaluation_frame = read_evaluation_frame(
        parquet_path,
        feature_columns=feature_columns,
        label_columns=labels,
        return_source_column=return_source_column,
        segment_columns=segment_columns,
        risk_column=risk_column,
        neutralization_group_columns=neutralization_group_columns,
        neutralization_exposure_columns=neutralization_exposure_columns,
        start_date=range_start,
        end_date=range_end,
    )
    run_id = manifest_run_id or _default_manifest_run_id(
        report_root,
        source_alias=source_alias,
    )
    payloads: list[EvaluationRunPayload] = []
    for kernel in kernels:
        for portfolio_mode in portfolio_modes:
            payloads.append(
                evaluate_a_share_matrix_payload(
                    panel=evaluation_frame,
                    source_file=parquet_path.name,
                    start_year=start_year,
                    end_year=end_year,
                    label_columns=labels,
                    feature_columns=feature_columns,
                    output_dir=report_root,
                    run_id=run_id,
                    n_groups=n_groups,
                    start_date=start_date,
                    end_date=end_date,
                    return_source_column=return_source_column,
                    cost_bps=cost_bps,
                    portfolio_mode=portfolio_mode,
                    kernel=kernel,
                    segment_columns=segment_columns,
                    risk_column=risk_column,
                    neutralization_method=neutralization_method,
                    neutralization_group_columns=neutralization_group_columns,
                    neutralization_exposure_columns=neutralization_exposure_columns,
                    diagnostics_spec=diagnostics_spec,
                )
            )
    combined_tables = _combine_payload_tables(payloads)
    combined_summary = _combined_summary(
        payloads,
        tables=combined_tables,
        run_id=run_id,
        labels=labels,
        kernels=kernels,
        portfolio_modes=portfolio_modes,
    )
    combined_run_context = _combined_run_context(
        payloads,
        run_id=run_id,
        labels=labels,
        kernels=kernels,
        portfolio_modes=portfolio_modes,
        input_reference=input_reference,
        source_alias=source_alias,
        start_year=start_year,
        end_year=end_year,
        start_date=start_date,
        end_date=end_date,
        return_source_column=return_source_column,
        feature_columns=feature_columns,
        cost_bps=cost_bps,
        segment_columns=segment_columns,
        risk_column=risk_column,
        neutralization_method=neutralization_method,
        neutralization_group_columns=neutralization_group_columns,
        neutralization_exposure_columns=neutralization_exposure_columns,
        source_file=parquet_path.name,
        date_id=date_id,
    )
    artifacts = write_evaluation_set_artifacts(
        report_root,
        run_context=combined_run_context,
        summary=combined_summary,
        tables=combined_tables,
        report=_combined_report(combined_summary, combined_tables),
    )
    run_records = [
        {
            "run_id": run_id,
            "start_year": start_year,
            "end_year": end_year,
            "labels": list(labels),
            "features": list(feature_columns),
            "start_date": start_date,
            "end_date": end_date,
            "return_source_column": return_source_column,
            "cost_bps": cost_bps,
            "portfolio_modes": list(portfolio_modes),
            "kernels": list(kernels),
            "segment_columns": list(segment_columns),
            "risk_column": risk_column,
            "neutralization_method": neutralization_method,
            "neutralization_group_columns": list(neutralization_group_columns),
            "neutralization_exposure_columns": list(neutralization_exposure_columns),
            **input_reference,
            "input_materialized": False,
            "output_dir": ".",
        }
    ]

    label_summary = pd.concat(label_summaries, ignore_index=True)
    return label_summary, artifacts.manifest_path, run_records


def _processed_factor_input_reference(parquet_path: Path) -> dict[str, object]:
    """Return private-path-safe metadata for the processed factor input."""

    match = re.search(r"_(\d{4})_(\d{4})\.parquet$", parquet_path.name)
    reference: dict[str, object] = {"input_source": "processed_factor_table"}
    if match:
        reference["input_year_range"] = f"{match.group(1)}_{match.group(2)}"
    return reference


def _combine_payload_tables(
    payloads: list[EvaluationRunPayload],
) -> dict[str, pd.DataFrame]:
    if not payloads:
        raise ValueError("at least one evaluation payload is required.")
    table_names = sorted({name for payload in payloads for name in payload.tables})
    combined: dict[str, pd.DataFrame] = {}
    for name in table_names:
        if _is_retired_breakdown_table(name):
            continue
        frames = [
            payload.tables[name]
            for payload in payloads
            if name in payload.tables and not payload.tables[name].empty
        ]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            template = next(
                (
                    payload.tables[name]
                    for payload in payloads
                    if name in payload.tables
                ),
                None,
            )
            if template is not None:
                frame = template.head(0).copy()
        if name in {"coverage_summary", "diagnostic_policies"}:
            frame = frame.drop_duplicates(ignore_index=True)
        if name.startswith("evaluation_metrics_summary_by_"):
            frame = _normalize_schema_07_breakdown_table(
                frame,
                period=name.rsplit("_by_", maxsplit=1)[1],
            )
        combined[name] = frame
    return combined


def _is_retired_breakdown_table(name: str) -> bool:
    return any(
        name.startswith(prefix) and "_by_" in name
        for prefix in ("ic_summary", "group_spread_summary", "performance_summary")
    )


def _normalize_schema_07_breakdown_table(
    frame: pd.DataFrame,
    *,
    period: str,
) -> pd.DataFrame:
    if frame.empty or "metric_name" not in frame.columns:
        return frame
    normalized = frame.copy()
    if "period" in normalized.columns and normalized["period"].notna().any():
        return normalized
    records: list[dict[str, object]] = []
    current_period: dict[str, object] = {}
    for row in normalized.to_dict(orient="records"):
        combination_id = str(row.get("combination_id", ""))
        metric_name = row.get("metric_name")
        if metric_name == period:
            current_period[combination_id] = row.get("metric_value")
            continue
        if metric_name in {"period_type", "period_start", "period_end"}:
            continue
        period_value = current_period.get(combination_id)
        record = dict(row)
        record["period_type"] = period
        record["period"] = period_value
        record["period_start"] = _period_start(period, period_value)
        record["period_end"] = _period_end(period, period_value)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=list(frame.columns))


def _period_start(period: str, value: object) -> str:
    if period == "year" and pd.notna(value):
        return f"{int(value)}-01-01"
    return ""


def _period_end(period: str, value: object) -> str:
    if period == "year" and pd.notna(value):
        return f"{int(value)}-12-31"
    return ""


def _combined_summary(
    payloads: list[EvaluationRunPayload],
    *,
    tables: dict[str, pd.DataFrame],
    run_id: str,
    labels: tuple[str, ...],
    kernels: tuple[str, ...],
    portfolio_modes: tuple[str, ...],
) -> dict[str, object]:
    summary = dict(payloads[0].summary)
    summary["run_id"] = run_id
    summary["n_combinations"] = int(len(tables["combinations"]))
    summary["kernels"] = list(kernels)
    summary["portfolio_modes"] = list(portfolio_modes)
    base_evaluation = dict(summary.get("base_evaluation", {}))
    base_evaluation["label_columns"] = list(labels)
    base_evaluation["kernel"] = kernels[0] if len(kernels) == 1 else "multiple"
    base_evaluation["kernels"] = list(kernels)
    base_evaluation["portfolio_modes"] = list(portfolio_modes)
    summary["base_evaluation"] = base_evaluation
    summary["tables"] = {name: int(len(frame)) for name, frame in tables.items()}
    return summary


def _combined_run_context(
    payloads: list[EvaluationRunPayload],
    *,
    run_id: str,
    labels: tuple[str, ...],
    kernels: tuple[str, ...],
    portfolio_modes: tuple[str, ...],
    input_reference: dict[str, object],
    source_alias: str,
    start_year: int,
    end_year: int,
    start_date: str | None,
    end_date: str | None,
    return_source_column: str,
    feature_columns: tuple[str, ...],
    cost_bps: float,
    segment_columns: tuple[str, ...],
    risk_column: str | None,
    neutralization_method: str,
    neutralization_group_columns: tuple[str, ...],
    neutralization_exposure_columns: tuple[str, ...],
    source_file: str,
    date_id: str | None,
) -> dict[str, object]:
    del source_alias
    run_context = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in payloads[0].run_context.items()
    }
    config = dict(run_context.get("config", {}))
    evaluation_spec = dict(config.get("evaluation_spec", {}))
    kernel_value = kernels[0] if len(kernels) == 1 else "multiple"
    metadata = dict(evaluation_spec.get("metadata", {}))
    metadata.update(
        {
            "run_id": run_id,
            "label_columns": list(labels),
            "kernel": kernel_value,
            "kernels": list(kernels),
            "portfolio_modes": list(portfolio_modes),
        }
    )
    metadata.pop("portfolio_mode", None)
    evaluation_spec["metadata"] = metadata
    evaluation_spec["kernel"] = kernel_value
    evaluation_spec["kernels"] = list(kernels)
    evaluation_spec["portfolio_modes"] = list(portfolio_modes)
    _remove_dynamic_portfolio_mode_config(evaluation_spec)
    schema = dict(evaluation_spec.get("schema", {}))
    schema["label_columns"] = list(labels)
    schema["return_source_column"] = return_source_column
    evaluation_spec["schema"] = schema
    evaluation_spec["matrix"] = {
        "features": list(feature_columns),
        "labels": list(labels),
        "kernels": list(kernels),
        "portfolio_modes": list(portfolio_modes),
    }
    config["evaluation_spec"] = evaluation_spec
    run_context["config"] = config
    provenance = dict(run_context.get("provenance", {}))
    provenance.update(
        {
            "run_id": run_id,
            "start_year": start_year,
            "end_year": end_year,
            "start_date": start_date,
            "end_date": end_date,
            "date_id": date_id,
            "source_file": source_file,
            "return_source_column": return_source_column,
            "cost_config": {"cost_bps": cost_bps},
            "segment_columns": list(segment_columns),
            "risk_column": risk_column,
            "neutralization": {
                "method": neutralization_method,
                "group_columns": list(neutralization_group_columns),
                "exposure_columns": list(neutralization_exposure_columns),
            },
            **input_reference,
        }
    )
    run_context["provenance"] = provenance
    return run_context


def _remove_dynamic_portfolio_mode_config(config: dict[str, object]) -> None:
    weight_config = config.get("weight_config")
    if isinstance(weight_config, dict):
        normalized = dict(weight_config)
        normalized.pop("portfolio_mode", None)
        config["weight_config"] = normalized

    weight_configs = config.get("weight_configs")
    if isinstance(weight_configs, list):
        normalized_configs: list[dict[str, object] | object] = []
        for item in weight_configs:
            if isinstance(item, dict):
                normalized_item = dict(item)
                normalized_item.pop("portfolio_mode", None)
                normalized_configs.append(normalized_item)
            else:
                normalized_configs.append(item)
        config["weight_configs"] = normalized_configs


def _combined_report(
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> str:
    combinations = tables.get("combinations", pd.DataFrame())
    return "\n".join(
        [
            "# Evaluation Set Report",
            "",
            f"- run_id: {summary['run_id']}",
            f"- Combinations: {len(combinations)}",
            f"- Kernels: {', '.join(summary.get('kernels', []))}",
            f"- Portfolio modes: {', '.join(summary.get('portfolio_modes', []))}",
            "",
        ]
    )


def _default_manifest_run_id(report_root: Path, *, source_alias: str) -> str:
    """Derive API-visible manifest identity for full and nested pipeline reports."""

    if source_alias == "pipeline" and report_root.name == "evaluation":
        return report_root.parent.name
    return report_root.name


def _validate_run_processed_options(
    *,
    start_year: int,
    end_year: int,
    kernels: tuple[str, ...],
    portfolio_modes: tuple[str, ...],
    requested_labels: tuple[str, ...],
    n_groups: int,
    start_date: str | None,
    end_date: str | None,
    cost_bps: float,
) -> None:
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    if cost_bps < 0:
        raise ValueError("cost_bps must be nonnegative.")
    if not kernels:
        raise ValueError("kernels must contain at least one kernel.")
    for kernel in kernels:
        try:
            normalize_kernel_name(kernel)
        except Exception as exc:
            raise ValueError(
                f"kernel must be one of {list(supported_kernel_names())}."
            ) from exc
    if not portfolio_modes:
        raise ValueError("portfolio_modes must contain at least one mode.")
    for portfolio_mode in portfolio_modes:
        if portfolio_mode not in PORTFOLIO_MODES:
            raise ValueError(
                f"portfolio_mode must be one of {sorted(PORTFOLIO_MODES)}."
            )
    for label in requested_labels:
        if not RUN_ID_PATTERN.fullmatch(f"csmar_2000_2000_{label}"):
            raise ValueError(f"unsafe label for evaluation run_id: {label!r}")
        validate_forward_return_label(label)
    start = normalize_date_arg(start_date) if start_date else None
    end = normalize_date_arg(end_date) if end_date else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be less than or equal to end_date.")
    if start is not None and start.year < start_year:
        raise ValueError("start_date/end_date must be within start_year/end_year.")
    if end is not None and end.year > end_year:
        raise ValueError("start_date/end_date must be within start_year/end_year.")
