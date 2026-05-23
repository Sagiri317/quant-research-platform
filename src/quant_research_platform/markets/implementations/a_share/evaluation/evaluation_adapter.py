"""把私有 A 股 factor table 接入平台 evaluation layer。

本模块只负责 A 股适配：定位 processed Parquet、读取 CLI 请求的 columns、
补充 A 股 metadata，并调用平台 evaluation layer。指标定义、持仓路径和
period breakdown 语义保留在 evaluations 与 kernels 模块内。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.evaluations import (
    EvaluationSpec,
    NeutralizationConfig,
    TimeSemanticsConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import PORTFOLIO_MODES
from quant_research_platform.kernels.kernel_registry import (
    normalize_engine_name,
    supported_engine_names,
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
PNL_RETURN_COLUMN = "forward_return_1d"
DATE_COLUMN = "trade_date"
DEFAULT_REPRESENTATIVE_FACTORS = ("mom_20d", "vol_20d", "amihud_20d")
LABEL_AVAILABILITY_COLUMNS = (
    "eligible",
    "can_enter",
    "can_exit",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FORWARD_RETURN_LABEL_PATTERN = re.compile(r"^forward_return_\d+d$")


def schema_columns(path: Path) -> tuple[str, ...]:
    """返回 Parquet schema column names。"""

    return tuple(pq.ParquetFile(path).schema.names)


def discover_factor_columns(path: Path) -> tuple[str, ...]:
    """发现 slim factor file 中的价量 factor columns。"""

    return tuple(
        sorted(
            column
            for column in schema_columns(path)
            if column.startswith(FACTOR_PREFIXES)
        )
    )


def discover_label_columns(path: Path) -> tuple[str, ...]:
    """发现 slim factor file 中的 forward-return label columns。"""

    return tuple(
        sorted(
            column
            for column in schema_columns(path)
            if column.startswith(LABEL_PREFIX)
        )
    )


def choose_default_factor_columns(factor_columns: tuple[str, ...]) -> tuple[str, ...]:
    """在调用方未指定 columns 时选择代表性 factors。"""

    selected = [
        column for column in DEFAULT_REPRESENTATIVE_FACTORS if column in factor_columns
    ]
    if selected:
        return tuple(selected)
    return factor_columns[:3]


def parse_columns(value: str) -> tuple[str, ...]:
    """解析逗号分隔的 CLI column list。"""

    return tuple(column.strip() for column in value.split(",") if column.strip())


def summarize_label_availability(
    frame: pd.DataFrame,
    *,
    label_columns: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    """汇总 label 非空率和 filter-column 计数。"""

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
    """返回 filter column 中 truthy values 的数量。"""

    if column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def normalize_date_arg(value: str | None) -> pd.Timestamp | None:
    """把 CLI date 参数规范化为 pandas timestamp。"""

    if not value:
        return None
    return pd.Timestamp(value).normalize()


def date_range_id(start_date: str | None, end_date: str | None) -> str:
    """为过滤后的 evaluation artifacts 返回紧凑 date-range id。"""

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
    """按闭区间 trade_date 边界过滤 frame。"""

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
    """读取 factor table，并可按 trade_date 过滤。"""

    read_columns = list(columns) if columns is not None else None
    if (start_date or end_date) and read_columns is not None:
        read_columns = list(dict.fromkeys((*read_columns, DATE_COLUMN)))
    frame = pd.read_parquet(path, columns=read_columns, engine="pyarrow")
    return filter_by_trade_date(frame, start_date=start_date, end_date=end_date)


def write_csv_utf8_sig(path: Path, frame: pd.DataFrame) -> Path:
    """写出带 BOM 的 CSV，便于 Windows Excel 直接打开。"""

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
    """只读取 availability summary 所需的 filter 与 label columns。"""

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
    """读取一次 evaluation layer 请求所需的过滤后 DataFrame。

    只加载 EvaluationSpec 引用的 columns，避免把 panel-only fields 带入
    evaluation handoff，同时在 CLI 请求时保留 segment、risk 和
    neutralization columns。
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
    """从已过滤的 DataFrame 构造 A 股 evaluation-set spec。

    年份范围描述请求的 research window；evaluation layer 接收连续 panel，
    并通过切分完整路径生成 metrics_by_year，而不是在自然年边界重启持仓。
    """

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
        "label_columns": [label_column],
        "start_date": start_date,
        "end_date": end_date,
        "pnl_return_column": pnl_return_column,
        "cost_bps": cost_bps,
        "portfolio_mode": portfolio_mode,
        "engine": normalize_engine_name(engine),
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
        time_semantics_config=TimeSemanticsConfig(
            decision_time_column="trade_date",
            frequency="1d",
            timezone="Asia/Shanghai",
        ),
        segment_columns=segment_columns,
        engine=normalize_engine_name(engine),
        pnl_return_column=pnl_return_column,
        time_breakdowns=("year",),
        output_dir=output_dir,
        domain_notes={
            "boundary": (
                "Processed DataFrame adapter only; evaluation layer does not generate "
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
    """在过滤后的全样本 frame 上运行一个 label 的 evaluation。

    每个 label 使用独立 evaluation run directory，因为 combinations
    以一个 label set、一个 weight config 和一个 cost config 为单位构造。
    """

    run_id = f"csmar_{start_year}_{end_year}_{label_column}"
    if date_id:
        run_id = f"{run_id}_{date_id}"
    output_dir = _evaluation_run_output_dir(output_root, run_id)
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


def _evaluation_run_output_dir(output_root: Path, run_id: str) -> Path:
    """返回 output_root/evaluation_runs 下的安全 evaluation run 输出路径。"""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"unsafe evaluation run_id: {run_id!r}")
    root = (output_root / "evaluation_runs").resolve()
    output_dir = (root / run_id).resolve()
    if output_dir.parent != root:
        raise ValueError(
            f"evaluation run output path escaped output root: {output_dir}"
        )
    return output_dir


def validate_forward_return_label(label: str) -> None:
    """要求 A 股 adapter 只接收已知 forward-return label 命名。"""

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
    """在一个全样本 slim factor Parquet file 上运行 evaluation layer。

    label availability 按自然年汇总供诊断使用；evaluation 本身读取连续日期区间，
    因此 tradability 允许时路径指标可以跨年持仓。
    """

    _validate_run_processed_options(
        start_year=start_year,
        end_year=end_year,
        engine=engine,
        portfolio_mode=portfolio_mode,
        requested_labels=requested_labels,
        n_groups=n_groups,
        start_date=start_date,
        end_date=end_date,
        cost_bps=cost_bps,
    )
    label_summaries = []
    evaluation_runs: list[dict[str, object]] = []
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
        evaluation_runs.append(
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
                "output_dir": f"evaluation_runs/{run_id}",
            }
        )

    label_summary = pd.concat(label_summaries, ignore_index=True)
    write_csv_utf8_sig(report_root / "label_availability.csv", label_summary)
    manifest_path = report_root / "evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {"evaluation_runs": evaluation_runs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return label_summary, manifest_path, evaluation_runs


def _validate_run_processed_options(
    *,
    start_year: int,
    end_year: int,
    engine: str,
    portfolio_mode: str,
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
    try:
        normalize_engine_name(engine)
    except Exception as exc:
        raise ValueError(
            f"engine must be one of {list(supported_engine_names())}."
        ) from exc
    if portfolio_mode not in PORTFOLIO_MODES:
        raise ValueError(f"portfolio_mode must be one of {sorted(PORTFOLIO_MODES)}.")
    for label in requested_labels:
        if not RUN_ID_PATTERN.fullmatch(f"csmar_2000_2000_{label}"):
            raise ValueError(f"unsafe label for evaluation run_id: {label!r}")
        validate_forward_return_label(label)
    start = normalize_date_arg(start_date) if start_date else None
    end = normalize_date_arg(end_date) if end_date else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be less than or equal to end_date.")
