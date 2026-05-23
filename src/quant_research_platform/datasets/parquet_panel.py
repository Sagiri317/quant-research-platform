"""面向 DataFrame evaluation 请求的轻量 Parquet 输入 helper。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from quant_research_platform.evaluations.evaluation_spec import (
    DEFAULT_PNL_RETURN_COLUMN,
    CostConfig,
    EvaluationSpec,
    ExecutionConfig,
    NeutralizationConfig,
    TradabilityConfig,
    WeightConfig,
)


class ParquetSchemaInspection(BaseModel):
    """用于 CLI dry-run 检查的轻量 Parquet schema metadata。"""

    columns: tuple[str, ...]
    n_rows: int | None = None
    input_count: int = 1
    merge_mode: str = "single_file"
    paths: tuple[str, ...] = ()


def read_factor_panel_parquet(
    parquet_path: str | Path | Sequence[str | Path],
    *,
    timestamp_column: str,
    asset_column: str,
    feature_columns: tuple[str, ...],
    label_column: str,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """只读取构造 evaluation layer panel 所需的列。"""

    columns = _unique_columns(
        (
            timestamp_column,
            asset_column,
            *feature_columns,
            label_column,
            *extra_columns,
        )
    )
    paths = _normalize_parquet_paths(parquet_path)
    merge_plan = _infer_parquet_merge_plan(
        paths,
        required_columns=tuple(columns),
        key_columns=(timestamp_column, asset_column),
    )
    panel = _read_parquet_merge_plan(
        merge_plan,
        requested_columns=tuple(columns),
        key_columns=(timestamp_column, asset_column),
    )
    if panel.duplicated([timestamp_column, asset_column]).any():
        raise ValueError(
            "merged Parquet panel must have unique timestamp-asset keys."
        )
    panel[asset_column] = panel[asset_column].astype(str)
    return panel


def inspect_parquet_schema(parquet_path: str | Path) -> ParquetSchemaInspection:
    """不加载完整 panel，只检查 Parquet columns 和 row count。"""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional engine
        raise RuntimeError(
            "Parquet schema inspection requires pyarrow in the active environment."
        ) from exc

    parquet_file = pq.ParquetFile(parquet_path)
    return ParquetSchemaInspection(
        columns=tuple(parquet_file.schema_arrow.names),
        n_rows=parquet_file.metadata.num_rows if parquet_file.metadata else None,
        paths=(str(parquet_path),),
    )


def validate_parquet_evaluation_columns(
    parquet_path: str | Path | Sequence[str | Path],
    *,
    timestamp_column: str,
    asset_column: str,
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    pnl_return_column: str | None = None,
    extra_columns: tuple[str, ...] = (),
) -> ParquetSchemaInspection:
    """校验 evaluation set 所需的全部 Parquet columns。"""

    pnl_return_column = pnl_return_column or DEFAULT_PNL_RETURN_COLUMN
    required = _unique_columns(
        (
            timestamp_column,
            asset_column,
            *feature_columns,
            *label_columns,
            pnl_return_column,
            *extra_columns,
        )
    )
    paths = _normalize_parquet_paths(parquet_path)
    inspections = tuple(inspect_parquet_schema(path) for path in paths)
    merge_plan = _infer_parquet_merge_plan(
        paths,
        required_columns=tuple(required),
        key_columns=(timestamp_column, asset_column),
        inspections=inspections,
    )
    available_columns = tuple(
        dict.fromkeys(
            column
            for inspection in inspections
            for column in inspection.columns
        )
    )
    missing = [column for column in required if column not in available_columns]
    if missing:
        raise ValueError(f"Parquet file is missing required columns: {missing}.")
    n_rows = _inspection_row_count(merge_plan, inspections)
    return ParquetSchemaInspection(
        columns=available_columns,
        n_rows=n_rows,
        input_count=len(paths),
        merge_mode=merge_plan["mode"],
        paths=tuple(str(path) for path in paths),
    )


def make_parquet_evaluation_spec(
    parquet_path: str | Path | Sequence[str | Path],
    *,
    timestamp_column: str,
    asset_column: str,
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    metadata: dict[str, Any],
    output_dir: str | Path,
    n_groups: int = 5,
    weight_configs: tuple[WeightConfig, ...] | None = None,
    cost_configs: tuple[CostConfig, ...] | None = None,
    execution_config: ExecutionConfig | None = None,
    tradability_config: TradabilityConfig | None = None,
    neutralization_config: NeutralizationConfig | None = None,
    segment_columns: tuple[str, ...] = (),
    time_breakdowns: tuple[str, ...] = (),
    engine: str = "dataframe",
    pnl_return_column: str | None = None,
    domain_notes: dict[str, Any] | None = None,
    extra_columns: tuple[str, ...] = (),
) -> EvaluationSpec:
    """从一个或多个 Parquet panel 文件构造 EvaluationSpec。"""

    weight_configs = weight_configs or (WeightConfig(),)
    cost_configs = cost_configs or (CostConfig(),)
    execution_config = execution_config or ExecutionConfig()
    tradability_config = tradability_config or TradabilityConfig()
    neutralization_config = neutralization_config or NeutralizationConfig()
    resolved_pnl_return_column = pnl_return_column or DEFAULT_PNL_RETURN_COLUMN
    panel = read_factor_panel_parquet(
        parquet_path,
        timestamp_column=timestamp_column,
        asset_column=asset_column,
        feature_columns=feature_columns,
        label_column=label_columns[0],
        extra_columns=_request_extra_columns(
            weight_config=WeightConfig(),
            execution_config=execution_config,
            tradability_config=tradability_config,
            neutralization_config=neutralization_config,
            segment_columns=segment_columns,
            extra_columns=(
                *label_columns[1:],
                resolved_pnl_return_column,
                *_risk_columns(weight_configs),
                *extra_columns,
            ),
        ),
    )
    return EvaluationSpec(
        panel=panel,
        feature_columns=feature_columns,
        label_columns=label_columns,
        metadata=metadata,
        output_dir=output_dir,
        timestamp_column=timestamp_column,
        asset_column=asset_column,
        n_groups=n_groups,
        weight_configs=weight_configs,
        cost_configs=cost_configs,
        execution_config=execution_config,
        tradability_config=tradability_config,
        neutralization_config=neutralization_config,
        engine=engine,
        pnl_return_column=pnl_return_column,
        segment_columns=segment_columns,
        time_breakdowns=time_breakdowns,
        domain_notes={} if domain_notes is None else domain_notes,
    )


def _request_extra_columns(
    *,
    weight_config: WeightConfig,
    execution_config: ExecutionConfig,
    tradability_config: TradabilityConfig,
    neutralization_config: NeutralizationConfig,
    segment_columns: tuple[str, ...],
    extra_columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        _unique_columns(
            (
                *segment_columns,
                weight_config.risk_column,
                execution_config.execution_price_column,
                tradability_config.eligibility_column,
                tradability_config.can_enter_column,
                tradability_config.can_exit_column,
                *neutralization_config.group_columns,
                *neutralization_config.exposure_columns,
                *extra_columns,
            )
        )
    )


def _risk_columns(
    weight_configs: tuple[WeightConfig, ...],
) -> tuple[str, ...]:
    return tuple(
        config.risk_column
        for config in weight_configs
        if config.risk_column is not None
    )


def _unique_columns(columns: tuple[str | None, ...]) -> list[str]:
    return list(dict.fromkeys(column for column in columns if column is not None))


def _normalize_parquet_paths(
    parquet_path: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(parquet_path, str | Path):
        paths = (Path(parquet_path),)
    else:
        paths = tuple(Path(path) for path in parquet_path)
    if not paths:
        raise ValueError("at least one parquet path is required.")
    return paths


def _infer_parquet_merge_plan(
    paths: tuple[Path, ...],
    *,
    required_columns: tuple[str, ...],
    key_columns: tuple[str, str],
    inspections: tuple[ParquetSchemaInspection, ...] | None = None,
) -> dict[str, object]:
    inspections = (
        tuple(inspect_parquet_schema(path) for path in paths)
        if inspections is None
        else inspections
    )
    column_sets = [set(inspection.columns) for inspection in inspections]
    required_set = set(required_columns)
    if len(paths) == 1:
        missing = required_set - column_sets[0]
        if missing:
            raise ValueError(
                f"Parquet file is missing required columns: {sorted(missing)}."
            )
        return {"mode": "single_file", "paths": paths, "inspections": inspections}

    if all(required_set.issubset(columns) for columns in column_sets):
        return {"mode": "row_concat", "paths": paths, "inspections": inspections}

    key_set = set(key_columns)
    union_columns = set().union(*column_sets)
    if not all(key_set.issubset(columns) for columns in column_sets):
        raise ValueError(
            "multi-file column join requires every Parquet file to contain "
            f"key columns: {list(key_columns)}."
        )
    if not required_set.issubset(union_columns):
        missing = sorted(required_set - union_columns)
        raise ValueError(f"Parquet files are missing required columns: {missing}.")

    non_key_required = required_set - key_set
    owners: dict[str, list[Path]] = {
        column: [
            path
            for path, columns in zip(paths, column_sets, strict=True)
            if column in columns
        ]
        for column in non_key_required
    }
    duplicate_columns = {
        column: owner_paths
        for column, owner_paths in owners.items()
        if len(owner_paths) > 1
    }
    if duplicate_columns:
        duplicate_names = sorted(duplicate_columns)
        raise ValueError(
            "multi-file column join is ambiguous because non-key columns appear "
            f"in multiple files: {duplicate_names}."
        )
    return {"mode": "column_join", "paths": paths, "inspections": inspections}


def _read_parquet_merge_plan(
    merge_plan: dict[str, object],
    *,
    requested_columns: tuple[str, ...],
    key_columns: tuple[str, str],
) -> pd.DataFrame:
    paths = merge_plan["paths"]
    if not isinstance(paths, tuple):
        raise TypeError("merge plan paths must be a tuple.")
    mode = merge_plan["mode"]
    if mode == "single_file":
        return pd.read_parquet(paths[0], columns=list(requested_columns))
    if mode == "row_concat":
        frames = [
            pd.read_parquet(path, columns=list(requested_columns)) for path in paths
        ]
        return pd.concat(frames, ignore_index=True)
    if mode == "column_join":
        return _read_column_joined_parquets(
            paths,
            requested_columns=requested_columns,
            key_columns=key_columns,
        )
    raise ValueError(f"unsupported parquet merge mode {mode!r}.")


def _read_column_joined_parquets(
    paths: tuple[Path, ...],
    *,
    requested_columns: tuple[str, ...],
    key_columns: tuple[str, str],
) -> pd.DataFrame:
    key_set = set(key_columns)
    result: pd.DataFrame | None = None
    for path in paths:
        inspection = inspect_parquet_schema(path)
        available = set(inspection.columns)
        columns = [
            *key_columns,
            *(
                column
                for column in requested_columns
                if column in available and column not in key_set
            ),
        ]
        frame = pd.read_parquet(path, columns=columns)
        if frame.duplicated(list(key_columns)).any():
            raise ValueError(
                f"Parquet file {path} has duplicate timestamp-asset keys and "
                "cannot be column-joined."
            )
        result = (
            frame
            if result is None
            else result.merge(
                frame,
                on=list(key_columns),
                how="inner",
                validate="one_to_one",
            )
        )
    if result is None:
        raise ValueError("at least one parquet path is required.")
    return result[[column for column in requested_columns if column in result.columns]]


def _inspection_row_count(
    merge_plan: dict[str, object],
    inspections: tuple[ParquetSchemaInspection, ...],
) -> int | None:
    mode = merge_plan["mode"]
    counts = [inspection.n_rows for inspection in inspections]
    if any(count is None for count in counts):
        return None
    if mode in {"single_file", "column_join"}:
        return counts[0]
    if mode == "row_concat":
        return sum(int(count) for count in counts if count is not None)
    return None
