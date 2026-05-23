"""定义跨资产 alpha research evaluation 的公开合同。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.kernels.kernel_protocol import (
    PORTFOLIO_MODES,
    WEIGHT_DIRECTIONS,
    WEIGHT_METHODS,
    CostConfig,
    PanelSpec,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.kernels.kernel_protocol import (
    SUPPORTED_ENGINES as EVALUATION_ENGINES,
)
from quant_research_platform.kernels.kernel_registry import normalize_engine_name
from quant_research_platform.runtime.profiling import ProfileConfig
from quant_research_platform.runtime.run_metadata import (
    MINIMUM_RUN_METADATA_FIELDS,
    validate_run_metadata,
)

NEUTRALIZATION_METHODS = {"none", "demean_by_group", "cross_sectional_ols"}
NEUTRALIZATION_FIT_SCOPES = {"per_timestamp"}
TIME_BREAKDOWNS = {"year", "quarter", "month"}
DEFAULT_PNL_RETURN_COLUMN = "forward_return_1d"
TENSOR_BACKENDS = {"numpy"}
TENSOR_DEVICES = {"cpu"}
TENSOR_VALUE_DTYPES = {"float32", "float64"}
TENSOR_AGGREGATION_DTYPES = {"float64"}
__all__ = [
    "CostConfig",
    "DEFAULT_PNL_RETURN_COLUMN",
    "EVALUATION_ENGINES",
    "EvaluationSpec",
    "ExecutionConfig",
    "MINIMUM_RUN_METADATA_FIELDS",
    "NeutralizationConfig",
    "PORTFOLIO_MODES",
    "PanelSpec",
    "TensorBatchConfig",
    "TimeSemanticsConfig",
    "TradabilityConfig",
    "WEIGHT_DIRECTIONS",
    "WEIGHT_METHODS",
    "WeightConfig",
    "validate_run_metadata",
]


@dataclass(frozen=True)
class EvaluationSpec:
    """定义 batch-native evaluation set 输入合同。"""

    panel: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    metadata: dict[str, Any]
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"
    n_groups: int = 5
    weight_configs: tuple[WeightConfig, ...] = field(
        default_factory=lambda: (WeightConfig(),)
    )
    execution_config: ExecutionConfig = field(
        default_factory=lambda: ExecutionConfig()
    )
    tradability_config: TradabilityConfig = field(
        default_factory=lambda: TradabilityConfig()
    )
    cost_configs: tuple[CostConfig, ...] = field(
        default_factory=lambda: (CostConfig(),)
    )
    neutralization_config: NeutralizationConfig = field(
        default_factory=lambda: NeutralizationConfig()
    )
    time_semantics_config: TimeSemanticsConfig = field(
        default_factory=lambda: TimeSemanticsConfig()
    )
    tensor_config: TensorBatchConfig = field(
        default_factory=lambda: TensorBatchConfig()
    )
    segment_columns: tuple[str, ...] = ()
    time_breakdowns: tuple[str, ...] = ()
    engine: str = "dataframe"
    pnl_return_column: str | None = None
    output_dir: str | Path = Path("reports/runs/demo")
    profile_config: ProfileConfig = field(default_factory=ProfileConfig)
    domain_notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验 evaluation set、metadata 和共享配置。"""

        if not isinstance(self.panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame.")
        _validate_column_names(self.feature_columns, "feature_columns")
        _validate_column_names(self.label_columns, "label_columns")
        if set(self.feature_columns).intersection(self.label_columns):
            raise ValueError("feature_columns and label_columns must not overlap.")
        _validate_optional_column_name(self.timestamp_column, "timestamp_column")
        _validate_optional_column_name(self.asset_column, "asset_column")
        reserved = {self.timestamp_column, self.asset_column}
        if reserved.intersection((*self.feature_columns, *self.label_columns)):
            raise ValueError(
                "feature_columns and label_columns must not reuse timestamp "
                "or asset columns."
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        validate_run_metadata(self.metadata)
        if not isinstance(self.domain_notes, dict):
            raise TypeError("domain_notes must be a dictionary.")
        if not isinstance(self.n_groups, int) or isinstance(self.n_groups, bool):
            raise TypeError("n_groups must be an integer.")
        if self.n_groups < 2:
            raise ValueError("n_groups must be at least 2.")
        if not self.weight_configs:
            raise ValueError("weight_configs must contain at least one config.")
        if any(not isinstance(config, WeightConfig) for config in self.weight_configs):
            raise TypeError("weight_configs must contain WeightConfig objects.")
        if not self.cost_configs:
            raise ValueError("cost_configs must contain at least one config.")
        if any(not isinstance(config, CostConfig) for config in self.cost_configs):
            raise TypeError("cost_configs must contain CostConfig objects.")
        if not isinstance(self.execution_config, ExecutionConfig):
            raise TypeError("execution_config must be an ExecutionConfig.")
        if not isinstance(self.tradability_config, TradabilityConfig):
            raise TypeError("tradability_config must be a TradabilityConfig.")
        if not isinstance(self.neutralization_config, NeutralizationConfig):
            raise TypeError("neutralization_config must be a NeutralizationConfig.")
        if not isinstance(self.time_semantics_config, TimeSemanticsConfig):
            raise TypeError("time_semantics_config must be a TimeSemanticsConfig.")
        self.time_semantics_config.validate_panel(
            self.panel,
            label_columns=self.label_columns,
            feature_columns=self.feature_columns,
        )
        if not isinstance(self.tensor_config, TensorBatchConfig):
            raise TypeError("tensor_config must be a TensorBatchConfig.")
        _validate_column_names(self.segment_columns, "segment_columns")
        if any(
            not isinstance(value, str) or value not in TIME_BREAKDOWNS
            for value in self.time_breakdowns
        ):
            raise ValueError(
                f"time_breakdowns must contain only {sorted(TIME_BREAKDOWNS)}."
            )
        if len(set(self.time_breakdowns)) != len(self.time_breakdowns):
            raise ValueError("time_breakdowns must not contain duplicates.")
        try:
            normalized_engine = normalize_engine_name(self.engine)
        except Exception as exc:
            raise ValueError(
                f"engine must be one of {sorted(EVALUATION_ENGINES)}."
            ) from exc
        object.__setattr__(self, "engine", normalized_engine)
        if self.pnl_return_column is not None and (
            not isinstance(self.pnl_return_column, str) or not self.pnl_return_column
        ):
            raise ValueError("pnl_return_column must be a non-empty string.")
        if not isinstance(self.profile_config, ProfileConfig):
            raise TypeError("profile_config must be a ProfileConfig.")

    def panel_spec_for_label(self, label_column: str) -> PanelSpec:
        """构造单个 label 的 PanelSpec，供 metric kernels 使用。"""

        if label_column not in self.label_columns:
            raise ValueError(f"unknown label_column {label_column!r}.")
        return PanelSpec(
            feature_columns=self.feature_columns,
            label_column=label_column,
            timestamp_column=self.timestamp_column,
            asset_column=self.asset_column,
        )

    def resolve_pnl_return_column(self, panel: pd.DataFrame) -> str:
        """解析 portfolio PnL 使用的 realized-return column。"""

        if self.pnl_return_column is not None:
            return self.pnl_return_column
        if DEFAULT_PNL_RETURN_COLUMN in panel.columns:
            return DEFAULT_PNL_RETURN_COLUMN
        raise ValueError(
            "PnL curve requires pnl_return_column, but none was provided and "
            f"default column {DEFAULT_PNL_RETURN_COLUMN!r} was not found."
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """市场无关的 execution timing 和 price selection metadata。"""

    execution_lag: int = 0
    execution_price_column: str | None = None

    def __post_init__(self) -> None:
        """校验 execution lag 和可选 execution price column。"""

        if not isinstance(self.execution_lag, int) or isinstance(
            self.execution_lag,
            bool,
        ):
            raise TypeError("execution_lag must be an integer.")
        if self.execution_lag < 0:
            raise ValueError("execution_lag must be nonnegative.")
        _validate_optional_column_name(
            self.execution_price_column,
            "execution_price_column",
        )


@dataclass(frozen=True)
class NeutralizationConfig:
    """市场无关的 cross-sectional feature neutralization metadata。"""

    method: str = "none"
    group_columns: tuple[str, ...] = ()
    exposure_columns: tuple[str, ...] = ()
    fit_scope: str = "per_timestamp"

    def __post_init__(self) -> None:
        """校验 neutralization method 和 group/exposure columns。"""

        if self.method not in NEUTRALIZATION_METHODS:
            raise ValueError(
                f"method must be one of {sorted(NEUTRALIZATION_METHODS)}."
            )
        if self.fit_scope not in NEUTRALIZATION_FIT_SCOPES:
            raise ValueError(
                f"fit_scope must be one of {sorted(NEUTRALIZATION_FIT_SCOPES)}."
            )
        _validate_column_names(self.group_columns, "group_columns")
        _validate_column_names(self.exposure_columns, "exposure_columns")
        if self.method == "demean_by_group" and not self.group_columns:
            raise ValueError("group_columns are required for demean_by_group.")
        if self.method == "cross_sectional_ols" and not self.exposure_columns:
            raise ValueError("exposure_columns are required for cross_sectional_ols.")
        if self.method == "none" and (self.group_columns or self.exposure_columns):
            raise ValueError("neutralization columns require a non-none method.")


@dataclass(frozen=True)
class TimeSemanticsConfig:
    """声明 factor evaluation 的 decision、label 和 as-of 时间语义。"""

    decision_time_column: str | None = None
    label_time_columns: dict[str, str] = field(default_factory=dict)
    as_of_time_columns: dict[str, str] = field(default_factory=dict)
    frequency: str = "1d"
    timezone: str = "Asia/Shanghai"

    def __post_init__(self) -> None:
        """校验时间语义配置字段。"""

        _validate_optional_column_name(
            self.decision_time_column,
            "decision_time_column",
        )
        if not isinstance(self.label_time_columns, dict):
            raise TypeError("label_time_columns must be a dictionary.")
        if not isinstance(self.as_of_time_columns, dict):
            raise TypeError("as_of_time_columns must be a dictionary.")
        for mapping_name, mapping in (
            ("label_time_columns", self.label_time_columns),
            ("as_of_time_columns", self.as_of_time_columns),
        ):
            for key, value in mapping.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{mapping_name} keys must be non-empty strings.")
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{mapping_name} values must be non-empty strings."
                    )
        if not isinstance(self.frequency, str) or not self.frequency:
            raise ValueError("frequency must be a non-empty string.")
        if not isinstance(self.timezone, str) or not self.timezone:
            raise ValueError("timezone must be a non-empty string.")

    def validate_panel(
        self,
        panel: pd.DataFrame,
        *,
        label_columns: tuple[str, ...],
        feature_columns: tuple[str, ...],
    ) -> None:
        """校验 panel 满足已声明的时间可知性合同。"""

        if self.decision_time_column is None:
            if self.label_time_columns or self.as_of_time_columns:
                raise ValueError(
                    "decision_time_column is required when label_time_columns "
                    "or as_of_time_columns are configured."
                )
            return
        if self.decision_time_column not in panel.columns:
            raise ValueError(
                f"decision_time_column {self.decision_time_column!r} missing."
            )
        decision_time = _datetime_column(panel, self.decision_time_column)
        for label in label_columns:
            column = self.label_time_columns.get(label)
            if column is None:
                continue
            if column not in panel.columns:
                raise ValueError(f"label_time column {column!r} missing.")
            label_time = _datetime_column(panel, column)
            invalid = label_time <= decision_time
            if invalid.any():
                raise ValueError(
                    f"label_time column {column!r} must be after decision_time."
                )
        for feature in feature_columns:
            column = self.as_of_time_columns.get(feature)
            if column is None:
                continue
            if column not in panel.columns:
                raise ValueError(f"as_of_time column {column!r} missing.")
            as_of_time = _datetime_column(panel, column)
            invalid = as_of_time > decision_time
            if invalid.any():
                raise ValueError(
                    f"as_of_time column {column!r} must not be after decision_time."
                )


@dataclass(frozen=True)
class TensorBatchConfig:
    """内部 tensor kernel execution 和 memory-budget metadata。"""

    backend: str = "numpy"
    device: str = "cpu"
    value_dtype: str = "float64"
    aggregation_dtype: str = "float64"
    signal_chunk_size: int | None = None
    label_chunk_size: int | None = None
    max_memory_mb: float | None = None

    def __post_init__(self) -> None:
        """校验 tensor backend、dtypes、chunk sizes 和 memory budget。"""

        if self.backend not in TENSOR_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(TENSOR_BACKENDS)}.")
        if self.device not in TENSOR_DEVICES:
            raise ValueError(f"device must be one of {sorted(TENSOR_DEVICES)}.")
        if self.value_dtype not in TENSOR_VALUE_DTYPES:
            raise ValueError(
                f"value_dtype must be one of {sorted(TENSOR_VALUE_DTYPES)}."
            )
        if self.aggregation_dtype not in TENSOR_AGGREGATION_DTYPES:
            raise ValueError(
                "aggregation_dtype must be one of "
                f"{sorted(TENSOR_AGGREGATION_DTYPES)}."
            )
        _validate_optional_positive_int(self.signal_chunk_size, "signal_chunk_size")
        _validate_optional_positive_int(self.label_chunk_size, "label_chunk_size")
        if self.max_memory_mb is not None:
            if isinstance(self.max_memory_mb, bool) or not isinstance(
                self.max_memory_mb,
                (int, float),
            ):
                raise TypeError("max_memory_mb must be numeric.")
            if float(self.max_memory_mb) < 0:
                raise ValueError("max_memory_mb must be nonnegative.")


def _validate_optional_positive_int(value: int | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")


def _datetime_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column} must contain valid datetime values.")
    return values


def _validate_column_names(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
