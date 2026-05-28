"""Schema 0.7 evaluation-set registry.

This module is the code-owned source of truth for schema 0.7 artifact tables,
generated files, display eligibility, and retired schema 0.6 public identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EVALUATION_SET_SCHEMA_VERSION = "0.7"
EVALUATION_SET_ARTIFACT_TYPE = "evaluation_set"

Materialization = Literal["required", "conditional", "generated"]


@dataclass(frozen=True)
class DisplaySpec:
    eligible: bool
    download_eligible: bool
    default_visible: bool = True
    section: str = "metrics"
    section_order: int = 0
    display_order: int = 0
    supports_combination_filter: bool = True


@dataclass(frozen=True)
class TableSpec:
    key: str
    path: str
    columns: tuple[str, ...]
    grain: tuple[str, ...]
    unique_key: tuple[str, ...]
    sort_key: tuple[str, ...] = ()
    materialization: Materialization = "required"
    status_column: str | None = None
    required_if: tuple[str, ...] = ()
    nullable_if: dict[str, tuple[str, ...]] = field(default_factory=dict)
    sentinel_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    display: DisplaySpec = field(default_factory=lambda: DisplaySpec(True, True))


@dataclass(frozen=True)
class ChartSpec:
    key: str
    kind: str
    source_table_key: str
    source_path: str
    x_column: str
    grouping_column: str
    input_series: str
    transform: str
    output_series: str
    supports_combination_filter: bool


REQUIRED_METADATA_FILES = (
    "manifest.json",
    "summary.json",
    "run_context.json",
)
OPTIONAL_METADATA_FILES = ("report.md",)

_COMMON_STATUS_COLUMNS = (
    "diagnostic_status",
    "status_reason",
    "warning",
)

_DIAGNOSTIC_STATUSES = (
    "ok",
    "available",
    "partial",
    "unavailable",
    "skipped",
    "disabled",
)

_STATUS_SENTINELS = {"diagnostic_status": _DIAGNOSTIC_STATUSES}

_PERIOD_BREAKDOWN_COLUMNS = (
    "combination_id",
    "period_type",
    "period",
    "period_start",
    "period_end",
    "metric_family",
    "metric_name",
    "metric_value",
    "unit",
    "n_observations",
    *_COMMON_STATUS_COLUMNS,
)

_MONTHLY_IC_BREAKDOWN_COLUMNS = (
    "combination_id",
    "period_type",
    "period",
    "period_start",
    "period_end",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "n_observations",
    *_COMMON_STATUS_COLUMNS,
)

_YEARLY_IC_BREAKDOWN_COLUMNS = (
    "combination_id",
    "period_type",
    "period",
    "period_start",
    "period_end",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "n_observations",
    *_COMMON_STATUS_COLUMNS,
)

_TABLE_SPECS = (
    TableSpec(
        key="combinations",
        path="dimensions/combinations.csv",
        columns=("combination_id", "feature", "label", "kernel", "portfolio_mode"),
        grain=("combination_id",),
        unique_key=("combination_id",),
        sort_key=("combination_id",),
        display=DisplaySpec(
            eligible=True,
            download_eligible=True,
            section="dimensions",
            section_order=0,
            display_order=0,
            supports_combination_filter=False,
        ),
    ),
    TableSpec(
        key="ic",
        path="metrics/ic.csv",
        columns=("combination_id", "timestamp", "ic", "rankic", "n_obs"),
        grain=("combination_id", "timestamp"),
        unique_key=("combination_id", "timestamp"),
        sort_key=("timestamp", "combination_id"),
        display=DisplaySpec(True, True, section="metrics", display_order=10),
    ),
    TableSpec(
        key="grouped_returns",
        path="metrics/grouped_returns.csv",
        columns=(
            "combination_id",
            "timestamp",
            "group",
            "mean_forward_return",
            "n_obs",
        ),
        grain=("combination_id", "timestamp", "group"),
        unique_key=("combination_id", "timestamp", "group"),
        sort_key=("timestamp", "combination_id", "group"),
        display=DisplaySpec(True, True, section="metrics", display_order=20),
    ),
    TableSpec(
        key="evaluation_metrics_summary",
        path="metrics/evaluation_metrics_summary.csv",
        columns=(
            "combination_id",
            "metric_family",
            "metric_name",
            "metric_value",
            "unit",
            "n_observations",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "metric_family", "metric_name"),
        unique_key=("combination_id", "metric_family", "metric_name"),
        sort_key=("combination_id", "metric_family", "metric_name"),
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="metrics", display_order=30),
    ),
    TableSpec(
        key="returns",
        path="portfolio/returns.csv",
        columns=(
            "combination_id",
            "timestamp",
            "portfolio_mode",
            "long_return",
            "short_return",
            "gross_portfolio_return",
            "turnover",
            "cost",
            "net_return",
            "long_count",
            "short_count",
            "return_source_column",
        ),
        grain=("combination_id", "timestamp"),
        unique_key=("combination_id", "timestamp"),
        sort_key=("timestamp", "combination_id"),
        display=DisplaySpec(True, True, section="portfolio", display_order=40),
    ),
    TableSpec(
        key="coverage_summary",
        path="input_quality/coverage_summary.csv",
        columns=(
            "input_family",
            "input_name",
            "available_rows",
            "missing_rows",
            "coverage_rate",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("input_family", "input_name"),
        unique_key=("input_family", "input_name"),
        sort_key=("input_family", "input_name"),
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(
            True,
            True,
            section="input_quality",
            display_order=50,
            supports_combination_filter=False,
        ),
    ),
    TableSpec(
        key="diagnostic_policies",
        path="dimensions/diagnostic_policies.csv",
        columns=(
            "policy_id",
            "policy_family",
            "enabled",
            "requested",
            "materialized",
            "policy_fingerprint",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("policy_id",),
        unique_key=("policy_id",),
        sort_key=("policy_family", "policy_id"),
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(
            True,
            True,
            section="dimensions",
            display_order=60,
            supports_combination_filter=False,
        ),
    ),
    TableSpec(
        key="diagnostic_summary",
        path="diagnostics/diagnostic_summary.csv",
        columns=(
            "combination_id",
            "policy_id",
            "policy_family",
            "decision_usable",
            "materialized_views",
            "missing_inputs",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "policy_id"),
        unique_key=("combination_id", "policy_id"),
        sort_key=("combination_id", "policy_family", "policy_id"),
        status_column="diagnostic_status",
        required_if=("status_reason",),
        nullable_if={"status_reason": ("ok", "available")},
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=70),
    ),
    TableSpec(
        key="diagnostic_metrics",
        path="diagnostics/diagnostic_metrics.csv",
        columns=(
            "combination_id",
            "policy_id",
            "policy_family",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            "required_inputs_met",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "policy_id", "metric_name"),
        unique_key=("combination_id", "policy_id", "metric_name"),
        sort_key=("combination_id", "policy_family", "policy_id", "metric_name"),
        status_column="diagnostic_status",
        required_if=("metric_name", "unit", "metric_value", "status_reason"),
        nullable_if={
            "metric_value": ("unavailable", "skipped", "disabled"),
            "status_reason": ("ok", "available"),
        },
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=80),
    ),
    TableSpec(
        key="portfolio_assumptions",
        path="dimensions/portfolio_assumptions.csv",
        columns=(
            "assumption_id",
            "assumption_family",
            "assumption_name",
            "assumption_value",
            "unit",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("assumption_id",),
        unique_key=("assumption_id",),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(
            True,
            True,
            section="dimensions",
            display_order=90,
            supports_combination_filter=False,
        ),
    ),
    TableSpec(
        key="weights",
        path="portfolio/weights.parquet",
        columns=("combination_id", "timestamp", "asset", "target_weight"),
        grain=("combination_id", "timestamp", "asset"),
        unique_key=("combination_id", "timestamp", "asset"),
        sort_key=("timestamp", "combination_id", "asset"),
        materialization="conditional",
        display=DisplaySpec(False, True, default_visible=False, section="portfolio"),
    ),
    TableSpec(
        key="tradability_diagnostics",
        path="diagnostics/tradability_diagnostics.csv",
        columns=(
            "combination_id",
            "timestamp",
            "asset",
            "can_enter",
            "can_exit",
            "reason",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "timestamp", "asset"),
        unique_key=("combination_id", "timestamp", "asset"),
        sort_key=("timestamp", "combination_id", "asset"),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=100),
    ),
    TableSpec(
        key="tradability_summary",
        path="diagnostics/tradability_summary.csv",
        columns=(
            "combination_id",
            "eligible_count",
            "blocked_count",
            "blocked_rate",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id",),
        unique_key=("combination_id",),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=110),
    ),
    TableSpec(
        key="scenario_metrics",
        path="diagnostics/scenario_metrics.csv",
        columns=(
            "combination_id",
            "scenario_id",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "scenario_id", "metric_name"),
        unique_key=("combination_id", "scenario_id", "metric_name"),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=120),
    ),
    TableSpec(
        key="exposure_capacity_summary",
        path="diagnostics/exposure_capacity_summary.csv",
        columns=(
            "combination_id",
            "exposure_family",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "exposure_family", "metric_name"),
        unique_key=("combination_id", "exposure_family", "metric_name"),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=130),
    ),
    TableSpec(
        key="portfolio_risk_summary",
        path="diagnostics/portfolio_risk_summary.csv",
        columns=(
            "combination_id",
            "risk_family",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "risk_family", "metric_name"),
        unique_key=("combination_id", "risk_family", "metric_name"),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="diagnostics", display_order=140),
    ),
    TableSpec(
        key="diagnostic_observations",
        path="diagnostics/diagnostic_observations.parquet",
        columns=(
            "combination_id",
            "timestamp",
            "asset",
            "policy_id",
            "observation_name",
            "observation_value",
            "unit",
            *_COMMON_STATUS_COLUMNS,
        ),
        grain=("combination_id", "timestamp", "asset", "policy_id", "observation_name"),
        unique_key=(
            "combination_id",
            "timestamp",
            "asset",
            "policy_id",
            "observation_name",
        ),
        materialization="conditional",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(
            True,
            True,
            default_visible=False,
            section="diagnostics",
            display_order=150,
        ),
    ),
    TableSpec(
        key="ic_breakdown_by_year",
        path="breakdowns/year/ic.csv",
        columns=_YEARLY_IC_BREAKDOWN_COLUMNS,
        grain=("combination_id", "period_type", "period"),
        unique_key=("combination_id", "period_type", "period"),
        sort_key=("period_start", "combination_id"),
        materialization="generated",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="breakdowns", display_order=180),
    ),
    TableSpec(
        key="ic_breakdown_by_month",
        path="breakdowns/month/ic.csv",
        columns=_MONTHLY_IC_BREAKDOWN_COLUMNS,
        grain=("combination_id", "period_type", "period"),
        unique_key=("combination_id", "period_type", "period"),
        sort_key=("period_start", "combination_id"),
        materialization="generated",
        status_column="diagnostic_status",
        sentinel_values=_STATUS_SENTINELS,
        display=DisplaySpec(True, True, section="breakdowns", display_order=190),
    ),
    *[
        TableSpec(
            key=f"evaluation_metrics_summary_by_{period}",
            path=f"breakdowns/{period}/evaluation_metrics_summary.csv",
            columns=_PERIOD_BREAKDOWN_COLUMNS,
            grain=(
                "combination_id",
                "period_type",
                "period",
                "metric_family",
                "metric_name",
            ),
            unique_key=(
                "combination_id",
                "period_type",
                "period",
                "metric_family",
                "metric_name",
            ),
            sort_key=(
                "period_type",
                "period_start",
                "combination_id",
                "metric_family",
                "metric_name",
            ),
            materialization="generated",
            status_column="diagnostic_status",
            sentinel_values=_STATUS_SENTINELS,
            display=DisplaySpec(True, True, section="breakdowns", display_order=200),
        )
        for period in ("year", "quarter", "month")
    ],
)

_TABLE_SPECS_BY_KEY = {spec.key: spec for spec in _TABLE_SPECS}

_CHART_SPECS = {
    "primary_cumulative_return": ChartSpec(
        key="primary_cumulative_return",
        kind="derived_cumulative_return",
        source_table_key="returns",
        source_path="portfolio/returns.csv",
        x_column="timestamp",
        grouping_column="combination_id",
        input_series="net_return",
        transform="compound_cumulative_return",
        output_series="cumulative_net_return",
        supports_combination_filter=True,
    )
}

RETIRED_SCHEMA_06_IDENTITIES = frozenset(
    {
        "metrics/ic_summary.csv",
        "ic_summary",
        "metrics/group_spread_summary.csv",
        "group_spread_summary",
        "metrics/performance_summary.csv",
        "performance_summary",
        "summaries/evaluation_summary.csv",
        "evaluation_summary",
        "portfolio/cumulative_returns.csv",
        "cumulative_returns",
        "input_quality/label_availability.csv",
        "label_availability",
        "input_quality/factor_coverage.csv",
        "factor_coverage",
        "input_quality/tradability_coverage.csv",
        "tradability_coverage",
        "config.json",
        "lineage.json",
        "/diagnostics/{metric_name}",
        "/metrics/{metric_name}",
        "/portfolio/{metric_name}",
        "/breakdowns/{group}/{metric_name}",
    }
)


def required_metadata_files() -> tuple[str, ...]:
    return REQUIRED_METADATA_FILES


def optional_metadata_files() -> tuple[str, ...]:
    return OPTIONAL_METADATA_FILES


def table_specs() -> tuple[TableSpec, ...]:
    return _TABLE_SPECS


def required_table_keys() -> tuple[str, ...]:
    return tuple(
        spec.key for spec in _TABLE_SPECS if spec.materialization == "required"
    )


def get_table_spec(key: str) -> TableSpec:
    try:
        return _TABLE_SPECS_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown schema 0.7 table key: {key}") from exc


def chart_specs() -> tuple[ChartSpec, ...]:
    return tuple(_CHART_SPECS.values())


def get_chart_spec(key: str) -> ChartSpec:
    try:
        return _CHART_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"unknown schema 0.7 chart key: {key}") from exc


def validate_table_contract(
    spec: TableSpec,
    *,
    rows: list[dict[str, object]],
    columns: list[str] | tuple[str, ...] | None = None,
) -> None:
    available_columns = (
        set(columns) if columns is not None else set(rows[0] if rows else ())
    )
    missing = sorted(set(spec.columns).difference(available_columns))
    if missing:
        raise ValueError(
            f"{spec.key} missing columns: {missing}; "
            f"{spec.key} is missing columns: {missing}."
        )

    seen: set[tuple[object, ...]] = set()
    for row in rows:
        _validate_row_conditions(spec, row)
        key = tuple(row.get(column) for column in spec.unique_key)
        if key in seen:
            raise ValueError(f"{spec.key} duplicate unique key: {key}.")
        seen.add(key)


def _validate_row_conditions(spec: TableSpec, row: dict[str, object]) -> None:
    status = _row_status(spec, row)
    for column, allowed_values in spec.sentinel_values.items():
        value = row.get(column)
        if _is_missing(value):
            continue
        if str(value) not in allowed_values:
            raise ValueError(
                f"{spec.key} column {column!r} has invalid sentinel value: "
                f"{value!r}."
            )
    for column in spec.required_if:
        if not _is_missing(row.get(column)):
            continue
        nullable_statuses = spec.nullable_if.get(column, ())
        if status is not None and status in nullable_statuses:
            continue
        raise ValueError(
            f"{spec.key} column {column!r} is required for status {status!r}."
        )


def _row_status(spec: TableSpec, row: dict[str, object]) -> str | None:
    if spec.status_column is None:
        return None
    value = row.get(spec.status_column)
    return None if _is_missing(value) else str(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return isinstance(value, float) and value != value


__all__ = [
    "ChartSpec",
    "DisplaySpec",
    "EVALUATION_SET_ARTIFACT_TYPE",
    "EVALUATION_SET_SCHEMA_VERSION",
    "OPTIONAL_METADATA_FILES",
    "REQUIRED_METADATA_FILES",
    "RETIRED_SCHEMA_06_IDENTITIES",
    "TableSpec",
    "chart_specs",
    "get_chart_spec",
    "get_table_spec",
    "optional_metadata_files",
    "required_metadata_files",
    "required_table_keys",
    "table_specs",
    "validate_table_contract",
]
