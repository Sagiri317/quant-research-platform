from __future__ import annotations

import pytest

from quant_research_platform.artifacts.schema_registry import (
    EVALUATION_SET_SCHEMA_VERSION,
    RETIRED_SCHEMA_06_IDENTITIES,
    TableSpec,
    get_chart_spec,
    get_table_spec,
    required_metadata_files,
    required_table_keys,
    table_specs,
    validate_table_contract,
)


def test_schema_07_registry_names_required_metadata_and_tables() -> None:
    assert EVALUATION_SET_SCHEMA_VERSION == "0.7"
    assert required_metadata_files() == (
        "manifest.json",
        "summary.json",
        "run_context.json",
    )
    assert required_table_keys() == (
        "combinations",
        "ic",
        "grouped_returns",
        "evaluation_metrics_summary",
        "returns",
        "coverage_summary",
        "diagnostic_policies",
        "diagnostic_summary",
        "diagnostic_metrics",
    )


def test_schema_07_registry_owns_table_paths_keys_and_display_eligibility() -> None:
    specs = {spec.key: spec for spec in table_specs()}

    assert specs["diagnostic_summary"].path == "diagnostics/diagnostic_summary.csv"
    assert specs["diagnostic_summary"].grain == (
        "combination_id",
        "policy_id",
    )
    assert specs["diagnostic_summary"].unique_key == (
        "combination_id",
        "policy_id",
    )
    assert specs["diagnostic_summary"].display.eligible is True
    assert specs["diagnostic_summary"].display.download_eligible is True
    assert specs["diagnostic_summary"].status_column == "diagnostic_status"

    assert specs["diagnostic_observations"].path == (
        "diagnostics/diagnostic_observations.parquet"
    )
    assert specs["diagnostic_observations"].materialization == "conditional"
    assert specs["diagnostic_observations"].display.default_visible is False

    assert specs["weights"].path == "portfolio/weights.parquet"
    assert specs["weights"].materialization == "conditional"

    assert specs["evaluation_metrics_summary_by_year"].path == (
        "breakdowns/year/evaluation_metrics_summary.csv"
    )
    assert specs["evaluation_metrics_summary_by_year"].unique_key == (
        "combination_id",
        "period_type",
        "period",
        "metric_family",
        "metric_name",
    )
    assert specs["ic_breakdown_by_year"].path == "breakdowns/year/ic.csv"
    assert specs["ic_breakdown_by_year"].unique_key == (
        "combination_id",
        "period_type",
        "period",
    )
    assert specs["ic_breakdown_by_month"].path == "breakdowns/month/ic.csv"
    assert specs["ic_breakdown_by_month"].unique_key == (
        "combination_id",
        "period_type",
        "period",
    )


def test_schema_07_chart_registry_declares_derived_cumulative_return() -> None:
    chart = get_chart_spec("primary_cumulative_return")

    assert chart.kind == "derived_cumulative_return"
    assert chart.source_table_key == "returns"
    assert chart.source_path == "portfolio/returns.csv"
    assert chart.input_series == "net_return"
    assert chart.output_series == "cumulative_net_return"
    assert chart.transform == "compound_cumulative_return"
    assert chart.supports_combination_filter is True


def test_schema_07_registry_rejects_missing_columns_and_duplicate_keys() -> None:
    spec = get_table_spec("diagnostic_summary")
    with pytest.raises(ValueError, match="missing columns"):
        validate_table_contract(
            spec,
            rows=[
                {
                    "combination_id": "combo",
                    "policy_id": "__diagnostics_disabled__",
                }
            ],
        )

    complete = {column: "" for column in spec.columns}
    complete["combination_id"] = "combo"
    complete["policy_id"] = "__diagnostics_disabled__"
    complete["diagnostic_status"] = "disabled"
    complete["status_reason"] = "disabled by config"
    with pytest.raises(ValueError, match="duplicate unique key"):
        validate_table_contract(spec, rows=[complete, dict(complete)])


def test_schema_07_registry_enforces_conditional_and_sentinel_rules() -> None:
    spec = TableSpec(
        key="demo",
        path="demo.csv",
        columns=("id", "value", "diagnostic_status"),
        grain=("id",),
        unique_key=("id",),
        status_column="diagnostic_status",
        required_if=("value",),
        nullable_if={"value": ("unavailable",)},
        sentinel_values={"diagnostic_status": ("ok", "unavailable")},
    )

    validate_table_contract(
        spec,
        rows=[{"id": "a", "value": "", "diagnostic_status": "unavailable"}],
    )
    with pytest.raises(ValueError, match="required for status"):
        validate_table_contract(
            spec,
            rows=[{"id": "a", "value": "", "diagnostic_status": "ok"}],
        )
    with pytest.raises(ValueError, match="invalid sentinel"):
        validate_table_contract(
            spec,
            rows=[{"id": "a", "value": "1", "diagnostic_status": "stale"}],
        )


def test_schema_07_registry_lists_retired_schema_06_identities() -> None:
    retired = RETIRED_SCHEMA_06_IDENTITIES

    assert "metrics/performance_summary.csv" in retired
    assert "portfolio/cumulative_returns.csv" in retired
    assert "summaries/evaluation_summary.csv" in retired
    assert "config.json" in retired
    assert "lineage.json" in retired
    assert "/diagnostics/{metric_name}" in retired
