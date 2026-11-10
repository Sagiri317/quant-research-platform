"""Multi-table semantic artifact for scientific research statistics."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.evaluations import CORRELATION_METRICS
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import read_qrp_json_file, write_manifest
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    parquet_table_writer,
)

_ARROW_SCHEMAS = {
    "factor_stratified_diagnostics.parquet": pa.schema(
        [
            *[
                pa.field(name, pa.string(), nullable=name == "stratum")
                for name in (
                    "signal",
                    "label",
                    "sample_scope",
                    "stratification",
                    "stratum",
                    "statistic",
                )
            ],
            *[
                pa.field(name, pa.int64(), nullable=False)
                for name in (
                    "input_date_count",
                    "observed_date_count",
                    "observation_count",
                    "observation_pairs",
                )
            ],
            *[
                pa.field(name, pa.float64())
                for name in (
                    "mean",
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                )
            ],
            pa.field("status", pa.string(), nullable=False),
            pa.field("reason", pa.string()),
            pa.field("hac_lag", pa.int64(), nullable=False),
            pa.field("fdr_method", pa.string(), nullable=False),
            pa.field("fdr_alpha", pa.float64(), nullable=False),
            pa.field("fdr_adjusted_p_value", pa.float64()),
            pa.field("defined_hypothesis_count", pa.int64(), nullable=False),
        ]
    ),
    "factor_correlation.parquet": pa.schema(
        [
            *[
                pa.field(name, pa.string(), nullable=False)
                for name in ("factor_id", "comparison_factor_id", "statistic")
            ],
            *[
                pa.field(name, pa.int64(), nullable=False)
                for name in (
                    "input_date_count",
                    "observation_count",
                    "observation_pairs",
                )
            ],
            pa.field("status", pa.string(), nullable=False),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("mean_absolute", pa.float64()),
        ]
    ),
    "factor_diagnostics.parquet": pa.schema(
        [
            pa.field("family_id", pa.string()),
            pa.field("factor_id", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("fdr_method", pa.string()),
            pa.field("fdr_alpha", pa.float64()),
            pa.field("fdr_adjusted_p_value", pa.float64()),
            pa.field("planned_hypothesis_count", pa.int64()),
            pa.field("defined_hypothesis_count", pa.int64()),
            pa.field("missing_hypothesis_count", pa.int64()),
        ]
    ),
    "prediction_diagnostics.parquet": pa.schema(
        [
            pa.field("prediction_id", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("bootstrap_status", pa.string()),
            pa.field("bootstrap_reason", pa.string()),
            pa.field("bootstrap_lower", pa.float64()),
            pa.field("bootstrap_upper", pa.float64()),
            pa.field("bootstrap_block_size", pa.int64()),
            pa.field("bootstrap_repetitions", pa.int64()),
            pa.field("bootstrap_seed", pa.int64()),
        ]
    ),
    "model_trials.parquet": pa.schema(
        [
            pa.field("selection_group", pa.string()),
            pa.field("decision_id", pa.string()),
            pa.field("family_id", pa.string()),
            pa.field("config_id", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("selection_status", pa.string()),
        ]
    ),
    "robustness.parquet": pa.schema(
        [
            pa.field("test_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
        ]
    ),
    "fama_macbeth.parquet": pa.schema(
        [
            pa.field("regression_id", pa.string()),
            pa.field("coefficient_id", pa.string()),
            pa.field("period_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("white_defined_period_count", pa.int64()),
            pa.field("white_undefined_period_count", pa.int64()),
        ]
    ),
    "fama_macbeth_periods.parquet": pa.schema(
        [
            pa.field("regression_id", pa.string()),
            pa.field("coefficient_id", pa.string()),
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("value", pa.float64()),
            pa.field("white_p_value", pa.float64()),
            pa.field("white_reason", pa.string()),
        ]
    ),
    "model_prediction_metrics.parquet": pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("family_id", pa.string()),
            pa.field("prediction_count", pa.int64()),
            pa.field("target_count", pa.int64()),
            pa.field("mse", pa.float64()),
            pa.field("mse_reason", pa.string()),
            pa.field("baseline_mse", pa.float64()),
            pa.field("oos_r2", pa.float64()),
            pa.field("oos_r2_reason", pa.string()),
            pa.field("head_selected_count", pa.int64()),
            pa.field("head_evaluable_count", pa.int64()),
            pa.field("head_coverage", pa.float64()),
            pa.field("pool_evaluable_count", pa.int64()),
            pa.field("head_mean_raw_return", pa.float64()),
            pa.field("pool_mean_raw_return", pa.float64()),
            pa.field("head_excess_raw_return", pa.float64()),
        ]
    ),
    "model_losses.parquet": pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns")),
            pa.field("sample_role", pa.string()),
            pa.field("period_id", pa.string()),
            pa.field("family_id", pa.string()),
            pa.field("config_id", pa.string()),
            pa.field("loss", pa.float64()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("common_date", pa.int64()),
            pa.field("common_observation_count", pa.int64()),
            pa.field("asset_coverage_id", pa.string()),
        ]
    ),
    "model_comparison.parquet": pa.schema(
        [
            pa.field("sample_role", pa.string()),
            pa.field("benchmark_id", pa.string()),
            pa.field("prediction_id", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("mean", pa.float64()),
            pa.field("standard_error", pa.float64()),
            pa.field("t_statistic", pa.float64()),
            pa.field("p_value", pa.float64()),
            pa.field("ci_lower", pa.float64()),
            pa.field("ci_upper", pa.float64()),
            pa.field("bootstrap_status", pa.string()),
            pa.field("bootstrap_reason", pa.string()),
            pa.field("bootstrap_lower", pa.float64()),
            pa.field("bootstrap_upper", pa.float64()),
            pa.field("bootstrap_block_size", pa.int64()),
            pa.field("bootstrap_repetitions", pa.int64()),
            pa.field("bootstrap_seed", pa.int64()),
        ]
    ),
    "model_group_comparison.parquet": pa.schema(
        [
            pa.field(name, pa.string())
            for name in (
                "family_id",
                "sample_role",
                "status",
                "reason",
                "fdr_method",
                "bootstrap_status",
                "bootstrap_reason",
            )
        ]
        + [
            pa.field(name, pa.int64())
            for name in (
                "observation_count",
                "calendar_count",
                "fdr_family_size",
                "hac_lag",
                "bootstrap_block_size",
                "bootstrap_repetitions",
                "bootstrap_seed",
            )
        ]
        + [
            pa.field(name, pa.float64())
            for name in (
                "global_mean",
                "quarterly_mean",
                "mean_difference",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
                "fdr_adjusted_p_value",
                "fdr_alpha",
                "confidence",
                "bootstrap_lower",
                "bootstrap_upper",
            )
        ]
    ),
    "model_reality_checks.parquet": pa.schema(
        [
            pa.field("test_id", pa.string()),
            pa.field("sample_role", pa.string()),
            pa.field("family_id", pa.string()),
            pa.field("statistic_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("block_size", pa.int64()),
            pa.field("repetitions", pa.int64()),
            pa.field("alpha", pa.float64()),
            pa.field("seed", pa.int64()),
        ]
    ),
    "portfolio_daily.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("nav_before", pa.float64()),
            pa.field("nav_post", pa.float64()),
            pa.field("cash_post", pa.float64()),
            pa.field("cash_weight_post", pa.float64()),
            pa.field("nav_end", pa.float64()),
            pa.field("gross_return", pa.float64()),
            pa.field("net_return", pa.float64()),
            pa.field("cost_drag", pa.float64()),
            pa.field("target_turnover", pa.float64()),
            pa.field("requested_turnover", pa.float64()),
            pa.field("realized_turnover", pa.float64()),
            pa.field("return_status", pa.string()),
            pa.field("return_reason", pa.string()),
            pa.field("sample_role", pa.string()),
        ]
    ),
    "portfolio_group_comparison.parquet": pa.schema(
        [
            *(
                (name, pa.string())
                for name in (
                    "portfolio_id",
                    "comparison_role",
                    "sample_role",
                    "status",
                    "reason",
                    "bootstrap_status",
                    "bootstrap_reason",
                    "fdr_method",
                    "global_terminal_wealth_reason",
                    "quarterly_terminal_wealth_reason",
                )
            ),
            *(
                (name, pa.int64())
                for name in (
                    "observation_count",
                    "hac_lag",
                    "bootstrap_block_size",
                    "bootstrap_repetitions",
                    "bootstrap_seed",
                    "fdr_family_size",
                )
            ),
            *(
                (name, pa.float64())
                for name in (
                    "global_mean",
                    "quarterly_mean",
                    "mean_difference",
                    "global_terminal_wealth",
                    "quarterly_terminal_wealth",
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                    "bootstrap_lower",
                    "bootstrap_upper",
                    "confidence",
                    "fdr_alpha",
                    "fdr_adjusted_p_value",
                )
            ),
        ]
    ),
    "portfolio_performance.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("sample_role", pa.string()),
        ]
    ),
    "portfolio_capacity.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("decision_timestamp", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("participation", pa.float64()),
            pa.field("capacity", pa.float64()),
            pa.field("bottleneck_asset_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "portfolio_capacity_summary.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("participation", pa.float64()),
            pa.field("period", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
        ]
    ),
    "portfolio_attribution.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("component_id", pa.string()),
            pa.field("component_type", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
        ]
    ),
    "portfolio_robustness.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("test_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("status", pa.string()),
            pa.field("reason", pa.string()),
            pa.field("observation_count", pa.int64()),
            pa.field("candidate_count", pa.int64()),
            pa.field("sample_role", pa.string()),
        ]
    ),
    "portfolio_exposures.parquet": pa.schema(
        [
            pa.field("candidate_id", pa.string()),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("factor_id", pa.string()),
            pa.field("value", pa.float64()),
        ]
    ),
    "holding_factor_returns.parquet": pa.schema(
        [
            pa.field("attribution_group", pa.string(), nullable=False),
            pa.field("decision_at", pa.timestamp("ns")),
            pa.field("entry_at", pa.timestamp("ns")),
            pa.field("end_at", pa.timestamp("ns")),
            pa.field("factor_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("observation_count", pa.int64()),
        ]
    ),
}
_COMPARISON_MEMBERS = (
    "portfolio_benchmark_intervals.parquet",
    "portfolio_benchmark_metrics.parquet",
    "portfolio_market_exposure.parquet",
    "portfolio_regime_comparison.parquet",
)
_COMPARISON_IDENTITY = (
    "candidate_id",
    "fold_id",
    "reference_id",
    "reference_kind",
    "sample_role",
)
for _member in _COMPARISON_MEMBERS:
    _fields = [
        pa.field(name, pa.string(), nullable=False) for name in _COMPARISON_IDENTITY
    ]
    if _member == "portfolio_benchmark_intervals.parquet":
        _fields += [
            pa.field(name, pa.timestamp("ns"), nullable=False)
            for name in ("start_at", "end_at")
        ]
        _fields += [
            pa.field(name, pa.float64())
            for name in (
                "portfolio_return",
                "benchmark_return",
                "return_difference",
                "portfolio_wealth",
                "benchmark_wealth",
                "relative_wealth",
            )
        ]
        _fields += [
            pa.field(name, pa.string(), nullable=False)
            for name in (
                "portfolio_status",
                "portfolio_reason",
                "benchmark_status",
                "benchmark_reason",
                "status",
                "reason",
            )
        ]
    else:
        if _member == "portfolio_regime_comparison.parquet":
            _fields.append(pa.field("regime", pa.string(), nullable=False))
        _fields.append(
            pa.field(
                "coefficient_id"
                if _member == "portfolio_market_exposure.parquet"
                else "metric_id",
                pa.string(),
                nullable=False,
            )
        )
        _fields.append(pa.field("value", pa.float64()))
        if _member == "portfolio_market_exposure.parquet":
            _fields += [
                pa.field(name, pa.float64())
                for name in (
                    "standard_error",
                    "t_statistic",
                    "p_value",
                    "ci_lower",
                    "ci_upper",
                )
            ]
        _fields += [
            pa.field(name, pa.int64(), nullable=False)
            for name in ("observation_count", "planned_observation_count")
        ]
        _fields += [
            pa.field(name, pa.string(), nullable=False) for name in ("status", "reason")
        ]
        if _member == "portfolio_market_exposure.parquet":
            _fields += [
                pa.field(name, pa.timestamp("ns"), nullable=False)
                for name in ("start_at", "end_at")
            ]
            _fields += [
                pa.field("hac_lag", pa.int64(), nullable=False),
                pa.field("confidence", pa.float64(), nullable=False),
                pa.field("inference_status", pa.string(), nullable=False),
                pa.field("inference_reason", pa.string(), nullable=False),
            ]
    if _member == "portfolio_market_exposure.parquet":
        _fields += [
            pa.field("fdr_method", pa.string(), nullable=False),
            pa.field("planned_hypothesis_count", pa.int64(), nullable=False),
            pa.field("defined_hypothesis_count", pa.int64(), nullable=False),
            pa.field("fdr_adjusted_p_value", pa.float64()),
        ]
    _ARROW_SCHEMAS[_member] = pa.schema(_fields)

STATISTICS_TABLE_GROUPS = {
    "factor": (
        "factor_stratified_diagnostics.parquet",
        "factor_diagnostics.parquet",
        "factor_annual_diagnostics.parquet",
        "factor_correlation.parquet",
        "fama_macbeth.parquet",
        "fama_macbeth_periods.parquet",
    ),
    "research": (
        "prediction_diagnostics.parquet",
        "model_trials.parquet",
        "robustness.parquet",
        "model_losses.parquet",
        "model_prediction_metrics.parquet",
        "model_comparison.parquet",
        "model_group_comparison.parquet",
        "model_reality_checks.parquet",
    ),
    "portfolio": (
        *_COMPARISON_MEMBERS,
        "portfolio_daily.parquet",
        "portfolio_group_comparison.parquet",
        "portfolio_performance.parquet",
        "portfolio_capacity.parquet",
        "portfolio_capacity_summary.parquet",
        "portfolio_attribution.parquet",
        "portfolio_robustness.parquet",
        "portfolio_exposures.parquet",
        "holding_factor_returns.parquet",
    ),
}
_EMPTY_TABLE_MEMBERS = frozenset(
    {
        *_COMPARISON_MEMBERS,
        "factor_annual_diagnostics.parquet",
        "factor_correlation.parquet",
        "fama_macbeth_periods.parquet",
        "holding_factor_returns.parquet",
        "model_comparison.parquet",
        "model_losses.parquet",
        "model_group_comparison.parquet",
        "portfolio_capacity.parquet",
        "portfolio_performance.parquet",
        "portfolio_group_comparison.parquet",
        "portfolio_exposures.parquet",
    }
)
for _member in (
    "factor_diagnostics.parquet",
    "prediction_diagnostics.parquet",
    "fama_macbeth.parquet",
    "model_comparison.parquet",
    "model_reality_checks.parquet",
    "robustness.parquet",
):
    _ARROW_SCHEMAS[_member] = _ARROW_SCHEMAS[_member].append(
        pa.field("input_date_count", pa.int64(), nullable=False)
    )

_ARROW_SCHEMAS["factor_annual_diagnostics.parquet"] = _ARROW_SCHEMAS[
    "factor_diagnostics.parquet"
].append(pa.field("year", pa.int64(), nullable=False))

# Original columns remain required. Correlation results are declared optional
# output groups, supplied only when the analysis actually requests them.
FACTOR_CORRELATION_FIELDS = (
    "observation_count",
    "status",
    "reason",
    "mean",
    "standard_error",
    "t_statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
    "fdr_adjusted_p_value",
)
PREDICTION_CORRELATION_FIELDS = (
    "observation_count",
    "status",
    "reason",
    "mean",
    "standard_error",
    "t_statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
    "bootstrap_status",
    "bootstrap_reason",
    "bootstrap_lower",
    "bootstrap_upper",
)
_CORRELATION_FIELDS = {
    "factor_diagnostics.parquet": FACTOR_CORRELATION_FIELDS,
    "factor_annual_diagnostics.parquet": FACTOR_CORRELATION_FIELDS,
    "prediction_diagnostics.parquet": PREDICTION_CORRELATION_FIELDS,
}
_REQUIRED_COLUMNS = {
    member: tuple(schema.names) for member, schema in _ARROW_SCHEMAS.items()
}
for _member, _names in _CORRELATION_FIELDS.items():
    _schema = _ARROW_SCHEMAS[_member]
    _ARROW_SCHEMAS[_member] = pa.schema(
        [
            *_schema,
            *(
                pa.field(f"{metric}_{name}", _schema.field(name).type)
                for metric in CORRELATION_METRICS[1:]
                for name in _names
            ),
        ]
    )


_TABLE_COLUMNS = {
    member: tuple(schema.names) for member, schema in _ARROW_SCHEMAS.items()
}
_STRING_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_string(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}
_COUNT_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_integer(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}
_TIMESTAMP_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_timestamp(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}
STATISTICS_TABLE_COLUMNS = _REQUIRED_COLUMNS


def _nullable_count(member, column):
    return member in {
        "prediction_diagnostics.parquet",
        "model_comparison.parquet",
    } and column in {"bootstrap_block_size", "bootstrap_repetitions", "bootstrap_seed"}


def _restored_frame(table, member):
    frame = table.to_pandas(integer_object_nulls=True)
    for column in set(_COUNT_COLUMNS[member]).intersection(frame.columns):
        frame[column] = frame[column].astype(
            "Int64" if _nullable_count(member, column) else "int64"
        )
    if member == "factor_stratified_diagnostics.parquet":
        frame["stratum"] = frame.stratum.astype("string")
    return frame


@dataclass(frozen=True, slots=True)
class StatisticsArtifact:
    metadata: dict
    root: Path
    table_group: str
    _members: tuple[str, ...]

    @contextmanager
    def table(self, member: str):
        if member not in self._members:
            raise ValueError("statistics_table_member_invalid")
        with _opened_parquet(self.root / member) as (parquet, _handle):
            frame = _restored_frame(parquet.read(), member)
        yield frame


def write_statistics_artifact(
    output_path: Path,
    *,
    metadata: dict,
    tables: Mapping[str, pd.DataFrame],
    table_group: str = "research",
) -> Path:
    if table_group not in STATISTICS_TABLE_GROUPS:
        raise ValueError("statistics_tables_invalid")
    _validate_metadata(metadata)
    known_members = tuple(STATISTICS_TABLE_GROUPS[table_group])
    expected = {member.removesuffix(".parquet") for member in known_members}
    if not isinstance(tables, Mapping) or not tables or set(tables) - expected:
        raise ValueError("statistics_tables_invalid")
    members = tuple(
        member for member in known_members if member.removesuffix(".parquet") in tables
    )

    def write(staging):
        for name, frame in tables.items():
            member = f"{name}.parquet"
            validated = _validated_frame(frame, member=member, metadata=metadata)
            schema = pa.schema(
                [field for field in _ARROW_SCHEMAS[member] if field.name in validated]
            )
            with closing_iterator(
                open_filesystem_file(staging / member, "xb")
            ) as handle:
                with parquet_table_writer(
                    handle,
                    compression="zstd",
                    schema=schema,
                    statistics=tuple(
                        column
                        for column in ("fold_id", "side", "event_id", "sample_id")
                        if column in _TABLE_COLUMNS[member]
                    ),
                ) as writer:
                    writer(
                        pa.Table.from_pandas(
                            validated,
                            schema=schema,
                            preserve_index=False,
                        )
                    )
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": "qrp-statistics",
                "table_group": table_group,
                "metadata": dict(metadata),
                "members": list(members),
            },
        )

    atomic_write_directory(Path(output_path), write)
    return Path(output_path)


def read_statistics_artifact(path: Path) -> StatisticsArtifact:
    root = Path(path)
    _, manifest = read_qrp_json_file(root / "manifest.json")
    group = manifest.get("table_group")
    if (
        manifest.get("schema_id") != "qrp-statistics"
        or group not in STATISTICS_TABLE_GROUPS
    ):
        raise ValueError("statistics_artifact_contract_invalid")
    members = tuple(manifest["members"])
    metadata = manifest["metadata"]
    artifact = StatisticsArtifact(metadata, root, group, members)
    return artifact


def _validate_metadata(metadata):
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("sample_role")) is not str
        or not metadata["sample_role"].strip()
    ):
        raise ValueError("statistics_sample_role_invalid")
    window = metadata.get("window")
    if not isinstance(window, dict) or set(window) != {"start", "end"}:
        raise ValueError("statistics_window_invalid")
    TimeWindow(window["start"], window["end"])
    candidates = metadata.get("candidate_ids", [])
    if (
        not isinstance(candidates, list)
        or any(not isinstance(x, str) or not x for x in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("statistics_candidate_ids_invalid")


def _validate_table_binding(frame, *, member, metadata):
    if (
        "sample_role" in frame
        and not frame.sample_role.eq(metadata["sample_role"]).all()
    ):
        raise ValueError("statistics_sample_role_mismatch")
    if "candidate_id" in frame:
        valid = frame.candidate_id.isin(metadata.get("candidate_ids", []))
        if member == "portfolio_robustness.parquet":
            valid |= frame.candidate_id.eq("joint") & frame.test_id.isin(
                ("cscv_pbo", "validation_only")
            )
        if not valid.all():
            raise ValueError("statistics_candidate_id_mismatch")
    if (
        "portfolio_id" in frame
        and not frame.portfolio_id.isin(
            metadata.get("comparison_portfolio_ids", [])
        ).all()
    ):
        raise ValueError("statistics_candidate_id_mismatch")
    if not _TIMESTAMP_COLUMNS[member] and member != "factor_annual_diagnostics.parquet":
        return
    start, stop, inclusive = time_window_timestamp_interval(
        TimeWindow(**metadata["window"])
    )
    if member == "factor_annual_diagnostics.parquet":
        last = stop if inclusive else stop - pd.Timedelta(nanoseconds=1)
        if not frame.year.between(start.year, last.year).all():
            raise ValueError("statistics_table_window_mismatch")
    for column in _TIMESTAMP_COLUMNS[member]:
        valid = frame[column].ge(start)
        valid &= frame[column].le(stop) if inclusive else frame[column].lt(stop)
        if not valid.all():
            raise ValueError("statistics_table_window_mismatch")


def _validate_inference_values(result, member):
    if not result.status.isin(("DEFINED", "UNDEFINED")).all():
        raise ValueError("statistics_table_status_invalid")
    defined = result.status.eq("DEFINED")
    measurements = [
        column
        for column in (
            "value",
            "loss",
            "mean",
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
            "bootstrap_lower",
            "bootstrap_upper",
        )
        if column in result
        and not (
            member in {"prediction_diagnostics.parquet", "model_comparison.parquet"}
            and column in {"bootstrap_lower", "bootstrap_upper"}
        )
    ]
    if (
        result.loc[defined, measurements].isna().any().any()
        or result.loc[
            ~defined,
            [
                column
                for column in measurements
                if column != "mean"
                or member
                not in {
                    "prediction_diagnostics.parquet",
                    "model_comparison.parquet",
                    "factor_stratified_diagnostics.parquet",
                    "factor_diagnostics.parquet",
                    "factor_annual_diagnostics.parquet",
                    "fama_macbeth.parquet",
                }
            ],
        ]
        .notna()
        .any()
        .any()
        or result.loc[defined, "reason"].ne("").any()
        or result.loc[~defined, "reason"].str.strip().eq("").any()
    ):
        raise ValueError("statistics_table_status_value_invalid")
    if member in {"prediction_diagnostics.parquet", "model_comparison.parquet"}:
        bootstrap_defined = result.bootstrap_status.eq("DEFINED")
        if (
            not result.bootstrap_status.isin(("DEFINED", "UNDEFINED")).all()
            or result.loc[bootstrap_defined, ["bootstrap_lower", "bootstrap_upper"]]
            .isna()
            .any()
            .any()
            or result.loc[~bootstrap_defined, ["bootstrap_lower", "bootstrap_upper"]]
            .notna()
            .any()
            .any()
            or result.loc[bootstrap_defined, "bootstrap_reason"].ne("").any()
            or result.loc[~bootstrap_defined, "bootstrap_reason"]
            .str.strip()
            .eq("")
            .any()
            or result.loc[
                bootstrap_defined,
                ["bootstrap_block_size", "bootstrap_repetitions", "bootstrap_seed"],
            ]
            .isna()
            .any()
            .any()
        ):
            raise ValueError("statistics_table_bootstrap_status_value_invalid")


def _validated_frame(frame: pd.DataFrame, *, member: str, metadata) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame:
        raise ValueError(f"statistics_table_schema_invalid:{member}")
    required = set(_REQUIRED_COLUMNS[member])
    for metric in CORRELATION_METRICS[1:]:
        fields = {f"{metric}_{name}" for name in _CORRELATION_FIELDS.get(member, ())}
        if fields.intersection(frame.columns):
            required.update(fields)
    if (
        required - set(frame.columns)
        or set(frame.columns) - set(_TABLE_COLUMNS[member])
        or tuple(frame.columns)
        != tuple(column for column in _TABLE_COLUMNS[member] if column in frame)
    ):
        raise ValueError(f"statistics_table_schema_invalid:{member}")
    result = frame.copy(deep=False)
    for field in _ARROW_SCHEMAS[member]:
        if (
            field.name in result
            and not field.nullable
            and result[field.name].isna().any()
        ):
            raise ValueError("statistics_table_null_invalid")
    for column in set(_STRING_COLUMNS[member]).intersection(result.columns):
        if member == "factor_stratified_diagnostics.parquet" and column == "stratum":
            if any(type(value) is not str for value in result[column].dropna()):
                raise ValueError("statistics_stratum_invalid")
            result[column] = result[column].astype("string")
            continue
        values = result[column]
        if values.isna().any():
            result[column] = values.astype("string").fillna("")
        result[column] = result[column].astype(str)
    for column in set(_COUNT_COLUMNS[member]).intersection(result.columns):
        values = strict_integer_series(
            result[column],
            error="statistics_table_count_invalid",
            allow_null=_nullable_count(member, column),
        )
        if values.lt(0).any():
            raise ValueError("statistics_table_count_invalid")
        result[column] = values
    for column in _TIMESTAMP_COLUMNS.get(member, ()):
        values = pd.to_datetime(result[column], errors="raise")
        if values.isna().any():
            raise ValueError("statistics_table_timestamp_invalid")
        result[column] = values.astype("datetime64[ns]")
    numeric_columns = tuple(
        column
        for column in _TABLE_COLUMNS[member]
        if column in result
        and column not in _STRING_COLUMNS[member]
        and column not in _COUNT_COLUMNS[member]
        and (column not in _TIMESTAMP_COLUMNS.get(member, ()))
    )
    for column in numeric_columns:
        result[column] = strict_finite_numeric_series(
            result[column], error="statistics_table_numeric_invalid", allow_null=True
        ).astype("float64")
    if result.empty and member not in _EMPTY_TABLE_MEMBERS:
        raise ValueError("statistics_table_empty")
    if member in (
        *STATISTICS_TABLE_GROUPS["research"],
        *STATISTICS_TABLE_GROUPS["factor"],
    ) and member not in {
        "model_group_comparison.parquet",
        "model_prediction_metrics.parquet",
    }:
        _validate_inference_values(result, member)
        for metric in CORRELATION_METRICS[1:]:
            names = _CORRELATION_FIELDS.get(member, ())
            if names and f"{metric}_status" in result:
                projected = result.loc[
                    :,
                    [
                        column
                        for column in _REQUIRED_COLUMNS[member]
                        if column in result
                    ],
                ].copy()
                for name in names:
                    projected[name] = result[f"{metric}_{name}"]
                _validate_inference_values(projected, member)
    if member in {
        "model_losses.parquet",
        "model_comparison.parquet",
        "model_reality_checks.parquet",
    }:
        if (
            result["sample_role"].isna().any()
            or result["sample_role"].str.strip().eq("").any()
            or result["sample_role"].nunique() > 1
        ):
            raise ValueError("statistics_sample_role_invalid")
        keys = {
            "model_losses.parquet": ("timestamp", "family_id"),
            "model_comparison.parquet": ("prediction_id",),
            "model_reality_checks.parquet": ("test_id", "family_id", "statistic_id"),
        }[member]
        if result.duplicated(list(keys)).any():
            raise ValueError("statistics_table_duplicate_keys")
    if member in {
        "model_group_comparison.parquet",
        "portfolio_group_comparison.parquet",
    }:
        key = (
            "family_id"
            if member == "model_group_comparison.parquet"
            else "portfolio_id"
        )
        defined = result.status.eq("DEFINED")
        bootstrap_defined = result.bootstrap_status.eq("DEFINED")
        measurements = (
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        )
        bootstrap = ("bootstrap_lower", "bootstrap_upper")
        if (
            result.duplicated([key]).any()
            or not result.status.isin(("DEFINED", "UNDEFINED")).all()
            or (not result.bootstrap_status.isin(("DEFINED", "UNDEFINED")).all())
            or result.sample_role.isna().any()
            or result.sample_role.str.strip().eq("").any()
            or (result.sample_role.nunique() > 1)
            or result.loc[
                bootstrap_defined, ["global_mean", "quarterly_mean", "mean_difference"]
            ]
            .isna()
            .any()
            .any()
            or result.loc[defined, list(measurements)].isna().any().any()
            or result.loc[defined, ["global_mean", "quarterly_mean", "mean_difference"]]
            .isna()
            .any()
            .any()
            or result.loc[~defined, list(measurements)].notna().any().any()
            or result.loc[defined, "reason"].ne("").any()
            or result.loc[~defined, "reason"].str.strip().eq("").any()
            or result.loc[bootstrap_defined, list(bootstrap)].isna().any().any()
            or result.loc[~bootstrap_defined, list(bootstrap)].notna().any().any()
            or result.loc[bootstrap_defined, "bootstrap_reason"].ne("").any()
            or result.loc[~bootstrap_defined, "bootstrap_reason"]
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError("statistics_group_comparison_invalid")
        if (
            member == "portfolio_group_comparison.parquet"
            and not result.empty
            and (
                not result.comparison_role.isin(("primary", "secondary")).all()
                or result.comparison_role.eq("primary").sum() != 1
            )
        ):
            raise ValueError("statistics_group_comparison_primary_invalid")
    if (
        member == "model_prediction_metrics.parquet"
        and result.duplicated(["timestamp", "family_id"]).any()
    ):
        raise ValueError("statistics_table_duplicate_keys")
    if (
        member == "model_losses.parquet"
        and not result.asset_coverage_id.str.fullmatch("[0-9a-f]{64}").all()
    ):
        raise ValueError("statistics_model_loss_invalid")
    if member in _COMPARISON_MEMBERS:
        if (
            not result.reference_kind.isin(("benchmark", "control")).all()
            or result[["candidate_id", "reference_id", "sample_role"]]
            .eq("")
            .any()
            .any()
        ):
            raise ValueError("statistics_comparison_identity_invalid")
        suffix = {
            "portfolio_benchmark_intervals.parquet": ("start_at", "end_at"),
            "portfolio_benchmark_metrics.parquet": ("metric_id",),
            "portfolio_market_exposure.parquet": ("coefficient_id",),
            "portfolio_regime_comparison.parquet": ("regime", "metric_id"),
        }[member]
        if result.duplicated([*_COMPARISON_IDENTITY, *suffix]).any():
            raise ValueError("statistics_table_duplicate_keys")
        status_values = [
            (
                "status",
                "reason",
                "return_difference"
                if member == "portfolio_benchmark_intervals.parquet"
                else "value",
            )
        ]
        if member == "portfolio_benchmark_intervals.parquet":
            status_values += [
                ("portfolio_status", "portfolio_reason", "portfolio_return"),
                ("benchmark_status", "benchmark_reason", "benchmark_return"),
            ]
        for status, reason, value in status_values:
            defined = result[status].eq("DEFINED")
            if (
                not result[status]
                .isin(("DEFINED", "UNDEFINED", "NOT_APPLICABLE"))
                .all()
                or not defined.eq(result[value].notna()).all()
                or result.loc[defined, reason].ne("").any()
                or result.loc[~defined, reason].eq("").any()
            ):
                raise ValueError("statistics_table_status_value_invalid")
        if "start_at" in result and not result.start_at.lt(result.end_at).all():
            raise ValueError("statistics_comparison_interval_invalid")
    if member in STATISTICS_TABLE_GROUPS["portfolio"] and member not in (
        "portfolio_group_comparison.parquet",
        *_COMPARISON_MEMBERS,
    ):
        keys = {
            "portfolio_daily.parquet": ("candidate_id", "entry_at"),
            "portfolio_performance.parquet": ("candidate_id", "metric_id"),
            "portfolio_capacity.parquet": ("candidate_id", "entry_at", "participation"),
            "portfolio_capacity_summary.parquet": (
                "candidate_id",
                "participation",
                "period",
                "metric_id",
            ),
            "portfolio_attribution.parquet": (
                "candidate_id",
                "entry_at",
                "component_id",
            ),
            "portfolio_robustness.parquet": ("candidate_id", "test_id"),
            "portfolio_exposures.parquet": ("candidate_id", "entry_at", "factor_id"),
            "holding_factor_returns.parquet": (
                "attribution_group",
                "entry_at",
                "end_at",
                "factor_id",
            ),
        }[member]
        if result.duplicated(list(keys)).any():
            raise ValueError("statistics_table_duplicate_keys")
        if "status" in result:
            if not result.status.isin(("DEFINED", "UNDEFINED", "BLOCKED")).all():
                raise ValueError("statistics_table_status_invalid")
            value_column = (
                "capacity" if member == "portfolio_capacity.parquet" else "value"
            )
            defined = result.status.eq("DEFINED")
            if (
                result.loc[defined, value_column].isna().any()
                or result.loc[~defined, value_column].notna().any()
                or result.loc[~defined, "reason"].eq("").any()
            ):
                raise ValueError("statistics_table_status_value_invalid")
        if "end_at" in result and (not result.entry_at.lt(result.end_at).all()):
            raise ValueError("statistics_holding_interval_invalid")
        if "decision_at" in result:
            ordered = (
                result.decision_at.le(result.entry_at)
                if metadata.get("attribution_clock") == "inventory_intervals"
                else result.decision_at.lt(result.entry_at)
            )
            if not ordered.all():
                raise ValueError("statistics_holding_interval_invalid")
        if member == "portfolio_attribution.parquet" and (
            not result.component_type.isin(("additive", "detail")).all()
        ):
            raise ValueError("statistics_attribution_component_invalid")
    keys = {
        "model_trials.parquet": (
            "selection_group",
            "decision_id",
            "family_id",
            "config_id",
            "metric_id",
        ),
        "prediction_diagnostics.parquet": ("prediction_id",),
        "robustness.parquet": ("test_id",),
        "factor_diagnostics.parquet": ("family_id", "factor_id", "fdr_method"),
        "factor_annual_diagnostics.parquet": (
            "year",
            "family_id",
            "factor_id",
            "fdr_method",
        ),
        "factor_correlation.parquet": (
            "factor_id",
            "comparison_factor_id",
            "statistic",
        ),
        "fama_macbeth.parquet": ("regression_id", "coefficient_id"),
        "fama_macbeth_periods.parquet": (
            "regression_id",
            "coefficient_id",
            "timestamp",
        ),
    }.get(member)
    if keys is not None and result.duplicated(list(keys)).any():
        raise ValueError("statistics_table_duplicate_keys")
    if member == "factor_stratified_diagnostics.parquet":
        _validate_factor_stratified(result, metadata)
    _validate_table_binding(result, member=member, metadata=metadata)
    result.index = pd.RangeIndex(len(result))
    return result


def _validate_factor_stratified(frame, metadata):
    """Bind concrete factor summaries to their persisted EVAL inputs and family."""
    binding = metadata.get("evaluation_binding", {})
    policy = metadata.get("statistics", {})
    allowed_domains = {"side", *binding.get("strata", {}).get("kinds", [])}
    if set(metadata.get("stratum_domains", {})) - allowed_domains:
        raise ValueError("statistics_stratified_domain_invalid")
    if frame.duplicated(
        ["signal", "label", "sample_scope", "stratification", "stratum", "statistic"]
    ).any():
        raise ValueError("statistics_stratified_duplicate_keys")
    signals = {item["signal_id"] for item in binding.get("signals", [])}
    scopes = set(metadata.get("family_scopes", []))
    expected = {
        (signal, label, scope, "all", None, statistic)
        for signal in signals
        for label in binding.get("label_columns", [])
        for scope in scopes
        for statistic in ("ic", "rankic")
    }
    expected.update(
        (signal, binding["strata"]["primary_label"], scope, kind, stratum, statistic)
        for kind, strata in metadata.get("stratum_domains", {}).items()
        if kind != "side"
        for stratum in strata
        for signal in signals
        for scope in scopes
        for statistic in ("ic", "rankic")
    )
    if "side" in metadata.get("stratum_domains", {}):
        if metadata["stratum_domains"]["side"] != ["long", "short"]:
            raise ValueError("statistics_stratified_domain_invalid")
        expected.update(
            (signal, label, scope, "side", side, statistic)
            for signal in signals
            for label in binding.get("label_columns", [])
            for scope in scopes
            for side in ("long", "short")
            for statistic in ("ic", "rankic")
        )
    actual = {
        (
            row.signal,
            row.label,
            row.sample_scope,
            row.stratification,
            None if pd.isna(row.stratum) else row.stratum,
            row.statistic,
        )
        for row in frame.itertuples(index=False)
    }
    if actual != expected:
        raise ValueError("statistics_stratified_domain_incomplete")
    for scope, block in frame.groupby("sample_scope"):
        if not block.input_date_count.eq(
            metadata.get("input_date_counts", {}).get(scope)
        ).all():
            raise ValueError("statistics_stratified_count_invalid")
    for key in ("hac_lag", "fdr_method", "fdr_alpha"):
        if not frame[key].eq(policy.get(key)).all():
            raise ValueError("statistics_stratified_policy_invalid")
