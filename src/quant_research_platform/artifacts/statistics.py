"""Multi-table semantic artifact for scientific research statistics."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.evaluations import CORRELATION_METRICS
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import (
    read_qrp_json_file,
    write_qrp_json_file,
)
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
        "model_reality_checks.parquet",
    ),
    "portfolio": (
        *_COMPARISON_MEMBERS,
        "portfolio_daily.parquet",
        "portfolio_performance.parquet",
        "portfolio_capacity.parquet",
        "portfolio_capacity_summary.parquet",
        "portfolio_attribution.parquet",
        "portfolio_robustness.parquet",
        "portfolio_exposures.parquet",
        "holding_factor_returns.parquet",
    ),
}
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
STATISTICS_TABLE_COLUMNS = {
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


_COUNT_COLUMNS = {
    member: tuple(field.name for field in schema if pa.types.is_integer(field.type))
    for member, schema in _ARROW_SCHEMAS.items()
}


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
    known_members = tuple(STATISTICS_TABLE_GROUPS[table_group])
    members = tuple(
        member for member in known_members if member.removesuffix(".parquet") in tables
    )

    def write(staging):
        for name, frame in tables.items():
            member = f"{name}.parquet"
            schema = pa.schema(
                [field for field in _ARROW_SCHEMAS[member] if field.name in frame]
            )
            frame = frame.assign(
                **{
                    field.name: frame[field.name].astype("string").fillna("")
                    for field in schema
                    if pa.types.is_string(field.type) and field.name != "stratum"
                }
            )
            frame = frame.assign(
                **{
                    field.name: pd.to_datetime(frame[field.name])
                    for field in schema
                    if pa.types.is_timestamp(field.type)
                }
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
                        if column in schema.names
                    ),
                ) as writer:
                    writer(
                        pa.Table.from_pandas(
                            frame,
                            schema=schema,
                            preserve_index=False,
                        )
                    )
        write_qrp_json_file(
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
    group = manifest["table_group"]
    members = tuple(manifest["members"])
    metadata = manifest["metadata"]
    artifact = StatisticsArtifact(metadata, root, group, members)
    return artifact
