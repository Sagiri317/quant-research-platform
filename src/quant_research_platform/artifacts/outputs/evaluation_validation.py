"""Validate evaluation metadata at atomic publication."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.evaluations import (
    EVALUATION_FORECAST_AGE_BOUNDS,
    EvaluationDiagnostics,
    EvaluationSpec,
    evaluation_sessions,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec

EVALUATION_ARTIFACTS = [
    "metrics.parquet",
    "grouped_label_values.parquet",
    "ic_summary.parquet",
    "group_spread_summary.parquet",
]
_EVALUATION_OUTPUT_METADATA_KEYS = frozenset(
    {
        "evaluation_id",
        "run_type",
        "market_id",
        "research_dataset_id",
        "evaluation_axis",
        "signals",
        "label_columns",
        "label_availability_columns",
        "filter_mode",
        "evaluation_diagnostics",
    }
)
_EVALUATION_DIAGNOSTIC_KEYS = frozenset(
    {
        "filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_signal_label_dropna",
        "metric_rows",
        "grouped_label_value_rows",
        "ic_summary_rows",
        "group_spread_summary_rows",
        "timestamp_count",
        "min_timestamp",
        "max_timestamp",
        "n_groups",
        "skipped_timestamp_count",
    }
)


EVALUATION_SCIENCE_KEYS = frozenset(
    {
        "sample_scope",
        "input_facts",
        "comparison_inputs",
        "strata",
        "stratum_domains",
        "optional_member_rows",
        "common_rule",
    }
)


def validate_evaluation_science(metadata):
    """Validate actual input, window, units and optional daily-member facts."""
    from quant_research_platform.contracts.time import TimeWindow

    base = {"window", "cutoff", "sample_role"}
    if isinstance(metadata, dict) and "sessions" in metadata:
        from quant_research_platform.contracts.time import TimeWindow

        dates = evaluation_sessions(metadata["sessions"])
        window = TimeWindow(**metadata["window"])
        if len(dates) and (
            dates[0] < pd.Timestamp(window.start).normalize()
            or dates[-1] > pd.Timestamp(window.end).normalize()
        ):
            raise ValueError("evaluation_sessions_outside_window")
        base.add("sessions")
    if isinstance(metadata, dict) and "classification_metrics" in metadata:
        from quant_research_platform.contracts.models import (
            CLASSIFICATION_METRICS,
            validate_classification_metrics,
        )

        summaries = metadata["classification_metrics"]
        if not isinstance(summaries, dict) or not summaries:
            raise ValueError("evaluation_classification_metrics_invalid")
        for signal, summary in summaries.items():
            if (
                type(signal) is not str
                or not signal
                or set(summary)
                != {"target_column", "class_threshold", "rows", *CLASSIFICATION_METRICS}
            ):
                raise ValueError("evaluation_classification_metrics_invalid")
            if (
                type(summary["target_column"]) is not str
                or not summary["target_column"]
                or type(summary["class_threshold"]) not in (int, float)
                or not 0 <= summary["class_threshold"] < float("inf")
            ):
                raise ValueError("evaluation_classification_metrics_invalid")
            validate_classification_metrics(
                {name: summary[name] for name in CLASSIFICATION_METRICS},
                summary["rows"],
            )
        metadata = {
            key: value
            for key, value in metadata.items()
            if key != "classification_metrics"
        }
    has_response = isinstance(metadata, dict) and "response_diagnostics" in metadata
    if isinstance(metadata, dict) and "response_diagnostics" in metadata:
        from quant_research_platform.contracts.evaluations import (
            validate_response_diagnostics,
        )

        validate_response_diagnostics(metadata["response_diagnostics"])
        metadata = {
            key: value
            for key, value in metadata.items()
            if key != "response_diagnostics"
        }
    if type(metadata) is not dict or set(metadata) not in (
        base,
        base | EVALUATION_SCIENCE_KEYS,
    ):
        raise ValueError("evaluation_science_metadata_invalid")
    if type(metadata["sample_role"]) is not str or not metadata["sample_role"].strip():
        raise ValueError("evaluation_sample_role_invalid")
    window = TimeWindow(**metadata["window"])
    if window.end is None or any(
        pd.isna(pd.Timestamp(value))
        for value in (window.start, window.end, metadata["cutoff"])
    ):
        raise ValueError("evaluation_science_time_invalid")
    if "sample_scope" not in metadata:
        return
    if (
        metadata["sample_scope"] not in {"own", "common"}
        or metadata["common_rule"] != "mask_only_no_retransform"
    ):
        raise ValueError("evaluation_sample_scope_invalid")
    peers = metadata["comparison_inputs"]
    if type(peers) is not list or (metadata["sample_scope"] == "own" and peers):
        raise ValueError("evaluation_comparison_inputs_invalid")
    for facts in [metadata["input_facts"], *peers]:
        if type(facts) is not dict or set(facts) != {
            "selector",
            "columns",
            "column_units",
            "feature_transforms",
            "signal_units",
            "label_units",
            "label_availability_columns",
            "label_semantics",
            "market_id",
        }:
            raise ValueError("evaluation_input_facts_invalid")
        if (
            type(facts["selector"]) is not dict
            or type(facts["selector"].get("path")) is not str
            or not facts["selector"]["path"]
        ):
            raise ValueError("evaluation_input_facts_invalid")
        for key in (
            "columns",
            "column_units",
            "signal_units",
            "label_units",
            "label_availability_columns",
            "label_semantics",
        ):
            if type(facts[key]) is not dict:
                raise ValueError("evaluation_input_facts_invalid")
        if type(facts["feature_transforms"]) is not list or any(
            unit is not None and (type(unit) is not str or not unit)
            for key in ("column_units", "signal_units", "label_units")
            for unit in facts[key].values()
        ):
            raise ValueError("evaluation_input_units_invalid")
        if set(facts["column_units"]) != set(facts["columns"]) or set(
            facts["label_units"]
        ) != set(facts["label_availability_columns"]):
            raise ValueError("evaluation_input_facts_invalid")
        for label, availability in facts["label_availability_columns"].items():
            if (
                facts["columns"].get(label) != "label"
                or availability not in facts["columns"]
                or facts["label_units"][label] != facts["column_units"].get(label)
            ):
                raise ValueError("evaluation_input_facts_invalid")
        for signal, unit in facts["signal_units"].items():
            if facts["columns"].get(signal) not in {
                "factor",
                "feature",
            } or unit != facts["column_units"].get(signal):
                raise ValueError("evaluation_input_facts_invalid")
        if facts["label_units"] != metadata["input_facts"]["label_units"]:
            raise ValueError("evaluation_label_units_mismatch")
    strata = metadata["strata"]
    if type(strata) is not dict or type(metadata["stratum_domains"]) is not dict:
        raise ValueError("evaluation_strata_invalid")
    if strata:
        allowed = {
            "primary_label",
            "kinds",
            "forecast_freshness_days",
            "population",
            "age_column",
            "age_bounds",
        }
        if (
            set(strata) - allowed
            or not {"primary_label", "kinds", "population", "age_bounds"} <= set(strata)
            or strata["population"]
            != "focal_eligible_primary_label_mature_before_common"
        ):
            raise ValueError("evaluation_strata_invalid")
        kinds = strata["kinds"]
        if (
            type(kinds) is not list
            or not kinds
            or len(set(kinds)) != len(kinds)
            or set(kinds) - {"industry", "size", "liquidity", "forecast_age"}
        ):
            raise ValueError("evaluation_strata_invalid")
        if "forecast_age" in kinds:
            if (
                strata.get("forecast_freshness_days") not in {90, 180}
                or strata.get("age_column") != "median_forecast_age_days_fy1"
                or strata["age_bounds"]
                != [
                    list(item)
                    for item in EVALUATION_FORECAST_AGE_BOUNDS
                    if item[2] <= strata["forecast_freshness_days"]
                ]
            ):
                raise ValueError("evaluation_age_bounds_invalid")
    if set(metadata["stratum_domains"]) != set(strata.get("kinds", [])) or any(
        type(values) is not list
        or any(value is not None and type(value) is not str for value in values)
        or len(values) != len(set(values))
        for values in metadata["stratum_domains"].values()
    ):
        raise ValueError("evaluation_stratum_domains_invalid")
    counts = metadata["optional_member_rows"]
    if has_response != (
        isinstance(counts, dict)
        and set(counts) == {"response_metrics", "response_groups"}
    ):
        raise ValueError("evaluation_response_members_invalid")
    if (
        type(counts) is not dict
        or set(counts)
        not in (
            set(),
            {"daily_coverage", "daily_strata"},
            {"response_metrics", "response_groups"},
        )
        or any(type(value) is not int or value < 0 for value in counts.values())
        or ((strata or metadata["sample_scope"] == "common") and not counts)
    ):
        raise ValueError("evaluation_optional_members_invalid")


def validate_evaluation_output_metadata_shape(
    metadata: object,
) -> dict[str, object]:
    science = {"window", "cutoff", "sample_role"}
    if isinstance(metadata, dict) and "sessions" in metadata:
        science.add("sessions")
    if isinstance(metadata, dict) and "classification_metrics" in metadata:
        science.add("classification_metrics")
    if isinstance(metadata, dict) and "response_diagnostics" in metadata:
        science.add("response_diagnostics")
    base = _EVALUATION_OUTPUT_METADATA_KEYS | (
        {"correlation_clock"}
        if isinstance(metadata, dict) and "correlation_clock" in metadata
        else set()
    )
    if type(metadata) is not dict or set(metadata) not in (
        base,
        base | science,
        base | science | EVALUATION_SCIENCE_KEYS,
    ):
        raise ValueError("output_manifest_metadata_invalid")
    if science <= set(metadata):
        validate_evaluation_science(
            {
                key: metadata[key]
                for key in science
                | (EVALUATION_SCIENCE_KEYS if "sample_scope" in metadata else set())
            }
        )
    for key in (
        "evaluation_id",
        "run_type",
        "market_id",
        "research_dataset_id",
        "evaluation_axis",
        "filter_mode",
    ):
        require_metadata_string(metadata, key, "output_manifest_metadata_invalid")
    label_columns = require_metadata_string_list(
        metadata, "label_columns", "output_manifest_metadata_invalid"
    )
    label_availability_columns = metadata["label_availability_columns"]
    if type(label_availability_columns) is not dict or any(
        type(label) is not str or type(column) is not str
        for label, column in label_availability_columns.items()
    ):
        raise ValueError("output_manifest_metadata_invalid")
    signals = require_signal_metadata_list(
        metadata, "signals", "output_manifest_metadata_invalid"
    )
    if "input_facts" in metadata:
        facts = metadata["input_facts"]
        if (
            facts["label_availability_columns"] != label_availability_columns
            or set(facts["signal_units"])
            != {
                signal.source_column
                for signal in signals
                if signal.source_kind != "prediction"
            }
            or set(facts["label_units"]) != set(label_columns)
        ):
            raise ValueError("evaluation_input_facts_invalid")
        if (
            metadata["strata"]
            and metadata["strata"]["primary_label"] not in label_columns
        ):
            raise ValueError("evaluation_strata_invalid")
    if "response_diagnostics" in metadata:
        from quant_research_platform.contracts.evaluations import (
            validate_response_diagnostics,
        )

        validate_response_diagnostics(
            metadata["response_diagnostics"],
            signals=signals,
        )
        if (
            metadata["evaluation_axis"] != "single_asset_time_series"
            or metadata["sample_scope"] != "own"
            or metadata["strata"]
            or any(signal.transform != "identity" for signal in signals)
            or any(
                unit != "decimal_return"
                for unit in metadata["input_facts"]["label_units"].values()
            )
        ):
            raise ValueError("evaluation_response_domain_invalid")
    if not label_columns:
        raise ValueError("output_manifest_metadata_invalid")

    diagnostics = metadata["evaluation_diagnostics"]
    if type(diagnostics) is not dict or set(diagnostics) != _EVALUATION_DIAGNOSTIC_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    require_metadata_string(
        diagnostics, "filter_mode", "output_manifest_metadata_invalid"
    )
    require_metadata_string(
        diagnostics, "filter_definition", "output_manifest_metadata_invalid"
    )
    require_metadata_string_list(
        diagnostics,
        "filter_columns_used",
        "output_manifest_metadata_invalid",
        allow_empty=True,
    )
    for key in (
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_signal_label_dropna",
        "metric_rows",
        "grouped_label_value_rows",
        "ic_summary_rows",
        "group_spread_summary_rows",
        "timestamp_count",
        "n_groups",
        "skipped_timestamp_count",
    ):
        require_non_negative_metadata_int(
            diagnostics, key, "output_manifest_metadata_invalid"
        )
    min_timestamp = require_optional_metadata_string(
        diagnostics, "min_timestamp", "output_manifest_metadata_invalid"
    )
    max_timestamp = require_optional_metadata_string(
        diagnostics, "max_timestamp", "output_manifest_metadata_invalid"
    )
    if (diagnostics["min_timestamp"] is None) != (diagnostics["max_timestamp"] is None):
        raise ValueError("output_manifest_metadata_invalid")
    if science <= set(metadata) and min_timestamp is not None:
        from quant_research_platform.contracts.time import (
            TimeWindow,
            time_window_timestamp_interval,
        )

        start, stop, closed = time_window_timestamp_interval(
            TimeWindow(**metadata["window"])
        )
        maximum = pd.Timestamp(max_timestamp)
        if pd.Timestamp(min_timestamp) < pd.Timestamp(start) or (
            maximum > pd.Timestamp(stop) if closed else maximum >= pd.Timestamp(stop)
        ):
            raise ValueError("evaluation_diagnostics_outside_window")
    if not (
        diagnostics["rows_after_filter_mode"] <= diagnostics["rows_input"]
        and diagnostics["rows_after_signal_label_dropna"]
        <= diagnostics["rows_after_filter_mode"]
        * len(metadata["signals"])
        * len(label_columns)
    ):
        raise ValueError("output_manifest_metadata_invalid")
    if metadata["filter_mode"] != diagnostics["filter_mode"]:
        raise ValueError("output_manifest_metadata_invalid")
    if metadata["evaluation_axis"] == "cross_sectional":
        if diagnostics["metric_rows"] < diagnostics["timestamp_count"]:
            raise ValueError("output_manifest_metadata_invalid")
    elif metadata["evaluation_axis"] == "single_asset_time_series":
        expected_metric_rows = (
            len(signals) * len(label_columns) if diagnostics["timestamp_count"] else 0
        )
        if (
            (
                diagnostics["metric_rows"] != expected_metric_rows
                if metadata.get("correlation_clock", "observation") == "observation"
                else not expected_metric_rows
                <= diagnostics["metric_rows"]
                <= expected_metric_rows * diagnostics["timestamp_count"]
            )
            or diagnostics["grouped_label_value_rows"] != 0
            or diagnostics["group_spread_summary_rows"] != 0
        ):
            raise ValueError("output_manifest_metadata_invalid")
    else:
        raise ValueError("output_manifest_metadata_invalid")
    try:
        evaluation_diagnostics = EvaluationDiagnostics(
            filter_mode=diagnostics["filter_mode"],
            filter_columns_used=tuple(diagnostics["filter_columns_used"]),
            filter_definition=diagnostics["filter_definition"],
            rows_input=diagnostics["rows_input"],
            rows_after_filter_mode=diagnostics["rows_after_filter_mode"],
            rows_after_signal_label_dropna=diagnostics[
                "rows_after_signal_label_dropna"
            ],
            metric_rows=diagnostics["metric_rows"],
            grouped_label_value_rows=diagnostics["grouped_label_value_rows"],
            ic_summary_rows=diagnostics["ic_summary_rows"],
            group_spread_summary_rows=diagnostics["group_spread_summary_rows"],
            timestamp_count=diagnostics["timestamp_count"],
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            n_groups=diagnostics["n_groups"],
            skipped_timestamp_count=diagnostics["skipped_timestamp_count"],
        )
        validate_filter_metadata(
            mode=evaluation_diagnostics.filter_mode,
            columns=list(evaluation_diagnostics.filter_columns_used),
            definition=evaluation_diagnostics.filter_definition,
            error_name="output_manifest_metadata_invalid",
        )
        EvaluationSpec(
            evaluation_id=metadata["evaluation_id"],
            market_id=MarketId(metadata["market_id"]),
            research_dataset_id=metadata["research_dataset_id"],
            signals=signals,
            label_columns=tuple(label_columns),
            label_availability_columns=label_availability_columns,
            evaluation_axis=metadata["evaluation_axis"],
            correlation_clock=metadata.get("correlation_clock", "observation"),
            filter_mode=metadata["filter_mode"],
            n_groups=evaluation_diagnostics.n_groups,
            run_type=metadata["run_type"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc
    return metadata


_SIGNAL_METADATA_KEYS = frozenset(
    {"signal_id", "source_kind", "source_column", "direction", "transform"}
)


def validate_filter_metadata(
    *,
    mode: str,
    columns: object,
    definition: str,
    error_name: str,
) -> None:
    if type(columns) is not list or columns != list(required_filter_columns(mode)):
        raise ValueError(error_name)
    if definition != filter_definition(mode):
        raise ValueError(error_name)


def require_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str:
    value = metadata[key]
    if type(value) is not str or not value:
        raise ValueError(error_name)
    return value


def require_optional_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str | None:
    value = metadata[key]
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(error_name)
    return value


def require_metadata_string_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = metadata[key]
    if type(value) is not list:
        raise ValueError(error_name)
    if (not allow_empty and not value) or any(
        type(item) is not str or not item for item in value
    ):
        raise ValueError(error_name)
    if len(value) != len(set(value)):
        raise ValueError(error_name)
    return value


def require_non_negative_metadata_int(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> int:
    value = metadata[key]
    if type(value) is not int or value < 0:
        raise ValueError(error_name)
    return value


def require_signal_metadata_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> tuple[SignalSpec, ...]:
    value = metadata[key]
    if type(value) is not list or not value:
        raise ValueError(error_name)
    signals: list[SignalSpec] = []
    for signal in value:
        if type(signal) is not dict or set(signal) != _SIGNAL_METADATA_KEYS:
            raise ValueError(error_name)
        for field in _SIGNAL_METADATA_KEYS:
            require_metadata_string(signal, field, error_name)
        try:
            signals.append(
                SignalSpec(
                    signal_id=signal["signal_id"],
                    source_kind=signal["source_kind"],
                    source_column=signal["source_column"],
                    direction=signal["direction"],
                    transform=signal["transform"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(error_name) from exc
    return tuple(signals)
