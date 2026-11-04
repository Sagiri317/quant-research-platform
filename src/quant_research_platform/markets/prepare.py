"""Explicit market-data preparation from caller-selected source paths."""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets.reading import _dataset_input_options
from quant_research_platform.storage.datasets.writing import write_dataset


def _request(config):
    required = {"market_id", "data_id", "inputs", "time_window", "output_root"}
    if not required <= set(config) or set(config) - required - {
        "source",
        "parameters",
        "factor_columns",
        "label_columns",
        "base_panel",
    }:
        raise ValueError("data_operation_fields_invalid")
    if any(
        type(config[key]) is not str or not config[key]
        for key in ("market_id", "data_id", "output_root")
    ):
        raise ValueError("data_operation_value_invalid")
    inputs = config["inputs"]
    if not isinstance(inputs, Mapping) or any(
        type(k) is not str or not k for k in inputs
    ):
        raise ValueError("data_operation_inputs_invalid")
    for value in inputs.values():
        if (isinstance(value, str) and not value) or (
            isinstance(value, Mapping) and value.get("path") == ""
        ):
            raise ValueError("data_operation_inputs_invalid")
        _dataset_input_options(value)
    window = config["time_window"]
    if not isinstance(window, Mapping) or set(window) != {"start", "end"}:
        raise ValueError("data_operation_window_invalid")
    return dict(inputs), TimeWindow(**window)


def prepare(config: Mapping) -> Path:
    """Compute one declared market dataset without discovering upstream paths."""
    config = qrp_json_mapping_snapshot(config)
    inputs, window = _request(config)
    market = config["market_id"]
    source = config.get("source")
    if market == "a_share":
        if source is not None:
            raise ValueError("a_share_prepare_source_not_consumed")
        from quant_research_platform.catalog.processed import InterimDataCatalog
        from quant_research_platform.markets.implementations.a_share.processed.catalog import (  # noqa: E501
            processed_catalog,
        )
        from quant_research_platform.markets.implementations.a_share.processed.producer import (  # noqa: E501
            produce_a_share_processed_data,
        )
        from quant_research_platform.sources.choice.interim import (
            interim_catalog as choice,
        )
        from quant_research_platform.sources.csmar.interim import (
            interim_catalog as csmar,
        )
        from quant_research_platform.sources.reference.interim import (
            interim_catalog as reference,
        )

        # The canonical source schema is the same for shared financial fields.
        specs = {
            spec.data_id: spec
            for catalog in (csmar(), choice(), reference())
            for spec in catalog.list()
        }
        catalog = processed_catalog()
        spec = catalog.get(config["data_id"])
        producer = produce_a_share_processed_data
        kwargs = {"source_catalog": InterimDataCatalog(specs.values())}
        from quant_research_platform.research.protocols.a_share_hft import DATA_IDS

        if spec.data_id in DATA_IDS:
            from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
                produce,
            )

            producer, kwargs = produce, {}
    else:
        raise ValueError("data_operation_market_invalid")
    spec = _configured_spec(spec, config)
    required = set(spec.processed_input_data_ids) | (
        set(spec.source_input_data_ids) - set(spec.optional_source_input_data_ids)
    )
    allowed = set(spec.processed_input_data_ids) | set(spec.source_input_data_ids)
    if not required <= set(inputs) or not set(inputs) <= allowed:
        raise ValueError("data_operation_input_paths_mismatch")
    output = Path(config["output_root"])
    metadata = {}

    def frames():
        with producer(
            spec, inputs=inputs, time_window=window, processed_catalog=catalog, **kwargs
        ) as (data, producer_metadata):
            producer_metadata["inputs"] = dict(config["inputs"])
            producer_metadata["parameters"] = {
                "time_window": dict(config["time_window"]),
                **json.loads(spec.parameters_json),
            }
            for frame in data:
                try:
                    yield frame
                except GeneratorExit:
                    return
        metadata.update(producer_metadata)

    write_dataset(output, frames(), spec=spec, metadata=metadata)
    return output


def _configured_spec(spec, config):
    """Apply only options consumed by this explicitly requested mathematical recipe."""
    from quant_research_platform.contracts.columns import ColumnRole

    parameters = config.get("parameters", {})
    if type(parameters) is not dict:
        raise ValueError("data_operation_parameters_invalid")
    if spec.market_id == "a_share" and spec.data_id == "on_id_decomposition":
        from quant_research_platform.markets.implementations.a_share.processed.catalog import (  # noqa: E501
            on_id_decomposition_spec,
        )

        if set(parameters) - {"horizons"}:
            raise ValueError("data_operation_parameter_not_consumed")
        if {"factor_columns", "label_columns", "base_panel"}.intersection(config):
            raise ValueError("on_id_selection_uses_horizons")
        return on_id_decomposition_spec(parameters.get("horizons", [20]))
    dynamic = {
        "expectations_factors": {"freshness_days"},
        "price_volume_facts": {"min_fresh_fraction"},
        "market_model_factors": {"min_fresh_fraction"},
        "risk_residual_factors": {"min_fresh_fraction"},
        "hft_panel": {"max_gap_seconds"},
    }.get(spec.data_id, set())
    if set(parameters) - dynamic:
        raise ValueError("data_operation_parameter_not_consumed")
    if "min_fresh_fraction" in parameters and (
        type(parameters["min_fresh_fraction"]) not in (int, float)
        or not 0 < parameters["min_fresh_fraction"] <= 1
    ):
        raise ValueError("data_operation_min_fresh_fraction_invalid")
    if "freshness_days" in parameters and (
        type(parameters["freshness_days"]) is not int
        or parameters["freshness_days"] <= 0
    ):
        raise ValueError("data_operation_freshness_days_invalid")
    if "max_gap_seconds" in parameters and (
        type(parameters["max_gap_seconds"]) not in (int, float)
        or not 0 < parameters["max_gap_seconds"] < float("inf")
    ):
        raise ValueError("hft_max_gap_seconds_invalid")
    actual = {**json.loads(spec.parameters_json), **parameters}
    if "base_panel" in config:
        raise ValueError("data_operation_parameter_not_consumed")
    if "factor_columns" in config:
        factors = config["factor_columns"]
        if (
            type(factors) is not list
            or not factors
            or any(type(factor) is not str or not factor for factor in factors)
            or len(set(factors)) != len(factors)
        ):
            raise ValueError("data_operation_factor_columns_invalid")
        if spec.data_id == "fundamental_pit":
            from quant_research_platform.markets.implementations.a_share.fundamentals import (  # noqa: E501
                FINANCIAL_VALUE_COLUMNS,
            )
            from quant_research_platform.research.factors.fundamental.definitions import (  # noqa: E501
                FUNDAMENTAL_FACTORS,
                fundamental_inputs,
            )

            if not set(factors) <= set(FUNDAMENTAL_FACTORS):
                raise ValueError("data_operation_factor_columns_invalid")
            metrics = {
                column.removeprefix("beginning_")
                .removeprefix("average_")
                .removeprefix("prior_year_")
                .removesuffix("_ttm")
                for factor in factors
                for column in fundamental_inputs(factor)
            }
            sources = {
                name
                for name, fields in FINANCIAL_VALUE_COLUMNS.items()
                if metrics.intersection(fields)
            }
            spec = replace(
                spec,
                source_input_data_ids=tuple(
                    name
                    for name in spec.source_input_data_ids
                    if name not in FINANCIAL_VALUE_COLUMNS or name in sources
                ),
            )
            actual["factor_columns"] = factors
        else:
            available = {
                name
                for name, role in spec.column_roles.items()
                if role == ColumnRole.FACTOR
            }
            if not set(factors) <= available:
                raise ValueError("data_operation_factor_columns_invalid")
            roles = {
                name: role
                for name, role in spec.column_roles.items()
                if role != ColumnRole.FACTOR
            }
            # Scientific input order is explicit; supporting keys never become factors.
            roles.update((name, ColumnRole.FACTOR) for name in factors)
            spec = replace(
                spec,
                column_roles=roles,
                column_units={
                    name: unit
                    for name, unit in spec.column_units.items()
                    if name in roles
                },
            )
            if spec.data_id == "fundamental_factors":
                from quant_research_platform.research.factors.fundamental.definitions import (  # noqa: E501
                    fundamental_inputs,
                )

                if not any(
                    "total_market_cap" in fundamental_inputs(name) for name in factors
                ):
                    spec = replace(spec, processed_input_data_ids=("fundamental_pit",))
    if "label_columns" in config:
        labels = config["label_columns"]
        if (
            type(labels) is not list
            or not labels
            or any(type(label) is not str or not label for label in labels)
            or len(set(labels)) != len(labels)
            or not set(labels) <= set(spec.label_availability_columns)
        ):
            raise ValueError("data_operation_label_columns_invalid")
        from quant_research_platform.research.protocols import a_share, a_share_hft

        definitions = {
            item.output_column: item
            for item in (
                *a_share.label_catalog().list(),
                *a_share_hft.label_catalog().list(),
            )
        }
        selected = [definitions[name] for name in labels]
        label_fields = {
            *spec.label_availability_columns,
            *spec.label_availability_columns.values(),
        }
        roles = {
            name: role
            for name, role in spec.column_roles.items()
            if name not in label_fields
        }
        for definition in selected:
            for name in (definition.output_column, definition.available_at_column):
                roles[name] = spec.column_roles[name]
        spec = replace(
            spec,
            column_roles=roles,
            column_units={
                name: unit for name, unit in spec.column_units.items() if name in roles
            },
            label_availability_columns={
                item.output_column: item.available_at_column for item in selected
            },
        )
        if "labels" in actual:
            actual["labels"] = [item.to_manifest() for item in selected]
    if spec.market_id == "a_share" and spec.data_id == "price_volume_facts":
        from quant_research_platform.research.factors.engine import factor_lookback_bars
        from quant_research_platform.research.protocols.a_share import factor_catalog

        selected = tuple(
            definition
            for definition in factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
        if not factor_lookback_bars(selected):
            spec = replace(spec, optional_source_input_data_ids=("trading_calendar",))
    return replace(spec, parameters_json=json.dumps(actual))
