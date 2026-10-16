"""Explicit market-data preparation from caller-selected source paths."""

import json
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
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
    elif market == "crypto":
        from quant_research_platform.markets.implementations.crypto.ids import (
            HFT_CROSS_ASSET_DATA_ID,
            HFT_DERIVATIVE_DATA_ID,
            HFT_EXTERNAL_DATA_IDS,
            HFT_INFORMATION_PANEL_DATA_ID,
            HFT_PANEL_DATA_ID,
            HFT_SOL_PANEL_DATA_ID,
            HFT_SPOT_CONTEXT_DATA_ID,
            HFT_SPOT_PANEL_DATA_ID,
        )
        from quant_research_platform.markets.implementations.crypto.processed import (
            producer as crypto,
        )
        from quant_research_platform.markets.implementations.crypto.processed.catalog import (  # noqa: E501
            hft_processed_catalog,
        )
        from quant_research_platform.sources.binance.ids import (
            FUTURES_MARKET_TYPE,
            RESEARCH_SYMBOLS,
            SPOT_MARKET_TYPE,
        )
        from quant_research_platform.sources.binance.replay import (
            ReconstructionStats,
            iter_agg_trades,
            iter_reconstructed_depth,
        )
        from quant_research_platform.sources.market import open_market_source

        catalog = hft_processed_catalog()
        spec = catalog.get(config["data_id"])
        kwargs = {"source_catalog": None}
        if spec.data_id in (
            HFT_PANEL_DATA_ID,
            HFT_SPOT_PANEL_DATA_ID,
            HFT_SOL_PANEL_DATA_ID,
            HFT_DERIVATIVE_DATA_ID,
        ):
            if (
                not isinstance(source, Mapping)
                or "root" not in source
                or set(source) - {"root", "archive_paths"}
                or type(source["root"]) is not str
                or not source["root"]
            ):
                raise ValueError("native_source_request_invalid")
            paths = source.get("archive_paths")
            if paths is not None and (
                type(paths) is not list
                or not paths
                or any(type(p) is not str or not p for p in paths)
                or len(set(paths)) != len(paths)
            ):
                raise ValueError("native_source_archive_paths_invalid")
            opener = partial(
                open_market_source,
                archive_paths=None if paths is None else tuple(paths),
            )
            kwargs["raw_root"] = Path(source["root"])
            if spec.data_id == HFT_DERIVATIVE_DATA_ID:
                producer = crypto.produce_derivative_processed_data
                kwargs["open_archive"] = opener
            else:
                producer = crypto.produce_hft_processed_data
                kwargs.update(
                    open_run=opener,
                    reconstruct_depth=iter_reconstructed_depth,
                    read_trades=iter_agg_trades,
                    make_stats=ReconstructionStats,
                    panel_id=spec.data_id,
                    market_type=SPOT_MARKET_TYPE
                    if spec.data_id == HFT_SPOT_PANEL_DATA_ID
                    else FUTURES_MARKET_TYPE,
                    symbols=("SOLUSDT",)
                    if spec.data_id == HFT_SOL_PANEL_DATA_ID
                    else RESEARCH_SYMBOLS,
                )
        else:
            if spec.data_id in HFT_EXTERNAL_DATA_IDS.values():
                from quant_research_platform.sources.tardis.normalized import (
                    SCHEMA_ID,
                    iter_records,
                )

                if (
                    not isinstance(source, Mapping)
                    or set(source) != {"root", "archives"}
                    or type(source["root"]) is not str
                    or not source["root"]
                ):
                    raise ValueError("normalized_source_request_invalid")
                producer = crypto.produce_normalized_context
                kwargs.update(
                    raw_root=Path(source["root"]),
                    read_records=iter_records,
                    source_schema_id=SCHEMA_ID,
                    archives=source["archives"],
                )
            elif source is not None:
                raise ValueError("processed_source_not_consumed")
            else:
                producer = {
                    HFT_SPOT_CONTEXT_DATA_ID: crypto.produce_spot_perp_context,
                    HFT_INFORMATION_PANEL_DATA_ID: crypto.produce_information_panel,
                    HFT_CROSS_ASSET_DATA_ID: crypto.produce_cross_asset_context,
                }.get(spec.data_id, crypto.produce_hft_processed_data)
            if producer is crypto.produce_hft_processed_data:
                kwargs.update(
                    open_run=None,
                    reconstruct_depth=None,
                    read_trades=None,
                    make_stats=None,
                )
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
    with producer(
        spec, inputs=inputs, time_window=window, processed_catalog=catalog, **kwargs
    ) as (data, metadata):
        metadata["inputs"] = dict(config["inputs"])
        metadata["parameters"] = {
            "time_window": dict(config["time_window"]),
            **json.loads(spec.parameters_json),
        }
        write_dataset(output, data, spec=spec, metadata=metadata)
    return output


def _asset_ids(value, *, field):
    if (
        type(value) is not list
        or not value
        or any(
            type(item) is not str or item not in {"btc_usdt", "eth_usdt", "sol_usdt"}
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"data_operation_{field}_invalid")
    return value


def _configured_spec(spec, config):
    """Apply only options consumed by this explicitly requested mathematical recipe."""
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.markets.implementations.crypto import ids
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_DERIVATIVE_CHANNELS,
        derivative_channels,
    )

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
        ids.HFT_DERIVATIVE_DATA_ID: {"max_age_ms", "input_delay_ms"},
        ids.HFT_SPOT_CONTEXT_DATA_ID: {"max_age_ms", "input_delay_ms"},
        ids.HFT_CROSS_ASSET_DATA_ID: {
            "max_age_ms",
            "input_delay_ms",
            "information_assets",
            "target_assets",
        },
        ids.HFT_PANEL_DATA_ID: {"asset_ids"},
        ids.HFT_SPOT_PANEL_DATA_ID: {"asset_ids"},
        ids.HFT_SOL_PANEL_DATA_ID: {"asset_ids"},
        ids.HFT_INFORMATION_STATE_DATA_ID: {"asset_ids"},
        ids.HFT_FACTOR_DATA_ID: {"asset_ids"},
        ids.HFT_LABEL_DATA_ID: {"asset_ids"},
        **{
            name: {"max_age_ms", "input_delay_ms"}
            for name in ids.HFT_EXTERNAL_DATA_IDS.values()
        },
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
    if "input_delay_ms" in parameters and (
        type(parameters["input_delay_ms"]) is not int
        or parameters["input_delay_ms"] < 0
    ):
        raise ValueError("data_operation_input_delay_ms_invalid")
    if "max_age_ms" in parameters:
        ages = parameters["max_age_ms"]
        if spec.data_id == ids.HFT_DERIVATIVE_DATA_ID:
            if (
                type(ages) is not dict
                or not ages
                or not set(ages) <= set(HFT_DERIVATIVE_CHANNELS)
            ):
                raise ValueError("data_operation_max_age_ms_invalid")
            ages = list(ages.values())
        else:
            ages = [ages]
        if any(type(age) is not int or age <= 0 for age in ages):
            raise ValueError("data_operation_max_age_ms_invalid")
    for field in ("asset_ids", "information_assets", "target_assets"):
        if field in parameters:
            _asset_ids(parameters[field], field=field)
    actual = {**json.loads(spec.parameters_json), **parameters}
    if (
        spec.data_id
        in {
            ids.HFT_PANEL_DATA_ID,
            ids.HFT_SPOT_PANEL_DATA_ID,
            ids.HFT_SOL_PANEL_DATA_ID,
        }
        and "asset_ids" in actual
    ):
        supported = (
            {"sol_usdt"}
            if spec.data_id == ids.HFT_SOL_PANEL_DATA_ID
            else {"btc_usdt", "eth_usdt"}
        )
        if not set(actual["asset_ids"]) <= supported:
            raise ValueError("data_operation_asset_ids_invalid")
    if "max_age_ms" in dynamic:
        if "max_age_ms" not in actual:
            raise ValueError("data_operation_max_age_ms_required")
        actual.setdefault("input_delay_ms", 0)
    if spec.data_id == ids.HFT_CROSS_ASSET_DATA_ID and not {
        "information_assets",
        "target_assets",
    } <= set(actual):
        raise ValueError("data_operation_cross_asset_scopes_required")
    if "base_panel" in config:
        allowed = {
            ids.HFT_FACTOR_DATA_ID,
            ids.HFT_LABEL_DATA_ID,
            ids.HFT_INFORMATION_STATE_DATA_ID,
            ids.HFT_CROSS_ASSET_DATA_ID,
        }
        panel = config["base_panel"]
        if (
            spec.data_id not in allowed
            or type(panel) is not str
            or panel
            not in {
                ids.HFT_PANEL_DATA_ID,
                ids.HFT_SOL_PANEL_DATA_ID,
                ids.HFT_INFORMATION_PANEL_DATA_ID,
            }
        ):
            raise ValueError("data_operation_base_panel_invalid")
        spec = replace(
            spec, processed_input_data_ids=(panel, *spec.processed_input_data_ids[1:])
        )
    if spec.data_id in ids.HFT_EXTERNAL_DATA_IDS.values():
        from quant_research_platform.markets.implementations.crypto.processed.producer import (  # noqa: E501
            EXTERNAL_INSTRUMENTS,
        )

        venue = next(
            key
            for key, value in ids.HFT_EXTERNAL_DATA_IDS.items()
            if value == spec.data_id
        )
        instrument = EXTERNAL_INSTRUMENTS[venue]
        actual.update(
            venue=venue,
            symbols=instrument["symbols"],
            base_units_per_amount=instrument["base_units_per_amount"],
        )
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
            if spec.data_id == ids.HFT_DERIVATIVE_DATA_ID:
                channels = derivative_channels(factors)
                roles = {
                    name: role
                    for name, role in roles.items()
                    if name in spec.primary_key
                    or any(name.startswith(channel + "_") for channel in channels)
                }
            elif spec.data_id in ids.HFT_EXTERNAL_DATA_IDS.values():
                from quant_research_platform.research.protocols.crypto_hft import (
                    external_channels,
                )

                channels = external_channels(factors)
                prefixes = tuple(kind.removesuffix("s") + "_" for kind in channels)
                roles = {
                    name: role
                    for name, role in roles.items()
                    if name in spec.primary_key
                    or any(
                        name.split("_", 1)[1].startswith(prefix) for prefix in prefixes
                    )
                }
            elif spec.data_id == ids.HFT_CROSS_ASSET_DATA_ID:
                assets = {name.split("_", 1)[0] for name in factors}
                roles = {
                    name: role
                    for name, role in roles.items()
                    if name in spec.primary_key or name.split("_", 1)[0] in assets
                }
            # Scientific input order is explicit; supporting keys never become factors.
            roles.update((name, ColumnRole.FACTOR) for name in factors)
            spec = replace(spec, column_roles=roles)
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
        from quant_research_platform.research.protocols import a_share, crypto_hft

        definitions = {
            item.output_column: item
            for item in (
                a_share.label_catalog()
                if spec.market_id == "a_share"
                else crypto_hft.label_catalog()
            ).list()
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
    if spec.data_id == ids.HFT_DERIVATIVE_DATA_ID and not set(
        derivative_channels(spec.required_columns)
    ) <= set(actual["max_age_ms"]):
        raise ValueError("data_operation_max_age_ms_invalid")
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
