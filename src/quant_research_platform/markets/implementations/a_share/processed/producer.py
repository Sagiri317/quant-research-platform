"""A-share processed target producers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog as a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    compute_market_panel,
)
from quant_research_platform.research.factors import engine as factor_engine
from quant_research_platform.research.factors.expectations.calculations import (
    build_expectations_features,
)
from quant_research_platform.research.factors.fundamental.calculations import (
    construct_ttm_events,
)
from quant_research_platform.research.labels import engine as label_engine
from quant_research_platform.research.protocols import a_share
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.layout import StorageLayout

_REFERENCE_SOURCE_IDS = frozenset(
    {
        "stock_master",
        "market_cap",
        "industry_classification",
        "industry_classification_sets",
        "industry_classification_versions",
        "listing_status_changes",
        "stock_info_changes",
        "stock_info_annual",
        "trading_no_limit",
        "special_treatment_changes",
    }
)


def produce_a_share_processed_frame(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    processed_catalog: ProcessedDataCatalog | None = None,
    source_catalog: InterimDataCatalog,
    resource_policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    processed = processed_catalog or a_share_processed_catalog()
    sources = source_catalog
    source = {
        data_id: _read_source(
            layout,
            sources.get(data_id),
            time_window=time_window,
            policy=resource_policy,
            optional=data_id in _REFERENCE_SOURCE_IDS,
        )
        for data_id in spec.source_input_data_ids
        if _source_available(layout, sources.get(data_id))
        or data_id in _REFERENCE_SOURCE_IDS
    }
    dependencies = {
        data_id: _read_processed(
            layout,
            processed.get(data_id),
            time_window=time_window,
            policy=resource_policy,
        )
        for data_id in spec.processed_input_data_ids
    }
    if spec.data_id == "market_panel_1d":
        required = {"market_bars_1d", "stock_master", "market_cap", "trading_calendar"}
        if not required.issubset(source):
            raise ValueError("a_share_panel_source_data_missing")
        return compute_market_panel(
            source["market_bars_1d"],
            stock_master=source["stock_master"],
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
            industry_history=source.get("industry_classification"),
            special_treatment_history=source.get("special_treatment_changes"),
        ).loc[:, list(spec.required_columns)]
    if spec.data_id == "execution_mask":
        return derive_execution_mask(dependencies["market_panel_1d"]).loc[
            :, list(spec.required_columns)
        ]
    if spec.data_id == "fundamental_pit":
        return _fundamental_pit_frame(
            spec,
            source=source,
            panel=dependencies.get("market_panel_1d"),
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        definitions = tuple(
            definition
            for definition in a_share.factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
        if "expectations" in {definition.family for definition in definitions}:
            analyst = source.get("analyst_forecast")
            if analyst is None or analyst.empty:
                raise ValueError("BLOCKED_EXTERNAL_DATA: analyst_forecast")
            records = analyst.rename(columns={"source_asset_id": "asset_id"})
            panel = dependencies["market_panel_1d"]
            features = build_expectations_features(
                records,
                panel.loc[:, ["asset_id", "timestamp", "close"]],
                trading_sessions=tuple(sorted(panel["timestamp"].unique())),
            )
            return (
                panel.loc[:, ["market_id", "asset_id", "timestamp"]]
                .merge(
                    features,
                    on=["asset_id", "timestamp"],
                    how="left",
                    validate="one_to_one",
                )
                .loc[:, list(spec.required_columns)]
            )
        input_frame = dependencies[next(iter(spec.processed_input_data_ids))]
        if any(definition.family == "fundamental" for definition in definitions):
            market = next(
                frame
                for frame in dependencies.values()
                if "total_market_cap" in frame.columns
            ).loc[:, ["market_id", "asset_id", "timestamp", "total_market_cap"]]
            input_frame = input_frame.merge(
                market,
                on=["market_id", "asset_id", "timestamp"],
                how="left",
                validate="one_to_one",
            )
        return factor_engine.calculate_factors_for_panel(
            input_frame,
            definitions=definitions,
        ).loc[:, list(spec.required_columns)]
    if spec.kind is ProcessedDataKind.LABEL:
        panel = dependencies["market_panel_1d"].merge(
            dependencies["execution_mask"],
            on=["market_id", "asset_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        definitions = a_share.label_catalog().list()
        sessions = tuple(sorted(pd.to_datetime(panel["timestamp"]).unique()))
        return label_engine.calculate_labels_for_panel(
            panel,
            definitions=definitions,
            sessions=sessions,
        ).loc[:, list(spec.required_columns)]
    raise ValueError(f"unsupported_a_share_processed_kind: {spec.kind}")


def _fundamental_pit_frame(
    spec: ProcessedDataSpec,
    *,
    source: dict[str, pd.DataFrame],
    panel: pd.DataFrame | None,
) -> pd.DataFrame:
    if panel is None:
        raise ValueError("fundamental_pit_market_panel_required")
    required = {
        "financial_balance",
        "financial_income",
        "financial_cashflow",
        "financial_first_disclosure",
    }
    if not required.issubset(source):
        raise ValueError("BLOCKED_EXTERNAL_DATA: financial_first_disclosure")
    calendar = source.get("trading_calendar")
    if calendar is None:
        raise ValueError("fundamental_pit_calendar_required")
    sessions = pd.DatetimeIndex(
        calendar.loc[calendar["is_trading_day"], "calendar_date"]
    )
    events = financial_events_from_interim(
        {
            data_id: source[data_id]
            for data_id in (
                "financial_balance",
                "financial_income",
                "financial_cashflow",
            )
        },
        source["financial_first_disclosure"],
        trading_sessions=sessions,
    )
    ttm_events = construct_ttm_events(events)
    flow_metrics = {
        "operating_revenue",
        "operating_cost",
        "consolidated_net_income",
        "parent_net_income",
        "operating_cash_flow",
        "capex_cash_paid",
    }
    events = events.copy()
    events["value"] = events["value"].where(
        ~events["metric_id"].isin(flow_metrics), ttm_events["ttm_value"]
    )
    pit = materialize_pit_fundamentals(
        panel.loc[:, ["asset_id", "timestamp"]],
        events,
    )
    values = pit.pivot_table(
        index=["asset_id", "timestamp"],
        columns="metric_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    previous = (
        pit.loc[:, ["asset_id", "timestamp", "metric_id", "previous_value"]]
        .pivot_table(
            index=["asset_id", "timestamp"],
            columns="metric_id",
            values="previous_value",
            aggfunc="first",
        )
        .reset_index()
    )
    previous = previous.rename(
        columns={
            column: f"previous_{column}"
            for column in previous.columns
            if column not in {"asset_id", "timestamp"}
        }
    )
    output = (
        panel.loc[:, ["market_id", "asset_id", "timestamp"]]
        .merge(values, on=["asset_id", "timestamp"], how="left", validate="one_to_one")
        .merge(
            previous,
            on=["asset_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )
    )
    output["average_total_assets"] = (
        output["total_assets"] + output["previous_total_assets"]
    ) / 2.0
    output["average_parent_equity"] = (
        output["parent_equity"] + output["previous_parent_equity"]
    ) / 2.0
    for column in (
        "operating_revenue",
        "operating_cost",
        "consolidated_net_income",
        "parent_net_income",
        "operating_cash_flow",
        "capex_cash_paid",
    ):
        output[f"{column}_ttm"] = output[column]
    for column in spec.required_columns:
        if column not in output:
            output[column] = np.nan
    return output.loc[:, list(spec.required_columns)]


def _source_available(layout: StorageLayout, spec: Any) -> bool:
    return layout.interim_data_root(spec.source_id, spec.data_id).exists()


def _read_source(
    layout: StorageLayout,
    spec: Any,
    *,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    optional: bool,
) -> pd.DataFrame:
    root = layout.interim_data_root(spec.source_id, spec.data_id)
    if not root.exists() and optional:
        return pd.DataFrame(columns=spec.required_columns)
    if not root.exists():
        raise ValueError(f"missing_interim_data: {spec.data_id}")
    read_window = None if spec.data_id in _REFERENCE_SOURCE_IDS else time_window
    return read_dataset(root, time_window=read_window, resource_policy=policy)


def _read_processed(
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    *,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    dataset_root = layout.processed_data_root(
        spec.market_id,
        spec.kind,
        spec.data_id,
    )
    if not dataset_root.exists():
        raise ValueError(f"missing_processed_dependency: {spec.data_id}")
    return read_dataset(dataset_root, time_window=time_window, resource_policy=policy)
