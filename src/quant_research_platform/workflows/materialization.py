"""The single processed target-DAG planner and executor."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import MarketId
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
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.datasets.inspection import _inspect_dataset_owned
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import require_artifact_root
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class MaterializationPlan:
    requested_targets: tuple[str, ...]
    transitive_dependencies: tuple[str, ...]
    source_groups: dict[str, tuple[str, ...]]
    reuse: tuple[str, ...]
    rebuild: tuple[str, ...]
    blocked: tuple[str, ...]
    coverage: dict[str, str]


def source_groups_for_data_ids(
    catalog: InterimDataCatalog,
    data_ids: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Resolve each data id through its one InterimDataSpec owner."""

    groups: dict[str, list[str]] = {}
    for data_id in data_ids:
        spec = catalog.get(data_id)
        values = groups.setdefault(str(spec.source_id), [])
        if data_id not in values:
            values.append(data_id)
    return {source_id: tuple(values) for source_id, values in groups.items()}


def build_materialization_plan(
    *,
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    interim_catalog: InterimDataCatalog,
    market_id: MarketId,
    targets: Iterable[str] | None = None,
    kinds: frozenset[ProcessedDataKind] | None = None,
    rebuild: bool = False,
) -> MaterializationPlan:
    """Expand the unique processed DAG and classify existing artifact roots."""

    if (targets is None) == (kinds is None):
        raise ValueError("materialization_targets_xor_kinds_required")
    if targets is not None:
        requested = tuple(targets)
        specs = tuple(processed_catalog.get(data_id) for data_id in requested)
    else:
        assert kinds is not None
        specs = tuple(spec for spec in processed_catalog.list() if spec.kind in kinds)
        requested = tuple(spec.data_id for spec in specs)
    if not specs:
        raise ValueError("no_materialization_targets")
    expanded: dict[str, ProcessedDataSpec] = {}
    visiting: set[str] = set()

    def visit(spec: ProcessedDataSpec) -> None:
        if spec.data_id in expanded:
            return
        if spec.data_id in visiting:
            raise ValueError(f"processed_dependency_cycle: {spec.data_id}")
        visiting.add(spec.data_id)
        for dependency_id in spec.processed_input_data_ids:
            visit(processed_catalog.get(dependency_id))
        visiting.remove(spec.data_id)
        expanded[spec.data_id] = spec

    for spec in specs:
        if spec.market_id != market_id:
            raise ValueError("materialization_market_id_mismatch")
        for source_data_id in spec.source_input_data_ids:
            interim_catalog.get(source_data_id)
        visit(spec)
    source_ids = (
        source_data_id
        for spec in expanded.values()
        for source_data_id in spec.source_input_data_ids
    )
    grouped_sources = source_groups_for_data_ids(interim_catalog, source_ids)
    source_groups = {
        source_id: list(
            data_id
            for data_id in expanded
            if any(
                source_data_id in grouped_sources[source_id]
                for source_data_id in expanded[data_id].source_input_data_ids
            )
        )
        for source_id in grouped_sources
    }
    reuse: list[str] = []
    rebuild_ids: list[str] = []
    coverage: dict[str, str] = {}
    for data_id, spec in expanded.items():
        root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
        if root.exists():
            (rebuild_ids if rebuild else reuse).append(data_id)
            coverage[data_id] = "existing"
        else:
            coverage[data_id] = "missing"
    return MaterializationPlan(
        requested_targets=requested,
        transitive_dependencies=tuple(
            data_id for data_id in expanded if data_id not in requested
        ),
        source_groups={key: tuple(value) for key, value in source_groups.items()},
        reuse=tuple(reuse),
        rebuild=tuple(rebuild_ids),
        blocked=(),
        coverage=coverage,
    )


def materialize_processed_data(
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    data_ids: tuple[str, ...] | None = None,
    kinds: frozenset[ProcessedDataKind] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
    rebuild: bool = False,
    market_id: MarketId | str = "a_share",
) -> dict[str, DatasetInspection]:
    """Materialize or reuse a market's targets through this workflow owner."""

    normalized_market = MarketId(market_id)
    with runtime_resource_policy_scope(resource_policy) as policy:
        if normalized_market == MarketId("a_share"):
            return _materialize_a_share(
                layout=layout,
                time_window=time_window,
                data_ids=data_ids,
                kinds=kinds,
                policy=policy,
                rebuild=rebuild,
            )
        if normalized_market == MarketId("crypto"):
            raise ValueError("crypto_hft_materialization_requires_recorder_run")
    raise ValueError("unsupported_materialization_market")


def _materialize_a_share(
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    data_ids: tuple[str, ...] | None,
    kinds: frozenset[ProcessedDataKind] | None,
    policy: RuntimeResourcePolicy,
    rebuild: bool,
) -> dict[str, DatasetInspection]:
    processed = a_share_processed_catalog()
    sources = InterimDataCatalog(
        (*__import_csmar_catalog().list(), *__import_choice_catalog().list())
    )
    plan = build_materialization_plan(
        layout=layout,
        processed_catalog=processed,
        interim_catalog=sources,
        market_id=MarketId("a_share"),
        targets=data_ids,
        kinds=kinds,
        rebuild=rebuild,
    )
    if rebuild and plan.rebuild:
        raise ValueError("rebuild_requires_new_content_addressed_target")
    requested = set(plan.requested_targets)
    written: dict[str, DatasetInspection] = {}
    for data_id in (*plan.transitive_dependencies, *plan.requested_targets):
        spec = processed.get(data_id)
        root = require_artifact_root(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        )
        if root.exists():
            inspection = _inspect_dataset(root, policy)
            if data_id in requested or data_id not in written:
                written[data_id] = inspection
            continue
        frame = _produce_a_share_frame(
            spec,
            layout=layout,
            time_window=time_window,
            processed_catalog=processed,
            source_catalog=sources,
            written=written,
            policy=policy,
        )
        write_dataset(
            root,
            frame,
            spec=spec,
            resource_policy=policy,
            borrowed_input=True,
        )
        written[data_id] = _inspect_dataset(root, policy)
    return written


def _produce_a_share_frame(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    processed_catalog: ProcessedDataCatalog,
    source_catalog: InterimDataCatalog,
    written: dict[str, DatasetInspection],
    policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    source = {
        data_id: _read_source(
            layout,
            source_catalog.get(data_id),
            time_window=time_window,
            policy=policy,
            optional=data_id
            in {
                "industry_classification",
                "industry_classification_sets",
                "industry_classification_versions",
                "listing_status_changes",
                "stock_info_changes",
                "stock_info_annual",
                "trading_no_limit",
                "special_treatment_changes",
            },
        )
        for data_id in spec.source_input_data_ids
        if _source_available(layout, source_catalog.get(data_id))
        or data_id
        in {
            "industry_classification",
            "industry_classification_sets",
            "industry_classification_versions",
            "listing_status_changes",
            "stock_info_changes",
            "stock_info_annual",
            "trading_no_limit",
            "special_treatment_changes",
        }
    }
    dependencies = {
        data_id: _read_processed(
            layout,
            processed_catalog.get(data_id),
            time_window=time_window,
            policy=policy,
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
            policy=policy,
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
    policy: RuntimeResourcePolicy,
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
            previous, on=["asset_id", "timestamp"], how="left", validate="one_to_one"
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


def _source_available(layout: StorageLayout, spec) -> bool:
    return layout.interim_data_root(spec.source_id, spec.data_id).exists()


def _read_source(
    layout: StorageLayout,
    spec,
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
    reference_ids = {
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
    read_window = None if spec.data_id in reference_ids else time_window
    return read_dataset(root, time_window=read_window, resource_policy=policy)


def _read_processed(
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    *,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    if not root.exists():
        raise ValueError(f"missing_processed_dependency: {spec.data_id}")
    return read_dataset(root, time_window=time_window, resource_policy=policy)


def _inspect_dataset(root: Path, policy: RuntimeResourcePolicy) -> DatasetInspection:
    owned = _inspect_dataset_owned(root, resource_policy=policy)
    try:
        return owned.value
    finally:
        owned.release(owner="workflow.materialization.reused.inspection")


def __import_csmar_catalog():
    from quant_research_platform.sources.csmar.interim import interim_catalog

    return interim_catalog()


def __import_choice_catalog():
    from quant_research_platform.sources.choice.interim import interim_catalog

    return interim_catalog()
