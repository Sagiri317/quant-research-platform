"""A-share processed materialization validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
    market_panel_data_id,
)


@dataclass(frozen=True)
class _ProcessedInputContract:
    required_input_data_ids: tuple[str, ...]
    primary_input_data_id: str


_MARKET_PANEL_INPUT_CONTRACT = _ProcessedInputContract(
    required_input_data_ids=(market_panel_data_id(),),
    primary_input_data_id=market_panel_data_id(),
)
_FACTOR_FAMILY_INPUT_CONTRACTS = {
    "price_volume": _MARKET_PANEL_INPUT_CONTRACT,
    "fundamental": _ProcessedInputContract(
        required_input_data_ids=(FUNDAMENTAL_PIT_PROCESSED_DATA_ID,),
        primary_input_data_id=FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
    ),
}
_LABEL_FAMILY_INPUT_CONTRACTS = {
    "forward_return": _MARKET_PANEL_INPUT_CONTRACT,
}


def validate_processed_dependency_graph(
    processed_catalog: ProcessedDataCatalog,
    targets: tuple[ProcessedDataSpec, ...],
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(spec: ProcessedDataSpec) -> None:
        if spec.data_id in visited:
            return
        if spec.data_id in visiting:
            raise ValueError(f"processed_dependency_cycle: {spec.data_id}")
        visiting.add(spec.data_id)
        for dependency_id in spec.processed_input_data_ids:
            try:
                dependency = processed_catalog.get(dependency_id)
            except KeyError as exc:
                raise ValueError(
                    f"unknown_processed_dependency: {dependency_id}"
                ) from exc
            visit(dependency)
        visiting.remove(spec.data_id)
        visited.add(spec.data_id)

    for target in targets:
        visit(target)


def validate_active_processed_profile_frequencies(
    processed_catalog: ProcessedDataCatalog,
    targets: tuple[ProcessedDataSpec, ...],
) -> None:
    visited: set[str] = set()

    def visit(spec: ProcessedDataSpec) -> None:
        if spec.data_id in visited:
            return
        visited.add(spec.data_id)
        if spec.frequency is not DataFrequency.DAILY:
            raise ValueError(
                "unsupported_processed_frequency: "
                f"{spec.data_id}: {spec.frequency.value}"
            )
        for dependency_id in spec.processed_input_data_ids:
            visit(processed_catalog.get(dependency_id))

    for target in targets:
        visit(target)


def annotation_input_contract(spec: ProcessedDataSpec) -> _ProcessedInputContract:
    if spec.kind is not ProcessedDataKind.ANNOTATION:
        raise ValueError(f"invalid_annotation_spec_kind: {spec.data_id}")
    return _MARKET_PANEL_INPUT_CONTRACT


def factor_input_contract(
    definitions: Sequence[FactorDefinition],
) -> _ProcessedInputContract:
    families = tuple(dict.fromkeys(definition.family for definition in definitions))
    if not families:
        return _MARKET_PANEL_INPUT_CONTRACT
    if len(families) > 1:
        raise ValueError("mixed_factor_families: " + ", ".join(families))
    try:
        return _FACTOR_FAMILY_INPUT_CONTRACTS[families[0]]
    except KeyError as exc:
        raise ValueError(f"unknown_factor_family: {families[0]}") from exc


def label_input_contract(
    definitions: Sequence[LabelDefinition],
) -> _ProcessedInputContract:
    families = tuple(dict.fromkeys(definition.family for definition in definitions))
    if not families:
        return _MARKET_PANEL_INPUT_CONTRACT
    if len(families) > 1:
        raise ValueError("mixed_label_families: " + ", ".join(families))
    try:
        return _LABEL_FAMILY_INPUT_CONTRACTS[families[0]]
    except KeyError as exc:
        raise ValueError(f"unknown_label_family: {families[0]}") from exc


def validate_processed_input_contract(
    spec: ProcessedDataSpec,
    *,
    contract: _ProcessedInputContract,
    purpose: str,
    processed_inputs: dict[str, pd.DataFrame] | None = None,
) -> None:
    declared = tuple(spec.processed_input_data_ids)
    required = tuple(contract.required_input_data_ids)
    available = tuple(processed_inputs) if processed_inputs is not None else declared
    missing = [data_id for data_id in required if data_id not in available]
    if missing:
        raise ValueError(
            f"missing_processed_input_for_{purpose}: {spec.data_id}: "
            + ", ".join(missing)
        )
    unexpected = [data_id for data_id in available if data_id not in required]
    if unexpected:
        raise ValueError(
            f"unexpected_processed_input_for_{purpose}: {spec.data_id}: "
            + ", ".join(unexpected)
        )
    if contract.primary_input_data_id not in required:
        raise ValueError(
            f"invalid_primary_processed_input_for_{purpose}: {spec.data_id}: "
            f"{contract.primary_input_data_id}"
        )


def require_source_inputs_for_spec(
    spec: ProcessedDataSpec,
    source_inputs: dict[str, pd.DataFrame],
) -> None:
    if spec.kind is not ProcessedDataKind.PANEL:
        return
    missing = [
        data_id
        for data_id in ("stock_master", "market_cap", "trading_calendar")
        if data_id not in source_inputs
    ]
    if missing:
        raise ValueError("missing_source_input_for_panel: " + ", ".join(missing))


def require_source_input_columns(
    data_id: str,
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> None:
    require_source_input_columns_from_schema(data_id, frame.columns, columns=columns)


def require_source_input_columns_from_schema(
    data_id: str,
    available_columns: Iterable[str],
    *,
    columns: tuple[str, ...],
) -> None:
    available = set(available_columns)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            f"missing_source_input_columns: {data_id}: " + ", ".join(missing)
        )
