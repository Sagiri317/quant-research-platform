"""Minimal ResearchDataset assembly for QRP 0.1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.research.datasets.io import read_research_dataset
from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import (
    layout_from_processed_market_root,
)


def assemble_research_dataset(
    frame: pd.DataFrame,
    catalog: ResearchDatasetCatalog,
    dataset_id: str,
    processed_root: Path,
) -> Path:
    spec = catalog.get(dataset_id)
    validate_research_dataset(frame, spec)
    output = frame[list(spec.required_columns)].copy()
    layout = layout_from_processed_market_root(processed_root)
    artifact_root = layout.research_dataset_root(spec.market_id, spec.dataset_id)
    write_dataset(artifact_root, output, spec=spec)
    return artifact_root


def assemble_research_dataset_from_processed(
    *,
    processed_root: Path,
    dataset_catalog: ResearchDatasetCatalog,
    processed_catalog: ProcessedDataCatalog,
    dataset_id: str,
    years: tuple[int, ...] | None = None,
) -> Path:
    spec = dataset_catalog.get(dataset_id)
    layout = layout_from_processed_market_root(processed_root)
    artifact_root = layout.research_dataset_root(spec.market_id, spec.dataset_id)
    assembly_years = _assembly_years(
        processed_root=processed_root,
        dataset_spec=spec,
        processed_catalog=processed_catalog,
        years=years,
    )

    def _chunks():
        for year in assembly_years:
            frame = _join_processed_inputs(
                processed_root=processed_root,
                dataset_spec=spec,
                processed_catalog=processed_catalog,
                years=(year,),
            )
            validate_research_dataset(frame, spec)
            yield frame.loc[:, list(spec.required_columns)].copy()

    write_dataset(artifact_root, _chunks(), spec=spec)
    return artifact_root


def _join_processed_inputs(
    *,
    processed_root: Path,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    years: tuple[int, ...] | None,
) -> pd.DataFrame:
    assembled: pd.DataFrame | None = None
    for processed_data_id in dataset_spec.processed_data_ids:
        processed_spec = processed_catalog.get(processed_data_id)
        input_frame = _read_processed_input(
            processed_root=processed_root,
            dataset_spec=dataset_spec,
            processed_catalog=processed_catalog,
            processed_spec=processed_spec,
            existing_columns=tuple(assembled.columns) if assembled is not None else (),
            years=years,
        )
        if assembled is None:
            assembled = input_frame
            continue
        assembled = assembled.merge(
            input_frame,
            on=list(dataset_spec.primary_key),
            how="inner",
            validate="one_to_one",
        )
    if assembled is None:
        raise ValueError("empty_research_dataset_inputs")
    return assembled.loc[:, list(dataset_spec.required_columns)]


def _read_processed_input(
    *,
    processed_root: Path,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    processed_spec: ProcessedDataSpec,
    existing_columns: tuple[str, ...],
    years: tuple[int, ...] | None,
) -> pd.DataFrame:
    columns = _processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=existing_columns,
    )
    data_root = layout_from_processed_market_root(processed_root).processed_data_root(
        processed_spec.market_id,
        processed_spec.category,
        processed_spec.processed_data_id,
    )
    frame = read_dataset(
        data_root,
        partition_filter=_year_filter(years),
        columns=columns,
        mode=DatasetReadMode.MATERIALIZED,
    )
    if "year" in frame.columns:
        frame = frame.drop(columns=["year"])
    return frame


def _assembly_years(
    *,
    processed_root: Path,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    years: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if years is not None:
        return years
    first_processed_id = dataset_spec.processed_data_ids[0]
    first_spec = processed_catalog.get(first_processed_id)
    data_root = layout_from_processed_market_root(processed_root).processed_data_root(
        first_spec.market_id,
        first_spec.category,
        first_spec.processed_data_id,
    )
    manifest = inspect_dataset(data_root).manifest
    discovered = [
        int(partition.split("=", 1)[1])
        for partition in manifest.get("partitions", [])
        if str(partition).startswith("year=")
    ]
    if not discovered:
        raise FileNotFoundError(data_root)
    return tuple(discovered)


def _processed_input_columns(
    dataset_spec: ResearchDatasetSpec,
    processed_spec: ProcessedDataSpec,
    *,
    existing_columns: tuple[str, ...],
) -> tuple[str, ...]:
    existing = set(existing_columns)
    output_columns = set(dataset_spec.required_columns)
    columns: list[str] = []
    for column in dataset_spec.primary_key:
        columns.append(column)
    for column in processed_spec.required_columns:
        if column in dataset_spec.primary_key:
            continue
        if column not in output_columns:
            continue
        if column in existing:
            continue
        columns.append(column)
    return tuple(columns)


def _year_filter(years: tuple[int, ...] | None) -> dict[str, tuple[int, ...]] | None:
    if years is None:
        return None
    return {"year": years}


__all__ = [
    "assemble_research_dataset",
    "assemble_research_dataset_from_processed",
    "read_research_dataset",
    "validate_research_dataset",
]
