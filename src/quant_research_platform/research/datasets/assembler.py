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
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets import (
    DatasetAccessMode,
    _data_spec_payload,
    read_dataset,
    write_dataset_contents,
)
from quant_research_platform.storage.layout import (
    layout_from_processed_market_root,
    processed_data_parquet_root,
)
from quant_research_platform.storage.manifests import (
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.parquet import write_research_dataset


def assemble_research_dataset(
    frame: pd.DataFrame,
    catalog: ResearchDatasetCatalog,
    dataset_id: str,
    processed_root: Path,
) -> Path:
    spec = catalog.get(dataset_id)
    validate_research_dataset(frame, spec)
    output = frame[list(spec.required_columns)].copy()
    return write_research_dataset(
        output,
        layout=layout_from_processed_market_root(processed_root),
        catalog=catalog,
        dataset_id=dataset_id,
    )


def assemble_research_dataset_from_processed(
    *,
    processed_root: Path,
    dataset_catalog: ResearchDatasetCatalog,
    processed_catalog: ProcessedDataCatalog,
    dataset_id: str,
    years: tuple[int, ...] | None = None,
) -> Path:
    spec = dataset_catalog.get(dataset_id)
    artifact_root = Path(processed_root) / "research_datasets" / dataset_id
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

    def _write(temp_root: Path) -> None:
        accumulator = write_dataset_contents(
            _chunks(),
            temp_root / "data",
            catalog=dataset_catalog,
            data_id=dataset_id,
            access_mode=DatasetAccessMode.STREAMING,
        )
        write_internal_artifact_manifest(
            temp_root / "spec.json",
            _data_spec_payload(spec),
        )
        write_internal_artifact_manifest(
            temp_root / "_manifest" / "manifest.json",
            accumulator.payload(),
        )

    atomic_write_directory(artifact_root, _write)
    return artifact_root / "data"


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
    data_root = processed_data_parquet_root(
        processed_root,
        processed_spec.category,
        processed_spec.processed_data_id,
    )
    frame = read_dataset(
        data_root,
        catalog=processed_catalog,
        data_id=processed_spec.processed_data_id,
        years=years,
        columns=columns,
        access_mode=DatasetAccessMode.MATERIALIZED,
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
    data_root = processed_data_parquet_root(
        processed_root,
        first_spec.category,
        first_spec.processed_data_id,
    )
    discovered = []
    for path in sorted(data_root.glob("year=*")):
        if path.is_dir() and list(path.glob("*.parquet")):
            discovered.append(int(path.name.split("=", 1)[1]))
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


__all__ = [
    "assemble_research_dataset",
    "assemble_research_dataset_from_processed",
    "read_research_dataset",
    "validate_research_dataset",
]
