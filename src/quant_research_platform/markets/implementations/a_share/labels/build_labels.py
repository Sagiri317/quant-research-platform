"""Build split A-share forward-return label processed data from panels."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_label_catalog,
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    DAILY_PANEL_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)
LABEL_PANEL_INPUT_COLUMNS: tuple[str, ...] = (*LABEL_BASE_COLUMNS, "adj_open")


def calculate_panel_labels(
    panel: pd.DataFrame,
    *,
    horizons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    label_horizons = horizons or tuple(
        definition.horizon for definition in a_share_label_catalog().list()
    )
    return add_forward_return_labels(
        panel,
        horizons=label_horizons,
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
    )


def select_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    spec = a_share_processed_catalog().get("forward_returns")
    missing = [
        column for column in spec.required_columns if column not in frame.columns
    ]
    if missing:
        raise ValueError("missing_label_output_columns: " + ", ".join(missing))
    return frame[list(spec.required_columns)].copy()


def write_forward_labels(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    labels = select_label_columns(frame)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    mask = labels["timestamp"].dt.year.between(start_year, end_year)
    labels = labels.loc[mask].copy()
    labels = labels.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )
    catalog = a_share_processed_catalog()
    spec = catalog.get("forward_returns")
    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(artifact_root, labels, spec=spec)
    return artifact_root, len(labels)


def calculate_forward_returns_full(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    panel_frame: pd.DataFrame | None = None,
) -> tuple[Path, int]:
    if panel_frame is None:
        return calculate_forward_returns_asset_batches(
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
        )

    panel = _panel_frame_for_label_build(
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
        panel_frame=panel_frame,
    )
    labels = calculate_panel_labels(panel)
    return write_forward_labels(
        labels,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )


def calculate_forward_returns_asset_batches(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    asset_chunk_size: int | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[Path, int]:
    """Calculate forward-return labels in asset batches with future context."""

    catalog = a_share_processed_catalog()
    spec = catalog.get("forward_returns")
    data_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        "panels",
        DAILY_PANEL_PROCESSED_DATA_ID,
    )
    available_columns = set(inspect_dataset(data_root).columns)
    missing = [
        column
        for column in LABEL_PANEL_INPUT_COLUMNS
        if column not in available_columns
    ]
    if missing:
        raise ValueError("missing_label_panel_input_columns: " + ", ".join(missing))
    input_years = _available_panel_years(
        data_root=data_root,
        start_year=start_year,
        end_year=end_year + 1,
    )
    asset_ids = _panel_asset_ids(data_root, years=input_years)
    chunk_size = asset_chunk_size or _planned_asset_chunk_size(
        asset_ids=asset_ids,
        data_root=data_root,
        years=input_years,
        resource_policy=resource_policy,
    )
    row_count = 0

    def _output_chunks() -> Iterator[pd.DataFrame]:
        nonlocal row_count
        for asset_chunk in _chunked(asset_ids, chunk_size):
            panel = _read_panel_asset_chunk(
                data_root,
                years=input_years,
                asset_ids=asset_chunk,
            )
            if panel.empty:
                continue
            labels = calculate_panel_labels(panel)
            output = select_label_columns(labels)
            output["timestamp"] = pd.to_datetime(output["timestamp"])
            mask = output["timestamp"].dt.year.between(start_year, end_year)
            output = output.loc[mask].copy()
            output = output.sort_values(list(spec.primary_key)).reset_index(drop=True)
            if output.empty:
                continue
            row_count += len(output)
            yield output

    artifact_root = Path(processed_root) / "labels" / "forward_returns"
    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(
        artifact_root,
        _output_chunks(),
        spec=spec,
    )
    return artifact_root, row_count


def select_existing_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    wanted_unique = list(dict.fromkeys(wanted))
    columns = inspect_dataset(path).columns
    return [column for column in wanted_unique if column in columns]


def _panel_frame_for_label_build(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    panel_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if panel_frame is None:
        data_root = layout_from_processed_root(processed_root).processed_data_root(
            a_share_processed_catalog().get(DAILY_PANEL_PROCESSED_DATA_ID).market_id,
            "panels",
            DAILY_PANEL_PROCESSED_DATA_ID,
        )
        available_columns = set(inspect_dataset(data_root).columns)
        missing = [
            column
            for column in LABEL_PANEL_INPUT_COLUMNS
            if column not in available_columns
        ]
        if missing:
            raise ValueError(
                "missing_label_panel_input_columns: " + ", ".join(missing)
            )
        years = _available_panel_years(
            data_root=data_root,
            start_year=start_year,
            end_year=end_year + 1,
        )
        return read_dataset(
            data_root,
            partition_filter=_year_filter(years),
            columns=list(LABEL_PANEL_INPUT_COLUMNS),
        )
    missing = [
        column
        for column in LABEL_PANEL_INPUT_COLUMNS
        if column not in panel_frame.columns
    ]
    if missing:
        raise ValueError("missing_label_panel_input_columns: " + ", ".join(missing))
    panel = panel_frame.loc[:, list(LABEL_PANEL_INPUT_COLUMNS)].copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    return panel.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )


def _available_panel_years(
    *,
    data_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[int, ...]:
    years = tuple(
        year
        for year in range(start_year, end_year + 1)
        if _partition_exists(data_root, year)
    )
    if not years:
        raise ValueError("missing_requested_processed_input")
    return years


def _panel_asset_ids(data_root: Path, *, years: tuple[int, ...]) -> tuple[str, ...]:
    asset_ids: set[str] = set()
    chunks = read_dataset(
        data_root,
        partition_filter=_year_filter(years),
        columns=["asset_id"],
        mode=DatasetReadMode.CHUNKED,
    )
    for chunk in chunks:
        asset_ids.update(str(value) for value in chunk["asset_id"].dropna().unique())
    if not asset_ids:
        raise ValueError("missing_requested_processed_input")
    return tuple(sorted(asset_ids))


def _read_panel_asset_chunk(
    data_root: Path,
    *,
    years: tuple[int, ...],
    asset_ids: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    asset_id_set = set(asset_ids)
    chunks = read_dataset(
        data_root,
        partition_filter=_year_filter(years),
        columns=list(LABEL_PANEL_INPUT_COLUMNS),
        mode=DatasetReadMode.CHUNKED,
    )
    for chunk in chunks:
        filtered = chunk.loc[chunk["asset_id"].isin(asset_id_set)]
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame(columns=list(LABEL_PANEL_INPUT_COLUMNS))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    )


def _planned_asset_chunk_size(
    *,
    asset_ids: tuple[str, ...],
    data_root: Path,
    years: tuple[int, ...],
    resource_policy: RuntimeResourcePolicy | None,
) -> int:
    total_rows = _panel_row_count(data_root, years=years)
    rows_per_asset = max(1, total_rows // max(1, len(asset_ids)))
    output_width = len(
        a_share_processed_catalog().get("forward_returns").required_columns
    )
    working_width = len(LABEL_PANEL_INPUT_COLUMNS) + (output_width * 2)
    estimated_bytes_per_asset = max(
        1,
        rows_per_asset * working_width * 16,
    )
    label_horizons = tuple(
        definition.horizon for definition in a_share_label_catalog().list()
    )
    plan = ChunkPlanner(resource_policy).plan(
        axis=ChunkAxis.ASSET,
        item_count=len(asset_ids),
        estimated_bytes_per_item=estimated_bytes_per_asset,
        lookahead_window=max(label_horizons) + 1,
        output_width=output_width,
        disjoint=True,
    )
    return plan.chunk_size


def _panel_row_count(data_root: Path, *, years: tuple[int, ...]) -> int:
    return fingerprint_dataset(
        data_root,
        columns=["asset_id"],
        partition_filter=_year_filter(years),
    ).row_count


def _chunked(values: tuple[str, ...], size: int) -> Iterator[tuple[str, ...]]:
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]


def _partition_exists(data_root: Path, year: int) -> bool:
    try:
        fingerprint_dataset(
            data_root,
            columns=["asset_id"],
            partition_filter={"year": (year,)},
        )
        return True
    except ValueError as exc:
        if "partition_value_not_found" in str(exc):
            return False
        raise


def _year_filter(years: tuple[int, ...] | None) -> dict[str, tuple[int, ...]] | None:
    if years is None:
        return None
    return {"year": years}


__all__ = [
    "LABEL_BASE_COLUMNS",
    "LABEL_PANEL_INPUT_COLUMNS",
    "calculate_panel_labels",
    "calculate_forward_returns_asset_batches",
    "calculate_forward_returns_full",
    "select_existing_columns",
    "select_label_columns",
    "write_forward_labels",
]
