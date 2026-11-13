"""Publish already computed prediction rows; never fit or call a model."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.models import prediction_columns
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    open_filesystem_file,
)
from quant_research_platform.storage.manifests import write_qrp_json_file
from quant_research_platform.storage.parquet import parquet_table_writer


def write_prediction_outputs(
    *, output_path: Path, output_chunks, metadata: dict
) -> Path:
    from quant_research_platform.artifacts.outputs.manifest import (
        output_manifest_payload,
    )

    destination = Path(output_path)

    def write(staging):
        from itertools import chain

        first = next((frame for frame in chunks if not frame.empty), None)
        final_metadata = dict(metadata)
        if first is None:
            empty = pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="datetime64[ns]"),
                    "asset_id": pd.Series(dtype="str"),
                    "prediction_score": pd.Series(dtype="float64"),
                    "prediction_rank_score": pd.Series(dtype="float64"),
                }
            )
            for column in prediction_columns(metadata)[4:]:
                empty[column] = pd.Series(dtype="float64")
            empty.to_parquet(staging / "prediction.parquet", index=False)
            final_metadata.update(
                prediction_rows=0, effective_prediction_end=None, status="empty"
            )
        else:
            summary = write_parquet_chunks(
                chain((first,), chunks),
                staging / "prediction.parquet",
            )
            final_metadata.update(
                prediction_rows=summary.row_count,
                effective_prediction_end=summary.effective_prediction_end,
                status="complete",
            )
        resources.close()
        write_qrp_json_file(
            staging / "manifest.json",
            output_manifest_payload(
                definition_id="prediction_outputs",
                metadata=final_metadata,
                files=("prediction.parquet",),
            ),
        )

    from quant_research_platform.policy.lifecycle import closing_iterator

    with closing_iterator(ExitStack()) as resources:
        chunks = resources.enter_context(closing_iterator(iter(output_chunks)))
        atomic_write_directory(destination, write)
    return destination


def read_predictions(
    path: Path,
    *,
    columns=None,
    time_window=None,
    include_available_probabilities=False,
    asset_ids=None,
):
    """Read selected predictions without fitting or discovery."""
    from quant_research_platform.artifacts.outputs.manifest import (
        read_output_manifest,
    )

    manifest = read_output_manifest(Path(path))
    metadata = manifest["metadata"]
    if columns is not None and include_available_probabilities:
        columns = tuple(columns)
        if columns:
            columns += tuple(
                column
                for column in prediction_columns(metadata)[4:]
                if column not in columns
            )
    from quant_research_platform.policy.lifecycle import closing_iterator

    with closing_iterator(
        _iter_prediction_member(
            path,
            metadata,
            columns=columns,
            time_window=time_window,
            asset_ids=asset_ids,
        )
    ) as chunks:
        frames = list(chunks)
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "asset_id": pd.Series(dtype="str"),
                "prediction_score": pd.Series(dtype="float64"),
                "prediction_rank_score": pd.Series(dtype="float64"),
            }
        )
    )
    if frame.empty:
        for column in prediction_columns(metadata)[4:]:
            frame[column] = pd.Series(dtype="float64")
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    return frame, metadata


def iter_predictions(path: Path, *, columns=None, time_window=None):
    """Project requested prediction columns/window from the chosen native member."""
    from quant_research_platform.artifacts.outputs.manifest import (
        read_output_manifest,
    )

    manifest = read_output_manifest(Path(path))
    return _iter_prediction_member(
        path, manifest["metadata"], columns=columns, time_window=time_window
    )


def _iter_prediction_member(
    path, metadata, *, columns=None, time_window=None, asset_ids=None
):
    from quant_research_platform.storage.datasets.reading import (
        parquet_filters_for_time_window,
    )
    from quant_research_platform.storage.parquet import (
        PreparedMembershipValues,
        iter_dataset_part_batches,
    )

    requested = prediction_columns(metadata) if columns is None else tuple(columns)
    if (
        not requested
        or len(requested) != len(set(requested))
        or set(requested) - set(prediction_columns(metadata))
    ):
        raise ValueError("prediction_projection_columns_invalid")
    filters = parquet_filters_for_time_window(
        timestamp_column="timestamp", time_window=time_window
    )
    if asset_ids is not None:
        filters = [
            *(filters or ()),
            (
                "asset_id",
                "in",
                PreparedMembershipValues(pa.array(asset_ids, type=pa.string())),
            ),
        ]
    physical = tuple(
        dict.fromkeys((*requested, *(column for column, _, _ in filters or ())))
    )

    with closing_iterator(
        iter_dataset_part_batches(
            Path(path) / "prediction.parquet", columns=physical, parquet_filters=filters
        )
    ) as batches:
        for batch in batches:
            frame = batch.to_pandas()
            if not frame.empty:
                yield frame.loc[:, list(requested)]


@dataclass(frozen=True)
class _PredictionOutputWriteSummary:
    row_count: int
    effective_prediction_end: str


def write_parquet_chunks(
    frames: Iterable[pd.DataFrame],
    path: Path,
) -> _PredictionOutputWriteSummary:
    path = Path(path)
    row_count = 0
    effective_prediction_end = None
    with closing_iterator(iter(frames)) as active_frames:
        handle = open_filesystem_file(path, "xb")
        with closing_iterator(handle):
            with parquet_table_writer(
                handle, compression="zstd", statistics=("timestamp",)
            ) as write_table:
                for frame in active_frames:
                    if frame.empty:
                        continue
                    current_end = time_value_isoformat(
                        pd.Timestamp(frame.timestamp.max())
                    )
                    if effective_prediction_end is None or pd.Timestamp(
                        current_end
                    ) > pd.Timestamp(effective_prediction_end):
                        effective_prediction_end = current_end
                    write_table(
                        pa.Table.from_pandas(
                            frame.assign(
                                timestamp=frame.timestamp.astype("datetime64[ns]")
                            ),
                            preserve_index=False,
                        )
                    )
                    row_count += len(frame)
            handle.flush()
            os.fsync(handle.fileno())
    return _PredictionOutputWriteSummary(
        row_count=row_count, effective_prediction_end=effective_prediction_end
    )


class _PredictionLookup:
    def __init__(
        self,
        *,
        index: ExactPrimaryKeyIndex,
        primary_key: tuple[str, ...],
        prediction_column: str,
    ) -> None:
        self._index = index
        self.primary_key = primary_key
        self.prediction_column = prediction_column

    def close(self) -> None:
        self._index = None

    def values_for_frame(
        self, frame: pd.DataFrame, *, mark_matched: bool
    ) -> tuple[Sequence[float], bytearray]:
        if any(column not in frame for column in self.primary_key):
            raise ValueError("persisted_prediction_join_key_invalid")
        lookup_frame = frame.loc[:, list(self.primary_key)]
        table = pa.Table.from_pandas(lookup_frame, preserve_index=False)
        values, found = self._index.lookup_float_values(
            table,
            source_namespace="prediction",
            matched_namespace="prediction-matched",
            mark_matched=mark_matched,
        )
        return (values, found)

    def coverage_counts(self) -> tuple[int, int]:
        # Matching moves each key out of the outstanding partition.
        return (self._index.float_value_count(namespace="prediction"), 0)
