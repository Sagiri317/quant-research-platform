"""Publish already computed prediction rows; never fit or call a model."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_frame,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.parquet import parquet_table_writer


def prediction_model_metadata(metadata):
    """Project the single saved recipe into downstream eligibility/calibration facts.

    This is an in-memory view, not a model load or a persisted identity copy.
    """
    recipe = metadata["model_facts"]["model"]
    return {
        **metadata["model_metadata"],
        "model_algorithm": recipe["model_algorithm"],
        "input_columns": [
            *recipe["input_factor_columns"],
            *recipe["input_feature_columns"],
        ],
        "target_column": recipe["target_column"],
        "sample_filter_mode": recipe["sample_filter_mode"],
    }


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
            empty.to_parquet(staging / "prediction.parquet", index=False)
            final_metadata.update(
                prediction_rows=0, effective_prediction_end=None, status="empty"
            )
        else:
            summary = write_parquet_chunks(
                chain((first,), chunks),
                staging / "prediction.parquet",
                metadata=metadata,
            )
            final_metadata.update(
                prediction_rows=summary.row_count,
                effective_prediction_end=summary.effective_prediction_end,
                status="complete",
            )
        resources.close()
        if (
            metadata.get("prediction_rows") is not None
            and metadata["prediction_rows"] != final_metadata["prediction_rows"]
        ):
            raise ValueError("prediction_metadata_inconsistent")
        if metadata.get("effective_prediction_end") is not None and pd.Timestamp(
            metadata["effective_prediction_end"]
        ) != pd.Timestamp(final_metadata["effective_prediction_end"]):
            raise ValueError("prediction_metadata_inconsistent")
        atomic_write_json(
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
        if (
            metadata.get("prediction_rows") is not None
            and type(metadata["prediction_rows"]) is not int
        ):
            raise ValueError("invalid_prediction_row_count")
        atomic_write_directory(destination, write)
    return destination


def read_predictions(path: Path, *, columns=None, time_window=None):
    """Read selected predictions without fitting or discovery."""
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )

    manifest, _members = validate_output_bundle(
        Path(path), expected_definition_id="prediction_outputs"
    )
    metadata = manifest["metadata"]
    from quant_research_platform.policy.lifecycle import closing_iterator

    with closing_iterator(
        _iter_prediction_member(
            path, metadata, columns=columns, time_window=time_window
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
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    return frame, metadata


def iter_predictions(path: Path, *, columns=None, time_window=None):
    """Project requested prediction columns/window from the chosen native member."""
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )

    manifest, _members = validate_output_bundle(
        Path(path), expected_definition_id="prediction_outputs"
    )
    return _iter_prediction_member(
        path, manifest["metadata"], columns=columns, time_window=time_window
    )


def _iter_prediction_member(path, metadata, *, columns=None, time_window=None):
    from quant_research_platform.storage.datasets.reading import (
        parquet_filters_for_time_window,
    )
    from quant_research_platform.storage.parquet import iter_dataset_part_batches

    requested = (
        ("timestamp", "asset_id", "prediction_score", "prediction_rank_score")
        if columns is None
        else tuple(columns)
    )
    if (
        not requested
        or len(requested) != len(set(requested))
        or set(requested)
        - {"timestamp", "asset_id", "prediction_score", "prediction_rank_score"}
    ):
        raise ValueError("prediction_projection_columns_invalid")
    physical = tuple(dict.fromkeys(("timestamp", "asset_id", *requested)))
    filters = parquet_filters_for_time_window(
        timestamp_column="timestamp", time_window=time_window
    )
    count = 0
    maximum = None
    keys = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
    with closing_iterator(
        iter_dataset_part_batches(
            Path(path) / "prediction.parquet", columns=physical, parquet_filters=filters
        )
    ) as batches:
        for batch in batches:
            frame = batch.to_pandas()
            if frame.empty:
                continue
            validate_prediction_frame(frame, metadata, columns=physical)
            duplicates = keys.add_table(
                pa.Table.from_pandas(
                    frame.loc[:, ["timestamp", "asset_id"]], preserve_index=False
                ),
                namespace="prediction",
            )
            if duplicates:
                raise ValueError("prediction_coverage_invalid")
            count += len(frame)
            maximum = (
                frame.timestamp.max()
                if maximum is None
                else max(maximum, frame.timestamp.max())
            )
            yield frame.loc[:, list(requested)]
    if time_window is None and (
        count != metadata["prediction_rows"]
        or (count and maximum != pd.Timestamp(metadata["effective_prediction_end"]))
    ):
        raise ValueError("prediction_metadata_inconsistent")


@dataclass(frozen=True)
class _PredictionOutputWriteSummary:
    row_count: int
    effective_prediction_end: str


def write_parquet_chunks(
    frames: Iterable[pd.DataFrame],
    path: Path,
    *,
    metadata: dict,
) -> _PredictionOutputWriteSummary:
    path = Path(path)
    require_unaliased_path(path, error="invalid_output_write_path")
    row_count = 0
    effective_prediction_end = None
    with closing_iterator(iter(frames)) as active_frames:
        handle = open_filesystem_file(path, "xb")
        with closing_iterator(handle):
            require_unaliased_path(path.parent, error="invalid_output_write_path")
            with parquet_table_writer(handle, compression="zstd") as write_table:
                key_index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
                for frame in active_frames:
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("invalid_artifact_output_frame")
                    validate_prediction_frame(frame, metadata)
                    if frame.empty:
                        continue
                    timestamps = strict_timestamp_series(
                        frame["timestamp"],
                        error="prediction_metadata_inconsistent",
                    )
                    selected = frame.assign(timestamp=timestamps)
                    current_end = time_value_isoformat(pd.Timestamp(timestamps.max()))
                    if effective_prediction_end is None or pd.Timestamp(
                        current_end
                    ) > pd.Timestamp(effective_prediction_end):
                        effective_prediction_end = current_end
                    table = pa.Table.from_pandas(selected, preserve_index=False)
                    try:
                        duplicates = key_index.add_table(table)
                    except ValueError as exc:
                        raise ValueError("invalid_prediction_keys") from exc
                    if duplicates:
                        raise ValueError("duplicate_prediction_keys")
                    write_table(table)
                    row_count += len(selected)
            handle.flush()
            os.fsync(handle.fileno())
    if row_count == 0 or effective_prediction_end is None:
        raise ValueError("empty_prediction_window")
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
        self.row_count = 0

    def close(self) -> None:
        self._index = None

    def values_for_frame(
        self, frame: pd.DataFrame, *, mark_matched: bool
    ) -> tuple[Sequence[float], bytearray]:
        if any(column not in frame for column in self.primary_key):
            raise ValueError("persisted_prediction_join_key_invalid")
        lookup_frame = frame.loc[:, list(self.primary_key)]
        table = pa.Table.from_pandas(
            lookup_frame.loc[:, list(self.primary_key)], preserve_index=False
        )
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
