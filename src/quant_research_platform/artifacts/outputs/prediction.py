"""Publish already computed prediction rows; never fit or call a model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_frame,
    validate_prediction_output_metadata_shape,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


def write_prediction_outputs(
    *, output_path: Path, output_chunks, metadata: dict
) -> Path:
    from quant_research_platform.artifacts.outputs.manifest import (
        output_manifest_payload,
    )
    from quant_research_platform.artifacts.outputs.parquet import write_parquet_chunks

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
        validate_prediction_output_metadata_shape(final_metadata)
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

    with closing_iterator(iter(output_chunks)) as chunks:
        if (
            metadata.get("prediction_rows") is not None
            and type(metadata["prediction_rows"]) is not int
        ):
            raise ValueError("invalid_prediction_row_count")
        atomic_write_directory(destination, write)
    return destination


def read_predictions(path: Path, *, time_window=None):
    """Read selected predictions without fitting or discovery."""
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        validate_output_bundle,
    )

    manifest, _members = validate_output_bundle(
        Path(path), expected_definition_id="prediction_outputs"
    )
    metadata = manifest["metadata"]
    from quant_research_platform.policy.lifecycle import closing_iterator

    with closing_iterator(iter_predictions(path, time_window=time_window)) as chunks:
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
    return frame, metadata


def iter_predictions(path: Path, *, columns=None, time_window=None):
    """Project requested prediction columns/window from the chosen native member."""
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        validate_output_bundle,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.storage.datasets.reading import (
        parquet_filters_for_time_window,
    )
    from quant_research_platform.storage.parquet import iter_dataset_part_batches

    manifest, _members = validate_output_bundle(
        Path(path), expected_definition_id="prediction_outputs"
    )
    metadata = manifest["metadata"]
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
    physical = ("timestamp", "asset_id", "prediction_score", "prediction_rank_score")
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


def read_prediction_output_bundle(
    root: Path, *, time_window=None, prediction_column="prediction_score"
):
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        validate_output_bundle,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator

    manifest, _members = validate_output_bundle(
        Path(root), expected_definition_id="prediction_outputs"
    )
    if prediction_column not in {"prediction_score", "prediction_rank_score"}:
        raise ValueError("prediction_column_invalid")
    index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
    lookup = _PredictionLookup(
        index=index,
        primary_key=("timestamp", "asset_id"),
        prediction_column=prediction_column,
    )
    with closing_iterator(
        iter_predictions(
            root,
            columns=("timestamp", "asset_id", prediction_column),
            time_window=time_window,
        )
    ) as frames:
        for frame in frames:
            index.add_float_value_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                namespace="prediction",
                value_column=prediction_column,
                validation_column=prediction_column,
                retain_values=True,
            )
            lookup.row_count += len(frame)
    return manifest, lookup
