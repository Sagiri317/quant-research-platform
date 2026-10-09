"""Prediction output writer."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_research_platform.artifacts.outputs.common_validation import (
    derive_output_bundle_root,
)
from quant_research_platform.artifacts.outputs.manifest import (
    PREDICTION_ARTIFACTS,
    output_manifest_payload,
    prediction_output_metadata,
)
from quant_research_platform.artifacts.outputs.parquet import write_parquet_chunks
from quant_research_platform.artifacts.outputs.prediction_validation import (
    model_metadata_with_effective_prediction_end,
    validate_crypto_prediction_frame,
    validate_family_prediction_frame,
    validate_model_metadata_consistency,
    validate_model_metadata_shape,
    validate_model_prediction_metadata_consistency,
    validate_prediction_metadata_consistency_from_summary,
    validate_prediction_metadata_shape,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.predictions import (
    PredictionSpec,
    prediction_output_columns,
)
from quant_research_platform.contracts.time import is_datetime_value
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.dataframes import iter_complete_session_frames
from quant_research_platform.storage._scratch_ownership import (
    owned_scratch_name,
    remove_owned_scratch,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    dataset_file_identity,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest
from quant_research_platform.storage.parquet import iter_dataset_part_batches


def write_prediction_outputs(
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    output_chunks: Iterable[pd.DataFrame],
    model_metadata: Mapping[str, object],
    prediction_metadata_factory: Callable[[], Mapping[str, object]],
) -> Path:
    output_bundle = derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="prediction_outputs",
    )
    required = prediction_output_columns(
        spec.prediction_mode, spec.prediction_column, spec.primary_key
    )
    model_metadata_payload = validate_model_metadata_shape(dict(model_metadata))
    validate_model_metadata_consistency(model_metadata_payload, spec=spec)

    def validate_frame(frame: pd.DataFrame) -> None:
        timestamps = strict_timestamp_series(
            frame["timestamp"], error="prediction_metadata_inconsistent"
        )
        if timestamps.min() < pd.Timestamp(spec.test_start) or (
            spec.test_end is not None and timestamps.max() > pd.Timestamp(spec.test_end)
        ):
            raise ValueError("prediction_metadata_inconsistent")
        if spec.prediction_mode == "crypto_oos":
            validate_crypto_prediction_frame(frame, model_metadata_payload)

    def _write(temp_root: Path) -> None:
        paths = []
        primary = None
        try:
            active_frames = output_chunks
            if spec.prediction_mode == "a_share_oos":
                paths = _stage_family_predictions(
                    output_chunks, temp_root, required, spec, model_metadata_payload
                )
                active_frames = _merge_family_predictions(paths, required)
            write_summary = write_parquet_chunks(
                active_frames,
                temp_root / "prediction.parquet",
                required=required,
                key_columns=tuple(spec.primary_key),
                compression="zstd",
                date_only=not is_datetime_value(spec.test_start),
                validate_frame=validate_frame,
            )
        except BaseException as body_error:
            primary = body_error
            raise
        finally:
            for _quarter, _family, path in paths:
                try:
                    remove_owned_scratch(path)
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(
                        "family prediction cleanup also failed: "
                        f"{type(cleanup).__name__}"
                    )
        prediction_metadata = validate_prediction_metadata_shape(
            dict(prediction_metadata_factory())
        )
        validate_prediction_metadata_consistency_from_summary(
            prediction_metadata,
            spec=spec,
            row_count=write_summary.row_count,
            effective_test_end=write_summary.effective_test_end,
        )
        effective_model_metadata = model_metadata_with_effective_prediction_end(
            model_metadata_payload, prediction_metadata
        )
        validate_model_metadata_consistency(effective_model_metadata, spec=spec)
        validate_model_prediction_metadata_consistency(
            effective_model_metadata, prediction_metadata
        )
        write_manifest(
            temp_root / "manifest.json",
            output_manifest_payload(
                output_spec=output_spec,
                metadata=prediction_output_metadata(
                    spec=spec,
                    model_metadata=effective_model_metadata,
                    prediction_metadata=prediction_metadata,
                ),
                files=PREDICTION_ARTIFACTS,
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle


def _stage_family_predictions(frames, root, required, spec, metadata):
    paths = []
    expected = {
        (item["quarter_id"], item["family_id"]): index
        for index, item in enumerate(metadata["quarterly_models"])
    }
    seen = set()

    def group_key(item):
        frame = item
        if frame.empty:
            raise ValueError("empty_family_prediction_stream")
        validate_family_prediction_frame(frame, metadata)
        if len(frame[["quarter_id", "family_id"]].drop_duplicates()) != 1:
            raise ValueError("prediction_family_chunk_mixed")
        return (frame["quarter_id"].iloc[0], frame["family_id"].iloc[0])

    with closing_iterator(iter(frames)) as source:
        for coordinate, group in groupby(source, key=group_key):
            if coordinate not in expected or coordinate in seen:
                raise ValueError("prediction_family_stream_coordinate_invalid")
            seen.add(coordinate)
            path = root / owned_scratch_name(".s-", uuid4().hex[:12])
            write_parquet_chunks(
                group,
                path,
                required=required,
                key_columns=spec.primary_key,
                compression="zstd",
                date_only=True,
            )
            paths.append((*coordinate, path))
    return sorted(paths, key=lambda item: expected[item[:2]])


def _merge_family_predictions(paths, columns):
    for _quarter, group in groupby(paths, key=lambda item: item[0]):
        members = list(group)

        def frames(path):
            batches = iter_dataset_part_batches(
                path, expected_identity=dataset_file_identity(path), columns=columns
            )
            with closing_iterator(batches):
                for batch in batches:
                    frame = batch.to_pandas()
                    try:
                        yield frame
                    except GeneratorExit:
                        return

        with ExitStack() as readers:
            heap = []

            for _qid, family, path in members:
                source = readers.enter_context(
                    closing_iterator(
                        iter_complete_session_frames(
                            iter(frames(path)), timestamp_column="timestamp"
                        )
                    )
                )
                first = next(source, None)
                if first is None:
                    raise ValueError("empty_family_prediction_stream")
                heapq.heappush(
                    heap,
                    (
                        pd.Timestamp(first["timestamp"].iloc[0]),
                        family,
                        first,
                        source,
                    ),
                )

            while heap:
                timestamp, family, frame, source = heapq.heappop(heap)
                try:
                    yield frame
                except GeneratorExit:
                    return

                following = next(source, None)
                if following is not None:
                    next_timestamp = pd.Timestamp(following["timestamp"].iloc[0])
                    if next_timestamp <= timestamp:
                        raise ValueError("prediction_family_time_order_invalid")
                    heapq.heappush(heap, (next_timestamp, family, following, source))
