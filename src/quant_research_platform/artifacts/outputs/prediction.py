"""Prediction output writer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.outputs.common_validation import (
    derive_output_bundle_root,
)
from quant_research_platform.artifacts.outputs.manifest import (
    PREDICTION_ARTIFACTS,
    output_manifest_payload,
    prediction_output_metadata,
)
from quant_research_platform.artifacts.outputs.parquet import (
    write_parquet_chunks,
)
from quant_research_platform.artifacts.outputs.prediction_validation import (
    model_metadata_with_effective_prediction_end,
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
from quant_research_platform.contracts.time import is_intraday_time_value
from quant_research_platform.runtime.resources import OwnedValue
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest


def write_prediction_outputs(
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    output_chunks: Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]],
    model_metadata: Mapping[str, object],
    prediction_metadata_factory: Callable[[], Mapping[str, object]],
) -> Path:
    """Write bounded prediction chunks into an output bundle."""

    return _write_prediction_output_chunks(
        output_chunks,
        spec=spec,
        output_spec=output_spec,
        layout=layout,
        model_metadata=model_metadata,
        prediction_metadata_factory=prediction_metadata_factory,
    )


def _write_prediction_output_chunks(
    frames: Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]],
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    model_metadata: Mapping[str, object],
    prediction_metadata_factory: Callable[[], Mapping[str, object]],
) -> Path:
    output_bundle = derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="prediction_outputs",
    )
    required = prediction_output_columns(
        spec.prediction_mode,
        spec.prediction_column,
        spec.primary_key,
    )
    model_metadata_payload = validate_model_metadata_shape(dict(model_metadata))
    validate_model_metadata_consistency(model_metadata_payload, spec=spec)

    def _write(temp_root: Path) -> None:
        write_summary = write_parquet_chunks(
            frames,
            temp_root / "prediction.parquet",
            required=required,
            key_columns=tuple(spec.primary_key),
            compression="zstd",
            date_only=not is_intraday_time_value(spec.test_start),
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
            model_metadata_payload,
            prediction_metadata,
        )
        validate_model_metadata_consistency(effective_model_metadata, spec=spec)
        validate_model_prediction_metadata_consistency(
            effective_model_metadata,
            prediction_metadata,
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
