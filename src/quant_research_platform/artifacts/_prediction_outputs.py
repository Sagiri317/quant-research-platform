"""Prediction output writer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts._output_manifest import (
    _output_manifest_payload,
    _prediction_output_metadata,
)
from quant_research_platform.artifacts._output_validation import (
    _derive_output_bundle_root,
    _model_metadata_with_effective_prediction_end,
    _required_frame_metadata,
    _sanitize_required_metadata,
    _validate_model_metadata_consistency,
    _validate_model_metadata_shape,
    _validate_model_prediction_metadata_consistency,
    _validate_prediction_metadata_consistency,
    _validate_prediction_metadata_consistency_from_summary,
    _validate_prediction_metadata_shape,
)
from quant_research_platform.artifacts._parquet_writers import (
    _write_parquet,
    _write_parquet_chunks,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest


def write_prediction_outputs(
    frame: pd.DataFrame | None = None,
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    output_chunks: Iterable[pd.DataFrame] | None = None,
    model_metadata: Mapping[str, object] | None = None,
    prediction_metadata_factory: Callable[[], Mapping[str, object]] | None = None,
) -> Path:
    """Write local prediction outputs."""

    if output_chunks is not None:
        if model_metadata is None or prediction_metadata_factory is None:
            raise ValueError("prediction_chunk_metadata_required")
        return _write_prediction_output_chunks(
            output_chunks,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            model_metadata=model_metadata,
            prediction_metadata_factory=prediction_metadata_factory,
        )
    if frame is None:
        raise ValueError("prediction_output_frame_required")

    output_bundle = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="prediction_outputs",
    )
    required = (*spec.primary_key, spec.prediction_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_output_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_prediction_output_columns: {', '.join(extra)}")
    if bool(frame.duplicated(list(spec.primary_key)).any()):
        raise ValueError("duplicate_prediction_keys")
    model_metadata = _validate_model_metadata_shape(
        _required_frame_metadata(
            frame,
            key="model_metadata",
            error_name="missing_model_metadata",
        )
    )
    prediction_metadata = _validate_prediction_metadata_shape(
        _required_frame_metadata(
            frame,
            key="prediction_metadata",
            error_name="missing_prediction_metadata",
        )
    )
    _validate_model_metadata_consistency(model_metadata, spec=spec)
    _validate_prediction_metadata_consistency(
        prediction_metadata,
        spec=spec,
        frame=frame,
    )
    model_metadata = _model_metadata_with_effective_prediction_end(
        model_metadata,
        prediction_metadata,
    )
    _validate_model_metadata_consistency(model_metadata, spec=spec)
    _validate_model_prediction_metadata_consistency(
        model_metadata,
        prediction_metadata,
    )

    def _write(temp_root: Path) -> None:
        _write_parquet(
            frame.loc[:, list(required)],
            temp_root / "prediction.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_prediction_output_metadata(
                    spec=spec,
                    model_metadata=model_metadata,
                    prediction_metadata=prediction_metadata,
                ),
                files=("prediction.parquet",),
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle


def _write_prediction_output_chunks(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    model_metadata: Mapping[str, object],
    prediction_metadata_factory: Callable[[], Mapping[str, object]],
) -> Path:
    output_bundle = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="prediction_outputs",
    )
    required = (*spec.primary_key, spec.prediction_column)
    model_metadata_payload = _validate_model_metadata_shape(
        _sanitize_required_metadata(
            model_metadata,
            error_name="missing_model_metadata",
        )
    )
    _validate_model_metadata_consistency(model_metadata_payload, spec=spec)

    def _write(temp_root: Path) -> None:
        write_summary = _write_parquet_chunks(
            frames,
            temp_root / "prediction.parquet",
            required=required,
            key_columns=tuple(spec.primary_key),
            compression="zstd",
        )
        prediction_metadata = _validate_prediction_metadata_shape(
            _sanitize_required_metadata(
                prediction_metadata_factory(),
                error_name="missing_prediction_metadata",
            )
        )
        _validate_prediction_metadata_consistency_from_summary(
            prediction_metadata,
            spec=spec,
            row_count=write_summary.row_count,
            effective_test_end=write_summary.effective_test_end,
        )
        effective_model_metadata = _model_metadata_with_effective_prediction_end(
            model_metadata_payload,
            prediction_metadata,
        )
        _validate_model_metadata_consistency(effective_model_metadata, spec=spec)
        _validate_model_prediction_metadata_consistency(
            effective_model_metadata,
            prediction_metadata,
        )
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_prediction_output_metadata(
                    spec=spec,
                    model_metadata=effective_model_metadata,
                    prediction_metadata=prediction_metadata,
                ),
                files=("prediction.parquet",),
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle
