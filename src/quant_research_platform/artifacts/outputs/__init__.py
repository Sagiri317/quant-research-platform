"""Local output writers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.outputs.bundle_validation import (
    read_prediction_output_bundle as read_prediction_output_bundle,
)
from quant_research_platform.artifacts.outputs.bundle_validation import (
    validate_output_bundle as validate_output_bundle,
)
from quant_research_platform.artifacts.outputs.evaluation import (
    write_evaluation_outputs_from_chunks as evaluation_chunks_writer,
)
from quant_research_platform.artifacts.outputs.prediction import (
    write_prediction_outputs as prediction_writer,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout


def _write_in_resource_root(
    writer: Callable[[], Path],
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Path:
    active = current_runtime_resource_policy()
    if resource_policy is not None:
        if active is not resource_policy:
            raise RuntimeError("runtime_resource_policy_conflict")
        return writer()
    if active is not None:
        raise RuntimeError("runtime_resource_policy_competing_root")
    with runtime_resource_policy_scope(None):
        return writer()


def write_prediction_outputs(
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    output_chunks: Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]],
    model_metadata: Mapping[str, object],
    prediction_metadata_factory: Callable[[], Mapping[str, object]],
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Path:
    return _write_in_resource_root(
        lambda: prediction_writer(
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            output_chunks=output_chunks,
            model_metadata=model_metadata,
            prediction_metadata_factory=prediction_metadata_factory,
        ),
        resource_policy=resource_policy,
    )


def write_evaluation_outputs_from_chunks(
    chunks: Iterable[
        tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]
        | OwnedValue[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]]
    ],
    *,
    summary_factory: Callable[
        [],
        tuple[
            EvaluationDiagnostics,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ],
    ],
    spec: EvaluationSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    n_groups: int,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Path:
    return _write_in_resource_root(
        lambda: evaluation_chunks_writer(
            chunks,
            summary_factory=summary_factory,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            n_groups=n_groups,
        ),
        resource_policy=resource_policy,
    )
