from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.outputs import (
    write_prediction_outputs as write_prediction_outputs_in_workflow,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.research.predictions.outputs import (
    build_prediction_result_from_frames,
)
from quant_research_platform.storage.layout import StorageLayout


def write_prediction_output_and_read(
    project_root: Path,
    frames: tuple[pd.DataFrame, ...],
    spec: PredictionSpec,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object,
    model_metadata: dict[str, object],
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    layout = StorageLayout(project_root=project_root)
    result = build_prediction_result_from_frames(
        frames,
        spec,
        input_columns,
        prediction_model=prediction_model,
        model_metadata=model_metadata,
    )
    output_bundle = write_prediction_result(layout, result)
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    prediction = pd.read_parquet(output_bundle / "prediction.parquet")
    prediction.attrs["model_metadata"] = manifest["metadata"]["model_metadata"]
    prediction.attrs["prediction_metadata"] = manifest["metadata"][
        "prediction_metadata"
    ]
    return (prediction, output_bundle, manifest)


def write_prediction_result(layout: StorageLayout, result) -> Path:
    output_spec = OutputSpec(
        output_id=result.spec.prediction_id,
        output_definition_id="prediction_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
    )
    with result:
        return write_prediction_outputs_in_workflow(
            spec=result.spec,
            output_spec=output_spec,
            layout=layout,
            output_chunks=result.output_chunks,
            model_metadata=result.model_metadata,
            prediction_metadata_factory=result.prediction_metadata_factory,
        )
