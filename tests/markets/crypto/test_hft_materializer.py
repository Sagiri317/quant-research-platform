from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.outputs import write_prediction_outputs
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_datasets import (
    build_hft_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MODEL_ALGORITHMS,
    build_hft_model_catalog,
    build_hft_prediction_catalog,
)
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.models.dependencies import (
    preflight_model_dependency,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.replay import open_recorder_run
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import crypto_hft_materialization
from tests.helpers.binance_hft import write_binance_hft_run
from tests.helpers.runtime_resources import capture_test_policy


def test_hft_materialization_reuses_panel_and_standard_research_dataset(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    write_binance_hft_run(raw_root)
    layout = StorageLayout(output_root)
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    window = TimeWindow("1970-01-01T00:00:00", "1970-01-01T00:00:06")

    with (
        runtime_resource_policy_scope(policy),
        open_recorder_run(
            raw_root,
            policy,
        ) as run,
    ):
        inspections, stats = crypto_hft_materialization.materialize_hft_processed_data(
            layout=layout,
            run=run,
            time_window=window,
            resource_policy=policy,
        )
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=build_hft_research_dataset_catalog(),
            processed_catalog=hft_processed_catalog(),
            dataset_id=HFT_RESEARCH_DATASET_ID,
            time_window=window,
            resource_policy=policy,
        )
        model_catalog = build_hft_model_catalog(
            training_row_bound=inspections[HFT_PANEL_DATA_ID].row_count,
        )
        prediction_catalog = build_hft_prediction_catalog(
            train_end="1970-01-01T00:00:03.900000",
            test_start="1970-01-01T00:00:04",
            test_end="1970-01-01T00:00:06",
        )
        prediction_bundles = {}
        model_metadata = {}
        for algorithm in HFT_MODEL_ALGORITHMS:
            preflight_model_dependency(algorithm)
        for prediction_spec in prediction_catalog.list():
            result = build_prediction_result_from_research_dataset(
                layout=layout,
                dataset_catalog=build_hft_research_dataset_catalog(),
                model_catalog=model_catalog,
                prediction_catalog=prediction_catalog,
                prediction_id=prediction_spec.prediction_id,
                time_window=window,
                resource_policy=policy,
            )
            model_metadata[prediction_spec.model_id] = result.model_metadata
            try:
                prediction_bundles[prediction_spec.model_id] = write_prediction_outputs(
                    output_chunks=result.output_chunks,
                    spec=result.spec,
                    output_spec=OutputSpec(
                        output_id=f"{prediction_spec.prediction_id}_output",
                        output_definition_id="prediction_outputs",
                        run_id=prediction_spec.prediction_id,
                        market_id="crypto",
                        run_type="prediction",
                        scope=OutputScope.LOCAL_ONLY,
                    ),
                    layout=layout,
                    model_metadata=result.model_metadata,
                    prediction_metadata_factory=(result.prediction_metadata_factory),
                    resource_policy=policy,
                )
            finally:
                result.close()

    assert inspections[HFT_PANEL_DATA_ID].row_count == 61
    assert inspections[HFT_FACTOR_DATA_ID].row_count == 61
    assert inspections[HFT_LABEL_DATA_ID].row_count == 61
    assert stats.emitted_states == 61
    assert (
        inspections[HFT_PANEL_DATA_ID].manifest["source_artifact_identity"][
            "artifact_id"
        ]
        == "workflow-run"
    )
    assert set(
        inspections[HFT_FACTOR_DATA_ID].manifest["input_dataset_identities"]
    ) == {HFT_PANEL_DATA_ID}

    research_root = layout.research_dataset_root("crypto", HFT_RESEARCH_DATASET_ID)
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    with runtime_resource_policy_scope(policy):
        research = read_dataset(
            research_root,
            mode=DatasetReadMode.MATERIALIZED,
            resource_policy=policy,
        )

    assert len(research) == 61
    assert "segment_id" in research
    assert "ofi_100ms" in research
    assert "future_mid_return_500ms" in research
    assert set(prediction_bundles) == {
        f"crypto_hft_{algorithm}" for algorithm in HFT_MODEL_ALGORITHMS
    }
    assert all(path.is_dir() for path in prediction_bundles.values())
    assert all(
        metadata["max_train_rows"] == inspections[HFT_PANEL_DATA_ID].row_count
        for metadata in model_metadata.values()
    )
    assert all(
        metadata["rows_after_input_label_dropna"] <= metadata["max_train_rows"]
        for metadata in model_metadata.values()
    )
    assert all("T" in metadata["train_end"] for metadata in model_metadata.values())
