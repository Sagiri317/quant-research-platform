from __future__ import annotations

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, ProcessedDataKind
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_MODEL_ALGORITHMS,
    build_hft_model_catalog,
    build_hft_research_dataset_catalog,
)


def test_hft_catalog_has_one_panel_factor_label_and_research_path() -> None:
    processed = hft_processed_catalog()
    panel = processed.get(HFT_PANEL_DATA_ID)
    factors = processed.get(HFT_FACTOR_DATA_ID)
    labels = processed.get(HFT_LABEL_DATA_ID)
    research = build_hft_research_dataset_catalog().get(HFT_RESEARCH_DATASET_ID)

    assert panel.kind is ProcessedDataKind.PANEL
    assert factors.kind is ProcessedDataKind.FACTOR
    assert labels.kind is ProcessedDataKind.LABEL
    assert panel.frequency is DataFrequency.TICK
    assert panel.column_roles["segment_id"] is ColumnRole.CONTEXT
    assert "segment_id" in research.context_columns
    assert research.input_data_ids == (
        HFT_PANEL_DATA_ID,
        HFT_FACTOR_DATA_ID,
        HFT_LABEL_DATA_ID,
    )
    assert research.base_input_data_id == HFT_PANEL_DATA_ID


def test_hft_models_use_generic_specs_with_exact_training_admission_bound() -> None:
    models = build_hft_model_catalog(training_row_bound=100).list()

    assert {model.model_algorithm for model in models} == set(HFT_MODEL_ALGORITHMS)
    assert all(model.max_train_rows == 100 for model in models)
    assert all(not hasattr(model, "purge_gap_steps") for model in models)
    assert all(not model.model_id.startswith("hft_model_framework") for model in models)
