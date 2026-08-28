from __future__ import annotations

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, ProcessedDataKind
from quant_research_platform.markets.implementations.crypto.factors.definitions import (
    hft_factor_catalog,
)
from quant_research_platform.markets.implementations.crypto.hft import HFT_GRID_MS
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_COLUMNS,
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_COLUMNS,
    HFT_LABEL_DATA_ID,
    HFT_MODEL_LABEL_HORIZON_MS,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    hft_label_catalog,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_profiles import (
    build_hft_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MODEL_ALGORITHMS,
    build_hft_model_catalog,
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


def test_hft_factor_and_label_definitions_name_exact_inputs_and_lookahead() -> None:
    factors = {item.output_column: item for item in hft_factor_catalog().list()}
    labels = {item.output_column: item for item in hft_label_catalog().list()}

    assert set(factors) == set(HFT_FACTOR_COLUMNS)
    assert factors["ofi_100ms"].input_columns == ("book_ofi_100ms",)
    assert "ofi_100ms" not in factors["ofi_100ms"].input_columns
    assert set(labels) == set(HFT_LABEL_COLUMNS)
    assert labels["future_mid_return_500ms"].lookahead == 5
    assert labels["future_mid_direction_500ms"].family == "futures_mid_direction"


def test_hft_models_use_generic_specs_with_exact_training_admission_bound() -> None:
    models = build_hft_model_catalog(training_row_bound=100).list()

    assert [model.model_algorithm for model in models] == sorted(HFT_MODEL_ALGORITHMS)
    assert all(model.max_train_rows == 100 for model in models)
    assert all(
        model.purge_gap_steps == HFT_MODEL_LABEL_HORIZON_MS // HFT_GRID_MS
        for model in models
    )
    assert all(not model.model_id.startswith("hft_model_framework") for model in models)
