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
    build_hft_research_dataset_catalog,
)
from tests.helpers.research_configuration import model_specs, research_configuration


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


def test_hft_models_use_generic_specs() -> None:
    config = research_configuration("crypto", "baseline")
    models = model_specs(config["models"])
    assert {model.model_algorithm for model in models} == {
        item["model_algorithm"] for item in config["models"]
    }
    assert all(not hasattr(model, "purge_gap_steps") for model in models)


def test_hft_roster_preserves_explicit_model_inputs_and_targets():
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_FACTOR_COLUMNS,
        label_catalog,
    )

    config = research_configuration("crypto", "baseline")
    models = model_specs(config["models"])
    dataset = build_hft_research_dataset_catalog().get(HFT_RESEARCH_DATASET_ID)
    labels = {item.output_column for item in label_catalog().list()}
    assert models
    for model, declared in zip(models, config["models"], strict=True):
        assert model.input_columns == tuple(
            declared.get("input_factor_columns", [])
            + declared.get("input_feature_columns", [])
        )
        assert set(model.input_columns) <= set(dataset.factor_columns)
        assert model.target_column == declared["target_column"]
        assert model.target_column in labels
    assert dataset.factor_columns == HFT_FACTOR_COLUMNS
    assert config["evaluation"]["factors"]
    assert set(config["evaluation"]["factors"]) <= set(dataset.factor_columns)


def test_native_labels_declare_distinct_units_and_maturity_columns():
    from quant_research_platform.research.protocols.crypto_hft import label_catalog

    spec = hft_processed_catalog().get(HFT_LABEL_DATA_ID)
    for definition in label_catalog().list():
        assert definition.label_timing == "decision_time_plus_horizon"
        assert (
            spec.label_availability_columns[definition.output_column]
            == definition.available_at_column
        )
        assert (
            spec.column_units[definition.output_column]
            == {
                "future_mid_return": "decimal_return",
                "future_mid_direction": "dimensionless",
                "maker_sell_pressure_proxy": "quantity_ratio",
            }[definition.family]
        )
    assert not set(spec.label_availability_columns.values()) & set(spec.column_units)
