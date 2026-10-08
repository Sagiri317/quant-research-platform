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
    build_hft_model_catalog,
    build_hft_research_dataset_catalog,
)
from tests.helpers.research_configuration import research_configuration


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
    config = research_configuration("crypto", "baseline")
    for item in config["models"]:
        item["max_train_rows"] = 100
    models = build_hft_model_catalog(configuration=config).list()
    assert {model.model_algorithm for model in models} == {
        item["model_algorithm"] for item in config["models"]
    }
    assert all(model.max_train_rows == 100 for model in models)
    assert all(not hasattr(model, "purge_gap_steps") for model in models)
    assert all(not model.model_id.startswith("hft_model_framework") for model in models)


def test_ridge_asset_scopes_share_dimensionless_inputs_and_new_spec_identity():
    from dataclasses import replace

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.models import model_spec_payload
    from quant_research_platform.research.protocols.crypto_hft import HFT_FACTOR_COLUMNS

    comparable = (
        "relative_spread",
        "imbalance_1",
        "imbalance_5",
        "imbalance_10",
        "return_100ms",
        "realized_volatility_1s",
    )
    config = research_configuration("crypto", "baseline")
    for item in config["models"]:
        item["max_train_rows"] = None
    models = build_hft_model_catalog(configuration=config).list()
    ridge = [model for model in models if model.model_algorithm == "ridge_regression"]
    assert len(ridge) == 5
    for model in ridge:
        assert model.input_columns == comparable
        old = replace(model, input_factor_columns=HFT_FACTOR_COLUMNS)
        assert canonical_json_hash(model_spec_payload(model)) != canonical_json_hash(
            model_spec_payload(old)
        )
        assert model.label_column == "future_mid_return_500ms"
    assert len(models) == 10
    for model in models:
        if model.model_algorithm != "ridge_regression":
            assert model.input_columns == (
                ("ofi_100ms",)
                if model.model_algorithm == "linear_regression"
                else tuple(config["evaluation"]["factors"])
            )
    dataset = build_hft_research_dataset_catalog().get(HFT_RESEARCH_DATASET_ID)
    assert dataset.factor_columns == HFT_FACTOR_COLUMNS
    assert len(config["evaluation"]["factors"]) == 14
    assert len(dataset.factor_columns) == 30
