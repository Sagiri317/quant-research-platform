"""Explicit assembly choices preserve sparse auxiliary science, never the base."""

import pytest

from quant_research_platform.markets.implementations.crypto import ids
from quant_research_platform.research.datasets import assembler


def _config():
    return {
        "market_id": "crypto",
        "dataset_id": ids.HFT_RESEARCH_DATASET_ID,
        "inputs": {
            ids.HFT_PANEL_DATA_ID: "/panel",
            ids.HFT_FACTOR_DATA_ID: "/factors",
            ids.HFT_LABEL_DATA_ID: "/labels",
            ids.HFT_DERIVATIVE_DATA_ID: "/derivative",
        },
        "time_window": {"start": "2024-01-01", "end": "2024-01-02"},
        "output_root": "/result",
    }


def test_auxiliary_sparse_coverage_is_an_explicit_choice(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return kwargs["output_root"]

    monkeypatch.setattr(assembler, "assemble_research_dataset", capture)
    config = _config()
    config["sparse_input_data_ids"] = [ids.HFT_DERIVATIVE_DATA_ID]
    assembler.assemble(config)
    spec = captured["dataset_catalog"].get(ids.HFT_RESEARCH_DATASET_ID)
    assert spec.key_coverage[ids.HFT_DERIVATIVE_DATA_ID] == "sparse"
    assert spec.key_coverage[ids.HFT_PANEL_DATA_ID] == "strict"
    assert spec.key_coverage[ids.HFT_LABEL_DATA_ID] == "strict"


@pytest.mark.parametrize("data_id", (ids.HFT_PANEL_DATA_ID, ids.HFT_LABEL_DATA_ID))
def test_sparse_coverage_cannot_weaken_economic_base_or_labels(data_id):
    config = _config()
    config["sparse_input_data_ids"] = [data_id]
    with pytest.raises(ValueError, match="assemble_sparse_inputs_invalid"):
        assembler.assemble(config)
