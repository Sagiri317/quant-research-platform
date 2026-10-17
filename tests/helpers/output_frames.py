"""Small native fixtures; scientific behavior has real pipeline tests."""

import pandas as pd

from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

PRICE_VOLUME_SIGNAL_COLUMN = "momentum_20d"
FORWARD_RETURN_LABEL_COLUMN = forward_return_label_column(20)


def make_price_volume_prediction_output_frame():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-03"]),
            "asset_id": ["asset-a"],
            "prediction_score": [0.25],
            "prediction_rank_score": [0.0],
        }
    )
    frame.attrs["prediction_metadata"] = _price_volume_prediction_metadata()
    return frame


def make_artifact_signal_spec():
    return SignalSpec(
        signal_id="signal", source_column="feature", source_kind="feature"
    )


def _price_volume_prediction_metadata():
    return {
        "model": "model",
        "inputs": "inputs",
        "model_id": "linear-demo",
        "family_id": "linear-demo",
        "prediction_window": {"start": "2014-01-03", "end": "2014-01-03"},
        "effective_prediction_end": "2014-01-03",
        "sample_role": "validation",
        "prediction_rows": 1,
        "label_unit": "decimal_return",
        "model_metadata": {
            "training_asset_ids": ["a", "b"],
            "train_start": "2014-01-02T00:00:00",
            "train_end": "2014-01-02T00:00:00",
            "label_availability_column": f"{FORWARD_RETURN_LABEL_COLUMN}_available_at",
            "label_available_max": "2014-01-02T00:00:00",
        },
        "model_facts": {
            "family_id": "linear-demo",
            "asset_ids": None,
            "market_id": "a_share",
            "model": model_spec_payload(
                ModelSpec(
                    model_id="linear-demo",
                    model_algorithm="linear_regression",
                    input_factor_columns=(PRICE_VOLUME_SIGNAL_COLUMN,),
                    target_column=FORWARD_RETURN_LABEL_COLUMN,
                    random_seed=7,
                )
            ),
            "fit_windows": [{"start": "2014-01-02", "end": "2014-01-02"}],
            "fit_cutoff": "2014-01-02",
            "candidate_parameters": {},
            "input_units": {PRICE_VOLUME_SIGNAL_COLUMN: "decimal_return"},
            "label_unit": "decimal_return",
            "selection_cutoff": None,
        },
        "input_spec": None,
        "status": "complete",
    }
