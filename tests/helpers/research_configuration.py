"""Read ordered research-design examples, never runtime defaults or orchestration."""

import json
from pathlib import Path

from quant_research_platform.contracts.models import _MODEL_ALGORITHM_FACTS, _model_spec

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "docs" / "research-rosters"


def research_configuration(market="a_share", name="core6"):
    return json.loads(
        (CONFIG_ROOT / market / (name + ".json")).read_text(encoding="utf-8")
    )


def model_specs(items):
    """Unpack research-design examples into actual model contract objects."""
    from dataclasses import replace

    result = []
    for item in items:
        fields = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "parameter_candidates",
                "training_asset_ids",
                "evaluation_asset_ids",
            }
        }
        model = _model_spec(fields)
        for parameters in item.get("parameter_candidates", ()):
            replace(model, hyperparameters=parameters)
        result.append(model)
    return tuple(result)


def configured_models(scenario="pv_all", *, factor_scope="core"):
    name = (
        ("core6" if factor_scope == "core" else "pv25")
        if scenario == "pv_all"
        else scenario
    )
    return model_specs(research_configuration(name=name)["models"])


A_SHARE_MODEL_ALGORITHMS = tuple(_MODEL_ALGORITHM_FACTS)


PV25_FACTORS = tuple(research_configuration(name="pv25")["evaluation"]["factors"])
CORE6_FACTORS = tuple(research_configuration(name="core6")["evaluation"]["factors"])
