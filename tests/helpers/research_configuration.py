"""Read ordered research-design examples, never runtime defaults or orchestration."""

import json
from pathlib import Path

from quant_research_platform.contracts.models import _model_spec

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "docs" / "research-rosters"


def research_configuration(
    market="a_share", name="core6", *, validation=None, plan=None, small_models=False
):
    config = json.loads(
        (CONFIG_ROOT / market / (name + ".json")).read_text(encoding="utf-8")
    )
    if validation is not None:
        config["windows"]["validation"] = dict(
            zip(("start", "end"), validation, strict=True)
        )
    if plan is not None:
        config["research_plan"] = plan
        config["windows"]["test"] = dict(plan["test"])
        config["source_archives"] = None
    if small_models:
        for model in config["models"]:
            if model["model_algorithm"] in {
                "lightgbm_regressor",
                "xgboost_regressor",
                "gradient_boosting_regressor",
                "random_forest_regressor",
            }:
                candidates = model.get("parameter_candidates")
                if candidates is None:
                    candidates = (model.setdefault("hyperparameters", {}),)
                for parameters in candidates:
                    parameters["n_estimators"] = 16
    return config


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


A_SHARE_MODEL_ALGORITHMS = tuple(
    model["model_algorithm"] for model in research_configuration()["models"]
)


PV25_FACTORS = tuple(research_configuration(name="pv25")["evaluation"]["factors"])
CORE6_FACTORS = tuple(research_configuration(name="core6")["evaluation"]["factors"])
