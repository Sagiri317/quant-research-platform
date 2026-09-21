"""Read the same editable research configurations used by production."""

import json
from pathlib import Path

from quant_research_platform.catalog.models import ModelCatalog

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "research"


def research_configuration(
    market="a_share", name="core6", *, development=None, plan=None
):
    config = json.loads((CONFIG_ROOT / market / (name + ".json")).read_text())
    if development is not None:
        config["windows"]["development"] = dict(
            zip(("start", "end"), development, strict=True)
        )
    if plan is not None:
        config["research_plan"] = plan
        config["windows"]["holdout"] = dict(plan["final_holdout"])
    return config


def configured_models(scenario="pv_all", *, factor_scope="core"):
    name = (
        ("core6" if factor_scope == "core" else "pv25")
        if scenario == "pv_all"
        else scenario
    )
    return ModelCatalog.from_configuration(
        research_configuration(name=name)["models"]
    ).list()


A_SHARE_MODEL_ALGORITHMS = tuple(
    model["model_algorithm"] for model in research_configuration()["models"]
)


PV25_FACTORS = tuple(research_configuration(name="pv25")["evaluation"]["factors"])
CORE6_FACTORS = tuple(research_configuration(name="core6")["evaluation"]["factors"])
