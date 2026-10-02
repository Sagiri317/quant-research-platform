"""Read the same editable research configurations used by production."""

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import patch

from quant_research_platform.catalog.models import ModelCatalog

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "research"


def research_configuration(
    market="a_share", name="core6", *, development=None, plan=None, small_models=False
):
    config = json.loads(
        (CONFIG_ROOT / market / (name + ".json")).read_text(encoding="utf-8")
    )
    if development is not None:
        config["windows"]["development"] = dict(
            zip(("start", "end"), development, strict=True)
        )
    if plan is not None:
        config["research_plan"] = plan
        config["windows"]["holdout"] = dict(plan["test"])
        config["source_archives"] = None
    if small_models:
        for model in config["models"]:
            if model["model_algorithm"] in {
                "lightgbm_regressor",
                "xgboost_regressor",
                "gradient_boosting_regressor",
                "random_forest_regressor",
            }:
                for parameters in (
                    model["hyperparameters"],
                    *model.get("parameter_candidates", ()),
                ):
                    parameters["n_estimators"] = 16
    return config


@contextmanager
def compact_a_share_windows():
    """Restore production dates after each bounded synthetic acceptance call."""
    from quant_research_platform.research.protocols import a_share

    windows = {
        "DATA_START": "2016-07-21",
        "DATA_END": "2019-12-31",
        "FORMAL_DATASET_START": "2017-10-01",
        "FORMAL_DATASET_END": "2019-12-31",
        "DEVELOPMENT_END": "2019-06-30",
        "HOLDOUT_START": "2019-07-01",
        "HOLDOUT_END": "2019-12-31",
    }
    with patch.multiple(
        a_share,
        **{field: date.fromisoformat(value) for field, value in windows.items()},
    ):
        yield


def compact_a_share_configuration(*, name="core6"):
    """Bind the existing test configuration to the scoped protocol windows."""
    from quant_research_platform.research.protocols import a_share

    config = research_configuration(
        name=name,
        development=(
            a_share.DEVELOPMENT_START.isoformat(),
            a_share.DEVELOPMENT_END.isoformat(),
        ),
        small_models=True,
    )
    config["windows"]["holdout"] = {
        "start": a_share.HOLDOUT_START.isoformat(),
        "end": a_share.HOLDOUT_END.isoformat(),
    }
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
