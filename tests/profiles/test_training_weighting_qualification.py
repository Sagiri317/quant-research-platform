"""Formal A-share decay qualification is owned by its market protocol."""

import json

import pytest

from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.protocols import a_share
from tests.helpers.research_configuration import research_configuration


def _configuration(half_lives, market="a_share"):
    config = research_configuration(
        market, "core6" if market == "a_share" else "baseline"
    )
    config["models"] = config["models"][: len(half_lives)]
    config.update(portfolios=[], comparison_model_id=config["models"][0]["model_id"])
    if market == "a_share":
        config["primary_portfolio_id"] = None
    for model, half_life in zip(config["models"], half_lives, strict=True):
        model["training_weighting"] = {
            "method": "equal_timestamp",
            "half_life": half_life,
        }
    return config


@pytest.mark.parametrize("entrypoint", ("binding", "protocol"))
@pytest.mark.parametrize(
    "half_lives,reason",
    [
        ((None,), None),
        ((None, None), None),
        ((756,), None),
        ((756, 756), None),
        ((378,), "a_share_training_weighting_half_life_not_supported"),
        ((None, 378), "a_share_training_weighting_half_life_not_supported"),
        ((None, 756), "a_share_training_weighting_research_roles_mixed"),
        ((756, 378), "a_share_training_weighting_research_roles_mixed"),
    ],
)
def test_a_share_formal_decay_admission(half_lives, reason, entrypoint):
    config = _configuration(half_lives)

    def bind():
        if entrypoint == "protocol":
            return a_share.AShareProtocol(
                config["scenario_id"], configuration_json=json.dumps(config)
            )
        selected = bind_research_configuration(research_profile(), config)
        return selected.protocol(config["scenario_id"])

    if reason is not None:
        with pytest.raises(ValueError, match="^" + reason + "$"):
            bind()
        return
    protocol = bind()
    role = "primary" if half_lives[0] is None else "robustness"
    assert protocol.manifest()["training_weighting"] == {"role": role}
    blocker = None if role == "primary" else "training_weighting_robustness_excluded"
    for target_id in (
        *protocol.statistics_target_ids,
        config["scenario_id"] + "_portfolio_statistics",
    ):
        manifest = protocol.target_manifest(target_id, "STATISTIC")
        assert manifest["training_weighting"] == {"role": role}
        assert a_share._inference_blocker(manifest) == blocker


@pytest.mark.parametrize("half_lives", ((1,), (378,), (None, 378)))
def test_generic_and_crypto_keep_other_decay_contracts(half_lives):
    config = _configuration(half_lives, "crypto")
    assert {
        model.training_weighting.half_life
        for model in ModelCatalog.from_configuration(config["models"]).list()
    } == set(half_lives)
    selected = bind_research_configuration(
        research_profile(config["profile_id"]), config
    )
    assert {
        model.training_weighting.half_life for model in selected.models.list()
    } == set(half_lives)
