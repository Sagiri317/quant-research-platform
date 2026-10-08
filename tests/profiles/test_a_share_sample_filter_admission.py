"""Formal A-share sample-filter admission preserves shared prediction capability."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    iter_validation_prediction_chunks,
)
from quant_research_platform.research.predictions.outputs import (
    _add_a_share_scores,
    build_prediction_result_from_frames,
)
from quant_research_platform.research.protocols import a_share
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.research import research_run
from tests.helpers.research_configuration import research_configuration

ALGORITHMS = (
    "linear_regression",
    "equal_weight_rank_ensemble",
    "ic_weighted_rank_ensemble",
)


def _configuration(algorithm, market="a_share", mode="unfiltered"):
    config = research_configuration(
        market, "core6" if market == "a_share" else "baseline"
    )
    if market == "crypto":
        model = next(
            item
            for item in config["models"]
            if item["model_algorithm"] == "linear_regression"
        )
        model.update(
            model_id="unfiltered_" + algorithm,
            model_algorithm=algorithm,
            hyperparameters={},
        )
        model.pop("parameter_candidates", None)
    else:
        model = next(
            item for item in config["models"] if item["model_algorithm"] == algorithm
        )
    if mode is not None:
        model["sample_filter_mode"] = mode
    config.update(models=[model], portfolios=[], comparison_model_id=model["model_id"])
    if market == "a_share":
        config["primary_portfolio_id"] = None
    return config


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("entrypoint", ("binding", "development", "holdout"))
def test_a_share_rejects_unfiltered_models_at_formal_admission(algorithm, entrypoint):
    config = _configuration(algorithm)
    with pytest.raises(
        ValueError, match="^a_share_model_sample_filter_mode_not_supported$"
    ):
        if entrypoint == "binding":
            bind_research_configuration(research_profile(), config)
        else:
            a_share.AShareProtocol(
                config["scenario_id"],
                access_window=entrypoint,
                configuration_json=json.dumps(config),
            )


@pytest.mark.parametrize("mode", (None, "eligible"))
def test_a_share_default_and_explicit_eligible_keep_formal_selection(mode):
    config = _configuration("equal_weight_rank_ensemble", mode=mode)
    profile = bind_research_configuration(research_profile(), config)
    assert {model.sample_filter_mode for model in profile.models.list()} == {"eligible"}
    assert profile.protocol(config["scenario_id"]).target_specs("research run")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_generic_and_crypto_keep_unfiltered_model_admission(algorithm):
    config = _configuration(algorithm, market="crypto")
    assert (
        ModelCatalog.from_configuration(config["models"]).list()[0].sample_filter_mode
        == "unfiltered"
    )
    profile = bind_research_configuration(
        research_profile(config["profile_id"]), config
    )
    assert profile.models.list()[0].sample_filter_mode == "unfiltered"
    assert (
        PredictionSpec(
            "generic",
            "rank",
            "fixture",
            "2020-01-01",
            "2021-01-01",
            sample_filter_mode="unfiltered",
        ).sample_filter_mode
        == "unfiltered"
    )


@pytest.mark.parametrize("holdout", (False, True))
def test_public_a_share_workflow_rejects_unfiltered_before_any_artifact(
    tmp_path, holdout
):
    root = tmp_path / "research"
    with pytest.raises(
        ValueError, match="^a_share_model_sample_filter_mode_not_supported$"
    ):
        research_run(
            profile=research_profile(),
            layout=StorageLayout(root),
            research_plan=_configuration("equal_weight_rank_ensemble"),
            holdout=holdout,
            plan=True,
        )
    assert not root.exists()


@pytest.mark.parametrize("algorithm", ALGORITHMS[1:])
def test_raw_rank_filter_domains_explain_formal_unfiltered_rejection(algorithm):
    spec = ModelSpec(
        model_id="rank",
        model_algorithm=algorithm,
        input_factor_columns=("first", "second"),
        label_column="label",
        sample_filter_mode="unfiltered",
    )
    training = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": ["a", "b", "c"],
            "first": [0.0, 1.0, 2.0],
            "second": [0.0, 1.0, 2.0],
            "label": [0.0, 1.0, 2.0],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-03"),
        }
    )
    fit = fitting._fit_model_result_from_frames(
        (training,),
        spec,
        estimator=build_estimator_from_model_spec(spec),
        fitting_cutoff=pd.Timestamp("2020-01-04"),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2021-01-04"),
            "asset_id": ["a", "b", "c"],
            "first": [0.0, 2.0, 1.0],
            "second": [2.0, 1.0, 0.0],
            "label": [0.0, 1.0, 2.0],
            "eligible": [True, True, False],
            "label_available_at": pd.Timestamp("2021-01-05"),
            "family_id": "rank",
            "quarter_id": "2021Q1",
        }
    )

    def validation(model_spec):
        source = iter_validation_prediction_chunks(
            lambda: iter((frame.copy(),)),
            candidate_spec=model_spec,
            model_result=fit,
            validation_start=pd.Timestamp("2021-01-04"),
            validation_end=pd.Timestamp("2021-01-04"),
        )
        with closing_iterator(source):
            result = pd.concat(list(source), ignore_index=True)
        result["family_id"] = "rank"
        _add_a_share_scores(result, result.prediction_score)
        return result

    full_rank_then_eligible = validation(spec)
    assert full_rank_then_eligible.alpha_score.tolist() == [-0.25, 0.25]
    assert full_rank_then_eligible.asset_id.tolist() == ["a", "b"]
    prediction = PredictionSpec(
        "holdout",
        "rank",
        "fixture",
        None,
        "2021-01-04",
        "2021-01-04",
        prediction_mode="a_share_oos",
        prediction_column="raw_prediction",
    )
    with build_prediction_result_from_frames(
        (frame,),
        prediction,
        spec.input_columns,
        prediction_model=fit.prediction_model,
        model_metadata=fit.metadata,
    ) as result:
        eligible_first = pd.concat(list(result.output_chunks), ignore_index=True)
    assert eligible_first.alpha_score.tolist() == [0.0, 0.0]
    assert (
        validation(replace(spec, sample_filter_mode="eligible")).alpha_score.tolist()
        == eligible_first.alpha_score.tolist()
    )
