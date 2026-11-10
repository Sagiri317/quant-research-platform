"""A-share model eligibility and weighting follow the scientific protocol."""

import pytest

from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
    validate_model_selection,
)


@pytest.mark.parametrize(
    "half_lives",
    [(None, None), (756, 378), (None, 0.5)],
)
def test_a_share_accepts_explicit_positive_decay_candidates(half_lives):
    models = [
        ModelSpec(
            model_id=f"candidate_{i}",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
            training_weighting=TrainingWeighting(half_life_periods=value),
        )
        for i, value in enumerate(half_lives)
    ]
    validate_model_selection(models)


@pytest.mark.parametrize("periods", [1, 0.5])
def test_generic_weighting_keeps_other_positive_decay_contracts(periods):
    assert TrainingWeighting(half_life_periods=periods).half_life_periods == periods


@pytest.mark.parametrize("periods", [0, -1, True, float("inf"), float("nan"), "378"])
def test_decay_rejects_nonpositive_nonfinite_or_nonnumeric_values(periods):
    with pytest.raises(
        ValueError, match="training_weighting_half_life_periods_invalid"
    ):
        TrainingWeighting(half_life_periods=periods)


@pytest.mark.parametrize("role", ["primary", "robustness"])
def test_explicit_robustness_role_still_blocks_formal_joint_inference(role):
    assert _inference_blocker({"training_weighting": {"role": role}}) == (
        None if role == "primary" else "training_weighting_robustness_excluded"
    )


@pytest.mark.parametrize("mode", ("eligible", "unfiltered"))
def test_a_share_filter_admission(mode):
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        sample_filter_mode=mode,
    )
    if mode == "unfiltered":
        with pytest.raises(
            ValueError, match="a_share_model_sample_filter_mode_not_supported"
        ):
            validate_model_selection([spec])
    else:
        validate_model_selection([spec])
