"""A-share weighting accepts explicit decay independently of research roles."""

import pytest

from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
    validate_model_selection,
)


@pytest.mark.parametrize(
    "half_lives",
    [
        (None,),
        (None, None),
        (756,),
        (756, 756),
        (378,),
        (0.5,),
        (None, 378),
        (None, 756),
        (756, 378),
    ],
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


@pytest.mark.parametrize("periods", [1, 378, 756])
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
