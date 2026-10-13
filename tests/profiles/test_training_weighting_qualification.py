"""Formal A-share weighting qualification survives independent operation calls."""

import pytest

from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
from quant_research_platform.research.protocols.a_share import (
    _inference_blocker,
    validate_model_selection,
)


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
def test_a_share_formal_decay_admission(half_lives, reason):
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
    if reason:
        with pytest.raises(ValueError, match=reason):
            validate_model_selection(models)
    else:
        role = validate_model_selection(models)
        assert role == ("primary" if half_lives[0] is None else "robustness")
        assert _inference_blocker({"training_weighting": {"role": role}}) == (
            None if role == "primary" else "training_weighting_robustness_excluded"
        )


@pytest.mark.parametrize("periods", [1, 378, 756])
def test_generic_weighting_keeps_other_positive_decay_contracts(periods):
    assert TrainingWeighting(half_life_periods=periods).half_life_periods == periods
