"""A-share model eligibility and weighting follow the scientific protocol."""

import pytest

from quant_research_platform.contracts.models import TrainingWeighting
from quant_research_platform.research.protocols.a_share import _inference_blocker


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
