"""A-share admission at the scientific owner, with real rank-domain regression."""

import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.protocols.a_share import validate_model_selection

ALGORITHMS = ("linear_regression",)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("mode", ("eligible", "unfiltered"))
def test_a_share_filter_admission(algorithm, mode):
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm=algorithm,
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
