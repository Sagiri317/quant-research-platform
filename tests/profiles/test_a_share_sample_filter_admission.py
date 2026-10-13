"""A-share admission at the scientific owner, with real rank-domain regression."""

from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.protocols.a_share import validate_model_selection
from tests.research.direct_model_test_support import fit_frame, predict_frame

ALGORITHMS = (
    "linear_regression",
    "equal_weight_rank_ensemble",
    "ic_weighted_rank_ensemble",
)


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
        assert validate_model_selection([spec]) == "primary"


@pytest.mark.parametrize("algorithm", ALGORITHMS[1:])
def test_native_rank_filter_domains_explain_formal_unfiltered_rejection(
    tmp_path, algorithm
):
    spec = ModelSpec(
        model_id="rank",
        model_algorithm=algorithm,
        input_factor_columns=("first", "second"),
        target_column="label",
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
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2021-01-04"),
            "asset_id": ["a", "b", "c"],
            "first": [0.0, 2.0, 1.0],
            "second": [2.0, 1.0, 0.0],
            "label": [0.0, 1.0, 2.0],
            "eligible": [True, True, False],
            "label_available_at": pd.Timestamp("2021-01-05"),
        }
    )

    with pytest.raises(
        ValueError, match="a_share_model_sample_filter_mode_not_supported"
    ):
        fit_frame(
            tmp_path / "rejected",
            training,
            spec,
            cutoff=pd.Timestamp("2020-01-04"),
            market_id="a_share",
        )
    assert not (tmp_path / "rejected" / "model").exists()

    unfiltered = fit_frame(
        tmp_path / "unfiltered",
        training,
        spec,
        cutoff=pd.Timestamp("2020-01-04"),
        market_id="crypto",
    )
    eligible = fit_frame(
        tmp_path / "eligible",
        training,
        replace(spec, sample_filter_mode="eligible"),
        cutoff=pd.Timestamp("2020-01-04"),
        market_id="a_share",
    )
    window = {"start": "2021-01-04", "end": "2021-01-04"}
    full, _, full_metadata = predict_frame(
        tmp_path / "full", frame, unfiltered, window=window
    )
    eligible_first, _, metadata = predict_frame(
        tmp_path / "filtered", frame, eligible, window=window
    )
    assert full_metadata["model_metadata"]["sample_filter_mode"] == "unfiltered"
    assert metadata["model_metadata"]["sample_filter_mode"] == "eligible"
    assert full.asset_id.tolist() == ["a", "b", "c"]
    assert full.prediction_rank_score.tolist() == pytest.approx([0.0, 1 / 3, -1 / 3])
    # Filtering after scoring retains different ranks for a/b; filtering before
    # either factor rank makes a/b exact ties. This is why A-share rejects override.
    late_filter = full.loc[full.asset_id.isin(["a", "b"]), "prediction_score"]
    assert ((late_filter.rank(method="average") - 1.5) / 2).tolist() == [-0.25, 0.25]
    assert eligible_first.asset_id.tolist() == ["a", "b"]
    assert eligible_first.prediction_score.tolist() == [0.0, 0.0]
    assert eligible_first.prediction_rank_score.tolist() == [0.0, 0.0]
