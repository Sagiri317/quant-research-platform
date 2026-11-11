"""Independent member and return checks shared by quintiles, side IC and targets."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.signals import (
    DEFAULT_N_GROUPS,
    DEFAULT_TOP_QUANTILE,
)
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)


@pytest.mark.parametrize(
    "scores,low,high",
    [
        (list(range(11)), [0, 1], [9, 10]),
        ([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2], [9, 10]),
    ],
)
def test_quintile_extremes_match_side_and_equal_weight_members(scores, low, high):
    assert DEFAULT_N_GROUPS == 5 and DEFAULT_TOP_QUANTILE == 0.2
    assets = [f"a{i}" for i in range(11)]
    labels = np.arange(11, dtype=float) / 1000
    frame = pd.DataFrame(
        dict(
            timestamp=pd.Timestamp("2024-01-02"),
            asset=assets,
            score=scores,
            label=labels,
        )
    )
    metrics, grouped = _compute_ic_and_grouped_label_values(
        frame,
        ComputePanelSpec(signal_columns=("score",), label_columns=("label",)),
        top_quantile=DEFAULT_TOP_QUANTILE,
    )
    groups = grouped.set_index("group")
    assert groups.index.tolist() == [1, 2, 3, 4, 5]
    assert groups.loc[1, "n_obs"] == metrics.short_n_selected.item() == len(low)
    assert groups.loc[5, "n_obs"] == metrics.long_n_selected.item() == len(high)
    assert groups.loc[1, "mean_label_value"] == pytest.approx(labels[low].mean())
    assert groups.loc[5, "mean_label_value"] == pytest.approx(labels[high].mean())
    signal = pd.Series(scores, index=assets, dtype=float)
    weights = build_simple_target_weights(
        signal, eligible=pd.Series(True, index=assets), portfolio_mode="long_short"
    )
    assert set(weights.index[weights.gt(0)]) == {assets[i] for i in high}
    assert set(weights.index[weights.lt(0)]) == {assets[i] for i in low}
    assert float(weights @ pd.Series(labels, index=assets)) == pytest.approx(
        0.5 * (labels[high].mean() - labels[low].mean())
    )


def test_quintile_label_missingness_does_not_refill_selected_extremes():
    frame = pd.DataFrame(
        dict(
            timestamp=pd.Timestamp("2024-01-02"),
            asset=[f"a{i}" for i in range(11)],
            score=np.arange(11),
            label=np.arange(11, dtype=float),
        )
    )
    frame.loc[0, "label"] = np.nan
    _, grouped = _compute_ic_and_grouped_label_values(
        frame, ComputePanelSpec(signal_columns=("score",), label_columns=("label",))
    )
    low = grouped.set_index("group").loc[1]
    assert low.n_obs == 1 and low.mean_label_value == 1
