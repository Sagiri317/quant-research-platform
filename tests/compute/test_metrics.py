from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from tests.helpers.compute_inputs import make_perfect_cross_section_compute_panel


def test_validate_compute_panel_normalizes_and_sorts_valid_panel() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor",), label_columns=("forward_return",)
    )

    result = validate_compute_panel(
        make_perfect_cross_section_compute_panel().iloc[::-1],
        spec,
    )

    assert result["timestamp"].tolist()[0] == pd.Timestamp("2026-01-01")
    assert result["asset"].tolist()[:3] == ["A", "B", "C"]


def test_validate_compute_panel_rejects_missing_signal_column() -> None:
    spec = ComputePanelSpec(
        signal_columns=("missing",), label_columns=("forward_return",)
    )

    with pytest.raises(ValueError, match="panel is missing columns"):
        validate_compute_panel(
            make_perfect_cross_section_compute_panel(),
            spec,
        )


def test_validate_compute_panel_rejects_duplicate_timestamp_asset_keys() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor",), label_columns=("forward_return",)
    )
    panel = pd.concat(
        [
            make_perfect_cross_section_compute_panel(),
            make_perfect_cross_section_compute_panel().head(1),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        validate_compute_panel(panel, spec)


def test_compute_ic_and_groups_have_independent_nonperfect_oracles() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor",), label_columns=("forward_return",)
    )
    panel = make_perfect_cross_section_compute_panel()
    panel.loc[panel.timestamp.eq(panel.timestamp.min()), "factor"] = [1.0, 2.0, 4.0]
    panel.loc[panel.timestamp.eq(panel.timestamp.min()), "forward_return"] = [
        1.0,
        4.0,
        2.0,
    ]
    metrics, groups = _compute_ic_and_grouped_label_values(panel, spec, n_groups=3)

    assert metrics["timestamp"].nunique() == 2
    assert metrics["signal"].tolist() == ["factor", "factor"]
    # Centered dot products give 1/7; the rank permutations give 1/2.
    assert metrics["ic"].tolist() == pytest.approx([1 / 7, 1.0], rel=1e-13)
    assert metrics["rankic"].tolist() == pytest.approx([0.5, 1.0], rel=1e-13)
    assert metrics["n_obs"].tolist() == [3, 3]
    assert groups["group"].tolist() == [1, 2, 3, 1, 2, 3]
    assert groups["mean_label_value"].tolist() == pytest.approx(
        [1.0, 4.0, 2.0, 0.01, 0.02, 0.03], rel=1e-13
    )


def test_grouped_label_values_keep_tie_classes_and_real_tiny_variation() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_columns=("label",))
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"] * 4,
            "asset": ["d", "c", "b", "a"],
            "factor": [0.0, 1.0, 1.0, 2.0],
            "label": [0.0, 1.0, 3.0, 4.0],
        }
    )

    _, grouped = _compute_ic_and_grouped_label_values(frame, spec, n_groups=4)
    _, permuted = _compute_ic_and_grouped_label_values(
        frame.iloc[::-1].assign(asset=["w", "x", "y", "z"]),
        spec,
        n_groups=4,
    )
    _, tiny = _compute_ic_and_grouped_label_values(
        frame.iloc[:2].assign(factor=[0.0, np.nextafter(0.0, 1.0)]),
        spec,
        n_groups=2,
    )

    assert grouped["group"].tolist() == [1, 2, 4]
    assert grouped["n_obs"].tolist() == [1, 2, 1]
    pd.testing.assert_frame_equal(grouped, permuted)
    assert tiny["group"].tolist() == [1, 2]


@pytest.mark.parametrize(
    "labels", [(), ("label", "label"), ("",), ("factor",), ("timestamp",), ("asset",)]
)
def test_compute_panel_rejects_invalid_label_columns(labels):
    with pytest.raises(ValueError):
        ComputePanelSpec(signal_columns=("factor",), label_columns=labels)
