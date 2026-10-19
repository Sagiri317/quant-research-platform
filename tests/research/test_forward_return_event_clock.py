"""Persisted open-to-open labels are unusable until the adjustment-source close."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.storage.datasets.writing import write_dataset


@pytest.mark.parametrize(
    "cutoff,mature_rows",
    [
        ("2024-01-04", 0),
        ("2024-01-04T06:59:59.999999999Z", 0),
        ("2024-01-04T07:00:00Z", 3),
        ("2024-01-04T15:00:00+08:00", 3),
    ],
)
def test_native_fit_and_evaluate_respect_adjustment_source_close(
    tmp_path, cutoff, mature_rows
):
    sessions = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame(
        [
            dict(
                asset_id=asset,
                timestamp=date,
                adj_open=100.0 if date != sessions[-1] else 110.0 + 10.0 * index,
                filling_flag=0,
                is_trading=True,
                x=float(index + 1),
                eligible=True,
            )
            for index, asset in enumerate("abc")
            for date in sessions
        ]
    )
    label = "forward_return_1d"
    availability = f"{label}_available_at"
    frame = add_forward_return_labels(prices, horizons=(1,), sessions=sessions)
    assert (
        frame.loc[frame.timestamp.eq(sessions[0]), availability]
        .eq(pd.Timestamp("2024-01-04T07:00:00"))
        .all()
    )
    source = tmp_path / "labels"
    write_dataset(
        source,
        frame,
        spec=ProcessedDataSpec(
            market_id="a_share",
            kind="panel",
            data_id="clock",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
            column_roles={
                "timestamp": "timestamp",
                "asset_id": "key",
                "adj_open": "context",
                "filling_flag": "context",
                "is_trading": "context",
                "eligible": "mask",
                "x": "factor",
                label: "label",
                availability: "context",
            },
            column_units={"x": "dimensionless", label: "decimal_return"},
            label_availability_columns={label: availability},
        ),
    )
    window = dict(start="2024-01-02", end="2024-01-02")
    request = dict(
        inputs=str(source),
        output=str(tmp_path / "model"),
        market_id="a_share",
        model=dict(
            model_id="clock",
            model_algorithm="linear_regression",
            input_factor_columns=["x"],
            target_column=label,
            random_seed=1,
        ),
        fit_windows=[window],
        fit_cutoff=cutoff,
    )
    if not mature_rows:
        with pytest.raises(ValueError, match="model_training_sample_empty"):
            fit(request)
        assert not (tmp_path / "model").exists()
    else:
        metadata = load_model(fit(request))[0].metadata
        assert metadata["rows_after_input_label_dropna"] == mature_rows
        assert pd.Timestamp(metadata["label_available_max"]) == pd.Timestamp(
            "2024-01-04T07:00:00"
        )
    evaluated = evaluate(
        dict(
            inputs=str(source),
            output=str(tmp_path / "evaluation"),
            evaluation=dict(
                evaluation_id="clock",
                research_dataset_id="clock",
                market_id="a_share",
                signals=[dict(signal_id="x", source_kind="factor", source_column="x")],
                label_columns=[label],
                n_groups=2,
            ),
            window=window,
            cutoff=cutoff,
        )
    )
    metrics = read_evaluation_outputs(evaluated)[1]["metrics"]
    assert metrics.n_obs.tolist() == [mature_rows]
    if mature_rows:
        assert metrics.ic.iloc[0] == pytest.approx(1.0)


def test_wd_adjusted_open_depends_on_exit_close_even_with_fixed_raw_open():
    """Cash reinvestment leaves D/C in a cross-session adjusted-open ratio."""
    sessions = pd.bdate_range("2024-01-02", periods=3)
    labels = []
    for closing in (100.0, 200.0):
        # Hold raw opening, dividend and prior raw/adjusted close fixed.
        dividend = 10.0
        adjusted_close = 100.0 * (closing + dividend) / 100.0
        adjusted_open = 100.0 * adjusted_close / closing
        prices = pd.DataFrame(
            {
                "asset_id": "a",
                "timestamp": sessions,
                "adj_open": [100.0, 100.0, adjusted_open],
                "filling_flag": 0,
                "is_trading": True,
            }
        )
        row = add_forward_return_labels(prices, horizons=(1,), sessions=sessions).iloc[
            0
        ]
        labels.append(row.forward_return_1d)
        assert row.forward_return_1d_available_at == sessions[2] + pd.Timedelta(hours=7)
    assert labels == pytest.approx([0.1, 0.05])
