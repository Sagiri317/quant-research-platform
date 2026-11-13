from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.parametrize("operation", ("fit", "evaluate"))
@pytest.mark.parametrize(
    "case",
    (
        "valid",
        "undeclared_input",
        "wrong_input_role",
        "undeclared_label",
        "missing_label",
    ),
)
def test_direct_consumers_require_actual_declared_inputs(tmp_path, operation, case):
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.models.production import fit
    from tests.correctness.test_direct_model_operations import data, fit_config

    source = data(tmp_path)
    config = fit_config(tmp_path, source)
    declared = config["inputs"]["spec"]
    if case == "undeclared_input":
        del declared["column_roles"]["x"]
    elif case == "wrong_input_role":
        declared["column_roles"]["x"] = "context"
    elif case == "undeclared_label":
        del declared["column_roles"]["label"]
    elif case == "missing_label":
        pd.read_parquet(source).drop(columns="label").to_parquet(source, index=False)
    if operation == "fit":
        run = fit
        expected = {
            "undeclared_input": "model_feature_role_mismatch",
            "wrong_input_role": "model_feature_role_mismatch",
            "undeclared_label": "model_target_role_mismatch",
            "missing_label": "dataset_column_not_found",
        }
    else:
        run = evaluate
        config = {
            "inputs": {"path": str(source), "spec": declared},
            "output": str(tmp_path / "evaluation"),
            "evaluation": {
                "evaluation_id": "declared-inputs",
                "market_id": "a_share",
                "research_dataset_id": "synthetic",
                "signals": [
                    {"signal_id": "x", "source_kind": "feature", "source_column": "x"}
                ],
                "label_columns": ["label"],
                "n_groups": 2,
            },
            "window": {"start": "2020-01-01", "end": "2020-01-03"},
            "cutoff": "2020-01-03",
        }
        expected = {
            "undeclared_input": "evaluation_signal_role_mismatch",
            "wrong_input_role": "evaluation_signal_role_mismatch",
            "undeclared_label": "evaluation_label_role_mismatch",
            "missing_label": "dataset_column_not_found",
        }
    if case == "valid":
        assert run(config).is_dir()
    else:
        with pytest.raises(ValueError, match=expected[case]):
            run(config)
        assert not (
            tmp_path / ("linear" if operation == "fit" else "evaluation")
        ).exists()


@pytest.mark.parametrize(
    "primary_key",
    (
        ("timestamp", "asset_id"),
        ("asset_id", "timestamp"),
        ("timestamp", "asset_id", "venue"),
    ),
)
@pytest.mark.parametrize("duplicate", (False, True))
def test_evaluation_key_validation_has_one_owner_per_invariant(
    tmp_path, monkeypatch, primary_key, duplicate
):
    from quant_research_platform.research.evaluations import runner
    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

    dates = pd.date_range("2024-01-01", periods=4, freq="s")
    frame = pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id="asset_a",
            x=[1.0, 2.0, 3.0, 4.0],
            y=[2.0, 4.0, 6.0, 8.0],
            available=dates,
            eligible=True,
            venue="a",
        )
    )
    if duplicate:
        extra = frame.iloc[[0]].assign(venue="b")
        frame = pd.concat([frame.iloc[:1], extra, frame.iloc[1:]], ignore_index=True)
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    spec = dict(
        data_id="proof",
        frequency="tick",
        grain="asset_timestamp",
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "y": "label",
            "available": "context",
            "eligible": "mask",
            "venue": "key",
        },
        primary_key=primary_key,
        partition_columns=(),
    )
    indexes = []
    original_add = ExactPrimaryKeyIndex.add_table

    def observed_add(self, table, **kwargs):
        if {"timestamp", "asset_id"} <= set(table.column_names) and not any(
            index is self for index in indexes
        ):
            indexes.append(self)
        return original_add(self, table, **kwargs)

    monkeypatch.setattr(ExactPrimaryKeyIndex, "add_table", observed_add)
    original_chunks = runner.iter_dataset_time_chunks
    monkeypatch.setattr(
        runner,
        "iter_dataset_time_chunks",
        lambda session, **kwargs: original_chunks(session, batch_rows=1, **kwargs),
    )
    config = dict(
        inputs={"path": str(source), "spec": spec},
        output=str(tmp_path / "evaluation"),
        window={"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        cutoff="2024-01-02",
        evaluation=dict(
            evaluation_id="proof",
            market_id="other_market",
            research_dataset_id="proof",
            signals=[dict(signal_id="x", source_kind="factor", source_column="x")],
            label_columns=["y"],
            label_availability_columns={"y": "available"},
            evaluation_axis="single_asset_time_series",
            n_groups=2,
        ),
    )
    if duplicate:
        error = (
            "evaluation_input_duplicate_keys"
            if len(primary_key) > 2
            else "duplicate_dataset_primary_key"
        )
        with pytest.raises(ValueError, match=error):
            runner.evaluate(config)
    else:
        assert runner.evaluate(config).is_dir()
    # Projection omits venue: the wider-key case has only the narrower owner.
    assert len(indexes) == 1
