"""Native batches, explicit dependency authority, and materialized feature reuse."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.datasets import assembler
from quant_research_platform.research.features.transforms import (
    mask_unobserved_features,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
from tests.research.test_direct_assembly_config import _config, _frame, _source


def test_same_native_identity_supports_complementary_input_instances(tmp_path):
    frame = _frame(4)
    left, _ = _source(tmp_path / "left", frame, data_id="batch")
    right, _ = _source(
        tmp_path / "right",
        frame.drop(columns="momentum_20d").assign(log_float_market_cap=2.0),
        data_id="batch",
    )
    output = assembler.assemble(
        _config(
            tmp_path,
            left,
            inputs={"left": str(left), "right": str(right)},
            factor_columns=["momentum_20d", "log_float_market_cap"],
        )
    )
    actual = read_dataset(output)
    np.testing.assert_array_equal(actual.momentum_20d, frame.momentum_20d)
    assert actual.log_float_market_cap.eq(2).all()
    facts = inspect_dataset(output)
    assert facts.spec["input_data_ids"] == ["left", "right"]
    assert facts.manifest["parameters"]["input_data_ids"] == {
        "left": "batch",
        "right": "batch",
    }


@pytest.mark.parametrize("selected", [True, False])
def test_projection_selects_unique_support_producer_without_order_preference(
    tmp_path, selected
):
    frame = _frame(4).assign(eligible=True)
    left, _ = _source(tmp_path / "left", frame, data_id="batch")
    right, _ = _source(
        tmp_path / "right",
        frame.drop(columns="momentum_20d").assign(eligible=False),
        data_id="batch",
    )
    inputs = {
        "left": {
            "path": str(left),
            "columns": ["momentum_20d"] + (["eligible"] if selected else []),
        },
        "right": {"path": str(right), "columns": ["asset_id"]},
    }
    config = _config(
        tmp_path,
        left,
        inputs=inputs,
        feature_recipe="a_share_canonical",
        feature_transform={"industry_neutralize": False, "size_neutralize": False},
    )
    if not selected:
        with pytest.raises(
            ValueError, match="duplicate_research_output_columns: eligible"
        ):
            assembler.assemble(config)
        return
    actual = read_dataset(assembler.assemble(config))
    assert actual.feature_observed_momentum_20d.eq(1).all()
    config.update(
        inputs=dict(reversed(list(inputs.items()))),
        base_input_data_id="left",
        output_root=str(tmp_path / "reversed"),
    )
    pd.testing.assert_frame_equal(actual, read_dataset(assembler.assemble(config)))


def test_native_features_reuse_without_recomputation_preserves_model_contract(
    tmp_path, monkeypatch
):
    frame = _frame(4).assign(
        momentum_20d=[1.0, np.nan, 3.0, 4.0],
        eligible=True,
        return_next=[0.1, 0.2, 0.3, 0.4],
        matured_at=pd.Timestamp("2024-01-03"),
    )
    source, _ = _source(
        tmp_path,
        frame,
        units={"momentum_20d": "fraction", "return_next": "decimal_return"},
        availability={"return_next": "matured_at"},
    )
    produced = assembler.assemble(
        _config(
            tmp_path,
            source,
            feature_recipe="a_share_canonical",
            label_columns=["return_next"],
            filter_columns=["eligible"],
            feature_transform={"industry_neutralize": False, "size_neutralize": False},
        )
    )
    monkeypatch.setattr(
        assembler, "apply_feature_transforms", lambda *a, **k: pytest.fail("recomputed")
    )
    config = _config(
        tmp_path,
        produced,
        inputs={
            "features": {
                "path": str(produced),
                "columns": ["feature_momentum_20d", "return_next", "eligible"],
            }
        },
        factor_columns=[],
        feature_columns=["feature_momentum_20d"],
        label_columns=["return_next"],
        filter_columns=["eligible"],
        output_root=str(tmp_path / "reused"),
    )
    config.pop("feature_recipe")
    reused = assembler.assemble(config)
    actual = read_dataset(reused)
    columns = [
        "feature_momentum_20d",
        "feature_missing_momentum_20d",
        "feature_observed_momentum_20d",
    ]
    pd.testing.assert_frame_equal(actual[columns], read_dataset(produced)[columns])
    assert actual.feature_observed_momentum_20d.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert mask_unobserved_features(
        actual, ["feature_momentum_20d"], actual.columns
    ).feature_momentum_20d.isna().tolist() == [False, True, False, False]
    facts = inspect_dataset(reused)
    assert facts.spec["factor_columns"] == []
    assert facts.spec["feature_transforms"] == []
    assert facts.spec["column_units"]["feature_momentum_20d"] == "dimensionless"
    assert facts.spec["label_availability_columns"] == {"return_next": "matured_at"}
    assert (
        facts.manifest["producer_diagnostics"]["reused_feature_transforms"][
            "feature_momentum_20d"
        ]["scale"]
        is True
    )
    repeated = assembler.assemble(
        {
            **config,
            "inputs": {"features": str(reused)},
            "output_root": str(tmp_path / "reused_again"),
        }
    )
    pd.testing.assert_frame_equal(actual, read_dataset(repeated))
    assert (
        inspect_dataset(repeated).manifest["producer_diagnostics"][
            "reused_feature_transforms"
        ]
        == facts.manifest["producer_diagnostics"]["reused_feature_transforms"]
    )
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate

    evaluations = []
    for index, inputs in enumerate((produced, repeated)):
        output = evaluate(
            {
                "inputs": str(inputs),
                "output": str(tmp_path / f"evaluation_{index}"),
                "evaluation": {
                    "evaluation_id": "reuse",
                    "market_id": "a_share",
                    "research_dataset_id": "external_research",
                    "signals": [
                        {
                            "signal_id": "feature",
                            "source_kind": "feature",
                            "source_column": "feature_momentum_20d",
                            "transform": "identity",
                        }
                    ],
                    "label_columns": ["return_next"],
                    "filter_mode": "eligible",
                    "n_groups": 3,
                },
                "window": {"start": "2024-01-02", "end": "2024-01-02"},
                "cutoff": "2024-01-03",
            }
        )
        evaluations.append(read_evaluation_outputs(output))
    assert evaluations[1][1]["metrics"].n_obs.tolist() == [3]
    pd.testing.assert_frame_equal(
        evaluations[0][1]["metrics"], evaluations[1][1]["metrics"]
    )
    assert (
        evaluations[0][0]["metadata"]["input_facts"]["feature_transforms"]
        == evaluations[1][0]["metadata"]["input_facts"]["feature_transforms"]
    )
    model = fit(
        {
            "market_id": "a_share",
            "inputs": str(reused),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "reuse",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["feature_momentum_20d"],
                "target_column": "return_next",
                "random_seed": 17,
            },
            "fit_windows": [{"start": "2024-01-02", "end": "2024-01-02"}],
            "fit_cutoff": "2024-01-03",
        }
    )
    fitted, model_facts = load_model(model)
    assert fitted.metadata["rows_after_input_label_dropna"] == 4
    assert model_facts["input_units"] == {"feature_momentum_20d": "dimensionless"}


def test_selected_feature_cannot_collide_with_new_transform(tmp_path):
    source, _ = _source(tmp_path, _frame(4).assign(eligible=True))
    produced = assembler.assemble(
        _config(
            tmp_path,
            source,
            feature_recipe="a_share_canonical",
            feature_transform={"industry_neutralize": False, "size_neutralize": False},
        )
    )
    with pytest.raises(ValueError, match="assemble_feature_output_conflict"):
        assembler.assemble(
            _config(
                tmp_path,
                produced,
                feature_recipe="a_share_canonical",
                feature_columns=["feature_momentum_20d"],
                output_root=str(tmp_path / "collision"),
            )
        )


def test_projection_preserves_selected_units_and_label_clock_authority(tmp_path):
    frame = _frame(4).assign(
        eligible=True,
        return_next=[0.1, 0.2, 0.3, 0.4],
        matured_at=pd.Timestamp("2024-01-03"),
    )
    left, _ = _source(
        tmp_path / "left",
        frame,
        data_id="batch",
        units={"momentum_20d": "fraction", "return_next": "decimal_return"},
        availability={"return_next": "matured_at"},
    )
    right, _ = _source(
        tmp_path / "right",
        frame.assign(matured_at=pd.Timestamp("2030-01-01")),
        data_id="batch",
        units={"momentum_20d": "percent", "return_next": "percent"},
        availability={"return_next": "matured_at"},
    )
    output = assembler.assemble(
        _config(
            tmp_path,
            left,
            inputs={
                "left": {
                    "path": str(left),
                    "columns": ["momentum_20d", "return_next", "eligible"],
                },
                "right": {"path": str(right), "columns": ["asset_id"]},
            },
            label_columns=["return_next"],
            feature_recipe="a_share_canonical",
            feature_transform={
                "industry_neutralize": False,
                "size_neutralize": False,
                "center": False,
                "scale": False,
            },
        )
    )
    spec = inspect_dataset(output).spec
    assert spec["column_units"]["feature_momentum_20d"] == "fraction"
    assert spec["column_units"]["return_next"] == "decimal_return"
    assert read_dataset(output).matured_at.eq(pd.Timestamp("2024-01-03")).all()


def test_same_native_identity_never_allows_conflicting_business_columns(tmp_path):
    left, _ = _source(tmp_path / "left", _frame(3), data_id="batch")
    right, _ = _source(
        tmp_path / "right", _frame(3).assign(momentum_20d=999.0), data_id="batch"
    )
    with pytest.raises(
        ValueError, match="duplicate_research_output_columns: momentum_20d"
    ):
        assembler.assemble(
            _config(tmp_path, left, inputs={"left": str(left), "right": str(right)})
        )


@pytest.mark.parametrize("feature", ["previous_feature", "feature_momentum_20d"])
def test_borrowed_feature_does_not_disable_default_factor_transform(tmp_path, feature):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.storage.datasets import write_dataset

    frame = _frame().assign(
        eligible=True,
        industry_standard_id="test",
        industry_level1_code="A",
        log_total_market_cap=np.arange(120) / 10,
        **{feature: np.arange(120, dtype=float)},
    )
    _, spec = _source(tmp_path, frame)
    source = tmp_path / "with_features"
    write_dataset(
        source,
        frame,
        spec=replace(
            spec, column_roles={**spec.column_roles, feature: ColumnRole.FEATURE}
        ),
    )
    config = _config(tmp_path, source, feature_columns=[feature])
    config.pop("feature_recipe")
    if feature == "feature_momentum_20d":
        with pytest.raises(ValueError, match="assemble_feature_output_conflict"):
            assembler.assemble(config)
        config["feature_recipe"] = None
        output = assembler.assemble(config)
        np.testing.assert_array_equal(read_dataset(output)[feature], frame[feature])
        return
    output = assembler.assemble(config)
    actual = read_dataset(output)
    np.testing.assert_array_equal(actual[feature], frame[feature])
    assert actual.feature_observed_momentum_20d.eq(1).all()
    assert (
        inspect_dataset(output).spec["feature_transforms"][0]["recipe_id"]
        == "a_share_canonical"
    )
