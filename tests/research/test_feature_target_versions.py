"""Whole feature groups and target transforms retain scientific clocks and NULLs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.research.datasets.assembler import (
    _a_share_model_target,
    _add_a_share_semantic_columns,
    assemble,
)
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
    model_observed_input_mask,
)
from quant_research_platform.research.protocols.a_share import research_dataset_catalog


def _frame(n=120):
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": [f"a{i:03d}" for i in range(n)],
            "market_id": "a_share",
            "eligible": True,
            "factor": np.sin(x) + x / 100,
            "other": np.cos(x),
            "industry_standard_id": "test",
            "industry_level1_code": "I",
            "log_total_market_cap": x / 10,
            "forward_return_20d": np.sin(x) / 100 + x / 1000,
            "forward_return_20d_available_at": pd.Timestamp("2020-02-01"),
        }
    )


def test_feature_groups_share_clipping_and_keep_independent_domains():
    frame = _frame()
    frame.loc[0, "industry_level1_code"] = None
    frame.loc[1, "factor"] = np.nan
    frame.loc[2, "other"] = np.nan
    spec = FeatureTransformSpec(
        "a_share_canonical",
        ("factor", "other"),
        output_groups=("pre_neutral", "processed"),
    )
    result = apply_feature_transforms(frame, (spec,))
    for group in spec.output_groups:
        expected = apply_feature_transforms(
            frame, (replace(spec, output_groups=(group,)),)
        )
        for column in replace(spec, output_groups=(group,)).output_columns:
            pd.testing.assert_series_equal(result[column], expected[column])
    assert pd.notna(result.feature_pre_neutral_factor.iloc[0])
    assert pd.isna(result.feature_factor.iloc[0])
    assert result.feature_observed_pre_neutral_factor.iloc[1] == 0
    assert result.feature_observed_pre_neutral_other.iloc[1] == 1
    assert not model_observed_input_mask(
        result, ("feature_pre_neutral_factor",), result.columns
    ).iloc[1]
    pd.testing.assert_series_equal(result.factor, frame.factor)


def test_pre_neutral_only_needs_no_neutralization_facts():
    frame = _frame()[["timestamp", "eligible", "factor"]]
    spec = FeatureTransformSpec(
        "a_share_canonical", ("factor",), output_groups=("pre_neutral",)
    )
    assert spec.support_columns == ("eligible",)
    assert (
        apply_feature_transforms(frame, (spec,))
        .feature_pre_neutral_factor.notna()
        .all()
    )
    with pytest.raises(ValueError, match="duplicates"):
        FeatureTransformSpec(
            "a_share_canonical",
            ("factor", "pre_neutral_factor"),
            output_groups=("processed", "pre_neutral"),
        )


def test_target_constant_single_missing_and_percentile():
    frame = pd.DataFrame(
        {
            "timestamp": [1, 1, 1, 2, 2],
            "eligible": True,
            "forward_return_20d": [0.2, 0.2, np.nan, 0.5, np.nan],
        }
    )
    result = _a_share_model_target(frame)
    np.testing.assert_allclose(result.iloc[:2], [0, 0])
    assert result.iloc[2:].isna().all()
    result = _a_share_model_target(frame, transform="percentile")
    np.testing.assert_allclose(result.iloc[:2], [0.75, 0.75])
    assert result.iloc[2:].isna().all()
    pd.testing.assert_series_equal(
        _a_share_model_target(frame, transform="raw"),
        frame.forward_return_20d,
        check_names=False,
    )


def test_target_residual_uses_only_same_row_exposures_and_actual_clocks():
    frame = _frame()
    frame.loc[0, "industry_level1_code"] = None
    frame.loc[0, "forward_return_20d_available_at"] = pd.Timestamp("2020-03-01")
    frame.loc[1, "forward_return_20d_available_at"] = pd.Timestamp("2020-02-04")
    spec = replace(
        research_dataset_catalog().get("pv_all"),
        eligibility_recipe=None,
        target_transform="industry_size_residual",
        target_clip="mad",
    )
    result = _add_a_share_semantic_columns(frame, spec)
    assert pd.isna(result.model_target.iloc[0])
    assert pd.isna(result.model_target_available_at.iloc[0])
    assert (
        result.model_target_available_at.iloc[1:].eq(pd.Timestamp("2020-02-04")).all()
    )
    valid = result.model_target.dropna()
    assert abs(valid.sum()) < 1e-12
    assert abs(np.dot(valid, frame.loc[valid.index, "log_total_market_cap"])) < 1e-12
    changed = frame.copy()
    changed.loc[0, "forward_return_20d"] = 1e10
    pd.testing.assert_series_equal(
        _add_a_share_semantic_columns(changed, spec).model_target, result.model_target
    )
    pd.testing.assert_series_equal(result.forward_return_20d, frame.forward_return_20d)


@pytest.mark.parametrize(
    "groups",
    [["raw"], ["pre_neutral"], ["raw", "processed", "pre_neutral"]],
)
def test_assemble_persists_selected_groups_target_and_input_columns(tmp_path, groups):
    from quant_research_platform.contracts.columns import (
        PLATFORM_ASSET_TIME_KEY,
        ColumnRole,
    )
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.storage.datasets import write_dataset
    from quant_research_platform.storage.datasets.reading import (
        materialize_dataset_session,
        open_dataset_session,
    )

    frame = _frame()
    roles = {column: ColumnRole.CONTEXT for column in frame}
    roles.update(
        market_id=ColumnRole.KEY,
        asset_id=ColumnRole.KEY,
        timestamp=ColumnRole.TIMESTAMP,
        eligible=ColumnRole.UNIVERSE,
        factor=ColumnRole.FACTOR,
        other=ColumnRole.FACTOR,
        forward_return_20d=ColumnRole.LABEL,
    )
    source = ProcessedDataSpec(
        data_id="input",
        market_id="a_share",
        kind="panel",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
        column_roles=roles,
        column_units={
            "factor": "dimensionless",
            "forward_return_20d": "decimal_return",
        },
        label_availability_columns={
            "forward_return_20d": "forward_return_20d_available_at"
        },
    )
    write_dataset(tmp_path / "input", frame, spec=source)
    root = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "versions",
            "inputs": {
                "input": {
                    "path": str(tmp_path / "input"),
                    "columns": list(frame.columns),
                }
            },
            "output_root": str(tmp_path / "output"),
            "time_window": {"start": "2020-01-02", "end": "2020-01-02"},
            "factor_columns": ["factor", "other"],
            "feature_groups": groups,
            "filter_columns": ["eligible"],
            "label_columns": ["model_target"],
            "target_transform": "raw",
        }
    )
    session = open_dataset_session(root)
    result = materialize_dataset_session(session)
    assert ("factor" in result) == ("raw" in groups)
    assert ("feature_factor" in result) == ("processed" in groups)
    assert ("feature_pre_neutral_factor" in result) == ("pre_neutral" in groups)
    pd.testing.assert_series_equal(
        result.model_target, frame.forward_return_20d, check_names=False
    )
    assert session.spec["target_transform"] == "raw"
    assert session.spec["target_clip"] == "none"
    assert session.spec["column_units"]["model_target"] == "decimal_return"
    assert session.manifest["parameters"]["feature_groups"] == groups
    assert "factor" in session.manifest["parameters"]["input_columns"]["input"]
    if set(groups) == {"raw", "processed", "pre_neutral"}:
        default_config = {
            "market_id": "a_share",
            "dataset_id": "default_reuse",
            "inputs": {"versions": str(root)},
            "output_root": str(tmp_path / "default_reuse"),
            "time_window": {"start": "2020-01-02", "end": "2020-01-02"},
            "label_columns": [],
            "filter_columns": ["eligible"],
        }
        default_session = open_dataset_session(assemble(default_config))
        default_reused = materialize_dataset_session(default_session)
        assert default_session.spec["feature_transforms"] == []
        for column in session.spec["feature_columns"]:
            pd.testing.assert_series_equal(default_reused[column], result[column])
        with pytest.raises(ValueError, match="assemble_feature_output_conflict"):
            assemble({**default_config, "feature_recipe": "a_share_canonical"})

        new_frame = frame[
            [
                *PLATFORM_ASSET_TIME_KEY,
                "industry_standard_id",
                "industry_level1_code",
                "log_total_market_cap",
            ]
        ].copy()
        new_frame["new"] = np.cos(np.arange(len(frame)))
        new_spec = replace(
            source,
            data_id="new_input",
            column_roles={
                **{column: roles[column] for column in new_frame if column != "new"},
                "new": ColumnRole.FACTOR,
            },
            label_availability_columns={},
            column_units={},
        )
        write_dataset(tmp_path / "new_input", new_frame, spec=new_spec)
        mixed_session = open_dataset_session(
            assemble(
                {
                    **default_config,
                    "output_root": str(tmp_path / "mixed_reuse"),
                    "inputs": {
                        "versions": str(root),
                        "new_input": str(tmp_path / "new_input"),
                    },
                }
            )
        )
        mixed = materialize_dataset_session(mixed_session)
        assert mixed_session.spec["feature_transforms"][0]["source_columns"] == ["new"]
        assert mixed.feature_new.notna().all()
        for column in session.spec["feature_columns"]:
            pd.testing.assert_series_equal(mixed[column], result[column])
        for group in groups:
            reused_config = {
                "market_id": "a_share",
                "dataset_id": "reuse",
                "inputs": {"versions": str(root)},
                "output_root": str(tmp_path / ("reuse_" + group)),
                "time_window": {"start": "2020-01-02", "end": "2020-01-02"},
                "feature_recipe": None,
                "feature_groups": [group],
                "label_columns": [],
                "filter_columns": ["eligible"],
            }
            reused = materialize_dataset_session(
                open_dataset_session(assemble(reused_config))
            )
            assert ("factor" in reused) == (group == "raw")
            assert ("feature_factor" in reused) == (group == "processed")
            assert ("feature_pre_neutral_factor" in reused) == (group == "pre_neutral")
            if group == "raw":
                with pytest.raises(ValueError, match="assemble_feature_group_conflict"):
                    assemble({**reused_config, "feature_columns": ["feature_factor"]})


def test_raw_target_has_no_unrelated_cross_section_clock_dependency():
    frame = _frame(3)
    frame.loc[1, "forward_return_20d_available_at"] = pd.Timestamp("2020-02-04")
    frame.loc[2, "forward_return_20d_available_at"] = pd.NaT
    spec = replace(
        research_dataset_catalog().get("pv_all"),
        eligibility_recipe=None,
        target_transform="raw",
    )
    result = _add_a_share_semantic_columns(frame, spec)
    pd.testing.assert_series_equal(
        result.model_target_available_at,
        frame.forward_return_20d_available_at.astype("datetime64[ns]"),
        check_names=False,
    )
    clipped = _add_a_share_semantic_columns(
        frame, replace(spec, target_clip="quantile_1_99")
    )
    assert clipped.model_target_available_at.isna().all()
    assert clipped.model_target.notna().all()


def test_target_cross_section_is_independent_of_feature_availability():
    frame = _frame(3)
    frame.loc[2, "forward_return_20d_available_at"] = pd.Timestamp("2020-02-05")
    spec = replace(research_dataset_catalog().get("pv_all"), eligibility_recipe=None)
    expected = _add_a_share_semantic_columns(frame, spec)
    frame[["factor", "other"]] = np.nan
    actual = _add_a_share_semantic_columns(frame, spec)
    pd.testing.assert_series_equal(actual.model_target, expected.model_target)
    assert actual.model_target_available_at.eq(pd.Timestamp("2020-02-05")).all()
