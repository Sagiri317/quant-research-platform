"""Whole feature groups and target transforms retain scientific clocks and NULLs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.research.datasets.assembler import (
    _add_a_share_semantic_columns,
    assemble,
)
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.research.labels.targets import a_share_model_target
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
    assert pd.isna(result.feature_pre_neutral_factor.iloc[1])
    assert pd.notna(result.feature_pre_neutral_other.iloc[1])
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
            "forward_return_20d_available_at": pd.Timestamp("2020-02-01"),
        }
    )
    result = a_share_model_target(frame).model_target
    np.testing.assert_allclose(result.iloc[:2], [0, 0])
    assert result.iloc[2:].isna().all()
    result = a_share_model_target(frame, transform="percentile").model_target
    np.testing.assert_allclose(result.iloc[:2], [0.75, 0.75])
    assert result.iloc[2:].isna().all()
    pd.testing.assert_series_equal(
        a_share_model_target(frame, transform="raw").model_target,
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


def test_nonfinancial_risk_target_persists_orthogonal_returns_and_maturity(tmp_path):
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
    from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source

    # Independent binary signs are orthogonal, including under weights driven
    # by the seventh sign. The sixth sign is the unexplained return.
    signs = 2.0 * ((np.arange(128)[:, None] >> np.arange(7)) & 1) - 1
    residual = 0.001 * signs[:, 5]
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"a{i:03d}" for i in range(128)],
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            "total_market_cap": np.where(signs[:, 6] > 0, 4e8, 1e8),
            **dict(zip(RISK_STYLE_COLUMNS, signs[:, :5].T, strict=True)),
            "reversal_20d": signs[:, 5],
            "forward_return_20d": 0.02
            + signs[:, :5] @ (np.arange(5) / 1000)
            + residual,
            "forward_return_20d_available_at": pd.Timestamp("2024-02-01"),
        }
    )
    frame.loc[0, "forward_return_20d_available_at"] = pd.Timestamp("2024-02-04")
    outsider = frame.iloc[[0]].assign(
        asset_id="outside",
        eligible=False,
        forward_return_20d=10.0,
        forward_return_20d_available_at=pd.Timestamp("2024-03-01"),
    )
    frame = pd.concat([frame, outsider], ignore_index=True)
    source, _ = assembly_source(
        tmp_path,
        frame,
        units={"forward_return_20d": "decimal_return"},
        availability={"forward_return_20d": "forward_return_20d_available_at"},
    )
    root = assemble(
        assembly_config(
            tmp_path,
            source,
            label_columns=["model_target", "forward_return_20d"],
            filter_columns=["eligible"],
            target_transform="risk_residual",
        )
    )
    saved = read_dataset(root)
    np.testing.assert_allclose(saved.model_target.iloc[:128], residual, atol=1e-15)
    assert (
        saved.model_target_available_at.iloc[:128].eq(pd.Timestamp("2024-02-04")).all()
    )
    assert pd.isna(saved.model_target.iloc[-1])
    assert pd.isna(saved.model_target_available_at.iloc[-1])
    np.testing.assert_array_equal(saved.forward_return_20d, frame.forward_return_20d)
    spec = inspect_dataset(root).spec
    assert spec["target_transform"] == "risk_residual"
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.storage.datasets.schema import (
        validate_dataset_spec_payload,
    )

    assert spec["risk_model"] == FactorRiskModel().to_dict()
    legacy = {key: value for key, value in spec.items() if key != "risk_model"}
    validate_dataset_spec_payload(legacy)
    assert spec["column_units"]["model_target"] == "decimal_return"
    assert "earnings_yield" not in RISK_STYLE_COLUMNS

    missing = frame.copy()
    missing.loc[:128, "beta_252d"] = np.nan
    assert (
        a_share_model_target(missing, transform="risk_residual")
        .model_target.isna()
        .all()
    )


@pytest.mark.parametrize("clip", ["none", "mad", "quantile_1_99"])
@pytest.mark.parametrize("maturity", [pd.Timestamp("2024-03-01"), pd.NaT])
def test_risk_target_clipping_dependencies_survive_assembly_and_fit(
    tmp_path, clip, maturity
):
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source

    signs = 2.0 * ((np.arange(128)[:, None] >> np.arange(6)) & 1) - 1
    returns = np.arange(128) / 1000
    returns[0] = 1.0
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"a{i:03d}" for i in range(128)],
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            "total_market_cap": 1e8,
            **dict(zip(RISK_STYLE_COLUMNS, signs[:, :5].T, strict=True)),
            "reversal_20d": signs[:, 5],
            "forward_return_20d": returns,
            "return_maturity": pd.Timestamp("2024-02-01"),
        }
    )
    excluded = frame.iloc[[0]].assign(
        asset_id="excluded",
        beta_252d=np.nan,
        forward_return_20d=10.0,
        return_maturity=maturity,
    )
    frame = pd.concat(
        [
            frame,
            excluded,
            excluded.assign(
                asset_id="ineligible", eligible=False, return_maturity=pd.NaT
            ),
            excluded.assign(
                asset_id="missing_return",
                forward_return_20d=np.nan,
                return_maturity=pd.NaT,
            ),
        ],
        ignore_index=True,
    )
    source, _ = assembly_source(
        tmp_path,
        frame,
        units={"forward_return_20d": "decimal_return"},
        availability={"forward_return_20d": "return_maturity"},
    )
    root = assemble(
        assembly_config(
            tmp_path,
            source,
            label_columns=["model_target"],
            filter_columns=["eligible"],
            target_transform="risk_residual",
            target_clip=clip,
        )
    )
    saved = read_dataset(root)

    # Clipping sees 0.001, ..., 0.127, 1, 10, including the missing-style row.
    # The 129 values have median 0.065, MAD 0.032, and linear 1/99% bounds below.
    clipped = returns
    if clip == "mad":
        clipped = np.clip(
            returns, 0.065 - 3 * 1.4826 * 0.032, 0.065 + 3 * 1.4826 * 0.032
        )
    elif clip == "quantile_1_99":
        clipped = np.clip(returns, 0.002 + 0.28 * 0.001, 0.127 + 0.72 * (1 - 0.127))
    # Equal weights and binary signs give X'X = 128 I, independently of style scaling.
    design = np.column_stack([np.ones(128), signs[:, :5]])
    expected = clipped - design @ (design.T @ clipped / 128)
    np.testing.assert_allclose(
        saved.model_target.iloc[:128], expected, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_array_equal(saved.forward_return_20d, frame.forward_return_20d)
    assert saved.model_target.iloc[128:].isna().all()
    assert saved.model_target_available_at.iloc[128:].isna().all()
    expected_maturity = pd.Timestamp("2024-02-01") if clip == "none" else maturity
    clocks = saved.model_target_available_at.iloc[:128]
    if pd.isna(expected_maturity):
        assert clocks.isna().all()
    else:
        assert clocks.eq(expected_maturity).all()

    config = {
        "market_id": "a_share",
        "inputs": str(root),
        "output": str(tmp_path / "model"),
        "model": {
            "model_id": "linear",
            "model_algorithm": "linear_regression",
            "input_factor_columns": ["reversal_20d"],
            "target_column": "model_target",
        },
        "fit_windows": [{"start": "2024-01-02", "end": "2024-01-02"}],
        "fit_cutoff": "2024-02-05",
    }
    if clip == "none":
        assert fit(config).is_dir()
    else:
        with pytest.raises(ValueError, match="model_training_sample_empty"):
            fit(config)
        if pd.notna(maturity):
            assert fit(
                {
                    **config,
                    "output": str(tmp_path / "mature_model"),
                    "fit_cutoff": "2024-03-01",
                }
            ).is_dir()


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


def test_projection_feature_and_target_survive_assembly_fit_and_prediction(tmp_path):
    from fractions import Fraction

    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source

    sizes = 15 + np.arange(100) / 16
    signs = np.tile([1.0, -1.0, -1.0, 1.0], 25)
    raw = 3 + 2 * sizes + 1e-11 * signs
    # These dyadic coordinates make the nuisance slope exactly 2. Fraction
    # measures the residual of the actual stored floats, not the decimal literal.
    residual = float(Fraction(raw[0]) - 3 - 2 * Fraction(sizes[0]))
    frames = []
    for day, repetitions in (("2024-01-02", 1), ("2024-01-04", 10)):
        frames.append(
            pd.DataFrame(
                {
                    "market_id": "a_share",
                    "asset_id": [f"a{i:04d}" for i in range(100 * repetitions)],
                    "timestamp": pd.Timestamp(day),
                    "eligible": True,
                    "reversal_20d": np.tile(raw, repetitions),
                    "log_total_market_cap": np.tile(sizes, repetitions),
                    "industry_standard_id": "test",
                    "industry_level1_code": "A",
                    "forward_return_20d": np.tile(raw / 64, repetitions),
                    "forward_return_20d_available_at": pd.Timestamp(day)
                    + pd.Timedelta(days=1),
                }
            )
        )
    source, _ = assembly_source(
        tmp_path,
        pd.concat(frames, ignore_index=True),
        units={"forward_return_20d": "decimal_return"},
        availability={"forward_return_20d": "forward_return_20d_available_at"},
    )
    dataset = assemble(
        assembly_config(
            tmp_path,
            source,
            time_window={"start": "2024-01-02", "end": "2024-01-04"},
            feature_recipe="a_share_canonical",
            feature_transform={"clip": "none"},
            filter_columns=["eligible"],
            label_columns=["model_target"],
            target_transform="industry_size_residual",
            target_clip="none",
        )
    )
    saved = read_dataset(dataset)
    for day, group in saved.groupby("timestamp"):
        expected_sign = np.tile(signs, len(group) // 100)
        np.testing.assert_allclose(
            group.feature_reversal_20d, expected_sign, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            group.model_target, expected_sign * residual / 64, rtol=1e-12, atol=1e-17
        )
        assert group.feature_reversal_20d.notna().all()
        assert group.model_target_available_at.eq(day + pd.Timedelta(days=1)).all()
    model = fit(
        {
            "market_id": "a_share",
            "inputs": str(dataset),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "linear",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["feature_reversal_20d"],
                "target_column": "model_target",
            },
            "fit_windows": [{"start": "2024-01-02", "end": "2024-01-02"}],
            "fit_cutoff": "2024-01-03",
        }
    )
    output = predict(
        {
            "model": str(model),
            "inputs": str(dataset),
            "output": str(tmp_path / "prediction"),
            "prediction_window": {"start": "2024-01-04", "end": "2024-01-04"},
            "sample_role": "test",
        }
    )
    predictions = read_predictions(output)[0]
    np.testing.assert_allclose(
        predictions.prediction_score,
        np.tile(signs, 10) * residual / 64,
        rtol=1e-12,
        atol=1e-17,
    )


def test_explicit_risk_model_survives_assembly_and_model_facts(tmp_path, monkeypatch):
    from quant_research_platform.contracts.models import (
        _comparable_target_definition,
        _target_definition,
        _validate_target_definition,
    )
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source

    model = FactorRiskModel.from_mapping(
        {
            "style_columns": ["log_total_market_cap"],
            "config": {"weight_column": "float_market_cap", "factor_half_life": 90.0},
        }
    )
    frame = _frame().assign(
        reversal_20d=lambda rows: rows.factor,
        float_market_cap=np.exp(np.linspace(-2.0, 2.0, 120)),
    )
    source, _ = assembly_source(
        tmp_path,
        frame,
        units={"forward_return_20d": "decimal_return"},
        availability={"forward_return_20d": "forward_return_20d_available_at"},
    )
    from quant_research_platform.research.risk import FactorRiskState

    prepared_dates = []
    prepare = FactorRiskState.prepare_exposure_session

    def record_preparation(self, cross, **kwargs):
        prepared_dates.append(cross.timestamp.iloc[0])
        return prepare(self, cross, **kwargs)

    monkeypatch.setattr(FactorRiskState, "prepare_exposure_session", record_preparation)
    root = assemble(
        assembly_config(
            tmp_path,
            source,
            time_window={"start": "2020-01-02", "end": "2020-01-02"},
            label_columns=["model_target"],
            filter_columns=["eligible"],
            target_transform="risk_residual",
            risk_model=model.to_dict(),
            feature_recipe="a_share_canonical",
            feature_transform={
                "risk_neutralize": True,
                "clip": "none",
                "center": False,
                "scale": False,
            },
        )
    )
    assert prepared_dates == [pd.Timestamp("2020-01-02")]
    saved = read_dataset(root)
    values = frame.log_total_market_cap.to_numpy()
    design = np.column_stack((np.ones(120), (values - values.mean()) / values.std()))
    weights = frame.float_market_cap.to_numpy() ** 0.25
    returns = frame.forward_return_20d.to_numpy()
    coefficients = np.linalg.lstsq(
        design * weights[:, None], returns * weights, rcond=None
    )[0]
    np.testing.assert_allclose(
        saved.model_target, returns - design @ coefficients, atol=1e-14
    )
    np.testing.assert_allclose(
        saved.feature_reversal_20d, 100 * saved.model_target, atol=1e-12
    )
    declared = inspect_dataset(root).spec
    assert declared["risk_model"] == model.to_dict()
    assert declared["feature_transforms"][0]["risk_model"] == model.to_dict()
    with pytest.raises(ValueError, match="feature_risk_neutralization_conflict"):
        FeatureTransformSpec("a_share_canonical", ("reversal_20d",), risk_model=model)
    assert "total_market_cap" not in saved
    model_root = fit(
        {
            "market_id": "a_share",
            "inputs": str(root),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "linear",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["feature_reversal_20d"],
                "target_column": "model_target",
            },
            "fit_windows": [{"start": "2020-01-02", "end": "2020-01-02"}],
            "fit_cutoff": "2020-02-02",
        }
    )
    _, facts = load_model(model_root)
    definition = facts["target_definition"]
    assert definition["risk_model"] == model.to_dict()
    _validate_target_definition(definition, "model_target")
    alternative = _target_definition(
        {
            **declared,
            "risk_model": {
                **model.to_dict(),
                "config": {**model.to_dict()["config"], "factor_half_life": 126.0},
            },
        },
        "model_target",
    )
    assert _comparable_target_definition(alternative) == _comparable_target_definition(
        definition
    )
    changed_weight = {
        **alternative,
        "risk_model": {
            **model.to_dict(),
            "config": {"weight_column": "total_market_cap"},
        },
    }
    assert _comparable_target_definition(
        changed_weight
    ) != _comparable_target_definition(definition)
    legacy = {key: value for key, value in definition.items() if key != "risk_model"}
    _validate_target_definition(legacy, "model_target")
    default = {**legacy, "risk_model": FactorRiskModel().to_dict()}
    assert _comparable_target_definition(legacy) == _comparable_target_definition(
        default
    )


def test_shared_risk_missing_values_only_affect_dependent_outputs(tmp_path):
    from quant_research_platform.contracts.risk import FactorRiskModel

    frame = _frame().assign(total_market_cap=1e8)
    transform = FeatureTransformSpec(
        "a_share_canonical",
        ("factor", "other"),
        clip="none",
        industry_neutralize=False,
        size_neutralize=False,
        center=False,
        scale=False,
        output_groups=("processed", "pre_neutral"),
        risk_model=FactorRiskModel(("log_total_market_cap",)),
    )
    expected = apply_feature_transforms(frame, (transform,))
    changed = frame.copy()
    changed.loc[:9, "other"] = np.nan
    actual = apply_feature_transforms(changed, (transform,))
    pd.testing.assert_series_equal(actual.feature_factor, expected.feature_factor)
    assert actual.feature_other.iloc[:10].isna().all()
    assert actual.feature_other.iloc[10:].notna().all()
    changed.loc[0, "log_total_market_cap"] = np.nan
    actual = apply_feature_transforms(changed, (transform,))
    assert pd.isna(actual.feature_factor.iloc[0])
    pd.testing.assert_series_equal(
        actual.feature_pre_neutral_factor, frame.factor, check_names=False
    )
    independent = apply_feature_transforms(
        frame[["timestamp", "eligible", "factor", "other"]],
        (replace(transform, output_groups=("pre_neutral",)),),
    )
    pd.testing.assert_series_equal(
        independent.feature_pre_neutral_factor, frame.factor, check_names=False
    )

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source

    # Native assembly must not prepare risk descriptors for an unselected output.
    raw = frame.drop(
        columns=[
            "industry_standard_id",
            "industry_level1_code",
            "log_total_market_cap",
            "total_market_cap",
        ]
    ).assign(reversal_20d=frame.factor)
    source, _ = assembly_source(
        tmp_path,
        raw,
        units={"forward_return_20d": "decimal_return"},
        availability={"forward_return_20d": "forward_return_20d_available_at"},
    )
    root = assemble(
        assembly_config(
            tmp_path,
            source,
            time_window={"start": "2020-01-02", "end": "2020-01-02"},
            label_columns=["model_target"],
            filter_columns=["eligible"],
            target_transform="raw",
            feature_recipe="a_share_canonical",
            feature_groups=["raw", "pre_neutral"],
            feature_transform={
                "risk_neutralize": True,
                "clip": "none",
                "center": False,
                "scale": False,
            },
        )
    )
    saved = read_dataset(root)
    np.testing.assert_array_equal(saved.reversal_20d, raw.reversal_20d)
    np.testing.assert_array_equal(
        saved.feature_pre_neutral_reversal_20d, raw.reversal_20d
    )
    np.testing.assert_array_equal(saved.model_target, raw.forward_return_20d)


def test_full_risk_standardization_preserves_weighted_industry_and_style_neutrality():
    from quant_research_platform.contracts.risk import FactorRiskModel

    frame = _frame().assign(
        total_market_cap=np.exp(np.linspace(-4.0, 4.0, 120)),
        industry_level1_code=np.where(np.arange(120) < 60, "A", "B"),
    )
    recipe = FeatureTransformSpec(
        "a_share_canonical",
        ("factor", "other"),
        clip="none",
        industry_neutralize=False,
        size_neutralize=False,
        output_groups=("processed", "pre_neutral"),
        risk_model=FactorRiskModel(("log_total_market_cap",)),
    )
    actual = apply_feature_transforms(frame, (recipe,))
    styles = frame.log_total_market_cap.to_numpy()
    industries = np.column_stack((np.arange(120) < 60, np.arange(120) >= 60))
    design = np.column_stack((industries, (styles - styles.mean()) / styles.std()))
    weights = np.sqrt(frame.total_market_cap.to_numpy())
    square_root_weights = np.sqrt(weights)
    response = frame.factor.to_numpy()
    coefficients = np.linalg.lstsq(
        design * square_root_weights[:, None],
        response * square_root_weights,
        rcond=None,
    )[0]
    residual = response - design @ coefficients
    assert abs(residual.mean()) > 1e-4
    expected = (residual - np.average(residual, weights=weights)) / residual.std()
    np.testing.assert_allclose(actual.feature_factor, expected, atol=1e-13)
    np.testing.assert_allclose(
        design.T @ (weights * actual.feature_factor), 0.0, atol=1e-11
    )
    assert actual.feature_factor.std(ddof=0) == pytest.approx(1.0)
    np.testing.assert_allclose(
        actual.feature_pre_neutral_factor,
        (response - response.mean()) / response.std(),
        atol=1e-14,
    )
    changed = frame.copy()
    changed.loc[:9, "other"] = np.nan
    missing = apply_feature_transforms(changed, (recipe,))
    pd.testing.assert_series_equal(missing.feature_factor, actual.feature_factor)
    assert missing.feature_other.iloc[:10].isna().all()


@pytest.mark.parametrize("versioned_risk_keys", [False, True])
def test_intraday_full_risk_uses_published_prior_life_facts_without_changing_raw(
    tmp_path,
    versioned_risk_keys,
):
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.storage.datasets import (
        inspect_dataset,
        read_dataset,
        write_dataset,
    )
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source

    model = FactorRiskModel()
    signs = 2.0 * ((np.arange(128)[:, None] >> np.arange(7)) & 1) - 1
    times = pd.to_datetime(
        ["2024-01-03 01:30:00", "2024-01-03 01:30:03", "2024-01-03 01:30:06"]
    )
    daily = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"a{i:03d}" for i in range(128)],
            "timestamp": pd.Timestamp("2024-01-02"),
            "listing_date": pd.Timestamp("2010-01-01"),
            "risk_available_at": pd.Timestamp("2024-01-02 07:05:00"),
            "industry_standard_id": "test",
            "industry_level1_code": np.where(signs[:, 6] > 0, "A", "B"),
            "total_market_cap": np.exp(signs[:, 6]),
            **dict(zip(model.style_columns, signs[:, :5].T, strict=True)),
        }
    )
    old = daily.iloc[[5, 11]].assign(
        timestamp=pd.Timestamp("2023-12-29"),
        risk_available_at=pd.Timestamp("2023-12-29 07:05:00"),
    )
    old.loc[11, "risk_available_at"] = pd.Timestamp("2024-01-03 01:30:04")
    old.loc[11, "beta_252d"] = 99.0
    daily.loc[0, "risk_available_at"] = times[1]
    daily.loc[1, "risk_available_at"] = pd.NaT
    daily.loc[[3, 5], "beta_252d"] = np.nan
    daily.loc[5, "risk_available_at"] = times[2]
    daily.loc[4, "timestamp"] = pd.Timestamp("2024-01-03")
    daily.loc[4, "risk_available_at"] = pd.Timestamp("2024-01-03 01:00:00")
    daily.loc[7, "listing_date"] = pd.NaT
    daily.loc[8, "risk_available_at"] = pd.Timestamp("2024-01-04")
    risk_frame = pd.concat((old, daily), ignore_index=True)
    observations = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": daily.asset_id,
            "eligible": True,
            "listing_date": pd.Timestamp("2010-01-01"),
            "reversal_20d": signs[:, 5] + signs[:, :5] @ np.arange(1, 6),
            # The imported risk design must not overwrite a raw alpha with this name.
            "log_total_market_cap": np.arange(128) / 7,
        }
    )
    observations.loc[2, "listing_date"] = pd.Timestamp("2024-01-03")
    observations.loc[6, "listing_date"] = pd.NaT
    base = pd.concat(
        [observations.assign(timestamp=timestamp) for timestamp in times],
        ignore_index=True,
    )
    _, source_spec = assembly_source(tmp_path, base, data_id="raw")
    sampled = tmp_path / "sampled"
    write_dataset(
        sampled, base, spec=replace(source_spec, data_id="sampled", frequency="tick")
    )
    risk_root, risk_spec = assembly_source(tmp_path, risk_frame, data_id="daily_risk")
    if versioned_risk_keys:
        risk_root = tmp_path / "versioned_daily_risk"
        write_dataset(
            risk_root,
            risk_frame.dropna(subset=["listing_date", "risk_available_at"]),
            spec=replace(
                risk_spec,
                data_id="versioned_daily_risk",
                primary_key=(
                    *risk_spec.primary_key,
                    "listing_date",
                    "risk_available_at",
                ),
                column_roles={
                    **risk_spec.column_roles,
                    "listing_date": "key",
                    "risk_available_at": "key",
                },
            ),
        )
    config = assembly_config(
        tmp_path,
        sampled,
        inputs={"sampled": str(sampled), "daily_risk": str(risk_root)},
        time_window={"start": "2024-01-03", "end": "2024-01-03"},
        factor_columns=["reversal_20d", "log_total_market_cap"],
        filter_columns=["eligible"],
        feature_recipe="a_share_canonical",
        feature_groups=["raw", "processed", "pre_neutral"],
        feature_transform={
            "risk_neutralize": True,
            "clip": "none",
            "center": False,
            "scale": False,
        },
        risk_exposure_input={
            "data_id": "daily_risk",
            "available_at_column": "risk_available_at",
        },
    )
    saved = read_dataset(assemble(config))
    for index, (_timestamp, rows) in enumerate(saved.groupby("timestamp", sort=True)):
        unavailable = (
            {1, 2, 3, 4, 6, 7, 8}
            | ({0} if index == 0 else set())
            | ({5} if index == 2 else set())
        )
        valid = np.array([i for i in range(128) if i not in unavailable])
        complete = daily.iloc[valid].copy()
        if index < 2:
            complete.loc[5, "beta_252d"] = old.loc[5, "beta_252d"]
        # Independently compute each available style's MAD clipping and zscore.
        normalized = []
        available = [i for i in range(128) if i not in unavailable - {3, 5}]
        for style in model.style_columns:
            values = daily[style].copy()
            if index < 2:
                values.loc[5] = old.loc[5, style]
            domain = values.iloc[available].dropna()
            median = domain.median()
            mad = (domain - median).abs().median()
            clipped = (
                domain.clip(median - 3 * 1.4826 * mad, median + 3 * 1.4826 * mad)
                if mad
                else domain
            )
            normalized.append(
                (values.iloc[valid].clip(clipped.min(), clipped.max()) - clipped.mean())
                / clipped.std(ddof=0)
            )
        design = np.column_stack(
            (
                complete.industry_level1_code.eq("A"),
                complete.industry_level1_code.eq("B"),
                *normalized,
            )
        )
        response = observations.reversal_20d.iloc[valid].to_numpy()
        weights = complete.total_market_cap.to_numpy() ** 0.25
        coefficients = np.linalg.lstsq(
            design * weights[:, None], response * weights, rcond=None
        )[0]
        np.testing.assert_allclose(
            rows.feature_reversal_20d.iloc[valid],
            response - design @ coefficients,
            atol=1e-12,
        )
        assert rows.feature_reversal_20d.iloc[sorted(unavailable)].isna().all()
        np.testing.assert_array_equal(rows.reversal_20d, observations.reversal_20d)
        np.testing.assert_array_equal(
            rows.log_total_market_cap, observations.log_total_market_cap
        )
        np.testing.assert_array_equal(
            rows.feature_pre_neutral_reversal_20d, observations.reversal_20d
        )
    manifest = inspect_dataset(tmp_path / "assembled").manifest
    assert (
        manifest["parameters"]["risk_exposure_input"] == config["risk_exposure_input"]
    )

    # No publication clock is not permission to infer one from the daily date.
    missing_clock = risk_frame.drop(columns="risk_available_at")
    unknown_root, _ = assembly_source(tmp_path, missing_clock, data_id="unknown_clock")
    missing = read_dataset(
        assemble(
            {
                **config,
                "inputs": {"sampled": str(sampled), "daily_risk": str(unknown_root)},
                "output_root": str(tmp_path / "unknown_output"),
            }
        )
    )
    assert missing.feature_reversal_20d.isna().all()
    np.testing.assert_array_equal(missing.reversal_20d, saved.reversal_20d)
    np.testing.assert_array_equal(
        missing.feature_pre_neutral_reversal_20d, saved.feature_pre_neutral_reversal_20d
    )
