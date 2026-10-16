"""Optional steps preserve economic sample domains and independent controls."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.research.features import transforms


def _apply(frame, **options):
    return transforms.apply_feature_transforms(
        frame,
        (FeatureTransformSpec("a_share_canonical", ("x",), **options),),
    )


@pytest.mark.parametrize(
    "center,scale,expected",
    [
        (False, False, [1.0, 2.0, 3.0]),
        (True, False, [-1.0, 0.0, 1.0]),
        (True, True, [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]),
        (False, True, np.sqrt(1.5) * np.array([1.0, 2.0, 3.0])),
    ],
)
def test_off_steps_do_not_require_industry_or_size(center, scale, expected):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": [1.0, 2.0, 3.0],
        }
    )
    result = _apply(
        frame,
        clip="none",
        industry_neutralize=False,
        size_neutralize=False,
        center=center,
        scale=scale,
    )
    np.testing.assert_allclose(result.feature_x, expected, rtol=1e-14, atol=1e-14)
    assert result.feature_observed_x.eq(1).all()
    assert result.feature_missing_x.eq(0).all()
    pd.testing.assert_series_equal(result.x, frame.x)


def test_industry_only_uses_industry_intercepts_without_size():
    values = np.arange(120, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": values,
            "industry_standard_id": "test",
            "industry_level1_code": ["A"] * 60 + ["B"] * 60,
        }
    )
    result = _apply(
        frame, clip="none", size_neutralize=False, center=False, scale=False
    )
    expected = np.concatenate(
        (values[:60] - values[:60].mean(), values[60:] - values[60:].mean())
    )
    np.testing.assert_allclose(result.feature_x, expected, rtol=0, atol=1e-13)


def test_size_only_and_size_source_never_regress_on_itself():
    sizes = np.arange(120, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": 2 * sizes + np.sin(sizes),
            "log_float_market_cap": sizes,
        }
    )
    result = _apply(
        frame, clip="none", industry_neutralize=False, center=False, scale=False
    )
    design = np.column_stack((np.ones(len(sizes)), sizes))
    expected = frame.x - design @ np.linalg.lstsq(design, frame.x, rcond=None)[0]
    np.testing.assert_allclose(result.feature_x, expected, rtol=1e-11, atol=1e-12)
    self_result = transforms.apply_feature_transforms(
        frame,
        (
            FeatureTransformSpec(
                "a_share_canonical",
                ("log_float_market_cap",),
                clip="none",
                industry_neutralize=False,
                center=False,
                scale=False,
            ),
        ),
    )
    np.testing.assert_array_equal(self_result.feature_log_float_market_cap, sizes)


def test_optional_steps_keep_missing_observed_and_scope_distinct():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": [True, True, True, False],
            "x": [1.0, np.nan, 3.0, 5.0],
        }
    )
    result = _apply(
        frame, clip="none", industry_neutralize=False, size_neutralize=False
    )
    np.testing.assert_allclose(result.feature_x.iloc[:3], [-1.0, 0.0, 1.0])
    assert np.isnan(result.feature_x.iloc[3])
    assert result.feature_missing_x.tolist() == [0, 1, 0, 0]
    assert result.feature_observed_x.tolist() == [1, 0, 1, 0]


def test_neutralization_reuses_only_identical_valid_domains(monkeypatch):
    n = 120
    values = np.sin(np.arange(n))
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": values,
            "y": values * 2,
            "z": values * 3,
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            "log_float_market_cap": np.arange(n) / 10,
        }
    )
    frame.loc[0, "z"] = np.nan
    calls = []
    original = transforms._neutralization_design

    def measured(groups, sizes):
        calls.append(len(groups))
        return original(groups, sizes)

    monkeypatch.setattr(transforms, "_neutralization_design", measured)
    actual = transforms.apply_feature_transforms(
        frame, (FeatureTransformSpec("a_share_canonical", ("x", "y", "z")),)
    )
    assert calls == [120, 119]
    np.testing.assert_allclose(
        actual.feature_x, actual.feature_y, rtol=1e-14, atol=1e-14
    )
    separate = transforms.apply_feature_transforms(
        frame, (FeatureTransformSpec("a_share_canonical", ("z",)),)
    )
    np.testing.assert_allclose(
        actual.feature_z, separate.feature_z, rtol=1e-14, atol=1e-14
    )
    assert actual.feature_observed_x.iloc[0] == 1
    assert actual.feature_observed_z.iloc[0] == 0


@pytest.mark.parametrize(
    "options", [{"clip": "bad"}, {"scale": 1}, {"industry_neutralize": None}]
)
def test_feature_options_are_typed(options):
    with pytest.raises(ValueError, match="invalid_feature"):
        FeatureTransformSpec("a_share_canonical", ("x",), **options)


def test_imputation_can_be_disabled_without_changing_observation_flags():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": [1.0, np.nan, 3.0],
        }
    )
    result = _apply(
        frame,
        clip="none",
        industry_neutralize=False,
        size_neutralize=False,
        center=False,
        scale=False,
        impute_missing=False,
    )
    assert np.isnan(result.feature_x.iloc[1])
    assert result.feature_observed_x.tolist() == [1, 0, 1]
    assert result.feature_missing_x.tolist() == [0, 1, 0]


def test_observation_mask_excludes_fill_without_masking_indicator_signals():
    frame = pd.DataFrame(
        {
            "feature_x": [0.0, 1.0, 2.0],
            "feature_observed_x": [0, 1, 1],
            "feature_missing_x": [1, 0, 0],
            "feature_y": [3.0, 0.0, 5.0],
            "feature_observed_y": [1, 0, 1],
        }
    )
    original = frame.copy()
    columns = ("feature_x", "feature_y", "feature_missing_x", "feature_observed_x")
    assert transforms.feature_observation_columns(columns) == (
        "feature_observed_x",
        "feature_observed_y",
    )
    result = transforms.mask_unobserved_features(frame, columns)
    assert result.feature_x.isna().tolist() == [True, False, False]
    assert result.feature_y.isna().tolist() == [False, True, False]
    pd.testing.assert_series_equal(result.feature_missing_x, frame.feature_missing_x)
    pd.testing.assert_series_equal(result.feature_observed_x, frame.feature_observed_x)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="feature_observation_column_missing"):
        transforms.mask_unobserved_features(
            frame.drop(columns="feature_observed_x"), ("feature_x",)
        )
    with pytest.raises(ValueError, match="invalid_feature_observation_flag"):
        transforms.mask_unobserved_features(
            frame.assign(feature_observed_x=[0, 1, 2]), ("feature_x",)
        )


@pytest.mark.parametrize(
    "center,scale,defined",
    [(True, True, True), (False, True, False), (True, False, True)],
)
def test_zero_variance_convention_depends_on_requested_math(center, scale, defined):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": [3.0, 3.0, np.nan],
        }
    )
    diagnostic = {}
    result = transforms.apply_feature_transforms(
        frame,
        (
            FeatureTransformSpec(
                "a_share_canonical",
                ("x",),
                clip="none",
                industry_neutralize=False,
                size_neutralize=False,
                center=center,
                scale=scale,
            ),
        ),
        diagnostics=diagnostic,
    )
    if defined:
        assert result.feature_x.eq(0).all()
        assert result.feature_observed_x.tolist() == [1, 1, 0]
    else:
        assert result.feature_x.isna().all()
        assert result.feature_observed_x.eq(0).all()
        assert result.feature_missing_x.eq(1).all()
    assert diagnostic["zero_variance_by_factor"]["x"] == int(scale)
