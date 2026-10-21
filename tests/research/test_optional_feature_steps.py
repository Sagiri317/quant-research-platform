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


@pytest.mark.parametrize(
    "size_column", ["log_total_market_cap", "log_float_market_cap"]
)
def test_size_only_and_size_source_never_regress_on_itself(size_column):
    sizes = np.arange(120, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": 2 * sizes + np.sin(sizes),
            size_column: sizes,
        }
    )
    result = _apply(
        frame,
        clip="none",
        industry_neutralize=False,
        center=False,
        scale=False,
        size_column=size_column,
    )
    design = np.column_stack((np.ones(len(sizes)), sizes))
    expected = frame.x - design @ np.linalg.lstsq(design, frame.x, rcond=None)[0]
    np.testing.assert_allclose(result.feature_x, expected, rtol=1e-11, atol=1e-12)
    self_result = transforms.apply_feature_transforms(
        frame,
        (
            FeatureTransformSpec(
                "a_share_canonical",
                (size_column,),
                clip="none",
                size_column=size_column,
                industry_neutralize=False,
                center=False,
                scale=False,
            ),
        ),
    )
    np.testing.assert_array_equal(self_result["feature_" + size_column], sizes)


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
            "log_total_market_cap": np.arange(n) / 10,
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
    assert transforms.feature_observation_columns(columns, frame.columns) == (
        "feature_observed_x",
        "feature_observed_y",
    )
    result = transforms.mask_unobserved_features(frame, columns, frame.columns)
    assert result.feature_x.isna().tolist() == [True, False, False]
    assert result.feature_y.isna().tolist() == [False, True, False]
    pd.testing.assert_series_equal(result.feature_missing_x, frame.feature_missing_x)
    pd.testing.assert_series_equal(result.feature_observed_x, frame.feature_observed_x)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="feature_observation_column_missing"):
        transforms.mask_unobserved_features(
            frame.drop(columns="feature_observed_x"), ("feature_x",), frame.columns
        )
    with pytest.raises(ValueError, match="invalid_feature_observation_flag"):
        transforms.mask_unobserved_features(
            frame.assign(feature_observed_x=[0, 1, 2]), ("feature_x",), frame.columns
        )
    for invalid in (np.nan, 2, 1):
        with pytest.raises(ValueError, match="invalid_feature_missing_flag"):
            transforms.model_observed_input_mask(
                frame.assign(feature_missing_x=[1, 0, invalid]),
                ("feature_x",),
                frame.columns,
            )


@pytest.mark.parametrize(
    "columns",
    [("feature_x", "feature_y"), ("feature_missing_x", "feature_observed_y")],
)
def test_observed_mask_distinguishes_fill_missing_and_structural_unavailability(
    columns,
):
    frame = pd.DataFrame(
        {
            "feature_x": pd.array([100, 10, None, 100, 0, 100], dtype="Int64"),
            "feature_observed_x": [0, 1, 0, 0, 0, 0],
            "feature_missing_x": [1, 0, 1, 0, 1, 1],
            "feature_y": pd.array([5, 100, 8, 7, 0, None], dtype="Int64"),
            "feature_observed_y": [1, 0, 1, 1, 0, 1],
            "feature_missing_y": [0, 1, 0, 0, 1, 0],
        },
        index=[8, 3, 9, 4, 1, 2],
    )
    original = frame.copy()
    result = transforms.model_observed_input_mask(frame, columns, frame.columns)
    pd.testing.assert_series_equal(
        result, pd.Series([True, True, True, False, False, False], index=frame.index)
    )
    pd.testing.assert_frame_equal(frame, original)


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
        assert result.feature_missing_x.eq(0).all()
    assert diagnostic["zero_variance_by_factor"]["x"] == int(scale)


@pytest.mark.parametrize(
    "size_column", ["log_total_market_cap", "log_float_market_cap"]
)
def test_size_selection_changes_only_equal_weight_regression_covariate(size_column):
    # Four distinct capitalization pairs, repeated to meet the 100-row domain.
    total = np.tile([1.0, 2.0, 3.0, 4.0], 30)
    floating = np.tile([1.0, 3.0, 2.0, 4.0], 30)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": np.tile([1.0, 2.0, 4.0, 8.0], 30),
            "log_total_market_cap": total,
            "log_float_market_cap": floating,
        }
    )
    options = (
        {} if size_column == "log_total_market_cap" else {"size_column": size_column}
    )
    actual = _apply(frame, clip="none", industry_neutralize=False, **options)
    # Equal-weight intercept and slope: 2.3 for total, 1.9 for floating.
    slope = 2.3 if size_column == "log_total_market_cap" else 1.9
    residual = frame.x - 3.75 - slope * (frame[size_column] - 2.5)
    expected = residual / np.sqrt(np.mean(residual**2))
    np.testing.assert_allclose(actual.feature_x, expected, rtol=1e-14, atol=1e-14)
    for column in ("x", "log_total_market_cap", "log_float_market_cap"):
        pd.testing.assert_series_equal(actual[column], frame[column])
    spec = FeatureTransformSpec("a_share_canonical", ("x",), **options)
    assert spec.support_columns == (
        "eligible",
        "industry_standard_id",
        "industry_level1_code",
        size_column,
    )
    assert FeatureTransformSpec(
        "a_share_canonical", (size_column,), **options
    ).support_columns == ("eligible", "industry_standard_id", "industry_level1_code")


def test_missing_total_size_never_falls_back_to_float():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": np.sin(np.arange(120)),
            "log_float_market_cap": np.arange(120, dtype=float),
        }
    )
    with pytest.raises(ValueError, match="feature_size_context_missing"):
        _apply(frame, industry_neutralize=False)
    actual = _apply(
        frame.assign(log_total_market_cap=np.nan), industry_neutralize=False
    )
    assert actual.feature_x.isna().all()
    assert actual.feature_observed_x.eq(0).all()
    assert (
        _apply(frame, industry_neutralize=False, size_neutralize=False)
        .feature_x.notna()
        .all()
    )
    assert (
        _apply(frame, industry_neutralize=False, size_column="log_float_market_cap")
        .feature_x.notna()
        .all()
    )


@pytest.mark.parametrize("size_column", [None, 1, [], "float_market_cap", "anything"])
def test_size_column_requires_a_named_raw_log_factor(size_column):
    with pytest.raises(ValueError, match="invalid_feature_size_column"):
        FeatureTransformSpec("a_share_canonical", ("x",), size_column=size_column)


def test_size_default_does_not_change_raw_factor_or_market_model_dependencies():
    from quant_research_platform.research.protocols.a_share import (
        MARKET_MODEL_FACTORS,
        RISK_STYLE_COLUMNS,
        factor_catalog,
    )

    catalog = factor_catalog()
    assert RISK_STYLE_COLUMNS[0] == "log_total_market_cap"
    for capitalization in ("total", "float"):
        assert catalog.get(
            f"price_volume_log_{capitalization}_market_cap"
        ).input_columns == (f"{capitalization}_market_cap",)
    for column in MARKET_MODEL_FACTORS:
        assert "float_market_cap" in catalog.get(column).input_columns
        assert "total_market_cap" not in catalog.get(column).input_columns


@pytest.mark.parametrize("variation", [0.0, 1e-8])
def test_neutralization_keeps_true_zero_and_resolvable_small_signal(variation):
    axis = np.arange(120, dtype=float)
    industry = pd.Series(np.arange(120) % 3)
    sizes = 15 + axis / 16
    target = 3 + industry.to_numpy() * 4 + 2 * sizes + variation * np.sin(axis)
    actual = transforms._industry_size_residuals(target, industry, sizes, rescale=False)
    if not variation:
        np.testing.assert_array_equal(actual, np.zeros(120))
    else:
        design = np.column_stack(
            (pd.get_dummies(industry).to_numpy(dtype=float), sizes)
        )
        expected = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-13)
        assert np.max(np.abs(actual)) > 1e-9
        standardized = transforms._industry_size_residuals(
            target, industry, sizes, rescale=True
        )
        assert np.max(np.abs(standardized)) == pytest.approx(1.0)


def test_size_collinear_with_industry_keeps_projection_undefined():
    industry = pd.Series(["a", "b"] * 60)
    sizes = np.tile([15.0, 20.0], 60)
    actual = transforms._industry_size_residuals(
        np.sin(np.arange(120)), industry, sizes, rescale=True
    )
    assert np.isnan(actual).all()


def test_projection_preserves_residual_under_replication_units_and_group_locality():
    from fractions import Fraction

    sizes = 15 + np.arange(100) / 16
    signs = np.tile([1.0, -1.0, -1.0, 1.0], 25)
    target = 3 + 2 * sizes + 1e-11 * signs
    exact_x = list(map(Fraction, sizes))
    exact_y = list(map(Fraction, target))
    mx, my = sum(exact_x) / 100, sum(exact_y) / 100
    slope = sum(
        (x - mx) * (y - my) for x, y in zip(exact_x, exact_y, strict=True)
    ) / sum((x - mx) ** 2 for x in exact_x)
    expected = np.array(
        [
            float(y - my - slope * (x - mx))
            for x, y in zip(exact_x, exact_y, strict=True)
        ]
    )
    assert slope == 2
    assert np.min(np.abs(expected)) > 1e-12
    # Each change preserves the same projection: equal replication, an unrelated
    # exactly fitted industry, a unit conversion, or a row permutation.
    for repeat_a, repeat_b, unit, reverse in (
        (1, 1, 1, False),
        (10, 10, 1, False),
        (1, 10, 1, False),
        (1, 1, 16, True),
    ):
        x = np.concatenate((np.tile(sizes, repeat_a), np.tile(sizes, repeat_b)))
        y = (
            np.concatenate(
                (np.tile(target, repeat_a), np.tile(7 + 2 * sizes, repeat_b))
            )
            * unit
        )
        industry = np.repeat(["A", "B"], [100 * repeat_a, 100 * repeat_b])
        oracle = (
            np.concatenate((np.tile(expected, repeat_a), np.zeros(100 * repeat_b)))
            * unit
        )
        order = np.arange(len(x))[::-1] if reverse else np.arange(len(x))
        actual = transforms._industry_size_residuals(
            y[order], industry[order], x[order], rescale=False
        )
        np.testing.assert_allclose(actual, oracle[order], rtol=1e-12, atol=2e-14 * unit)


def test_non_dyadic_exact_fit_does_not_become_standardized_alpha():
    sizes = 15 + np.random.default_rng(18).uniform(0, 6, 120)
    industry = np.arange(120) % 3
    target = 0.07 + 0.13 * industry + 0.37 * sizes
    for order in (np.arange(120), np.arange(120)[::-1]):
        actual = transforms._industry_size_residuals(
            target[order], industry[order], sizes[order], rescale=True
        )
        np.testing.assert_array_equal(actual, np.zeros(120))


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_observation_mask_preserves_integer_order_and_nulls(dtype):
    start = 2**63 - 4 if dtype == "int64" else 2**63 + 1
    values = np.arange(4, dtype=dtype) + start
    frame = pd.DataFrame({"feature_x": values, "feature_observed_x": [1, 0, 1, 1]})
    result = transforms.mask_unobserved_features(frame, ("feature_x",), frame.columns)
    assert str(result.feature_x.dtype) == ("Int64" if dtype == "int64" else "UInt64")
    assert result.feature_x.iloc[[0, 2, 3]].tolist() == values[[0, 2, 3]].tolist()
    assert result.feature_x.isna().tolist() == [False, True, False, False]
    assert result.feature_x.rank().tolist()[2:] == [2.0, 3.0]
    np.testing.assert_array_equal(frame.feature_x, values)
