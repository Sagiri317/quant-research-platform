from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.research.datasets.assembler import (
    _add_a_share_semantic_columns,
)
from quant_research_platform.research.features.transforms import (
    _neutralize_cross_section,
    a_share_feature_pipeline,
)
from quant_research_platform.research.protocols.a_share import (
    LABEL_HORIZONS,
    research_dataset_catalog,
    research_dataset_specs,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    read_dataset,
    write_dataset,
)
from tests.helpers.paths import StorageLayout
from tests.helpers.research_specs import assemble_research_dataset


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
def test_a_share_assembler_owns_eligibility_and_population_model_target(scale) -> None:
    spec = research_dataset_catalog().get("pv_all")
    assert (
        "model_target"
        not in processed_catalog().get("forward_returns").required_columns
    )
    for dataset_spec in research_dataset_specs():
        assert {
            "scenario_universe",
            "can_enter",
            "can_exit",
        }.isdisjoint(dataset_spec.required_columns)
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": [False, False, True],
            "is_pt": False,
            "listing_age_trading_days": 252,
            "has_252_trading_sessions": True,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "turnover": 0.1,
            "forward_return_20d": [0.10 * scale, 0.20 * scale, np.nan],
        }
    )

    frame["forward_return_20d_available_at"] = frame["timestamp"] + pd.offsets.BDay(21)
    result = _add_a_share_semantic_columns(frame, spec)

    assert result["eligible"].tolist() == [True, True, False]
    assert result["model_target"].iloc[:2].tolist() == pytest.approx([-1.0, 1.0])
    assert pd.isna(result.loc[2, "model_target"])
    assert "can_enter" not in result
    assert "can_exit" not in result
    assert "scenario_universe" not in result

    fusion = frame.assign(
        industry_standard_id="CSRC2012",
        industry_level1_code="A01",
        industry_level1_name="industry",
        is_nonfinancial_industry=True,
        has_legal_fundamental_pit=[False, True, True],
    )
    fusion_result = _add_a_share_semantic_columns(
        fusion, research_dataset_catalog().get("fusion_nf_core")
    )
    assert fusion_result["eligible"].tolist() == [False, True, False]
    assert not _add_a_share_semantic_columns(
        fusion.assign(is_nonfinancial_industry=False),
        research_dataset_catalog().get("fusion_nf_core"),
    )["eligible"].any()


def test_a_share_model_target_zero_variance_does_not_invalidate_other_date() -> None:
    spec = research_dataset_catalog().get("pv_all")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": False,
            "is_pt": False,
            "listing_age_trading_days": 252,
            "has_252_trading_sessions": True,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "turnover": 0.1,
            "forward_return_20d": [0.10, 0.10, 0.10, 0.20],
        }
    )

    frame["forward_return_20d_available_at"] = frame["timestamp"] + pd.offsets.BDay(21)
    result = _add_a_share_semantic_columns(frame, spec)

    assert result.loc[result["timestamp"].eq("2024-01-02"), "model_target"].isna().all()
    assert result.loc[
        result["timestamp"].eq("2024-01-03"), "model_target"
    ].tolist() == pytest.approx([-1.0, 1.0])


def test_insufficient_neutralization_is_missing_not_observed_zero() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "log_total_market_cap": [1.0, 2.0],
            "factor": [1.0, 2.0],
        }
    )

    result = a_share_feature_pipeline(frame, factor_columns=("factor",))

    assert result["feature_factor"].isna().all()
    assert result["feature_missing_factor"].tolist() == [1, 1]
    assert result["feature_observed_factor"].tolist() == [0, 0]


def test_missing_factor_in_other_industry_does_not_change_valid_neutralization():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": ["A"] * 200 + ["B"],
            "log_total_market_cap": np.linspace(1.0, 3.0, 201),
            "factor": [*np.sin(np.arange(200)), np.nan],
        }
    )
    valid = frame.iloc[:200].copy()
    for operation in (_neutralize_cross_section, a_share_feature_pipeline):
        kwargs = (
            {
                "industry_column": "industry_level1_code",
                "size_column": "log_total_market_cap",
                "rescale": False,
            }
            if operation is _neutralize_cross_section
            else {}
        )
        expected = operation(valid, factor_columns=("factor",), **kwargs)
        result = operation(frame, factor_columns=("factor",), **kwargs)
        pd.testing.assert_frame_equal(result.iloc[:200], expected)
        if operation is a_share_feature_pipeline:
            assert result.feature_observed_factor.iloc[:200].eq(1).all()
            assert result.feature_missing_factor.iloc[-1] == 1
        else:
            assert result.factor.iloc[:200].notna().all()
            assert pd.isna(result.factor.iloc[-1])


@pytest.mark.parametrize("constant", (False, True))
def test_a_share_assembler_persists_only_eligible_sample_truth(
    tmp_path,
    constant,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed = processed_catalog()
    dataset_spec = research_dataset_catalog().get("pv_all")
    # Synthetic sessions: next-open entry, exit after the registered horizon.
    sessions = pd.bdate_range("2023-01-03", periods=max(LABEL_HORIZONS) + 2)
    timestamps = pd.DatetimeIndex([sessions[0]] * 3)
    keys = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c"],
            "timestamp": timestamps,
        }
    )
    panel_spec = processed.get("market_panel_1d")
    panel = keys.copy()
    values = {
        "session_date": timestamps,
        "available_at": timestamps,
        "stock_name": ["a", "b", "c"],
        "market_type": 1,
        "trading_status_code": 0,
        "filling_flag": 0,
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "adj_open": 10.0,
        "adj_close": 10.0,
        "prev_close": 10.0,
        "volume": 100.0,
        "amount": 1_000.0,
        "after_hours_volume": 0.0,
        "after_hours_amount": 0.0,
        "period_return": 0.0,
        "turnover": 0.1,
        "limit_status": 0.0,
        "is_trading": True,
        "is_a_share_scope": True,
        "is_st": False,
        "is_pt": False,
        "listing_date": pd.Timestamp("2020-01-01"),
        "delisting_date": pd.NaT,
        "listing_age_trading_days": 252,
        "has_252_trading_sessions": True,
        "industry_standard_id": "CSRC2012",
        "industry_level1_code": "A",
        "industry_level1_name": "industry",
        "is_nonfinancial_industry": True,
        "share_change_date": pd.Timestamp("2020-01-01"),
        "total_shares": 1_000.0,
        "a_float_shares": 1_000.0,
        "total_market_cap": 10_000.0,
        "float_market_cap": 10_000.0,
    }
    for column, value in values.items():
        panel[column] = value
    write_dataset(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        panel.loc[:, list(panel_spec.required_columns)],
        spec=panel_spec,
    )

    for factor_id in ("price_volume_facts", "market_model_factors"):
        factor_spec = processed.get(factor_id)
        factors = keys.copy()
        for column in factor_spec.required_columns:
            if column not in factors:
                factors[column] = [1.0, 2.0, 3.0]
        write_dataset(
            layout.processed_data_root(
                factor_spec.market_id, factor_spec.kind, factor_spec.data_id
            ),
            factors.loc[:, list(factor_spec.required_columns)],
            spec=factor_spec,
        )

    label_spec = processed.get("forward_returns")
    labels = keys.copy()
    for column in label_spec.required_columns:
        if column in labels:
            continue
        if column == "forward_return_20d":
            labels[column] = [0.10] * 3 if constant else [0.10, 0.20, np.nan]
        elif column.endswith("_available_at"):
            horizon = int(
                column.removeprefix("forward_return_").removesuffix("d_available_at")
            )
            labels[column] = sessions[horizon + 1]
        else:
            labels[column] = 0.10
    if not constant:
        labels.loc[2, "forward_return_20d_available_at"] = pd.NaT
    write_dataset(
        layout.processed_data_root(
            label_spec.market_id, label_spec.kind, label_spec.data_id
        ),
        labels.loc[:, list(label_spec.required_columns)],
        spec=label_spec,
    )

    output = assemble_research_dataset(
        inputs={
            key: layout.processed_data_root(
                processed.get(key).market_id, processed.get(key).kind, key
            )
            for key in dataset_spec.input_data_ids
        },
        output_root=tmp_path / "assembled",
        dataset_catalog=ResearchDatasetCatalog((dataset_spec,)),
        processed_catalog=processed,
        dataset_id=dataset_spec.data_id,
        time_window=TimeWindow("2023-01-03", "2023-01-03"),
    )
    result = pd.concat(
        read_dataset(
            output,
            time_window=TimeWindow("2023-01-03", "2023-01-03"),
            mode=DatasetReadMode.CHUNKED,
        ),
        ignore_index=True,
    )

    assert set(("eligible", "model_target")).issubset(result.columns)
    assert {"scenario_universe", "can_enter", "can_exit"}.isdisjoint(result)
    assert result["eligible"].tolist() == [True, True, True]
    if constant:
        assert result.model_target.isna().all()
    else:
        assert result["model_target"].iloc[:2].tolist() == pytest.approx([-1.0, 1.0])
        assert pd.isna(result.loc[2, "model_target"])


@pytest.mark.parametrize("variation", (0.0, 1e-8))
def test_neutralization_constant_has_no_artificial_alpha_and_keeps_small_variation(
    variation,
):
    axis = np.arange(120, dtype="float64")
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": ["A", "B"] * 60,
            "log_total_market_cap": np.linspace(15.17, 24.81, 120),
            "factor": 2.3 + variation * np.sin(axis),
        }
    )
    actual = a_share_feature_pipeline(frame, factor_columns=("factor",))
    assert actual.feature_observed_factor.eq(1).all()
    assert actual.feature_missing_factor.eq(0).all()
    if variation == 0:
        assert actual.feature_factor.eq(0.0).all()
    else:
        assert actual.feature_factor.std(ddof=0) == pytest.approx(1.0)
        assert actual.feature_factor.nunique() > 100


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "origin,step,groups", ((0.0, 1.0, 1), (15.0, 0.125, 2), (20.0, 0.0625, 3))
)
def test_exact_industry_size_span_is_zero_but_one_ulp_signal_survives(
    reverse, origin, step, groups
):
    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )

    index = np.arange(120)
    size = origin + index * step
    industry = index % groups
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": [f"asset_{i}" for i in index],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": industry.astype(str),
            "log_total_market_cap": size,
            "factor": 3.0 + 4.0 * industry + 2.0 * size,
            "eligible": True,
        }
    )
    if reverse:
        frame = frame.iloc[::-1]
    residual = _neutralize_cross_section(
        frame,
        factor_columns=("factor",),
        industry_column="industry_level1_code",
        size_column="log_total_market_cap",
        rescale=False,
    )
    assert residual.factor.eq(0.0).all()
    actual = a_share_feature_pipeline(frame, factor_columns=("factor",))
    assert actual.feature_factor.eq(0.0).all()
    assert actual.feature_observed_factor.eq(1).all()
    assert actual.feature_missing_factor.eq(0).all()
    changed = frame.copy()
    changed.loc[60, "factor"] = np.nextafter(changed.loc[60, "factor"], np.inf)
    perturbed = a_share_feature_pipeline(changed, factor_columns=("factor",))
    assert perturbed.feature_factor.ne(0.0).any()
    assert perturbed.feature_observed_factor.eq(1).all()


@pytest.mark.parametrize("offset", [0.0, 1024.0])
@pytest.mark.parametrize("changed", [0, 64, 127])
def test_neutralization_preserves_single_ulp_projection(offset, changed):
    from fractions import Fraction

    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )

    n = 128
    size = np.arange(1, n + 1, dtype=float)
    values = size + offset
    values[changed] = np.nextafter(values[changed], np.inf)
    centered_size = [Fraction(2 * i - n + 1, 2) for i in range(n)]
    exact_y = [Fraction(float(value)) for value in values]
    mean = sum(exact_y) / n
    slope = sum(x * y for x, y in zip(centered_size, exact_y, strict=True)) / sum(
        x * x for x in centered_size
    )
    residuals = np.array(
        [
            float(y - mean - slope * x)
            for x, y in zip(centered_size, exact_y, strict=True)
        ]
    )
    expected = (residuals - residuals.mean()) / residuals.std()
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "log_total_market_cap": size,
            "factor": values,
            "eligible": True,
        }
    )
    actual = _neutralize_cross_section(
        frame,
        factor_columns=("factor",),
        industry_column="industry_level1_code",
        size_column="log_total_market_cap",
        rescale=False,
    )
    np.testing.assert_array_equal(actual["factor"], residuals)
    features = a_share_feature_pipeline(frame, factor_columns=("factor",))
    np.testing.assert_allclose(
        features["feature_factor"], expected, rtol=1e-14, atol=1e-14
    )
    assert features["feature_observed_factor"].eq(1).all()
    assert features["feature_missing_factor"].eq(0).all()


def test_auxiliary_size_winsorization_matches_full_configuration():
    size = np.r_[np.linspace(1.0, 4.0, 119), 1e12]
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "log_total_market_cap": size,
            "factor": np.sin(np.arange(120)),
        }
    )
    single = a_share_feature_pipeline(frame, factor_columns=("factor",))
    full = a_share_feature_pipeline(
        frame, factor_columns=("factor", "log_total_market_cap")
    )
    pd.testing.assert_series_equal(
        single.feature_factor, full.feature_factor, check_exact=True
    )
    assert "feature_log_total_market_cap" not in single
    frame.loc[0, "factor"] = np.nan
    frame.loc[1, "log_total_market_cap"] = np.nan
    frame.loc[2, "industry_level1_code"] = None
    result = a_share_feature_pipeline(frame, factor_columns=("factor",))
    assert result.feature_factor.iloc[0] == 0.0
    assert result.feature_factor.iloc[1:3].isna().all()
    assert result.feature_missing_factor.iloc[:3].eq(1).all()


def test_exact_nonzero_size_variation_is_not_svd_rank_failure():
    size = np.resize([1.0, np.nextafter(1.0, np.inf)], 120)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "log_total_market_cap": size,
            "factor": np.resize([1.0, 3.0], 120),
        }
    )
    result = _neutralize_cross_section(
        frame,
        factor_columns=("factor",),
        industry_column="industry_level1_code",
        size_column="log_total_market_cap",
        rescale=False,
    )
    assert result.factor.eq(0.0).all()
    frame["log_total_market_cap"] = 1.0
    assert (
        _neutralize_cross_section(
            frame,
            factor_columns=("factor",),
            industry_column="industry_level1_code",
            size_column="log_total_market_cap",
            rescale=False,
        )
        .factor.isna()
        .all()
    )


def test_canonical_industry_residuals_keep_same_code_standards_distinct():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-06-01"),
            "eligible": True,
            "industry_standard_id": ["CSRC2012"] * 60 + ["CAPCO"] * 60,
            "industry_level1_code": "A",
            "factor": [10.0] * 60 + [20.0] * 60,
            "log_total_market_cap": np.arange(120, dtype="float64"),
        }
    )
    before = frame.copy(deep=True)
    result = a_share_feature_pipeline(frame, factor_columns=("factor",))
    assert result["feature_factor"].eq(0.0).all()
    assert result["feature_observed_factor"].eq(1.0).all()
    pd.testing.assert_frame_equal(frame, before)
    assert result["industry_level1_code"].eq("A").all()


@pytest.mark.parametrize(
    "scale,cancellation", ((1.0, False), (1e308, False), (1e308, True))
)
def test_native_neutralization_zscore_publishes_representable_extreme_cross_section(
    tmp_path, scale, cancellation
):
    from decimal import Decimal, localcontext
    from fractions import Fraction

    from quant_research_platform.catalog.processed import ProcessedDataCatalog
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.contracts.data import ProcessedDataKind
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import _processed_spec
    from tests.helpers.research_specs import make_demo_research_dataset_spec

    positions = np.arange(100)
    values = np.where(positions < 50, -1.0, 1.0)
    values[0] = 1.0
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i:03d}" for i in positions],
            "timestamp": pd.Timestamp("2023-01-03"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "log_total_market_cap": 4.0 + positions / 100,
            "earnings_yield": values * scale,
            "label": 0.1,
        }
    )
    if cancellation:
        frame["log_total_market_cap"] = [4.0] * 49 + [5.0] + [6.0] * 50
        frame["earnings_yield"] = [-1e308] * 49 + [1e-20] + [1e308] * 50
    x = [Fraction(float(v)) for v in frame.log_total_market_cap]
    y = [Fraction(float(v)) for v in frame.earnings_yield]
    mx, my = sum(x) / 100, sum(y) / 100
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sum(
        (a - mx) ** 2 for a in x
    )
    residual = [b - my - slope * (a - mx) for a, b in zip(x, y, strict=True)]
    with localcontext() as context:
        context.prec = 100
        decimal = [Decimal(v.numerator) / Decimal(v.denominator) for v in residual]
        center = sum(decimal) / 100
        deviation = (sum((v - center) ** 2 for v in decimal) / 100).sqrt()
        expected = np.asarray([float((v - center) / deviation) for v in decimal])
    if scale == 1e308 and not cancellation:
        assert max(map(abs, residual)) > Fraction(float(np.finfo("float64").max))
        with pytest.raises(OverflowError):
            _neutralize_cross_section(
                frame,
                factor_columns=("earnings_yield",),
                industry_column="industry_level1_code",
                size_column="log_total_market_cap",
                rescale=False,
            )
    else:
        actual_raw = _neutralize_cross_section(
            frame,
            factor_columns=("earnings_yield",),
            industry_column="industry_level1_code",
            size_column="log_total_market_cap",
            rescale=False,
        )
        np.testing.assert_array_equal(
            actual_raw.earnings_yield, [float(v) for v in residual]
        )

    recipe = (FeatureTransformSpec("a_share_canonical", ("earnings_yield",)),)
    spec = make_demo_research_dataset_spec(
        data_id="extreme_features",
        input_data_ids=("feature_inputs",),
        base_input_data_id="feature_inputs",
        factor_columns=("earnings_yield",),
        label_columns=("label",),
        filter_columns=("eligible",),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        context_columns=(
            "log_total_market_cap",
            "industry_standard_id",
            "industry_level1_code",
        ),
        feature_columns=recipe[0].output_columns,
        feature_transforms=recipe,
    )
    source = _processed_spec(
        data_id="feature_inputs",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.ANNOTATION,
        column_roles={column: spec.column_roles[column] for column in frame},
    )
    layout = StorageLayout(tmp_path)
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        frame,
        spec=source,
    )
    root = assemble_research_dataset(
        inputs={
            source.data_id: layout.processed_data_root(
                source.market_id, source.kind, source.data_id
            )
        },
        output_root=tmp_path / "assembled",
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        processed_catalog=ProcessedDataCatalog((source,)),
        dataset_id=spec.data_id,
        time_window=TimeWindow("2023-01-03", "2023-01-03"),
    )
    actual = read_dataset(root)
    assert len(actual) == 100
    assert actual.feature_observed_earnings_yield.eq(1).all()
    assert actual.feature_missing_earnings_yield.eq(0).all()
    np.testing.assert_allclose(
        actual.feature_earnings_yield, expected, rtol=3e-14, atol=3e-14
    )
