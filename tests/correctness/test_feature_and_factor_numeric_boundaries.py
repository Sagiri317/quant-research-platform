import pickle
from decimal import Decimal, localcontext
from fractions import Fraction
from statistics import median

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.engine import calculate_factors_for_panel
from quant_research_platform.research.factors.price_volume.definitions import (
    price_volume_factor_definitions,
)
from quant_research_platform.research.features.transforms import (
    _mad_winsorize,
    a_share_feature_pipeline,
)
from quant_research_platform.research.models.scaler import ModelMatrixScaler
from quant_research_platform.research.portfolios.risk import preprocess_risk_styles
from quant_research_platform.research.statistics import hac_mean
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.helpers.paths import StorageLayout
from tests.helpers.research_specs import (
    assemble_research_dataset,
    make_demo_research_dataset_spec,
)


@pytest.mark.parametrize("weights", ([1.0, 1.0], [1.0, 3.0]))
def test_scaler_preserves_weighted_center_across_chunks_and_pickle(weights):
    frame = pd.DataFrame({"x": [1.0, 3.0]})
    exact = [Fraction.from_float(value) for value in frame.x]
    exact_weights = [Fraction.from_float(weight) for weight in weights]
    center = sum(
        value * weight for value, weight in zip(exact, exact_weights, strict=True)
    ) / sum(exact_weights)
    variance = sum(
        (value - center) ** 2 * weight
        for value, weight in zip(exact, exact_weights, strict=True)
    ) / sum(exact_weights)
    with localcontext() as context:
        context.prec = 100
        scale = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        expected = [
            float(
                Decimal((value - center).numerator)
                / Decimal((value - center).denominator)
                / scale
            )
            for value in exact
        ]
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    actual = scaler.transform(frame).x.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
    assert np.dot(actual, weights) == pytest.approx(0, abs=1e-15)
    restored = pickle.loads(pickle.dumps(scaler))
    chunks = pd.concat(
        [restored.transform(frame.iloc[:1]), restored.transform(frame.iloc[1:])]
    )
    pd.testing.assert_frame_equal(chunks, scaler.transform(frame))
    assert restored.to_payload() == scaler.to_payload()
    assert restored.to_payload()["scaling"] is True
    assert set(restored.to_payload()) == {
        "columns",
        "scaling",
        "weighted_mean",
        "weighted_population_scale",
    }


def test_risk_styles_remain_zero_centered():
    values = np.array([1.0, 3.0])
    frame = pd.DataFrame(
        {"timestamp": pd.Timestamp("2020-01-01"), "eligible": True, "style": values}
    )
    result = preprocess_risk_styles(frame, style_columns=("style",))
    np.testing.assert_array_equal(result["style"].to_numpy(), [-1.0, 1.0])
    assert result["style"].mean() == 0


def test_constant_eligible_risk_style_projects_zero_to_complete_benchmark_domain():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["fit_a", "fit_b", "benchmark_only"],
            "eligible": [True, True, False],
            "style": [1.0, 1.0, 2.0],
        }
    )
    result = preprocess_risk_styles(frame, style_columns=("style",))
    pd.testing.assert_frame_equal(
        result.drop(columns="style"), frame.drop(columns="style")
    )
    pd.testing.assert_series_equal(result["style"], pd.Series([0.0] * 3, name="style"))


@pytest.mark.parametrize("unit", (1.0, 100.0, 0.01, -100.0))
def test_hac_units_match_independent_bartlett_oracle(unit):
    values = np.array([(0.1 + 0.01 * i) * unit for i in range(24)])
    exact = [Fraction.from_float(float(value)) for value in values]
    count = len(exact)
    center = sum(exact) / count
    residuals = [value - center for value in exact]
    meat = sum(value * value for value in residuals)
    for lag in range(1, 20):
        meat += (
            2
            * Fraction(20 - lag, 20)
            * sum(residuals[i] * residuals[i - lag] for i in range(lag, count))
        )
    variance = meat / count**2
    with localcontext() as context:
        context.prec = 100
        standard_error = (
            Decimal(variance.numerator) / Decimal(variance.denominator)
        ).sqrt()
        mean = Decimal(center.numerator) / Decimal(center.denominator)
        t_statistic = mean / standard_error
    actual = hac_mean(values, max_lags=19)
    assert actual.mean == pytest.approx(float(mean), rel=1e-14, abs=0)
    assert actual.standard_error == pytest.approx(
        float(standard_error), rel=1e-14, abs=0
    )
    assert actual.t_statistic == pytest.approx(float(t_statistic), rel=1e-14, abs=0)
    assert np.isfinite(actual.confidence_interval).all()


@pytest.mark.parametrize("window", (20, 60, 126))
def test_amihud_mean_survives_native_write_and_read(tmp_path, window):
    column = f"amihud_{window}d"
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": pd.bdate_range("2020-01-01", periods=window),
            "period_return": 0.01,
            "amount": 1e8,
            "filling_flag": 0,
        }
    )
    definitions = price_volume_factor_definitions(
        market_id="a_share", required_input_data_id="panel", factor_columns=(column,)
    )
    calculated = calculate_factors_for_panel(panel, definitions=definitions)
    expected = float(Fraction(0.01) / Fraction(1e8))
    assert calculated[column].iloc[:-1].isna().all()
    assert calculated[column].iloc[-1] == pytest.approx(expected, rel=2e-12, abs=0)
    spec = ProcessedDataSpec(
        data_id="amihud",
        market_id="a_share",
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            column: ColumnRole.FACTOR,
        },
    )
    root = StorageLayout(project_root=tmp_path).processed_data_root(
        "a_share", spec.kind, spec.data_id
    )
    write_dataset(root, calculated, spec=spec)
    pd.testing.assert_frame_equal(
        read_dataset(root),
        calculated.assign(
            market_id=calculated.market_id.astype("str"),
            asset_id=calculated.asset_id.astype("str"),
            timestamp=calculated.timestamp.astype("datetime64[ns]"),
        ),
        check_exact=True,
    )
    assert (
        inspect_dataset(root).manifest["non_finite_count_summary"].get(column, 0) == 0
    )


def test_amihud_mean_uses_valid_observations_and_freshness():
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2020-01-01", periods=20),
            "period_return": [0.1, *([0.0] * 19)],
            "amount": 1e8,
            "filling_flag": 0,
        }
    )
    from quant_research_platform.research.factors.price_volume.calculations import (
        calculate_price_volume_factors,
    )

    result = calculate_price_volume_factors(panel, factor_columns=("amihud_20d",))
    expected = float(Fraction(0.1) / Fraction(1e8) / 20)
    assert result.amihud_20d.iloc[-1] == pytest.approx(expected, rel=2e-12, abs=0)
    panel.loc[3, "amount"] = 0
    partial = calculate_price_volume_factors(panel, factor_columns=("amihud_20d",))
    expected_partial = float(Fraction(0.1) / Fraction(1e8) / 19)
    assert partial.amihud_20d.iloc[-1] == pytest.approx(
        expected_partial, rel=2e-12, abs=0
    )
    # Fewer than 15 valid observations fails the 75% freshness threshold.
    panel.loc[3:8, "amount"] = 0
    assert (
        calculate_price_volume_factors(panel, factor_columns=("amihud_20d",))
        .amihud_20d.isna()
        .all()
    )


@pytest.mark.parametrize(
    "values",
    (
        [0.8, 0.9, 1.0, 5.0],
        [-5.0, -1.0, -0.9, -0.8],
        [-2.0, -1.0, 0.0, 1.0, 8.0],
        [1.0, 1.0, 2.0],
    ),
)
def test_mad_bounds_match_independent_median_oracle(values):
    source = pd.Series([*values, np.nan])
    exact = [Fraction.from_float(value) for value in values]
    center = median(exact)
    mad = median(abs(value - center) for value in exact)
    radius = 3 * Fraction.from_float(1.4826) * mad
    expected = [
        float(min(max(value, center - radius), center + radius))
        if mad
        else float(value)
        for value in exact
    ]
    clipped = _mad_winsorize(source, source.groupby(pd.Series([0] * len(source))))
    np.testing.assert_allclose(clipped.iloc[:-1], expected, rtol=1e-13, atol=1e-15)
    assert pd.isna(clipped.iloc[-1])


def test_mad_preserves_complete_canonical_features_under_units_change(tmp_path):
    values = np.linspace(0.8, 1.0, 100)
    values[-1] = 1.7
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"a{i:03d}" for i in range(100)],
            "timestamp": pd.Timestamp("2020-01-01"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "log_total_market_cap": 4 + np.sin(np.arange(100)),
            "factor": values,
        }
    )
    actual = a_share_feature_pipeline(frame, factor_columns=("factor",))
    scaled = a_share_feature_pipeline(
        frame.assign(factor=values * 100), factor_columns=("factor",)
    )
    np.testing.assert_allclose(
        actual.feature_factor, scaled.feature_factor, rtol=1e-13, atol=1e-13
    )
    pd.testing.assert_frame_equal(
        actual[["feature_missing_factor", "feature_observed_factor"]],
        scaled[["feature_missing_factor", "feature_observed_factor"]],
    )
    roles = dict.fromkeys(actual.columns, ColumnRole.CONTEXT)
    roles.update(
        market_id=ColumnRole.KEY,
        asset_id=ColumnRole.KEY,
        timestamp=ColumnRole.TIMESTAMP,
        factor=ColumnRole.FACTOR,
    )
    roles.update(
        {
            column: ColumnRole.FEATURE
            for column in actual
            if column.startswith("feature_")
        }
    )
    spec = ProcessedDataSpec(
        data_id="canonical_units",
        market_id="a_share",
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
        column_roles=roles,
    )
    root = StorageLayout(project_root=tmp_path).processed_data_root(
        "a_share", spec.kind, spec.data_id
    )
    write_dataset(root, actual, spec=spec)
    persisted = read_dataset(root)
    pd.testing.assert_frame_equal(
        persisted.select_dtypes("number"),
        actual.select_dtypes("number"),
        check_exact=True,
    )
    assert persisted.timestamp.dtype == np.dtype("datetime64[ns]")
    assert persisted.asset_id.tolist() == actual.asset_id.tolist()


def test_canonical_degeneracy_diagnostics_accumulate_in_native_manifest(
    tmp_path, monkeypatch
):
    import quant_research_platform.storage.datasets.writing as writing

    monkeypatch.setattr(writing, "DEFAULT_BATCH_ROWS", 37)
    transforms = (FeatureTransformSpec("a_share_canonical", ("factor",)),)
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
        "eligible": ColumnRole.UNIVERSE,
        "industry_standard_id": ColumnRole.CONTEXT,
        "industry_level1_code": ColumnRole.CONTEXT,
        "log_total_market_cap": ColumnRole.CONTEXT,
    }
    source_spec = ProcessedDataSpec(
        data_id="feature_input",
        market_id="a_share",
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
        column_roles=roles,
    )
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "market_id": "a_share",
                    "asset_id": [f"a{i:03d}" for i in range(102)],
                    "timestamp": timestamp,
                    "factor": float(i + 1) if i < 2 else np.cos(np.arange(102)),
                    "label": 0.1,
                    "eligible": [True] * 101 + [False],
                    "industry_standard_id": "CSRC2012",
                    "industry_level1_code": "A",
                    "log_total_market_cap": 4 + np.sin(np.arange(102)),
                }
            )
            for i, timestamp in enumerate(pd.bdate_range("2020-01-01", periods=3))
        ],
        ignore_index=True,
    )
    source.loc[0, "factor"] = np.nan
    layout = StorageLayout(project_root=tmp_path)
    write_dataset(
        layout.processed_data_root("a_share", source_spec.kind, source_spec.data_id),
        source,
        spec=source_spec,
    )
    dataset_spec = make_demo_research_dataset_spec(
        data_id="numeric_features",
        input_data_ids=(source_spec.data_id,),
        base_input_data_id=source_spec.data_id,
        factor_columns=("factor",),
        feature_columns=transforms[0].output_columns,
        feature_transforms=transforms,
        label_columns=("label",),
        filter_columns=("eligible",),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        context_columns=(
            "industry_standard_id",
            "industry_level1_code",
            "log_total_market_cap",
        ),
    )
    root = assemble_research_dataset(
        inputs={
            source_spec.data_id: layout.processed_data_root(
                "a_share", source_spec.kind, source_spec.data_id
            )
        },
        output_root=tmp_path / "numeric-dataset",
        dataset_catalog=ResearchDatasetCatalog((dataset_spec,)),
        dataset_id=dataset_spec.data_id,
        time_window=TimeWindow("2020-01-01", "2020-01-03"),
    )
    result = read_dataset(root)
    diagnostic = inspect_dataset(root).manifest["producer_diagnostics"]
    assert diagnostic["canonical_features"]["mad_zero_by_input"] == {
        "factor": 2,
        "log_total_market_cap": 0,
    }
    assert diagnostic["canonical_features"]["zero_variance_by_factor"] == {"factor": 2}
    assert len(result) == len(source)
    assert not result.duplicated(list(PLATFORM_ASSET_TIME_KEY)).any()
    assert result.loc[~result.eligible, "feature_factor"].isna().all()
    observed_constant = (
        result.eligible
        & result.timestamp.lt(pd.Timestamp("2020-01-03"))
        & result.factor.notna()
    )
    assert result.loc[observed_constant, "feature_factor"].eq(0).all()
