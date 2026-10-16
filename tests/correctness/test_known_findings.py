"""Independent regression oracles for execution and statistical boundaries."""

import pickle
from math import sqrt

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _apply_no_limit_exceptions,
    _apply_special_treatment,
)
from quant_research_platform.research.datasets.assembler import _a_share_model_target
from quant_research_platform.research.labels.engine import calculate_labels_for_panel
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
)
from quant_research_platform.research.portfolios.production import (
    _mask_values,
)
from quant_research_platform.research.protocols.a_share import label_definitions
from quant_research_platform.research.statistics import (
    hac_mean,
    stationary_bootstrap_mean_ci,
    time_series_inference_summary,
)


def _panel():
    dates = pd.bdate_range("2020-01-01", periods=25)
    return pd.concat(
        [
            pd.DataFrame(
                dict(
                    listing_date=pd.Timestamp("2000-01-01"),
                    market_id="a_share",
                    asset_id=asset,
                    timestamp=dates,
                    is_trading=True,
                    filling_flag=0,
                    limit_status=0.0,
                    open=10.0,
                    high=10.0,
                    low=10.0,
                    close=10.0,
                    adj_open=10.0 + np.arange(25) * slope,
                )
            )
            for asset, slope in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize(
    "column,value", [("limit_status", np.nan), ("is_trading", None), ("open", np.nan)]
)
def test_unknown_execution_survives_persistence_as_unknown_portfolio_permission(
    tmp_path, column, value
):
    panel = _panel().astype({"is_trading": "boolean"})
    panel.loc[1, column] = value
    mask = derive_execution_mask(panel)
    mask["execution_at"] = panel.timestamp.shift(-1)
    assert pd.isna(mask.loc[0, "can_enter"])
    assert pd.isna(mask.loc[0, "can_exit"])
    path = tmp_path / "execution.parquet"
    mask.to_parquet(path)
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(mask, loaded)
    permissions = _mask_values(
        loaded.iloc[:1], pd.Index(["a"]), panel.timestamp.iloc[1], frozenset()
    )
    assert permissions["can_enter"].isna().all()
    assert permissions["can_exit"].isna().all()
    labeled = calculate_labels_for_panel(
        panel.merge(mask),
        definitions=label_definitions(),
        sessions=panel.timestamp.unique(),
    )
    if column == "is_trading":
        assert pd.isna(labeled.loc[0, "forward_return_1d"])
        assert pd.isna(labeled.loc[0, "forward_return_1d_available_at"])
    else:
        assert labeled.loc[0, "forward_return_1d"] == pytest.approx(1 / 11)
        assert labeled.loc[0, "forward_return_1d_available_at"] == panel.timestamp.iloc[
            2
        ] + pd.Timedelta(hours=7)


def test_positive_no_limit_evidence_and_unknown_special_treatment():
    panel = _panel()
    panel["limit_status"] = np.nan
    fixed = _apply_no_limit_exceptions(
        panel, pd.DataFrame({"asset_id": ["a"], "timestamp": [panel.timestamp.iloc[1]]})
    )
    mask = derive_execution_mask(fixed)
    assert mask.can_enter.iloc[0]
    assert pd.isna(mask.can_enter.iloc[1])
    panel["trading_status_code"] = pd.Series([pd.NA] * len(panel), dtype="Int64")
    facts = _apply_special_treatment(panel, None)
    assert facts[["is_st", "is_pt"]].isna().all().all()
    panel.loc[0, "trading_status_code"] = 16
    assert _apply_special_treatment(panel, None).is_pt.iloc[0]


def test_only_required_future_facts_determine_realized_target():
    panel = _panel()

    def targets(frame):
        labeled = calculate_labels_for_panel(
            frame.merge(derive_execution_mask(frame)),
            definitions=label_definitions(),
            sessions=frame.timestamp.unique(),
        )
        labeled["eligible"] = True
        return _a_share_model_target(labeled), labeled

    original, labels = targets(panel)
    assert labels.loc[0, "forward_return_20d"] == pytest.approx(31 / 11 - 1)
    assert labels.loc[0, "forward_return_20d_available_at"] == panel.timestamp.iloc[
        21
    ] + pd.Timedelta(hours=7)
    later = panel.copy()
    later.loc[23, "limit_status"] = np.nan
    changed, _ = targets(later)
    pd.testing.assert_series_equal(
        original.iloc[[0, 25, 50]], changed.iloc[[0, 25, 50]]
    )
    unknown_permission = panel.copy()
    unknown_permission.loc[1, "limit_status"] = np.nan
    changed, changed_labels = targets(unknown_permission)
    pd.testing.assert_series_equal(original, changed, rtol=1e-14, atol=0)
    pd.testing.assert_series_equal(
        labels.forward_return_20d,
        changed_labels.forward_return_20d,
        rtol=1e-14,
        atol=0,
    )
    for endpoint in (1, 21):
        required = panel.copy()
        required.loc[endpoint, "adj_open"] = np.nan
        changed, changed_labels = targets(required)
        assert pd.isna(changed.iloc[0]) and pd.notna(original.iloc[0])
        assert pd.isna(changed_labels.loc[0, "forward_return_20d_available_at"])


def test_frozen_scaler_constant_and_true_unit_scale_reload_predict(tmp_path):
    training = pd.DataFrame({"constant": [2.0, 2.0], "unit": [0.0, 2.0]})
    scaler = ModelMatrixScaler.fit(training)
    assert scaler.mean_ == (2.0, 1.0) and scaler.scale_ == (1.0, 1.0)
    np.testing.assert_array_equal(scaler.transform(training), [[0, -1], [0, 1]])
    predictor = LinearScaledPredictor((1.0, 2.0, 3.0), scaler)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(predictor))
    restored = pickle.loads(path.read_bytes())
    future = pd.DataFrame({"constant": [5.0], "unit": [4.0]})
    np.testing.assert_array_equal(restored.scaler.transform(future), [[3, 3]])
    assert restored.predict(future).tolist() == [16.0]
    for invalid in (np.nan, np.inf, "bad"):
        with pytest.raises(ValueError):
            restored.predict(pd.DataFrame({"constant": [invalid], "unit": [4.0]}))
    with pytest.raises(ValueError, match="model_scaler_columns_invalid"):
        ModelMatrixScaler.fit(training.iloc[:0])


def test_hac_hand_computed_bartlett_oracle_and_degeneracies():
    # Mean=2.5; sum of squared residuals=5; lag-one cross-products=1.25.
    result = hac_mean([1.0, 2.0, 3.0, 4.0], max_lags=1)
    assert result.mean == pytest.approx(2.5)
    assert result.standard_error == pytest.approx(sqrt(6.25 / 16))
    assert result.t_statistic == pytest.approx(4.0)
    assert result.confidence_interval == pytest.approx(
        (1.275022509662466, 3.724977490337534)
    )
    for values in ([0.0] * 4, [1.0] * 4, [1.0, 1.0 + np.finfo(float).eps] * 2):
        with pytest.raises(ValueError, match="hac_inference_undefined"):
            hac_mean(values, max_lags=0)
    for values in ([1.0], [1.0, np.nan], [1.0, np.inf]):
        with pytest.raises(ValueError):
            hac_mean(values, max_lags=0)
    for lag in (-1, 4):
        with pytest.raises(ValueError, match="hac_max_lags_invalid"):
            hac_mean([1.0, 2.0, 3.0, 4.0], max_lags=lag)
    point = stationary_bootstrap_mean_ci(
        [2.0] * 4, block_size=1, repetitions=10, seed=1
    )
    assert point.lower == point.mean == point.upper == 2.0
    with pytest.raises(ValueError, match="bootstrap_configuration_invalid"):
        stationary_bootstrap_mean_ci([1.0, 2.0], block_size=1, repetitions=1, seed=1)
    summary = time_series_inference_summary(
        pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01"]), "value": [1.0]}),
        identity_column="timestamp",
        max_lags=0,
        fdr_method=None,
        fdr_alpha=0.05,
    )
    assert summary.status.tolist() == ["UNDEFINED"]


@pytest.mark.parametrize(
    "se,pvalue", [(0.0, 0.5), (np.nan, 0.5), (1.0, np.inf), (1.0, -0.1)]
)
def test_hac_adapter_rejects_backend_non_inference(monkeypatch, se, pvalue):
    from types import SimpleNamespace

    from quant_research_platform.research import statistics

    fit = SimpleNamespace(bse=[se])
    tail_calls = []

    def tail(value):
        tail_calls.append(value)
        return pvalue / 2

    monkeypatch.setattr(statistics.norm, "sf", tail)
    monkeypatch.setattr(
        statistics, "OLS", lambda *args: SimpleNamespace(fit=lambda **kwargs: fit)
    )
    with pytest.raises(ValueError, match="hac_inference_undefined"):
        hac_mean([1.0, 2.0, 3.0], max_lags=1)
    assert len(tail_calls) == 1
    assert np.isfinite(tail_calls[0]) == (se == 1.0)


def test_non_one_word_prices_do_not_turn_missing_limit_status_into_permission():
    panel = _panel()
    panel.loc[1, "high"] = 12.0
    panel.loc[1, "limit_status"] = np.nan
    result = derive_execution_mask(panel)
    assert pd.isna(result.can_enter.iloc[0]) and pd.isna(result.can_exit.iloc[0])


def test_unobserved_next_session_is_unknown_not_forbidden():
    panel = _panel()
    mask = derive_execution_mask(panel)
    tails = mask.groupby("asset_id", sort=False).tail(1)
    assert tails[["can_enter", "can_exit"]].isna().all().all()


def test_perfect_fit_white_test_is_undefined_without_backend_division():
    from quant_research_platform.research import statistics

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b", "c", "d"],
            "factor": [0.0, 1.0, 2.0, 3.0],
            "return": [1.0, 3.0, 5.0, 7.0],
        }
    )
    with np.errstate(all="raise"):
        result = statistics.fama_macbeth_period(
            frame, return_column="return", factor_columns=("factor",)
        )
    assert result.coefficients.tolist() == pytest.approx([1.0, 2.0])
    assert (
        result.white_p_value is None and result.white_reason == "white_test_undefined"
    )
