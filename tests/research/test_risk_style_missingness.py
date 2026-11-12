"""Risk transforms use per-style fit domains before joint regressions."""

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.portfolios.daily as daily
from quant_research_platform.research.portfolios.attribution import (
    HoldingFactorInputs,
    holding_attribution,
)
from quant_research_platform.research.risk import (
    FactorRiskConfig,
    FactorRiskState,
    preprocess_risk_styles,
)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([7.0, 7.0, np.nan], [0.0, 0.0, np.nan]),
        ([7.0, np.nan], [np.nan, np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan]),
    ],
)
def test_constant_style_preserves_missingness_and_minimum_count(values, expected):
    frame = pd.DataFrame(
        {"timestamp": pd.Timestamp("2024-01-02"), "eligible": True, "style": values}
    )
    actual = preprocess_risk_styles(frame, style_columns=("style",))
    np.testing.assert_allclose(actual["style"], expected, equal_nan=True)


@pytest.mark.parametrize(
    "separation,signal,response_kind",
    [
        (1e-5, 0, "level"),
        (1e-5, 0, "difference"),
        (1e-5, 2**-30, "level"),
        (1.0, 2**-40, "level"),
    ],
)
def test_neutralization_preserves_resolvable_orthogonal_signal(
    separation, signal, response_kind
):
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )

    axis = np.arange(128)
    x, u, v = (2 * ((axis >> bit) & 1) - 1 for bit in range(3))
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"A{i}" for i in axis],
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            "total_market_cap": 1.0,
            "x": x,
            "z": x + separation * u,
            "target": x + signal * v,
        }
    )
    model = FactorRiskModel(style_columns=("x", "z"))
    state = FactorRiskState(style_columns=model.style_columns)
    prepared = state.prepare_exposure_session(frame)
    if response_kind == "difference":
        frame["target"] = prepared.frame.z.to_numpy() - prepared.frame.x.to_numpy()
    for scale in (1.0, 1e-9):
        work = frame.assign(target=frame.target * scale)
        result = a_share_feature_pipeline(
            work,
            factor_columns=("target",),
            clip="none",
            industry_neutralize=False,
            size_neutralize=False,
            risk_model=model,
            risk_sessions={frame.timestamp.iloc[0]: prepared},
        )
        if signal == 0:
            assert result.feature_target.isna().all()
        else:
            # The three bit vectors and the intercept are mutually orthogonal.
            np.testing.assert_allclose(
                result.feature_target, v, rtol=0, atol=32 * np.finfo(float).eps / signal
            )
            fitted = state.fit_return_session(
                work.assign(asset_return=work.target), prepared=prepared
            )
            assert fitted is not None
            np.testing.assert_allclose(
                fitted.residuals.reindex(work.asset_id),
                signal * scale * v,
                rtol=0,
                atol=32 * np.finfo(float).eps * scale,
            )


def test_style_fit_domains_are_independent_by_column_and_day():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 3
            + [pd.Timestamp("2024-01-03")] * 3,
            "eligible": True,
            "a": [1.0, 2.0, 3.0, 7.0, 7.0, np.nan],
            "b": [1.0, 2.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    actual = preprocess_risk_styles(frame, style_columns=("a", "b"))
    full = preprocess_risk_styles(frame.fillna({"b": 3.0}), style_columns=("a", "b"))
    pd.testing.assert_series_equal(actual.a.isna(), full.a.isna(), check_exact=True)
    pd.testing.assert_series_equal(
        actual.a, full.a, check_exact=False, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(actual.a[:3], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    np.testing.assert_allclose(actual.a[3:], [0.0, 0.0, np.nan], equal_nan=True)


def _cross():
    axis = np.arange(24, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": [f"asset_{i}" for i in range(len(axis))],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "total_market_cap": 100.0 + axis,
            "eligible": True,
            "a": axis,
            "b": np.sin(axis),
            "asset_return": 0.01 + 0.002 * axis + 0.003 * np.sin(axis),
        }
    )


def test_joint_risk_fit_uses_per_style_domain_then_complete_cases_and_reconciles():
    cross = _cross()
    cross.loc[23, "b"] = np.nan
    cross.loc[22, "asset_return"] = np.nan
    state = FactorRiskState(
        style_columns=("a", "b"), config=FactorRiskConfig(minimum_cross_section=15)
    )
    expected = preprocess_risk_styles(cross, style_columns=("a", "b"))
    assert len(state.prepare_exposures(cross)) == len(cross)
    fit = state.fit_return_session(cross)
    assert fit is not None
    selected = expected.dropna(subset=["a", "b", "asset_return"]).set_index("asset_id")
    pd.testing.assert_frame_equal(fit.exposures[["a", "b"]], selected[["a", "b"]])
    np.testing.assert_allclose(
        fit.exposures @ fit.factor_returns + fit.residuals,
        fit.asset_returns,
        atol=1e-15,
    )
    assert "asset_23" not in fit.exposures.index
    assert "asset_22" not in fit.exposures.index
    unknown = cross.iloc[[0]].assign(
        asset_id="unrelated_unknown",
        eligible=False,
        listing_date=pd.NaT,
        asset_return=np.nan,
    )
    state.observe_return_session(pd.concat([cross, unknown], ignore_index=True))
    assert "unrelated_unknown" not in state._listings
    np.testing.assert_allclose(
        [state._factor_rows[0][column] for column in fit.factor_returns.index],
        fit.factor_returns,
    )
    notionals = pd.Series(100 / len(fit.exposures), index=fit.exposures.index)
    values, reasons, _ = holding_attribution(
        HoldingFactorInputs(
            decision_at=pd.Timestamp("2024-01-02"),
            entry_at=pd.Timestamp("2024-01-03"),
            end_at=pd.Timestamp("2024-01-04"),
            fitted=fit,
            decision_close=pd.Series(10.0, index=fit.exposures.index),
            entry_open=pd.Series(10.0, index=fit.exposures.index),
            benchmark_weights=pd.DataFrame(),
        ),
        after_notionals=notionals,
        nav_before=100.0,
        nav_post=100.0,
        benchmark_id=None,
        benchmark_return=None,
        holding_returns=fit.asset_returns,
    )
    assert not reasons
    assert sum(
        values[key]
        for key in ("benchmark", "industry", "style", "residual_unexplained")
    ) == pytest.approx(notionals.dot(fit.asset_returns) / 100.0, abs=1e-15)


def test_enhanced_decision_preprocesses_before_joint_style_selection():
    cross = _cross()
    state = FactorRiskState(style_columns=("a", "b"))
    scheduled = pd.DataFrame(
        {
            name: pd.Series(dtype="datetime64[ns]")
            for name in ("timestamp", "entry_at", "available_at")
        }
    )

    def exposures(frame):
        return daily._enhanced_decision(
            benchmark_weights=pd.Series(1 / len(frame), index=frame.asset_id),
            alpha=pd.Series(0.0, index=frame.asset_id),
            factor_risk=state,
            current_cross=frame,
            calibration_rows=[],
            calibration_intervals=scheduled,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
            as_of=pd.Timestamp("2024-01-02"),
        ).exposures.set_index("asset_id")

    full = exposures(cross)
    cross.loc[23, "b"] = np.nan
    missing = exposures(cross)
    pd.testing.assert_series_equal(
        missing.a, full.a.loc[missing.index], check_exact=False, rtol=1e-12, atol=1e-14
    )
    pd.testing.assert_series_equal(
        missing.a.isna(), full.a.loc[missing.index].isna(), check_exact=True
    )
    assert "asset_23" not in missing.index


@pytest.mark.parametrize(
    "eligible", ([True, True, False, False], [True, False, False, False])
)
def test_constant_projection_requires_two_eligible_values_and_keeps_null(eligible):
    cross = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": eligible,
            "style": [7.0, 7.0, 20.0, np.nan],
        }
    )
    actual = preprocess_risk_styles(cross, style_columns=("style",))
    expected = [0.0, 0.0, 0.0, np.nan] if sum(eligible) == 2 else [np.nan] * 4
    np.testing.assert_allclose(actual["style"], expected, equal_nan=True)


def test_intraday_risk_reuses_unchanged_design_and_invalidates_real_dependencies(
    monkeypatch,
):
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.research.datasets.risk import (
        prepare_asof_risk_sessions,
    )

    model = FactorRiskModel(("a", "b"), FactorRiskConfig(minimum_cross_section=15))
    state = FactorRiskState(style_columns=model.style_columns, config=model.config)
    first = pd.Timestamp("2024-01-02T01:30:00")
    cross = _cross().assign(market_id="a_share", timestamp=first)
    old = cross.assign(timestamp=pd.Timestamp("2023-12-29"), published=first)
    newer = old.assign(
        timestamp=pd.Timestamp("2024-01-01"), published=first + pd.Timedelta(seconds=6)
    )
    newer.loc[0, "a"] = 100.0
    facts = pd.concat([old, newer], ignore_index=True)
    cache = {}
    svd = np.linalg.svd
    calls = []

    def observed_svd(*args, **kwargs):
        calls.append(args[0].shape)
        return svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", observed_svd)
    prepared = []
    for seconds in (0, 3, 6):
        at = first + pd.Timedelta(seconds=seconds)
        frame = cross.assign(timestamp=at)
        session = prepare_asof_risk_sessions(
            frame, facts, model=model, available_at_column="published", cache=cache
        )[at]
        prepared.append(session)
        response = frame.assign(asset_return=cross.asset_return + seconds * 0.01)
        fitted = state.fit_return_session(response, prepared=session)
        design = fitted.exposures.to_numpy()
        root_weight = (
            cross.set_index("asset_id")
            .total_market_cap.reindex(fitted.asset_returns.index)
            .to_numpy()
            ** 0.25
        )
        expected = np.linalg.lstsq(
            design * root_weight[:, None],
            fitted.asset_returns.to_numpy() * root_weight,
            rcond=None,
        )[0]
        np.testing.assert_allclose(
            fitted.factor_returns, expected, rtol=1e-11, atol=1e-13
        )
    assert prepared[0] is prepared[1]
    assert prepared[2] is not prepared[1]
    assert len(calls) == 2
    response.loc[0, "asset_return"] = np.nan
    partial = state.fit_return_session(response, prepared=prepared[2])
    assert len(calls) == 3
    assert len(partial.asset_returns) == 23
    # The prepared domain is unchanged by a response missing at its future endpoint.
    assert len(prepared[2].frame) == 24
    frame.loc[0, "listing_date"] = first.normalize()
    changed_life = prepare_asof_risk_sessions(
        frame, facts, model=model, available_at_column="published", cache=cache
    )[at]
    assert "asset_0" not in set(changed_life.frame.asset_id)


def test_risk_fact_window_keeps_life_baselines_late_updates_and_latest_nulls(
    tmp_path, monkeypatch
):
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.datasets import risk
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from tests.helpers.research_processed_inputs import assembly_source

    # Observation dates and publication clocks are intentionally out of order.
    rows = [
        ("A", "01", "01", "01", 1.0),
        ("A", "04", "04", "01", 4.0),
        ("A", "02", "06 12:00", "01", 200.0),
        ("A", "05", "06 12:00", "01", 5.0),
        ("A", "06", "06 12:00", "01", 6.0),
        ("B", "01", "01", "01", 1.0),
        ("B", "04", "06 12:00", "01", 4.0),
        ("B", "03", "08", "01", 3.0),
        ("C", "01", "01", "01", 1.0),
        ("C", "04", "04", "04", 44.0),
        ("C", "05", "05", "04", np.nan),
        ("D", "03", "03", "01", 3.0),
        ("D", "05", None, "01", 5.0),
    ]
    frame = pd.DataFrame(
        rows, columns=["asset_id", "timestamp", "published", "listing_date", "a"]
    )
    for column in ("timestamp", "published", "listing_date"):
        frame[column] = pd.to_datetime(
            "2024-01-" + frame[column], format="mixed"
        ).dt.as_unit("ns")
    frame = frame.assign(
        market_id="a_share",
        industry_standard_id="test",
        industry_level1_code="A",
        total_market_cap=100.0,
    ).astype({"market_id": "string", "asset_id": "string"})
    # Latest known observation wins, not latest publication of an older row.
    expected = frame.iloc[[1, 3, 5, 6, 8, 10, 11]]
    model = FactorRiskModel(("a",))
    original = risk.iter_dataset_chunks
    keys = ["asset_id", "listing_date", "timestamp"]
    for order, batch_rows in ((frame, 2), (frame.iloc[::-1], 1000)):
        root, _ = assembly_source(
            tmp_path / str(batch_rows), order.reset_index(drop=True)
        )
        consumed = []

        def measured(session, *, size=batch_rows, seen=consumed):
            for chunk in original(session, batch_rows=size):
                seen.extend(chunk.timestamp)
                yield chunk

        monkeypatch.setattr(risk, "iter_dataset_chunks", measured)
        actual = risk.read_asof_risk_facts(
            open_dataset_session(root),
            model=model,
            available_at_column="published",
            time_window=TimeWindow("2024-01-06", "2024-01-06"),
        )
        pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True),
            expected.loc[:, actual.columns].sort_values(keys).reset_index(drop=True),
        )
        assert max(consumed) < pd.Timestamp("2024-01-06")
        assert risk.read_asof_risk_facts(
            open_dataset_session(root),
            model=model,
            available_at_column="published",
            time_window=TimeWindow("2024-01-01", "2024-01-01"),
        ).empty

        # An observation published on its own date remains unavailable until
        # the following date, including across a multi-day decision batch.
        facts = risk.read_asof_risk_facts(
            open_dataset_session(root),
            model=model,
            available_at_column="published",
            time_window=TimeWindow("2024-01-06", "2024-01-07"),
        )
        identities = frame.iloc[[1, 5, 10, 11]][
            ["market_id", "asset_id", "listing_date"]
        ]
        decisions = pd.concat(
            [
                identities.assign(timestamp=pd.Timestamp(at), eligible=True)
                for at in ("2024-01-06 12:00", "2024-01-07 12:00")
            ],
            ignore_index=True,
        )
        prepared = risk.prepare_asof_risk_sessions(
            decisions, facts, model=model, available_at_column="published"
        )
        for session, values in zip(
            prepared.values(),
            ([np.sqrt(1.5), 0.0, -np.sqrt(1.5)], np.array([5, -1, -4]) / np.sqrt(14)),
            strict=True,
        ):
            styles = session.frame.set_index("asset_id").a
            np.testing.assert_allclose(styles.loc[["A", "B", "D"]], values, atol=1e-14)
            assert pd.isna(styles.loc["C"])


def test_risk_frames_keep_original_observation_clock_and_listing_life():
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.datasets.risk import (
        asof_risk_frame,
        compact_asof_risk_facts,
    )

    # Preopen publication describes the prior session, not its decision date.
    source = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "A",
            "listing_date": pd.Timestamp("2020-01-01"),
            "timestamp": pd.to_datetime(
                ["2024-01-08 01:00", "2024-01-09 01:00", "2024-01-09 03:00"]
            ),
            "signal_session": pd.to_datetime(
                ["2024-01-05", "2024-01-08", "2024-01-04"]
            ),
            "published": pd.to_datetime(
                ["2024-01-08 01:00:01", "2024-01-09 01:00:01", "2024-01-09 03:00:01"]
            ),
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            "total_market_cap": 100.0,
            "a": [10.0, 20.0, 99.0],
        }
    )
    model = FactorRiskModel(("a",))
    frames = (
        chunk.assign(timestamp=chunk.signal_session).drop(columns="signal_session")
        for chunk in (source.iloc[:1], source.iloc[1:])
    )
    facts = compact_asof_risk_facts(
        frames,
        model=model,
        available_at_column="published",
        time_window=TimeWindow("2024-01-08", "2024-01-09"),
    )
    decisions = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "A",
            "listing_date": pd.Timestamp("2020-01-01"),
            "timestamp": pd.to_datetime(
                [
                    "2024-01-08 01:00",
                    "2024-01-08 01:00:01",
                    "2024-01-08 01:35",
                    "2024-01-09 01:35",
                    "2024-01-09 04:00",
                ],
                format="mixed",
            ),
            "eligible": True,
        }
    )
    aligned = asof_risk_frame(
        decisions, facts, model=model, available_at_column="published"
    )
    assert pd.isna(aligned.a.iloc[0])
    assert aligned.a.iloc[1:].tolist() == [10.0, 10.0, 20.0, 20.0]
    changed_life = decisions.iloc[[-1]].assign(listing_date=pd.Timestamp("2024-01-09"))
    assert (
        asof_risk_frame(
            changed_life, facts, model=model, available_at_column="published"
        )
        .a.isna()
        .all()
    )
    assert facts.timestamp.tolist() == list(
        pd.to_datetime(["2024-01-05", "2024-01-08"])
    )


def test_unknown_eligibility_only_excludes_dependent_risk_and_target_values():
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )
    from quant_research_platform.research.labels.targets import a_share_model_target

    model = FactorRiskModel(("a", "b"), FactorRiskConfig(minimum_cross_section=15))
    cross = _cross().assign(
        factor=np.cos(np.arange(24)),
        forward_return_20d=np.cos(np.arange(24)) / 100,
        forward_return_20d_available_at=pd.Timestamp("2024-01-05"),
    )
    cross["eligible"] = cross.eligible.astype("boolean")
    cross.loc[23, "eligible"] = pd.NA
    excluded = cross.fillna({"eligible": False})
    state = FactorRiskState(style_columns=model.style_columns, config=model.config)
    pd.testing.assert_frame_equal(
        state.prepare_exposure_session(cross).frame,
        state.prepare_exposure_session(excluded).frame,
    )
    target = a_share_model_target(cross, transform="risk_residual", risk_model=model)
    expected = a_share_model_target(
        excluded, transform="risk_residual", risk_model=model
    )
    pd.testing.assert_frame_equal(target, expected)
    transformed = a_share_feature_pipeline(
        cross,
        factor_columns=("factor",),
        recipe_id="a_share_canonical",
        industry_neutralize=False,
        size_neutralize=False,
        risk_model=model,
    )
    pd.testing.assert_series_equal(transformed.factor, cross.factor)
    pd.testing.assert_series_equal(transformed.eligible, cross.eligible)
    assert transformed.feature_factor.iloc[:23].notna().all()
    assert pd.isna(transformed.feature_factor.iloc[23])
    assert pd.isna(target.model_target.iloc[23])
