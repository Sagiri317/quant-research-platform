"""Factor comparisons retain declared samples, clocks, and native artifacts."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.factors.statistics import factor_statistics
from tests.research.evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


def _configuration(source, output, *, factors, start, end, analyses):
    return {
        "dataset": str(source),
        "output": str(output),
        "window": {"start": start, "end": end},
        "evaluation": {
            "factors": list(factors),
            "primary_label": "label",
            "label_availability_columns": {"label": "label_available_at"},
        },
        "analyses": analyses,
        "statistics": {
            "hac_lag": 0,
            "factor_fdr_families": {"declared": list(factors)},
            "fama_macbeth_core_factors": list(factors),
        },
    }


@pytest.mark.parametrize("scope", [None, "own", "common"])
def test_fmb_common_family_uses_mature_eligible_rows_for_every_regression(
    tmp_path, scope
):
    date = pd.Timestamp("2020-01-02")
    x = np.arange(8, dtype=float)
    z = np.array([0, 1, 0, 1, 2, 2, 3, 4], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": date,
            "asset_id": list("abcdefgh"),
            "eligible": True,
            "x": x,
            "z": z,
            "label": 1 + 2 * x + 3 * z,
            "label_available_at": date,
            "total_market_cap": x + 1,
        }
    )
    frame.loc[4, "eligible"] = False
    frame.loc[5, "label_available_at"] = pd.Timestamp("2021-01-01")
    frame.loc[6, "total_market_cap"] = 0.0
    frame.loc[7, "z"] = np.nan
    source = write_evaluation_input(
        tmp_path / "input", frame, factors=("x", "z"), labels=("label",)
    )
    config = _configuration(
        source,
        tmp_path / "statistics",
        factors=("x", "z"),
        start="2020-01-02",
        end="2020-01-02",
        analyses=["fama_macbeth"],
    )
    if scope is not None:
        config["statistics"]["fama_macbeth_sample_scope"] = scope
    artifact = read_statistics_artifact(factor_statistics(config))
    assert artifact.metadata["statistics"]["fama_macbeth_sample_scope"] == (
        scope or "own"
    )
    with artifact.table("fama_macbeth_periods.parquet") as actual:
        if scope == "common":
            assert actual.observation_count.eq(4).all()
            assert actual.status.eq("DEFINED").all()
            for weighting in ("equal", "total_market_cap"):
                core = actual.loc[
                    actual.regression_id.eq(weighting + ":core_multifactor")
                ].set_index("coefficient_id")
                np.testing.assert_allclose(
                    core.loc[["intercept", "x", "z"], "value"],
                    [1, 2, 3],
                    rtol=1e-12,
                    atol=1e-12,
                )
            single = actual.loc[actual.regression_id.eq("equal:single:x")].set_index(
                "coefficient_id"
            )
            # On x=(0,1,2,3), z=(0,1,0,1), cov(x,z)/var(x)=1/5.
            assert single.loc["x", "value"] == pytest.approx(2.6)
            assert single.loc["intercept", "value"] == pytest.approx(1.6)
        else:
            counts = actual.groupby("regression_id").observation_count.first().to_dict()
            assert counts == {
                "equal:single:x": 6,
                "equal:single:z": 5,
                "equal:core_multifactor": 5,
                "total_market_cap:single:x": 5,
                "total_market_cap:single:z": 4,
                "total_market_cap:core_multifactor": 4,
            }


def _diagnostic_input(tmp_path):
    dates = pd.to_datetime(
        [
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2022-01-03",
        ]
    )
    labels = [
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [3, 0, 1, 2],
        [3, 2, 1, 0],
        [1, 0, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    frame = pd.DataFrame(
        [
            {
                "timestamp": date,
                "asset_id": str(asset),
                "eligible": True,
                "x": float(asset),
                "peer": float(asset * sign),
                "constant": 1.0,
                "label": float(labels[day][asset]),
                "label_available_at": date,
            }
            for day, (date, sign) in enumerate(
                zip(dates, [1, -1, 1, -1, 1, 1, -1], strict=True)
            )
            for asset in range(4)
        ]
    )
    frame.loc[0, "peer"] = np.nan
    # Missing a third factor also removes this row from the x/peer comparison.
    frame.loc[7, "constant"] = np.nan
    frame.loc[frame.timestamp.eq(dates[5]), "label_available_at"] = pd.Timestamp(
        "2023-01-01"
    )
    factors = ("x", "peer", "constant")
    source = write_evaluation_input(
        tmp_path / "input", frame, factors=factors, labels=("label",)
    )
    config = _configuration(
        source,
        tmp_path / "statistics",
        factors=factors,
        start="2020-01-02",
        end="2022-01-03",
        analyses=["annual_stability", "factor_correlation"],
    )
    return frame, source, config


def test_annual_and_correlation_diagnostics_keep_missing_clocks_and_signed_means(
    tmp_path,
):
    _, _, config = _diagnostic_input(tmp_path)
    artifact = read_statistics_artifact(factor_statistics(config))
    with artifact.table("factor_annual_diagnostics.parquet") as annual:
        x = annual.loc[annual.factor_id.eq("x") & annual.fdr_method.eq("BY")].set_index(
            "year"
        )
        assert x.loc[2020, "input_date_count"] == 3
        assert x.loc[2020, "observation_count"] == 3
        # Spearman's 1 - 6 sum(d^2) / (n(n^2)-n): 1, .8, -.2.
        assert x.loc[2020, "mean"] == pytest.approx(1.6 / 3)
        assert x.loc[2020, "status"] == "DEFINED"
        assert x.loc[2021, "observation_count"] == 2
        assert x.loc[2021, "input_date_count"] == 3
        assert x.loc[2021, "reason"] == "incomplete_session_clock"
        assert np.isnan(x.loc[2021, "p_value"])
        assert x.loc[2022, "observation_count"] == 1
        assert x.loc[2022, "status"] == "UNDEFINED"
        assert x.loc[2022, "reason"] == "hac_lag_exceeds_observations"
        assert x.loc[2022, "mean"] == pytest.approx(1.0)
        assert np.isnan(x.loc[2022, "p_value"])
        assert annual.planned_hypothesis_count.eq(3).all()
    with artifact.table("factor_correlation.parquet") as correlation:
        pair = correlation.loc[
            correlation.factor_id.eq("x") & correlation.comparison_factor_id.eq("peer")
        ]
        assert set(pair.statistic) == {"pearson", "spearman"}
        assert pair.input_date_count.eq(7).all()
        assert pair.observation_count.eq(6).all()
        assert pair.observation_pairs.eq(22).all()
        np.testing.assert_allclose(pair["mean"], 0.0, atol=1e-15)
        np.testing.assert_allclose(pair.mean_absolute, 1.0, atol=1e-15)
        constant = correlation.loc[correlation.comparison_factor_id.eq("constant")]
        assert constant.status.eq("UNDEFINED").all()
        assert constant.observation_count.eq(0).all()
        assert constant["mean"].isna().all()


def test_saved_evaluation_annual_summary_reuses_daily_facts(tmp_path, monkeypatch):
    _, source, config = _diagnostic_input(tmp_path)
    config["analyses"] = ["annual_stability"]
    native = read_statistics_artifact(factor_statistics(config))
    factors = tuple(config["evaluation"]["factors"])
    spec = EvaluationSpec(
        evaluation_id="annual",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=tuple(
            SignalSpec(signal_id=factor, source_kind="factor", source_column=factor)
            for factor in factors
        ),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        n_groups=2,
    )
    saved = evaluate(
        evaluation_config(
            source,
            tmp_path / "evaluation",
            spec,
            start=config["window"]["start"],
            end=config["window"]["end"],
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("saved annual inference must not reopen factor datasets")

    import quant_research_platform.research.factors.statistics as factor_module

    monkeypatch.setattr(factor_module, "iter_dataset_chunks", forbidden)
    published = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(saved)],
                "output": str(tmp_path / "saved-statistics"),
                "analyses": ["annual_stability"],
                "evaluation": {"factors": list(factors), "primary_label": "label"},
                "statistics": {
                    "hac_lag": 0,
                    "factor_fdr_families": {"declared": list(factors)},
                },
            }
        )
    )
    with native.table("factor_annual_diagnostics.parquet") as expected:
        with published.table("factor_annual_diagnostics.parquet") as actual:
            pd.testing.assert_frame_equal(
                actual, expected, check_exact=False, atol=1e-14
            )


def test_empty_diagnostic_window_and_fmb_empty_common_family_are_explicit(tmp_path):
    _, _, config = _diagnostic_input(tmp_path)
    config["window"] = {"start": "2019-01-01", "end": "2019-12-31"}
    artifact = read_statistics_artifact(factor_statistics(config))
    with artifact.table("factor_annual_diagnostics.parquet") as annual:
        assert annual.empty
    with artifact.table("factor_correlation.parquet") as correlation:
        assert correlation.input_date_count.eq(0).all()
        assert correlation.status.eq("UNDEFINED").all()

    from quant_research_platform.research.factors.statistics import (
        _fama_macbeth_period_records,
    )

    rows = _fama_macbeth_period_records(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-02")],
                "asset_id": ["a"],
                "x": [1.0],
                "label": [0.1],
                "total_market_cap": [np.nan],
            }
        ),
        return_column="label",
        factors=("x",),
        core_factors=(),
        sample_scope="common",
    )
    assert all(
        row["observation_count"] == 0 and row["status"] == "UNDEFINED" for row in rows
    )


@pytest.mark.parametrize("scope", ["pairwise", None, 1, []])
def test_invalid_fmb_sample_scope_is_rejected_before_publication(tmp_path, scope):
    _, _, config = _diagnostic_input(tmp_path)
    config["analyses"] = ["fama_macbeth"]
    config["statistics"]["fama_macbeth_sample_scope"] = scope
    with pytest.raises(ValueError, match="factor_statistics_policy_invalid"):
        factor_statistics(config)
    assert not (tmp_path / "statistics").exists()


def test_correlation_only_does_not_require_unused_inference_policy(tmp_path):
    _, _, config = _diagnostic_input(tmp_path)
    config["analyses"] = ["factor_correlation"]
    config["statistics"] = {}
    artifact = read_statistics_artifact(factor_statistics(config))
    with artifact.table("factor_correlation.parquet") as frame:
        assert len(frame) == 6
    with pytest.raises(ValueError, match="statistics_table_member_invalid"):
        with artifact.table("factor_diagnostics.parquet"):
            pass
