"""G/Q comparisons reevaluate explicit predictions on one chosen target stream."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
from quant_research_platform.research import statistics_production
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.statistics_production import (
    _paired_group_rankic,
    model_statistics,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.research.test_optional_statistics import _cash_ledger
from tests.workflows.test_portfolio_workflow_integration import _statistics_request
from tests.workflows.test_portfolio_workflow_integration import (
    native_comparison as native_comparison,
)


@pytest.fixture(scope="module")
def paired_sources(native_comparison, tmp_path_factory):
    case = native_comparison
    root = tmp_path_factory.mktemp("paired-targets")
    configs, paths = {}, {}
    for group in ("global", "quarterly"):
        configs[group] = _statistics_request(
            case, root / group, group=group, end="2019-02-15"
        )
        paths[group] = model_statistics(configs[group])
    return case, configs, paths


def _losses(path):
    with read_statistics_artifact(path).table("model_losses.parquet") as frame:
        return frame.copy()


def test_native_common_target_records_its_actual_owner(paired_sources):
    _case, _configs, paths = paired_sources
    artifact = read_statistics_artifact(paths["quarterly"])
    assert artifact.metadata["group_comparison_target"] == "explicit_current_evaluation"
    assert "global_statistics" not in artifact.metadata
    with artifact.table("model_group_comparison.parquet") as rows:
        assert len(rows) == 2
        assert rows.mean_difference.eq(0).all()
    assert not _losses(paths["global"]).empty


def test_native_comparison_reuses_the_single_dataset_scan(
    paired_sources, tmp_path, monkeypatch
):
    _case, configs, _paths = paired_sources
    actual_chunks = statistics_production.iter_dataset_chunks
    scans = []

    def counted_chunks(*args, **kwargs):
        scans.append(1)
        yield from actual_chunks(*args, **kwargs)

    monkeypatch.setattr(statistics_production, "iter_dataset_chunks", counted_chunks)
    model_statistics({**configs["quarterly"], "output": str(tmp_path / "right")})
    assert len(scans) == 1


@pytest.mark.parametrize(
    "variation",
    [
        "values",
        "maturity",
        "primary_label",
        "unconsumed_factor",
        "maturity_names_and_cutoff",
        "unknown_to_known_unit",
        "future_values",
        "signed_zero",
        "one_ulp",
    ],
)
def test_native_both_predictions_use_the_explicit_current_target(
    paired_sources, tmp_path, variation
):
    case, configs, paths = paired_sources
    frame, spec = case.frame.copy(), case.spec
    config = deepcopy(configs["quarterly"])
    if variation == "values":
        frame["forward_return_20d"] *= -1
    elif variation == "maturity":
        frame.loc[frame.asset_id.eq("a0"), "forward_return_20d_available_at"] = (
            pd.Timestamp("2021-01-01")
        )
    elif variation == "primary_label":
        frame["inverse_label"] = -frame.forward_return_20d
        spec = replace(
            spec,
            label_columns=(*spec.label_columns, "inverse_label"),
            column_roles={**spec.column_roles, "inverse_label": "label"},
            label_availability_columns={
                "inverse_label": "forward_return_20d_available_at"
            },
        )
        config["evaluation"]["primary_label"] = "inverse_label"
    elif variation == "unconsumed_factor":
        frame[case.factor] = np.nan
    elif variation == "maturity_names_and_cutoff":
        frame["actual_maturity"] = frame.forward_return_20d_available_at
        spec = replace(
            spec,
            context_columns=(*spec.context_columns, "actual_maturity"),
            column_roles={**spec.column_roles, "actual_maturity": "context"},
            label_availability_columns={"forward_return_20d": "actual_maturity"},
        )
        config["label_cutoff"] = "2019-02-16"
    elif variation == "unknown_to_known_unit":
        spec = replace(spec, column_units={"forward_return_20d": "decimal_return"})
    elif variation == "future_values":
        frame.loc[
            frame.forward_return_20d_available_at.gt(pd.Timestamp("2019-02-15")),
            "forward_return_20d",
        ] *= -9
    else:
        frame.loc[frame.asset_id.eq("a0"), "forward_return_20d"] = (
            -0.0 if variation == "signed_zero" else np.nextafter(0.1, 1.0)
        )
    write_dataset(tmp_path / "chosen-target", frame, spec=spec)
    config.update(
        dataset=str(tmp_path / "chosen-target"), output=str(tmp_path / "right")
    )
    original_losses = _losses(paths["global"])
    actual = model_statistics(config)
    with read_statistics_artifact(actual).table(
        "model_group_comparison.parquet"
    ) as rows:
        assert rows.observation_count.gt(0).all()
        assert rows.mean_difference.eq(0).all()
    # The prior independent statistic is neither an input nor rewritten.
    pd.testing.assert_frame_equal(_losses(paths["global"]), original_losses)
    assert not _losses(actual).empty


def test_native_reference_missing_family_only_undefines_its_pair(
    paired_sources, tmp_path
):
    case, configs, paths = paired_sources
    config = deepcopy(configs["quarterly"])
    config["global_predictions"] = config["global_predictions"][:1]
    config["output"] = str(tmp_path / "partial-reference")
    actual = model_statistics(config)
    pd.testing.assert_frame_equal(_losses(actual), _losses(paths["quarterly"]))
    with read_statistics_artifact(actual).table(
        "model_group_comparison.parquet"
    ) as rows:
        assert len(rows) == len(case.models)
        assert sorted(rows.observation_count.gt(0)) == [False, True]
        absent = rows.loc[rows.observation_count.eq(0)]
        assert absent.reason.eq("no_common_observations").all()
        assert absent.bootstrap_reason.eq("no_common_observations").all()


def test_native_reference_prediction_keys_must_be_unique(paired_sources, tmp_path):
    _case, configs, _paths = paired_sources
    config = deepcopy(configs["quarterly"])
    config["global_predictions"] *= 2
    config["output"] = str(tmp_path / "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        model_statistics(config)


@pytest.mark.parametrize("hac_lag,block", [(20, 1), (0, 4)])
def test_model_paired_inference_methods_are_independent(hac_lag, block):
    values = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4),
            "family_id": "one",
            "global": [0.1, 0.2, 0.3, 0.4],
            "quarterly": [0.3, 0.1, 0.5, 0.2],
        }
    )
    config = {
        "sample_role": "exploratory",
        "statistics": {
            "hac_lag": hac_lag,
            "bootstrap_block": block,
            "bootstrap_repetitions": 99,
            "group_comparison": {
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }
    row = _paired_group_rankic(
        values, selection_group="quarterly", family_ids=("one",), config=config
    ).iloc[0]
    assert row.status == ("UNDEFINED" if hac_lag == 20 else "DEFINED")
    assert row.bootstrap_status == ("DEFINED" if block == 1 else "UNDEFINED")
    values.loc[1, "global"] = np.nan
    missing = _paired_group_rankic(
        values, selection_group="quarterly", family_ids=("one",), config=config
    ).iloc[0]
    assert missing.status == missing.bootstrap_status == "UNDEFINED"
    assert missing.reason == missing.bootstrap_reason == "incomplete_session_clock"


def _portfolio_pair(tmp_path, *, change=None, base_changes=None):
    base = PortfolioCandidateSpec(
        candidate_id="global_cash",
        selection_group="global",
        construction="quantile_equal_weight",
        prediction_selector="same_predictions",
        rebalance_schedule="daily",
        top_quantile=0.2,
        benchmark_id=None,
    )
    base = replace(base, **(base_changes or {}))
    other = replace(
        base,
        **{
            "candidate_id": "quarterly_cash",
            "selection_group": "quarterly",
            **(change or {}),
        },
    )
    ledgers = []
    for candidate in (base, other):
        path = tmp_path / candidate.candidate_id
        window = _cash_ledger(path, candidate=candidate)
        ledgers.append(str(path))
    return {
        "ledgers": ledgers,
        "window": window,
        "output": str(tmp_path / "statistics"),
        "analyses": ["group_comparison"],
        "statistics": {
            "hac_lag": 0,
            "bootstrap_block": 1,
            "bootstrap_repetitions": 10,
            "group_comparison": {
                "portfolio_ids": ["cash"],
                "primary_portfolio_id": "cash",
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }


def test_native_identical_cash_paths_are_legal_group_comparison(tmp_path):
    output = portfolio_statistics(_portfolio_pair(tmp_path))
    with read_statistics_artifact(output).table(
        "portfolio_group_comparison.parquet"
    ) as rows:
        assert rows.global_mean.eq(0).all()
        assert rows.quarterly_mean.eq(0).all()
        assert rows.mean_difference.eq(0).all()
        for group in ("global", "quarterly"):
            assert rows[f"{group}_terminal_wealth"].eq(1.0).all()
            assert rows[f"{group}_terminal_wealth_reason"].eq("").all()


def test_native_paired_portfolio_benchmarks_must_match(tmp_path):
    config = _portfolio_pair(
        tmp_path,
        base_changes={
            "construction": "benchmark_enhanced",
            "top_quantile": None,
            "benchmark_id": "csi500",
            "risk_style_columns": ("size",),
        },
        change={"benchmark_id": "csi1000"},
    )
    with pytest.raises(
        ValueError, match="portfolio_group_comparison_strategy_mismatch"
    ):
        portfolio_statistics(config)


@pytest.mark.parametrize(
    "change",
    [
        {"selection_group": "global"},
        {"construction": "quantile_equal_weight_half_step"},
        {"rebalance_schedule": "weekly-last-trading-day"},
        {"top_quantile": 0.4},
        {"neutrality": "industry"},
        {"risk_style_columns": ("size",)},
    ],
)
def test_native_portfolio_slot_and_fixed_strategy_are_bound(tmp_path, change):
    config = _portfolio_pair(tmp_path, change=change)
    with pytest.raises(
        ValueError,
        match="portfolio_group_comparison_(selection_group_invalid|strategy_mismatch)",
    ):
        portfolio_statistics(config)
    assert not (tmp_path / "statistics").exists()
    # The group-specific invariant does not restrict independent performance analyses.
    config["analyses"] = ["performance"]
    portfolio_statistics(config)


@pytest.mark.parametrize("neutrality", ["none", "industry", "size", "industry_size"])
@pytest.mark.parametrize(
    "field,value,modes",
    [
        ("size_cap_column", "float_market_cap", {"size", "industry_size"}),
        ("size_neutrality_limit", 0.9, {"size", "industry_size"}),
        ("industry_neutrality_limit", 0.9, {"industry", "industry_size"}),
    ],
)
def test_native_paired_neutrality_matches_only_when_consumed(
    tmp_path, neutrality, field, value, modes
):
    config = _portfolio_pair(
        tmp_path,
        base_changes={"neutrality": neutrality},
        change={field: value},
    )
    if neutrality in modes:
        with pytest.raises(
            ValueError, match="portfolio_group_comparison_strategy_mismatch"
        ):
            portfolio_statistics(config)
        assert not (tmp_path / "statistics").exists()
        config["analyses"] = ["performance"]
    output = portfolio_statistics(config)
    assert output.is_dir()


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("kind", ["model", "portfolio"])
def test_paired_confidence_is_used_by_hac_and_persisted(tmp_path, confidence, kind):
    from statistics import NormalDist

    from quant_research_platform.research.portfolios.diagnostics import (
        _group_portfolio_comparison,
    )

    values = np.array([0.02, -0.01, 0.04, 0.01, 0.03, -0.02, 0.05, 0.01])
    dates = pd.date_range("2024-01-01", periods=len(values))
    metadata = {
        "selection_group": "global",
        "coverage_basis": "model_prediction_domain",
        "sample_role": "test",
        "window": {"start": str(dates[0]), "end": str(dates[-1])},
        "evaluation": {"primary_label": "label"},
        "evaluation_axis": "cross_sectional",
        "column_roles": {"label": "label"},
        "column_units": {},
    }
    declaration = {
        "fdr_method": "BY",
        "alpha": 0.05,
        "seed": 42,
        "confidence": confidence,
    }
    config = {
        "sample_role": "test",
        "statistics": {
            "hac_lag": 1,
            "bootstrap_block": 2,
            "bootstrap_repetitions": 20,
            "group_comparison": declaration,
        },
    }
    if kind == "model":
        left = pd.DataFrame(
            {
                "timestamp": dates,
                "family_id": "one",
                "sample_role": "test",
                "status": "DEFINED",
                "asset_coverage_id": "a" * 64,
                "observation_count": 100,
                "loss": 0.0,
            }
        )
        result = _paired_group_rankic(
            left.assign(**{"global": 0.0, "quarterly": values}),
            selection_group="quarterly",
            family_ids=("one",),
            config=config,
        )
    else:
        declaration.update(portfolio_ids=["one"], primary_portfolio_id="one")
        result = _group_portfolio_comparison(
            pd.DataFrame(
                {"global_one": np.zeros(len(values)), "quarterly_one": values},
                index=dates.rename("end_at"),
            ),
            config,
            daily_complete=True,
        )
        # The inference helper has no ledger endpoints. Supply the known
        # fixture endpoint wealth before exercising the statistics writer.
        result["global_terminal_wealth"] = 1.0
        result["quarterly_terminal_wealth"] = float(np.prod(1 + values))
        result["global_terminal_wealth_reason"] = ""
        result["quarterly_terminal_wealth_reason"] = ""
    centered = values - values.mean()
    standard_error = np.sqrt(centered @ centered + centered[1:] @ centered[:-1]) / len(
        values
    )
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    member = f"{kind}_group_comparison"
    path = write_statistics_artifact(
        tmp_path / "comparison",
        metadata={**metadata, "comparison_portfolio_ids": ["one"]},
        tables={member: result},
        table_group="portfolio" if kind == "portfolio" else "research",
    )
    with read_statistics_artifact(path).table(f"{member}.parquet") as saved:
        row = saved.iloc[0]
        assert row.confidence == confidence
        assert row.standard_error == pytest.approx(standard_error, rel=1e-14)
        np.testing.assert_allclose(
            [row.ci_lower, row.ci_upper],
            values.mean() + np.array([-1, 1]) * critical * standard_error,
            rtol=1e-14,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hac_lag", -1),
        ("hac_lag", True),
        ("hac_lag", 0.5),
        ("bootstrap_block", 0),
        ("bootstrap_repetitions", 1),
        ("seed", -1),
        ("seed", True),
        ("confidence", 1.0),
        ("alpha", -0.05),
        ("fdr_method", "invalid"),
    ],
)
def test_undefined_cash_comparison_still_rejects_invalid_configuration(
    tmp_path, field, value
):
    config = _portfolio_pair(tmp_path)
    policy = config["statistics"]
    (policy if field in policy else policy["group_comparison"])[field] = value
    with pytest.raises(
        ValueError, match="(comparison_policy|bootstrap_configuration)_invalid"
    ):
        portfolio_statistics(config)
    assert not (tmp_path / "statistics").exists()


@pytest.mark.parametrize(
    "hac_lag,block,status,bootstrap_status",
    [(20, 1, "UNDEFINED", "DEFINED"), (0, 4, "DEFINED", "UNDEFINED")],
)
def test_portfolio_group_inference_methods_have_independent_preconditions(
    hac_lag, block, status, bootstrap_status
):
    from quant_research_platform.research.portfolios.diagnostics import (
        _group_portfolio_comparison,
    )
    from quant_research_platform.research.statistics import stationary_bootstrap_mean_ci

    index = pd.MultiIndex.from_arrays(
        [
            pd.date_range("2024-01-01", periods=4),
            pd.date_range("2024-01-02", periods=4),
        ],
        names=["entry_at", "end_at"],
    )
    matrix = pd.DataFrame(
        {
            "global_p": [0.001, 0.002, 0.003, 0.004],
            "quarterly_p": [0.003, 0.001, 0.005, 0.002],
        },
        index=index,
    )
    config = {
        "sample_role": "exploratory",
        "statistics": {
            "hac_lag": hac_lag,
            "bootstrap_block": block,
            "bootstrap_repetitions": 99,
            "group_comparison": {
                "portfolio_ids": ["p"],
                "primary_portfolio_id": "p",
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }
    row = _group_portfolio_comparison(matrix, config, daily_complete=True).iloc[0]
    unavailable = _group_portfolio_comparison(
        matrix, config, daily_complete=False
    ).iloc[0]
    assert unavailable.mean_difference == row.mean_difference
    assert unavailable.reason == unavailable.bootstrap_reason == "incomplete_daily_axis"
    assert pd.isna(unavailable.t_statistic) and pd.isna(unavailable.bootstrap_lower)
    assert row.status == status
    assert row.bootstrap_status == bootstrap_status
    if bootstrap_status == "DEFINED":
        expected = stationary_bootstrap_mean_ci(
            matrix.quarterly_p - matrix.global_p,
            block_size=block,
            repetitions=99,
            seed=42,
            confidence=0.95,
        )
        assert row.bootstrap_lower == pytest.approx(expected.lower)
        assert row.bootstrap_upper == pytest.approx(expected.upper)
        assert row.reason == "hac_lag_exceeds_observations"
    else:
        assert row.bootstrap_reason == "bootstrap_block_exceeds_observations"


@pytest.mark.parametrize("split", [False, True])
def test_native_reference_scores_are_recomputed_on_common_label_keys(
    paired_sources, tmp_path, split
):
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )

    case, configs, _paths = paired_sources
    config = deepcopy(configs["quarterly"])
    references = []
    expected = {}
    for family, path in zip(case.models, config["global_predictions"], strict=True):
        frame, metadata = read_predictions(path)
        metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"prediction_rows", "effective_prediction_end"}
        }
        original = frame.copy()
        frame["prediction_score"] *= -1
        frame["prediction_rank_score"] *= -1
        # The shared inference must use the intersection, not each side's own IC.
        frame = frame.loc[frame.asset_id.ne("a0")].copy()
        partitions = [frame]
        if split:
            midpoint = sorted(frame.timestamp.unique())[
                len(frame.timestamp.unique()) // 2
            ]
            partitions = [
                frame.loc[frame.timestamp.lt(midpoint)],
                frame.loc[frame.timestamp.ge(midpoint)],
            ]
        for index, partition in enumerate(partitions):
            reference = write_prediction_outputs(
                output_path=tmp_path / f"{family}-{index}",
                output_chunks=[partition],
                metadata=metadata,
            )
            references.append(str(reference))
        eligible = case.frame.loc[
            case.frame.forward_return_20d_available_at.le(
                pd.Timestamp("2019-02-15 23:59:59.999999999")
            )
        ]
        joined = frame.merge(
            original, on=["timestamp", "asset_id"], suffixes=("_g", "_q")
        ).merge(eligible, on=["timestamp", "asset_id"])
        differences = [
            group.prediction_score_q.corr(group.forward_return_20d, method="spearman")
            - group.prediction_score_g.corr(group.forward_return_20d, method="spearman")
            for _, group in joined.groupby("timestamp")
        ]
        expected[family] = float(np.mean(differences))
    config.update(global_predictions=references, output=str(tmp_path / "paired"))
    artifact = read_statistics_artifact(model_statistics(config))
    with artifact.table("model_group_comparison.parquet") as rows:
        for row in rows.itertuples():
            assert row.mean_difference == pytest.approx(expected[row.family_id])
            assert row.mean_difference > 1.5


def test_unknown_reference_chronology_only_blocks_its_family():
    dates = pd.date_range("2024-01-01", periods=4)
    values = pd.DataFrame(
        {
            "timestamp": np.tile(dates, 2),
            "family_id": np.repeat(["known", "unknown"], 4),
            "global": 0.0,
            "quarterly": [0.1, 0.2, -0.1, 0.3] * 2,
        }
    )
    config = {
        "sample_role": "test",
        "statistics": {
            "hac_lag": 0,
            "bootstrap_block": 1,
            "bootstrap_repetitions": 20,
            "group_comparison": {
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }
    result = _paired_group_rankic(
        values,
        selection_group="quarterly",
        family_ids=("known", "unknown"),
        config=config,
        inference_blocker={"unknown": "selection_chronology_not_established"},
    ).set_index("family_id")
    assert (
        result.loc["known", "status"]
        == result.loc["known", "bootstrap_status"]
        == "DEFINED"
    )
    assert (
        result.loc["unknown", "status"]
        == result.loc["unknown", "bootstrap_status"]
        == "UNDEFINED"
    )
    assert result.loc["unknown", "reason"] == "selection_chronology_not_established"
    assert result.mean_difference.tolist() == pytest.approx([0.125, 0.125])
    assert result.fdr_family_size.eq(2).all()


def test_direct_and_selection_bearing_pairs_share_prediction_consumption(
    paired_sources, tmp_path
):
    case, configs, paths = paired_sources
    config = deepcopy(configs["quarterly"])
    config["family_ids"] = list(case.models)
    config["periods"][0].pop("selection")
    config["output"] = str(tmp_path / "direct")
    actual = model_statistics(config)
    for member in ("model_losses.parquet", "model_group_comparison.parquet"):
        with (
            read_statistics_artifact(paths["quarterly"]).table(member) as expected,
            read_statistics_artifact(actual).table(member) as observed,
        ):
            pd.testing.assert_frame_equal(observed, expected)


@pytest.mark.parametrize("unknown_side", ["current", "reference"])
def test_native_unknown_chronology_is_local_on_either_side(
    paired_sources, tmp_path, unknown_side
):
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )

    case, configs, paths = paired_sources
    config = deepcopy(configs["quarterly"])
    families = list(case.models)
    groups = {}
    for side in ("current", "reference"):
        groups[side] = []
        for family in families:
            frame, metadata = read_predictions(case.predictions["global", family])
            metadata = deepcopy(metadata)
            if side != unknown_side or family != families[0]:
                metadata["model_facts"]["selection_cutoff"] = "2018-10-30"
            if side == "reference":
                sign = frame.timestamp.dt.day.mod(2).mul(2).sub(1)
                frame["prediction_score"] *= sign
                frame["prediction_rank_score"] *= sign
            path = write_prediction_outputs(
                output_path=tmp_path / f"{side}-{family}",
                output_chunks=[frame],
                metadata=metadata,
            )
            groups[side].append(str(path))
    config["periods"][0].pop("selection")
    config["periods"][0]["predictions"] = groups["current"]
    config.update(
        global_predictions=groups["reference"],
        family_ids=families,
        label_cutoff="2019-04-01",
        analyses=["model_losses"],
        output=str(tmp_path / "paired"),
    )
    config["statistics"].update(hac_lag=0, bootstrap_block=1)
    actual = read_statistics_artifact(model_statistics(config))
    with actual.table("model_group_comparison.parquet") as rows:
        rows = rows.set_index("family_id")
        assert (
            rows.loc[families[0], "status"]
            == rows.loc[families[0], "bootstrap_status"]
            == "UNDEFINED"
        )
        assert (
            rows.loc[families[0], "reason"]
            == rows.loc[families[0], "bootstrap_reason"]
            == "selection_chronology_not_established"
        )
        assert (
            rows.loc[families[1], "status"]
            == rows.loc[families[1], "bootstrap_status"]
            == "DEFINED"
        )
        assert rows.mean_difference.notna().all()
        assert rows.fdr_family_size.eq(2).all()
    with actual.table("model_losses.parquet") as losses:
        assert losses.status.eq("DEFINED").all()


@pytest.mark.parametrize("method", ["hac", "bootstrap"])
def test_model_pair_writer_rejects_incomplete_clock_for_either_method(tmp_path, method):
    config = {
        "sample_role": "test",
        "statistics": {
            "hac_lag": 20 if method == "bootstrap" else 0,
            "bootstrap_block": 1 if method == "bootstrap" else 4,
            "bootstrap_repetitions": 20,
            "group_comparison": {
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }
    values = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4),
            "family_id": "one",
            "global": 0.0,
            "quarterly": [0.1, 0.2, -0.1, 0.3],
        }
    )
    rows = _paired_group_rankic(
        values, selection_group="quarterly", family_ids=("one",), config=config
    )
    assert (
        rows.iloc[0]["bootstrap_status" if method == "bootstrap" else "status"]
        == "DEFINED"
    )
    rows["calendar_count"] = 5
    with pytest.raises(ValueError, match="statistics_group_comparison_count_invalid"):
        write_statistics_artifact(
            tmp_path / "invalid",
            metadata={
                "sample_role": "test",
                "window": {"start": "2024-01-01", "end": "2024-01-05"},
            },
            tables={"model_group_comparison": rows},
        )
