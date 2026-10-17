"""Paired statistics bind actual group and consumed scientific target facts."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
from quant_research_platform.research import statistics_production
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.statistics_production import (
    _paired_group_rankic,
    model_statistics,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.research.test_optional_statistics import _cash_ledger
from tests.workflows.test_portfolio_workflow_integration import (
    _statistics_request,
)
from tests.workflows.test_portfolio_workflow_integration import (
    native_comparison as native_comparison,
)


@pytest.fixture(scope="module")
def paired_sources(native_comparison, tmp_path_factory):
    case = native_comparison
    root = tmp_path_factory.mktemp("paired-targets")
    configs, paths = {}, {}
    for group in ("global", "quarterly"):
        config = _statistics_request(
            case,
            root / group,
            group=group,
            end="2019-02-15",
            global_statistics=paths.get("global"),
        )
        configs[group] = config
        paths[group] = model_statistics(config)
    return case, configs, paths


def _losses(path):
    with read_statistics_artifact(path).table("model_losses.parquet") as frame:
        return frame.copy()


def test_native_same_models_predictions_and_unknown_units_remain_legal(paired_sources):
    _case, configs, paths = paired_sources
    assert (
        configs["global"]["periods"][0]["predictions"]
        == configs["quarterly"]["periods"][0]["predictions"]
    )
    global_losses, quarterly_losses = (
        _losses(paths[group]) for group in ("global", "quarterly")
    )
    pd.testing.assert_series_equal(
        global_losses.asset_coverage_id, quarterly_losses.asset_coverage_id
    )
    metadata = read_statistics_artifact(paths["global"]).metadata
    assert metadata["column_units"] == {}
    assert metadata["evaluation_axis"] == "cross_sectional"
    assert metadata["column_roles"]["forward_return_20d"] == "label"
    assert metadata["evaluation"]["label_availability_columns"] == {
        "forward_return_20d": "forward_return_20d_available_at"
    }
    assert pd.Timestamp(metadata["label_cutoff"]) == pd.Timestamp(
        "2019-02-15 23:59:59.999999999"
    )
    undefined = global_losses.loc[global_losses.status.eq("UNDEFINED")]
    assert not undefined.empty
    assert undefined.asset_coverage_id.eq(canonical_json_hash([])).all()
    with read_statistics_artifact(paths["quarterly"]).table(
        "model_group_comparison.parquet"
    ) as rows:
        assert len(rows) == 2
        assert rows.mean_difference.eq(0).all()


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


@pytest.mark.parametrize("source", ["quarterly", "direct"])
def test_native_global_slot_requires_actual_global_selection(
    paired_sources, tmp_path, source
):
    case, configs, paths = paired_sources
    left = paths["quarterly"]
    if source == "direct":
        direct = deepcopy(configs["global"])
        direct["family_ids"] = list(case.models)
        direct["periods"][0].pop("selection")
        direct["output"] = str(tmp_path / "direct")
        left = model_statistics(direct)
    config = {
        **configs["quarterly"],
        "global_statistics": str(left),
        "output": str(tmp_path / "rejected"),
    }
    with pytest.raises(
        ValueError, match="model_group_comparison_target_metadata_invalid"
    ):
        model_statistics(config)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mutation", ["primary_label", "values", "maturity"])
def test_native_declared_target_and_key_domain_control_comparison(
    paired_sources, tmp_path, mutation
):
    case, configs, paths = paired_sources
    frame, spec = case.frame.copy(), case.spec
    config = deepcopy(configs["global"])
    if mutation == "primary_label":
        frame["inverse_label"] = -frame.forward_return_20d
        frame["inverse_label_available_at"] = frame.forward_return_20d_available_at
        spec = replace(
            spec,
            label_columns=(*spec.label_columns, "inverse_label"),
            context_columns=(*spec.context_columns, "inverse_label_available_at"),
            column_roles={
                **spec.column_roles,
                "inverse_label": "label",
                "inverse_label_available_at": "context",
            },
            label_availability_columns={"inverse_label": "inverse_label_available_at"},
        )
        config["evaluation"]["primary_label"] = "inverse_label"
    elif mutation == "values":
        frame["forward_return_20d"] *= -1
    else:
        frame.loc[
            frame.timestamp.eq(pd.Timestamp("2019-01-02")) & frame.asset_id.eq("a0"),
            "forward_return_20d_available_at",
        ] = pd.Timestamp("2020-01-01")
    write_dataset(tmp_path / "different-target", frame, spec=spec)
    config.update(
        dataset=str(tmp_path / "different-target"), output=str(tmp_path / "left")
    )
    left = model_statistics(config)
    right = {
        **configs["quarterly"],
        "global_statistics": str(left),
        "output": str(tmp_path / "rejected"),
    }
    if mutation == "values":
        compared = model_statistics(right)
        with read_statistics_artifact(compared).table(
            "model_group_comparison.parquet"
        ) as result:
            assert result.observation_count.gt(0).all()
            assert result.mean_difference.ne(0).any()
    else:
        with pytest.raises(
            ValueError,
            match="model_group_comparison_(target_metadata|coverage)_invalid",
        ):
            model_statistics(right)
        assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "variation",
    [
        "unconsumed_factor",
        "maturity_names_and_cutoff",
        "unknown_to_known_unit",
        "future_values",
    ],
)
def test_native_equal_consumed_targets_do_not_bind_unrelated_inputs(
    paired_sources, tmp_path, variation
):
    case, configs, paths = paired_sources
    frame = case.frame.copy()
    spec = replace(case.spec, data_id="other_explicit_dataset")
    config = deepcopy(configs["quarterly"])
    if variation == "unconsumed_factor":
        frame["unused_factor"] = np.arange(len(frame), dtype=float)
        spec = replace(
            spec,
            factor_columns=(*spec.factor_columns, "unused_factor"),
            column_roles={**spec.column_roles, "unused_factor": "factor"},
        )
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
    else:
        frame.loc[
            frame.forward_return_20d_available_at.gt(pd.Timestamp("2019-02-15")),
            "forward_return_20d",
        ] *= -9
    write_dataset(tmp_path / "different-input", frame, spec=spec)
    config.update(
        dataset=str(tmp_path / "different-input"), output=str(tmp_path / "right")
    )
    actual = model_statistics(config)
    pd.testing.assert_series_equal(
        _losses(actual).asset_coverage_id, _losses(paths["global"]).asset_coverage_id
    )
    with read_statistics_artifact(actual).table(
        "model_group_comparison.parquet"
    ) as rows:
        assert rows.mean_difference.eq(0).all()


def test_native_explicit_unit_conflict_rejects_pair(paired_sources, tmp_path):
    case, configs, _paths = paired_sources
    left = None
    for group, unit in (("global", "decimal_return"), ("quarterly", "basis_points")):
        dataset = tmp_path / f"{group}-input"
        write_dataset(
            dataset,
            case.frame,
            spec=replace(case.spec, column_units={"forward_return_20d": unit}),
        )
        config = {
            **configs[group],
            "dataset": str(dataset),
            "output": str(tmp_path / group),
        }
        if group == "global":
            left = model_statistics(config)
        else:
            config["global_statistics"] = str(left)
            with pytest.raises(
                ValueError, match="model_group_comparison_target_metadata_invalid"
            ):
                model_statistics(config)


@pytest.mark.parametrize("variation", ["signed_zero", "one_ulp"])
def test_native_comparison_accepts_recomputed_float_targets(
    paired_sources, tmp_path, variation
):
    case, configs, _paths = paired_sources
    left_frame = case.frame.copy()
    changed = left_frame.timestamp.eq(
        pd.Timestamp("2019-01-02")
    ) & left_frame.asset_id.eq("a0")
    left_frame.loc[changed, "forward_return_20d"] = (
        0.0 if variation == "signed_zero" else 0.1
    )
    right_frame = left_frame.copy()
    right_frame.loc[changed, "forward_return_20d"] = (
        -0.0 if variation == "signed_zero" else np.nextafter(0.1, 1.0)
    )
    left = None
    for group, frame in (("global", left_frame), ("quarterly", right_frame)):
        dataset = tmp_path / f"{group}-input"
        write_dataset(dataset, frame, spec=case.spec)
        config = {
            **configs[group],
            "dataset": str(dataset),
            "output": str(tmp_path / group),
        }
        if group == "global":
            left = model_statistics(config)
        else:
            config["global_statistics"] = str(left)
            right = model_statistics(config)
            pd.testing.assert_series_equal(
                _losses(left).asset_coverage_id, _losses(right).asset_coverage_id
            )
            assert "target_domain_id" not in _losses(right)
            with read_statistics_artifact(right).table(
                "model_group_comparison.parquet"
            ) as result:
                assert result.observation_count.gt(0).all()


@pytest.mark.parametrize(
    "mutation",
    [
        "selection_group",
        "coverage_basis",
        "primary_label",
        "label_role",
        "evaluation_axis",
        "sample_role",
        "window",
        "unit",
    ],
)
def test_empty_native_losses_cannot_bypass_target_metadata(
    paired_sources, tmp_path, mutation
):
    _case, configs, paths = paired_sources
    original = read_statistics_artifact(paths["global"])
    metadata = deepcopy(dict(original.metadata))
    target = deepcopy(dict(read_statistics_artifact(paths["quarterly"]).metadata))
    if mutation == "primary_label":
        metadata["evaluation"]["primary_label"] = "inverse_label"
    elif mutation == "label_role":
        metadata["column_roles"]["forward_return_20d"] = "factor"
    elif mutation == "window":
        metadata["window"]["start"] = "2019-01-01"
    elif mutation == "unit":
        metadata["column_units"]["forward_return_20d"] = "basis_points"
        target["column_units"]["forward_return_20d"] = "decimal_return"
    else:
        metadata[mutation] = {
            "selection_group": "quarterly",
            "coverage_basis": "persisted_prediction_keys",
            "evaluation_axis": "time_series",
            "sample_role": "test",
        }[mutation]
    tables = {}
    for member in original._members:
        with original.table(member) as table:
            tables[member.removesuffix(".parquet")] = table.copy()
            if "sample_role" in table:
                tables[member.removesuffix(".parquet")]["sample_role"] = metadata[
                    "sample_role"
                ]
    tables["model_losses"] = tables["model_losses"].iloc[:0].copy()
    left = write_statistics_artifact(
        tmp_path / "empty", metadata=metadata, tables=tables
    )
    with pytest.raises(
        ValueError, match="model_group_comparison_target_metadata_invalid"
    ):
        _paired_group_rankic(
            tables["model_losses"],
            global_statistics=read_statistics_artifact(left),
            selection_group="quarterly",
            family_ids=("linear_regression", "equal_weight_rank_ensemble"),
            config=configs["quarterly"],
            target_metadata=target,
        )


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
def test_native_paired_size_basis_matches_only_when_consumed(tmp_path, neutrality):
    config = _portfolio_pair(
        tmp_path,
        base_changes={"neutrality": neutrality},
        change={"size_cap_column": "float_market_cap"},
    )
    if neutrality in {"size", "industry_size"}:
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
    from contextlib import nullcontext
    from statistics import NormalDist
    from types import SimpleNamespace

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
            left.assign(loss=-values),
            global_statistics=SimpleNamespace(
                metadata=metadata, table=lambda member: nullcontext(left)
            ),
            selection_group="quarterly",
            family_ids=("one",),
            config=config,
            target_metadata={**metadata, "selection_group": "quarterly"},
        )
    else:
        declaration.update(portfolio_ids=["one"], primary_portfolio_id="one")
        result = _group_portfolio_comparison(
            pd.DataFrame(
                {"global_one": np.zeros(len(values)), "quarterly_one": values},
                index=dates.rename("end_at"),
            ),
            config,
        )
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
