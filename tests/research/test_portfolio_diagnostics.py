from copy import deepcopy

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.a_share import (
    portfolio_sessions as market_sessions,
)
from quant_research_platform.research.protocols.a_share import (
    RISK_STYLE_COLUMNS,
)
from tests.helpers.statistics_artifacts import portfolio_statistics_config


def _holding_paths(
    roots,
    risk_targets=("price_volume_facts", "fundamental_factors", "market_model_factors"),
):
    result = {
        name: roots[source]
        for name, source in (
            ("panel", "market_panel_1d"),
            ("market_bars", "market_bars_1d"),
            ("stock_master", "stock_master"),
        )
    }
    result["risk_context"] = [roots[name] for name in risk_targets]
    for name, source in (
        ("benchmark", "benchmark_weights"),
        ("equity_events", "a_share_equity_events"),
        ("corporate_actions", "corporate_actions"),
        ("calendar", "trading_calendar"),
        ("index_market", "index_market"),
    ):
        if source in roots:
            result[name] = roots[source]
    return result


def test_published_ledgers_preserve_ordinary_group_bootstrap(tmp_path):
    import numpy as np
    from arch.bootstrap import StationaryBootstrap
    from statsmodels.api import OLS

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.portfolios.accounting import (
        portfolio_interval_economics,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from tests.helpers.portfolio_ledger import (
        CONTINUOUS_EXECUTION_COLUMNS,
        write_test_sessions,
    )

    returns = np.random.default_rng(42).normal(0.0005, 0.008, size=252)
    gross = 1 + returns
    dates = pd.bdate_range("2024-01-01", periods=len(gross) + 1)
    columns = {}
    ledgers = {}
    for group in ("global", "quarterly"):
        candidate = f"{group}_ordinary"
        capital = PortfolioCapital(
            **{
                "candidate_id": candidate,
                "initial_capital": 1000000.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        )
        metadata = {
            "sample_role": "test",
            "window": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        }
        frames = []
        nav = capital.initial_capital
        for index, factor in enumerate(gross):
            invested = group == "quarterly"
            end = nav * factor if invested else nav
            records = []
            for asset in ("asset", "__cash__"):
                cash = asset == "__cash__"
                before = (
                    nav
                    if cash
                    and (not invested or index == 0)
                    or (not cash and invested and (index > 0))
                    else 0.0
                )
                after = nav if cash != invested else 0.0
                row = dict.fromkeys(CONTINUOUS_EXECUTION_COLUMNS, np.nan)
                row.update(
                    decision_timestamp=dates[index],
                    execution_date=dates[index],
                    asset_id=asset,
                    adv20=10000000000.0,
                    pre_trade_notional=before,
                    target_notional=after,
                    executed_delta_notional=nav
                    if invested and index == 0 and (not cash)
                    else 0.0,
                    post_trade_notional=after,
                    holding_return=factor - 1.0 if invested and (not cash) else 0.0,
                    end_notional=end if cash != invested else 0.0,
                    decision_delta_weight=(-1.0 if cash else 1.0)
                    if invested and index == 0
                    else 0.0,
                    decision_reason="simple_rule" if cash else np.nan,
                )
                row.update(dict.fromkeys(PORTFOLIO_LEDGER_COST_COLUMNS, 0.0))
                records.append(row)
            frame = pd.DataFrame.from_records(
                records, columns=CONTINUOUS_EXECUTION_COLUMNS
            )
            frame.attrs["end_timestamp"] = dates[index + 1]
            frames.append(frame)
            nav = end
        assert np.isfinite(nav) and nav > 0
        root = write_test_sessions(
            tmp_path / candidate,
            metadata=metadata,
            ledger_chunks=iter(frames),
            capital=capital,
        )
        artifact = read_portfolio_ledger_artifact(root)
        ledgers[candidate] = artifact
        with closing_iterator(artifact.iter_chunks()) as batches:
            intervals = pd.DataFrame(
                [portfolio_interval_economics(batch) for batch in batches]
            )
        assert len(intervals) == len(returns)
        assert intervals.nav_end.gt(0).all() and np.isfinite(intervals.nav_end).all()
        columns[candidate] = intervals.set_index(["entry_at", "end_at"]).net_return
    matrix = pd.DataFrame(columns)
    np.testing.assert_array_equal(matrix.global_ordinary, np.zeros(len(returns)))
    np.testing.assert_allclose(
        matrix.quarterly_ordinary, returns, rtol=1e-12, atol=1e-14
    )
    expected = OLS(returns, np.ones((len(returns), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 19}
    )
    expected_interval = StationaryBootstrap(20, returns, seed=42).conf_int(
        lambda sample: np.array([sample.mean()]),
        reps=199,
        method="percentile",
        size=0.95,
    )
    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.contracts.time import TimeWindow

    candidates = tuple(
        PortfolioCandidateSpec(
            candidate_id=f"{group}_ordinary",
            selection_group=group,
            construction="quantile_equal_weight",
            prediction_selector="champion",
            rebalance_schedule="daily",
            top_quantile=1.0,
            benchmark_id=None,
        )
        for group in ("global", "quarterly")
    )
    result = _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        time_window=TimeWindow(dates[0], dates[-1]),
        benchmark_opens=pd.DataFrame(),
        calendar=pd.DataFrame({"timestamp": dates}),
        unified_trading_sessions=lambda _: dates,
        holding_inputs={},
        config={
            "analyses": ["group_comparison"],
            "sample_role": "test",
            "statistics": {
                "hac_lag": 19,
                "bootstrap_block": 20,
                "bootstrap_repetitions": 199,
                "group_comparison": {
                    "portfolio_ids": ["ordinary"],
                    "primary_portfolio_id": "ordinary",
                    "fdr_method": "BY",
                    "alpha": 0.05,
                    "seed": 42,
                    "confidence": 0.95,
                },
            },
        },
    )["portfolio_group_comparison"]
    row = result.iloc[0]
    assert row.status == row.bootstrap_status == "DEFINED"
    assert row.observation_count == len(returns)
    assert row.mean_difference == pytest.approx(
        expected.params[0], rel=1e-12, abs=1e-14
    )
    assert row.standard_error == pytest.approx(expected.bse[0], rel=1e-12, abs=1e-14)
    assert (row.bootstrap_lower, row.bootstrap_upper) == pytest.approx(
        expected_interval[:, 0], rel=1e-12, abs=1e-14
    )
    assert row.global_terminal_wealth == 1.0
    assert row.quarterly_terminal_wealth == pytest.approx(np.prod(gross), rel=1e-12)
    output = write_statistics_artifact(
        tmp_path / "statistics",
        table_group="portfolio",
        metadata={
            **metadata,
            "candidate_ids": list(columns),
            "comparison_portfolio_ids": ["ordinary"],
        },
        tables={"portfolio_group_comparison": result},
    )
    with read_statistics_artifact(output).table(
        "portfolio_group_comparison.parquet"
    ) as saved:
        assert saved.portfolio_id.tolist() == ["ordinary"]
        assert saved.status.tolist() == saved.bootstrap_status.tolist() == ["DEFINED"]
        measurements = [
            "mean_difference",
            "standard_error",
            "bootstrap_lower",
            "bootstrap_upper",
            "global_terminal_wealth",
            "quarterly_terminal_wealth",
        ]
        np.testing.assert_array_equal(saved[measurements], result[measurements])


def test_simple_only_configuration_reads_its_persisted_risk_context(tmp_path):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.portfolios.diagnostics import (
        iter_holding_factor_inputs,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.paths import StorageLayout
    from tests.helpers.portfolio_inputs import _frames, _processed_specs
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name="pv_expanded")
    candidates, config = portfolio_statistics_config(configuration=config)
    assert candidates and all(
        candidate.benchmark_id is None for candidate in candidates
    )
    candidate = candidates[0]
    risk_targets = ("price_volume_facts", "market_model_factors")
    dates = pd.bdate_range("2020-01-02", periods=3)
    panel = _frames(dates)["market_panel_1d"]
    layout = StorageLayout(tmp_path)
    roots = {}
    specs = (
        _processed_specs().get("market_panel_1d"),
        *(_processed_specs().get(target) for target in risk_targets),
    )
    for spec in specs:
        frame = (
            panel
            if spec.data_id == "market_panel_1d"
            else panel[["market_id", "asset_id", "timestamp"]].assign(
                **dict.fromkeys(
                    set(spec.required_columns) - {"market_id", "asset_id", "timestamp"},
                    1.0,
                )
            )
        )
        root = layout.processed_data_root("a_share", spec.kind, spec.data_id)
        write_dataset(root, frame, spec=spec)
        roots[spec.data_id] = root
    from tests.helpers.portfolio_ledger import write_market_evidence

    roots.update(write_market_evidence(layout.project_root, panel))
    source = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        inputs=_holding_paths(roots, risk_targets),
        calendar=pd.DataFrame(
            {"calendar_date": dates, "source_market_type": 1, "is_trading_day": True}
        ),
        time_window=TimeWindow(dates[0], dates[-1]),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        style_columns=candidate.risk_style_columns,
        benchmark_required=False,
    )
    with closing_iterator(source):
        first = next(source)
        assert first.entry_at == dates[1] and first.end_at == dates[2]
        assert first.reason == "holding_factor_fit_unidentified"


def test_five_ledger_interval_statistics_reconcile_fraction_oracle(
    tmp_path, monkeypatch
):
    from fractions import Fraction

    import numpy as np
    import pandas as pd

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from tests.helpers.portfolio_ledger import (
        CONTINUOUS_EXECUTION_COLUMNS,
        write_test_sessions,
    )
    from tests.helpers.research_configuration import research_configuration

    candidates, config = portfolio_statistics_config(
        configuration=research_configuration(name="pv25")
    )
    before, after, end, costs = (
        (1000.0, 1040.0, 1005.0),
        (990.0, 1020.0, 1005.0),
        (1040.0, 1005.0, 1035.0),
        (10.0, 20.0, 0.0),
    )
    frames = []
    for index in range(3):
        start = pd.Timestamp("2023-01-03") + pd.Timedelta(days=index)
        prior = 0.0 if index == 0 else before[index]
        delta = after[index] - prior
        target = before[index] if index == 0 else after[index]
        decision_delta = target / before[index] - prior / before[index]
        records = []
        for asset in ("a", "__cash__"):
            cash = asset == "__cash__"
            row = {column: np.nan for column in CONTINUOUS_EXECUTION_COLUMNS}
            row.update(
                decision_timestamp=start - pd.Timedelta(days=1),
                execution_date=start,
                asset_id=asset,
                pre_trade_notional=before[index] - prior if cash else prior,
                target_notional=before[index] - target if cash else target,
                executed_delta_notional=0.0 if cash else delta,
                post_trade_notional=0.0 if cash else after[index],
                holding_return=0.0 if cash else end[index] / after[index] - 1,
                end_notional=0.0 if cash else end[index],
                decision_delta_weight=-decision_delta if cash else decision_delta,
                adv20=10000.0,
                block_reason="cash_pro_rata" if index == 0 and (not cash) else None,
                decision_reason="simple_rule" if cash else np.nan,
            )
            row.update({name: 0.0 for name in PORTFOLIO_LEDGER_COST_COLUMNS})
            row["commission_cost"] = 0.0 if cash else costs[index]
            records.append(row)
        frame = pd.DataFrame(records, columns=CONTINUOUS_EXECUTION_COLUMNS)
        frame.attrs["end_timestamp"] = start + pd.Timedelta(days=1)
        frames.append(frame)
    artifacts = {}
    for candidate in candidates:
        capital = PortfolioCapital(
            **{
                "candidate_id": candidate.candidate_id,
                "initial_capital": 1000.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        )
        metadata = {
            "sample_role": "validation",
            "window": {"start": "2023-01-02", "end": "2023-01-06"},
        }
        root = write_test_sessions(
            tmp_path / candidate.candidate_id,
            metadata=metadata,
            ledger_chunks=iter(frames),
            capital=capital,
        )
        artifacts[candidate.candidate_id] = read_portfolio_ledger_artifact(root)
    metadata = {
        "sample_role": "validation",
        "window": {"start": "2023-01-02", "end": "2023-01-06"},
    }
    metadata = {
        **metadata,
        "candidate_ids": [c.candidate_id for c in candidates],
        "comparison_portfolio_ids": config["statistics"]["group_comparison"][
            "portfolio_ids"
        ],
    }
    from quant_research_platform.research.portfolios.risk import (
        FactorRiskState,
        HoldingFactorInputs,
    )

    rng = np.random.default_rng(179)
    assets = ["a", *[f"asset_{i:03d}" for i in range(1, 100)]]
    inputs = []
    for index, frame in enumerate(frames):
        entry = pd.Timestamp(frame.execution_date.iloc[0])
        cross = pd.DataFrame(
            {
                "listing_date": pd.Timestamp("2000-01-01"),
                "timestamp": entry - pd.Timedelta(days=1),
                "asset_id": assets,
                "industry_standard_id": "CSRC2012",
                "industry_level1_code": ["A01"] * 50 + ["B01"] * 50,
                "total_market_cap": np.geomspace(100.0, 100000000.0, 100),
                "float_market_cap": np.geomspace(100.0, 100000000.0, 100),
                "asset_return": rng.normal(scale=0.03, size=100),
                "eligible": True,
                **{name: rng.normal(size=100) for name in RISK_STYLE_COLUMNS},
            }
        )
        cross.loc[0, "asset_return"] = end[index] / after[index] - 1
        fitted = FactorRiskState(style_columns=RISK_STYLE_COLUMNS).fit_return_session(
            cross, require_complete=True
        )
        assert fitted is not None
        snapshots = pd.DataFrame(
            [
                {"benchmark_id": benchmark, "asset_id": asset, "weight": 0.01}
                for benchmark in ("csi500", "csi1000")
                for asset in assets
            ]
        )
        inputs.append(
            HoldingFactorInputs(
                entry - pd.Timedelta(days=1),
                entry,
                entry + pd.Timedelta(days=1),
                fitted,
                pd.Series(100.0, index=assets),
                pd.Series(np.linspace(95.0, 105.0, 100), index=assets),
                snapshots,
            )
        )
    benchmark = pd.DataFrame(
        [
            {
                "benchmark_id": name,
                "timestamp": pd.Timestamp("2023-01-03") + pd.Timedelta(days=index),
                "open": value,
            }
            for name in ("csi500", "csi1000")
            for index, value in enumerate((100.0, 101.0, 103.0, 104.0))
        ]
    )
    benchmark_scans = []

    class ObservedBenchmark(pd.DataFrame):
        def __getitem__(self, key):
            if isinstance(key, str) and key == "benchmark_id":
                benchmark_scans.append(len(self))
            return super().__getitem__(key)

    benchmark = ObservedBenchmark(benchmark)
    from tests.helpers.benchmark_references import reference_input

    config["analyses"].append("benchmark_comparison")
    config["comparison_benchmarks"] = [
        reference_input(
            tmp_path / "comparison.parquet",
            [100.0, 101.0, 103.0, 104.0],
            dates=pd.date_range("2023-01-03", periods=4),
        )
    ]
    comparison_metadata = {}
    owned = _portfolio_statistics_tables(
        ledgers=artifacts,
        candidates=candidates,
        config=deepcopy(config),
        time_window=TimeWindow("2023-01-02", "2023-01-06"),
        benchmark_opens=benchmark,
        comparison_metadata=comparison_metadata,
        holding_inputs={
            candidate.risk_style_columns: iter(inputs) for candidate in candidates
        },
        calendar=pd.DataFrame(
            {"calendar_date": pd.bdate_range("2023-01-02", "2023-01-06")}
        ),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
    )
    assert sum(benchmark_scans) <= 2 * len(benchmark)
    assert owned["portfolio_daily"].candidate_id.tolist() == [
        candidate.candidate_id for _ in frames for candidate in candidates
    ]
    assert (
        not owned["portfolio_daily"]
        .duplicated(["candidate_id", "entry_at", "end_at"])
        .any()
    )
    for candidate in candidates:
        actual = owned["portfolio_benchmark_intervals"].loc[
            owned["portfolio_benchmark_intervals"].candidate_id.eq(
                candidate.candidate_id
            )
        ]
        np.testing.assert_allclose(
            actual.relative_wealth,
            np.array([1.04, 1.005, 1.035]) / np.array([1.01, 1.03, 1.04]),
            rtol=1e-12,
            atol=0,
        )
    metadata.update(comparison_metadata, execution_mode="continuous")
    participation = deepcopy(config)["execution_assumptions"]["adv_sensitivities"]
    assert list(
        owned["portfolio_capacity"][["candidate_id", "participation"]].itertuples(
            index=False, name=None
        )
    ) == [
        (candidate.candidate_id, value)
        for _ in range(2)
        for candidate in candidates
        for value in participation
    ]
    root = write_statistics_artifact(
        tmp_path / "statistics",
        metadata=metadata,
        tables=owned,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(root)
    with artifact.table("portfolio_daily.parquet") as daily:
        rows = daily.loc[daily.candidate_id.eq(candidates[0].candidate_id)]
        np.testing.assert_allclose(
            rows.net_return,
            [
                float(Fraction(1, 25)),
                float(Fraction(-35, 1040)),
                float(Fraction(30, 1005)),
            ],
        )
        assert np.prod(1 + rows.net_return) == pytest.approx(1.035)
        np.testing.assert_allclose(rows.gross_return - rows.cost_drag, rows.net_return)
        assert rows.nav_post.tolist() == [990.0, 1020.0, 1005.0]
        assert rows.cash_post.eq(0.0).all()
        assert rows.cash_weight_post.eq(0.0).all()
        assert (
            (rows.nav_end - rows.nav_post).sum(),
            (rows.nav_end - rows.nav_before).sum(),
            (rows.cost_drag * rows.nav_before).sum(),
        ) == pytest.approx((65.0, 35.0, 30.0))
    with artifact.table("portfolio_performance.parquet") as performance:
        cagr = performance.loc[performance.metric_id.eq("cagr")]
        assert len(cagr) == len(candidates)
        assert cagr.status.eq("DEFINED").all()
        assert cagr.observation_count.eq(3).all()
        np.testing.assert_allclose(
            cagr.value, float(Fraction(207, 200) ** 84 - 1), rtol=1e-12
        )
    with artifact.table("portfolio_attribution.parquet") as attribution:
        assert attribution.status.eq("DEFINED").all()
        actual = (
            attribution.loc[attribution.component_type.eq("additive")]
            .groupby(["candidate_id", "entry_at"])
            .value.sum()
        )
        for candidate in candidates:
            np.testing.assert_allclose(
                actual.loc[candidate.candidate_id],
                [0.04, -35 / 1040, 30 / 1005],
                atol=1e-12,
            )
        detail = (
            attribution.loc[
                attribution.candidate_id.eq(candidates[-1].candidate_id)
                & attribution.entry_at.eq(inputs[0].entry_at)
            ]
            .set_index("component_id")
            .value
        )
        ref = inputs[0].entry_open / inputs[0].decision_close
        ref /= ref.sum()
        w = pd.Series(0.0, index=assets)
        w.loc["a"] = 0.99
        assert detail["stock_residual_difference"] == pytest.approx(
            (w - ref).dot(inputs[0].fitted.residuals)
        )
        assert detail["benchmark_replication_gap"] != 0.0
        assert detail["residual_unexplained"] == pytest.approx(
            detail["stock_residual_difference"] + detail["benchmark_replication_gap"]
        )
    with artifact.table("portfolio_exposures.parquet") as exposures:
        first = (
            exposures.loc[
                exposures.candidate_id.eq(candidates[0].candidate_id)
                & exposures.entry_at.eq(inputs[0].entry_at)
            ]
            .set_index("factor_id")
            .value
        )
        np.testing.assert_allclose(
            first.reindex(inputs[0].fitted.exposures.columns),
            inputs[0].fitted.exposures.loc["a"],
        )
    from dataclasses import replace

    from quant_research_platform.artifacts.portfolio_ledger import (
        PortfolioLedgerArtifact,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator

    def produce(*, factor_inputs=inputs, benchmark_frame=benchmark):
        return _portfolio_statistics_tables(
            ledgers=artifacts,
            candidates=candidates,
            config=deepcopy(config),
            time_window=TimeWindow("2023-01-02", "2023-01-06"),
            benchmark_opens=benchmark_frame,
            holding_inputs={
                candidate.risk_style_columns: iter(factor_inputs)
                for candidate in candidates
            },
            calendar=pd.DataFrame(
                {"calendar_date": pd.bdate_range("2023-01-02", "2023-01-06")}
            ),
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
        )

    original = PortfolioLedgerArtifact.iter_chunks
    for mode in ("missing_first", "missing_last", "interrupt", "missing_adv"):
        closed = []
        started = []

        def altered(self, mode=mode, closed=closed, started=started):
            started.append(str(self.root))
            try:
                with closing_iterator(original(self)) as source:
                    for index, item in enumerate(source):
                        if mode == "interrupt" and index == 1:
                            raise KeyboardInterrupt("fixture interruption")
                        if (
                            mode == "missing_first"
                            and index == 0
                            or (mode == "missing_last" and index == 2)
                        ):
                            continue
                        if mode == "missing_adv":
                            item = replace(
                                item,
                                diagnostics=tuple(
                                    row
                                    for row in item.diagnostics
                                    if row.name != "adv20"
                                ),
                            )
                        yield item
                        item = None
            finally:
                closed.append(str(self.root))

        with monkeypatch.context() as patch:
            patch.setattr(PortfolioLedgerArtifact, "iter_chunks", altered)
            if mode == "missing_adv":
                owned = produce()
                assert owned["portfolio_capacity"].status.eq("UNDEFINED").all()
                assert owned["portfolio_capacity_summary"].status.eq("UNDEFINED").all()
            elif mode == "interrupt":
                with pytest.raises(KeyboardInterrupt):
                    produce()
            else:
                expected_error = {
                    "missing_first": "portfolio_statistics_window_incomplete",
                    "missing_last": "portfolio_statistics_candidate_dates_misaligned",
                }[mode]
                with pytest.raises(ValueError, match=f"^{expected_error}$"):
                    produce()
        assert started
        assert sorted(closed) == sorted(started)
    no_fit = [
        replace(item, fitted=None, reason="attribution_pit_history_missing")
        for item in inputs
    ]
    absent_benchmark = benchmark.assign(open=np.nan)
    owned = produce(factor_inputs=no_fit, benchmark_frame=absent_benchmark)
    root = write_statistics_artifact(
        tmp_path / "data/artifacts/a_share/pv_all_portfolio_statistics",
        metadata=metadata,
        tables=owned,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(root)
    with artifact.table("holding_factor_returns.parquet") as rows:
        assert rows.empty
    with artifact.table("portfolio_exposures.parquet") as rows:
        assert rows.empty
    with artifact.table("portfolio_performance.parquet") as rows:
        assert (
            rows.loc[rows.metric_id.eq("tracking_error"), "status"].eq("BLOCKED").all()
        )
        assert rows.loc[rows.metric_id.eq("net_pnl"), "status"].eq("DEFINED").all()
    with artifact.table("portfolio_daily.parquet") as rows:
        assert len(rows) == 3 * len(candidates)
        assert rows.cash_weight_post.eq(0.0).all()
    with artifact.table("portfolio_capacity.parquet") as rows:
        assert {
            "candidate_id",
            "entry_at",
            "participation",
            "bottleneck_asset_id",
        } <= set(rows.columns)
    with artifact.table("portfolio_robustness.parquet") as rows:
        blocked_ids = [
            item.candidate_id for item in candidates if item.selection_group == "global"
        ]
        global_rows = rows.loc[rows.candidate_id.isin(blocked_ids)]
        assert len(global_rows) == 10 and global_rows.status.eq("UNDEFINED").all()
        assert not global_rows.reason.eq(
            "global_validation_selection_reuses_validation"
        ).any()
        assert global_rows.value.isna().all()
        assert rows.candidate_count.eq(len(candidates)).all()
    with artifact.table("portfolio_group_comparison.parquet") as rows:
        assert len(rows) == 5
        assert rows.status.eq("UNDEFINED").all()
        assert not rows.reason.eq("global_validation_selection_reuses_validation").any()
        assert rows.observation_count.eq(3).all()


@pytest.mark.parametrize("st_holding", (False, True))
@pytest.mark.parametrize("relist_step", (None, 1, 2))
def test_holding_input_reader_freezes_decision_universe_over_holding_interval(
    tmp_path, st_holding, relist_step
):
    import numpy as np

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.portfolios.diagnostics import (
        iter_holding_factor_inputs,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.paths import StorageLayout
    from tests.helpers.portfolio_inputs import _processed_specs

    dates = pd.bdate_range("2017-01-02", periods=4)
    assets = [f"asset_{index:03d}" for index in range(105)]
    rng = np.random.default_rng(818)
    returns = rng.normal(scale=0.008, size=(4, 1)) * rng.uniform(
        0.5, 1.5, size=(1, 105)
    ) + rng.normal(size=(4, 105)) * np.linspace(0.003, 0.009, 105)
    prices = 100 * np.cumprod(1 + returns, axis=0)
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_id": "a_share",
            "timestamp": dates.repeat(105),
            "asset_id": assets * 4,
            "open": prices.ravel(),
            "adj_open": prices.ravel(),
            "adj_close": prices.ravel(),
            "amount": 1000000.0,
            "market_type": 1,
            "period_return": returns.ravel(),
            "total_market_cap": np.tile(np.geomspace(100.0, 1000000.0, 105), 4),
            "float_market_cap": np.tile(np.geomspace(100.0, 1000000.0, 105), 4),
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": np.tile(["A01"] * 52 + ["B01"] * 53, 4),
        }
    )
    panel["industry_level1_code"] = np.tile(["A01"] * 52 + ["B01"] * 53, 4)
    panel["is_nonfinancial_industry"] = True
    panel = panel.assign(
        is_a_share_scope=True,
        is_trading=True,
        is_st=False,
        is_pt=False,
        listing_age_trading_days=300,
        has_252_trading_sessions=True,
        close=panel.adj_close,
        volume=1000.0,
        turnover=0.01,
    )
    styles = RISK_STYLE_COLUMNS
    if relist_step is not None:
        changed = panel.asset_id.eq(assets[-2]) & panel.timestamp.ge(dates[relist_step])
        panel.loc[changed, "listing_date"] = dates[relist_step]
        panel.loc[changed, ["adj_open", "adj_close"]] *= 100
    if st_holding:
        panel.loc[panel.asset_id.eq(assets[0]), "is_st"] = True
    data = panel[["market_id", "timestamp", "asset_id"]].copy()
    for name in styles:
        data[name] = rng.normal(size=len(data))
    data.loc[data.asset_id.eq(assets[-1]), styles[0]] = np.nan
    risk_data = data.loc[:, ["market_id", "timestamp", "asset_id", *styles]].copy()
    weights = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "benchmark_id": benchmark,
                "asset_id": asset,
                "timestamp": timestamp,
                "weight": 1 / 105,
            }
            for benchmark in ("csi500", "csi1000")
            for timestamp in dates[-4:]
            for asset in assets
        ]
    )
    base_layout = StorageLayout(tmp_path)
    roots = {
        "market_panel_1d": base_layout.processed_data_root(
            "a_share", _processed_specs().get("market_panel_1d").kind, "market_panel_1d"
        ),
        "benchmark_weights": base_layout.processed_data_root(
            "a_share",
            _processed_specs().get("benchmark_weights").kind,
            "benchmark_weights",
        ),
    }
    write_dataset(
        roots["market_panel_1d"], panel, spec=_processed_specs().get("market_panel_1d")
    )
    write_dataset(
        roots["benchmark_weights"],
        weights,
        spec=_processed_specs().get("benchmark_weights"),
    )
    risk_targets = ("price_volume_facts", "fundamental_factors", "market_model_factors")
    for target in risk_targets:
        context_spec = _processed_specs().get(target)
        roots[target] = base_layout.processed_data_root(
            "a_share", context_spec.kind, target
        )
        write_dataset(
            roots[target],
            risk_data.loc[:, list(context_spec.required_columns)],
            spec=context_spec,
        )
    from tests.helpers.portfolio_ledger import write_market_evidence

    roots.update(write_market_evidence(base_layout.project_root, panel))
    input_paths = _holding_paths(roots)
    calendar = pd.DataFrame(
        {"calendar_date": dates, "source_market_type": 1, "is_trading_day": True}
    )
    window = TimeWindow(dates[-4], dates[-1])
    source = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        benchmark_required=True,
        inputs=input_paths,
        calendar=calendar,
        time_window=window,
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        style_columns=RISK_STYLE_COLUMNS,
    )
    with closing_iterator(source):
        first = next(source)
        assert first.reason is None and first.fitted is not None
        assert len(first.fitted.asset_returns) == 104 - int(relist_step is not None)
        if relist_step is not None:
            assert assets[-2] not in first.fitted.exposures.index
        assert assets[-1] not in first.fitted.exposures.index
        assert (
            first.decision_at == dates[-4]
            and first.entry_at == dates[-3]
            and (first.end_at == dates[-2])
        )
        expected = (
            panel.loc[panel.timestamp.eq(first.end_at)].set_index("asset_id").adj_open
            / panel.loc[panel.timestamp.eq(first.entry_at)]
            .set_index("asset_id")
            .adj_open
            - 1
        )
        np.testing.assert_allclose(
            first.fitted.asset_returns,
            expected.reindex(first.fitted.asset_returns.index),
        )
        from quant_research_platform.research.portfolios.risk import holding_attribution

        values, reasons, exposure = holding_attribution(
            first,
            after_notionals=pd.Series({assets[0]: 100.0}),
            nav_before=100.0,
            nav_post=100.0,
            benchmark_id=None,
            benchmark_return=None,
            holding_returns=expected,
        )
        assert not reasons and (not exposure.empty)
        assert sum(
            values[name] for name in ("industry", "style", "residual_unexplained")
        ) == pytest.approx(expected[assets[0]])
        first = None

    def first_for_styles(selected_styles, selected_roots):
        stream = iter_holding_factor_inputs(
            iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
            benchmark_required=True,
            inputs=_holding_paths(selected_roots, risk_targets),
            calendar=calendar,
            time_window=window,
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
            style_columns=selected_styles,
        )
        with closing_iterator(stream):
            return next(stream)

    five = first_for_styles(RISK_STYLE_COLUMNS[:-1], roots)
    missing_earnings = risk_data.copy()
    missing_earnings["earnings_yield"] = np.nan
    fundamental_spec = _processed_specs().get("fundamental_factors")
    missing_earnings_layout = StorageLayout(tmp_path / "missing_earnings")
    missing_earnings_root = missing_earnings_layout.processed_data_root(
        "a_share", fundamental_spec.kind, fundamental_spec.data_id
    )
    write_dataset(
        missing_earnings_root,
        missing_earnings.loc[:, list(fundamental_spec.required_columns)],
        spec=fundamental_spec,
    )
    missing_earnings_inputs = {**roots, "fundamental_factors": missing_earnings_root}
    five_after = first_for_styles(RISK_STYLE_COLUMNS[:-1], missing_earnings_inputs)
    pd.testing.assert_series_equal(
        (five.fitted.factor_returns).isna(),
        (five_after.fitted.factor_returns).isna(),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        five.fitted.factor_returns,
        five_after.fitted.factor_returns,
        check_exact=False,
        rtol=2e-12,
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(
        (five.fitted.exposures).isna(),
        (five_after.fitted.exposures).isna(),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        five.fitted.exposures,
        five_after.fitted.exposures,
        check_exact=False,
        rtol=2e-12,
        atol=1e-14,
    )
    six = first_for_styles(RISK_STYLE_COLUMNS, missing_earnings_inputs)
    assert six.fitted is None and six.reason == "holding_factor_fit_unidentified"
    if not st_holding and relist_step is None:
        # Both groups receive the public combined input list. An unselected
        # source contributes neither a session requirement nor a key domain.
        sparse_root = tmp_path / "sparse_fundamental"
        write_dataset(
            sparse_root,
            risk_data.loc[
                risk_data.timestamp.ne(dates[0]),
                list(fundamental_spec.required_columns),
            ],
            spec=fundamental_spec,
        )
        sparse_inputs = {**roots, "fundamental_factors": sparse_root}
        five_sparse = first_for_styles(RISK_STYLE_COLUMNS[:-1], sparse_inputs)
        pd.testing.assert_series_equal(
            (five.fitted.factor_returns).isna(),
            (five_sparse.fitted.factor_returns).isna(),
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            five.fitted.factor_returns,
            five_sparse.fitted.factor_returns,
            check_exact=False,
            rtol=2e-12,
            atol=1e-14,
        )
        pd.testing.assert_frame_equal(
            (five.fitted.exposures).isna(),
            (five_sparse.fitted.exposures).isna(),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            five.fitted.exposures,
            five_sparse.fitted.exposures,
            check_exact=False,
            rtol=2e-12,
            atol=1e-14,
        )
        with pytest.raises(ValueError, match="holding_risk_context_session_missing"):
            first_for_styles(RISK_STYLE_COLUMNS, sparse_inputs)

        invalid_daily = panel.copy()
        invalid_daily["period_return"] = "vendor-invalid"
        invalid_daily_root = tmp_path / "invalid_daily_return"
        write_dataset(
            invalid_daily_root,
            invalid_daily,
            spec=_processed_specs().get("market_panel_1d"),
        )
        five_holding = first_for_styles(
            RISK_STYLE_COLUMNS[:-1],
            {**roots, "market_panel_1d": invalid_daily_root},
        )
        pd.testing.assert_series_equal(
            (five.fitted.factor_returns).isna(),
            (five_holding.fitted.factor_returns).isna(),
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            five.fitted.factor_returns,
            five_holding.fitted.factor_returns,
            check_exact=False,
            rtol=2e-12,
            atol=1e-14,
        )
        pd.testing.assert_series_equal(
            (five.fitted.asset_returns).isna(),
            (five_holding.fitted.asset_returns).isna(),
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            five.fitted.asset_returns,
            five_holding.fitted.asset_returns,
            check_exact=False,
            rtol=2e-12,
            atol=1e-14,
        )
        # Daily risk still requires and validates its own return input.
        from quant_research_platform.research.portfolios.production import (
            _enhanced_risk_cross,
        )

        with pytest.raises(ValueError, match="Unable to parse string"):
            _enhanced_risk_cross(
                panel=invalid_daily.loc[invalid_daily.timestamp.eq(dates[0])],
                risk_frames=[risk_data.loc[risk_data.timestamp.eq(dates[0])]],
                style_columns=RISK_STYLE_COLUMNS[:-1],
            )
    missing_panel = panel.copy()
    missing_panel.loc[
        missing_panel.timestamp.eq(dates[-2]) & missing_panel.asset_id.eq(assets[0]),
        "adj_open",
    ] = np.nan
    missing_layout = StorageLayout(tmp_path / "missing_price")
    missing_root = missing_layout.processed_data_root(
        "a_share", _processed_specs().get("market_panel_1d").kind, "market_panel_1d"
    )
    write_dataset(
        missing_root, missing_panel, spec=_processed_specs().get("market_panel_1d")
    )
    missing_inputs = _holding_paths({**roots, "market_panel_1d": missing_root})
    source = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        benchmark_required=True,
        inputs=missing_inputs,
        calendar=calendar,
        time_window=window,
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        style_columns=RISK_STYLE_COLUMNS,
    )
    with closing_iterator(source):
        first = next(source)
        assert first.fitted is None
        assert first.reason == "holding_returns_incomplete"
        first = None


@pytest.mark.parametrize(
    ("constant", "swap_group_metadata", "candidate_id"),
    (
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (False, False, "timestamp"),
    ),
)
def test_declared_ledger_family_preserves_descriptive_inference(
    tmp_path, constant, swap_group_metadata, candidate_id
):
    from dataclasses import replace
    from itertools import combinations

    import numpy as np
    from scipy.stats import kurtosis, norm, rankdata, skew

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
    from tests.helpers.portfolio_ledger import (
        CONTINUOUS_EXECUTION_COLUMNS,
        write_test_sessions,
    )
    from tests.helpers.research_configuration import research_configuration

    dates = pd.bdate_range("2023-01-02", periods=75)
    configuration = research_configuration(name="pv25")
    if candidate_id is not None:
        configuration["portfolios"][0]["candidate_id"] = candidate_id
        configuration["primary_portfolio_id"] = candidate_id
    candidates, config = portfolio_statistics_config(configuration=configuration)
    config["statistics"]["bootstrap_repetitions"] = 99
    if candidate_id is not None:
        assert (
            candidate_id
            in deepcopy(config)["statistics"]["group_comparison"]["portfolio_ids"]
        )
    if swap_group_metadata:
        candidates = tuple(
            replace(candidate, selection_group=other)
            for candidate in candidates
            for other in (
                "quarterly" if candidate.selection_group == "global" else "global",
            )
        )
    qualified = tuple(
        item for item in candidates if item.selection_group == "quarterly"
    )
    blocked = tuple(item for item in candidates if item.selection_group == "global")
    assert len(qualified) == len(blocked) == 5
    qualified_ids = [item.candidate_id for item in qualified]
    blocked_ids = [item.candidate_id for item in blocked]
    returns = np.random.default_rng(802).normal(0.0002, 0.008, (73, 5))
    if constant:
        returns[:, 0] = 0.1
    window = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "access": "validation",
    }

    def persisted_ledger(candidate, series, destination):
        capital = PortfolioCapital(
            **{
                "candidate_id": candidate.candidate_id,
                "initial_capital": 1000.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        )
        metadata = {
            "sample_role": window["access"],
            "window": {key: window[key] for key in ("start", "end")},
        }
        frames = []
        nav = 1000.0
        for index in range(73):
            rows = []
            for asset in ("asset", "__cash__"):
                cash = asset == "__cash__"
                value = 0.0 if cash else nav
                rate = 0.0 if cash else series[index]
                row = {name: np.nan for name in CONTINUOUS_EXECUTION_COLUMNS}
                row.update(
                    decision_timestamp=dates[index],
                    execution_date=dates[index + 1],
                    asset_id=asset,
                    pre_trade_notional=value,
                    target_notional=value,
                    executed_delta_notional=0.0,
                    post_trade_notional=value,
                    holding_return=rate,
                    end_notional=value * (1 + rate),
                    decision_delta_weight=0.0,
                    decision_reason="simple_rule" if cash else np.nan,
                )
                row.update({name: 0.0 for name in PORTFOLIO_LEDGER_COST_COLUMNS})
                rows.append(row)
            frame = pd.DataFrame(rows, columns=CONTINUOUS_EXECUTION_COLUMNS)
            frame.attrs["end_timestamp"] = dates[index + 2]
            frames.append(frame)
            nav *= 1 + series[index]
        root = write_test_sessions(
            destination, metadata=metadata, ledger_chunks=iter(frames), capital=capital
        )
        return read_portfolio_ledger_artifact(root)

    ledgers = {}
    for group in (qualified, blocked):
        for column, candidate in enumerate(group):
            series = returns[:, column]
            if candidate.selection_group == "global":
                series = 0.015 + 2 * series
            ledgers[candidate.candidate_id] = persisted_ledger(
                candidate, series, tmp_path / candidate.candidate_id
            )
    inputs = [
        HoldingFactorInputs(
            dates[i],
            dates[i + 1],
            dates[i + 2],
            None,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(columns=("benchmark_id", "asset_id", "weight")),
            "attribution_pit_history_missing",
        )
        for i in range(73)
    ]
    metadata = {
        "sample_role": window["access"],
        "window": {key: window[key] for key in ("start", "end")},
    }
    metadata = {
        **metadata,
        "candidate_ids": [c.candidate_id for c in candidates],
        "comparison_portfolio_ids": config["statistics"]["group_comparison"][
            "portfolio_ids"
        ],
    }

    def produce(selected_ledgers, *, robustness=False, sample_role="validation"):
        manifest = deepcopy(config)
        manifest["sample_role"] = sample_role
        if robustness:
            manifest["expectations"] = {"freshness_days": 180, "role": "robustness"}
        if robustness == "training_weighting":
            manifest["expectations"]["role"] = "primary"
            manifest["training_weighting"] = {"role": "robustness"}
        return _portfolio_statistics_tables(
            ledgers=selected_ledgers,
            candidates=candidates,
            config=manifest,
            time_window=TimeWindow(dates[0], dates[-1]),
            benchmark_opens=pd.DataFrame(columns=("benchmark_id", "timestamp", "open")),
            calendar=pd.DataFrame({"calendar_date": dates}),
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
            holding_inputs={
                candidate.risk_style_columns: iter(inputs) for candidate in candidates
            },
        )

    owned = produce(ledgers)
    root = write_statistics_artifact(
        tmp_path / "statistics",
        metadata=metadata,
        tables=owned,
        table_group="portfolio",
    )
    corrupt = dict(owned)
    corrupt["portfolio_daily"] = pd.concat(
        [corrupt["portfolio_daily"], corrupt["portfolio_daily"].iloc[:1]],
        ignore_index=True,
    )
    rejected = {
        "sample_role": window["access"],
        "window": {key: window[key] for key in ("start", "end")},
    }
    rejected = {
        **rejected,
        "candidate_ids": [c.candidate_id for c in candidates],
        "comparison_portfolio_ids": config["statistics"]["group_comparison"][
            "portfolio_ids"
        ],
    }
    with pytest.raises(ValueError, match="statistics_table_duplicate_keys"):
        write_statistics_artifact(
            tmp_path / "rejected_statistics",
            metadata=rejected,
            tables=corrupt,
            table_group="portfolio",
        )
    assert not (tmp_path / "rejected_statistics").exists()
    corrupt = None
    artifact = read_statistics_artifact(root)
    with artifact.table("portfolio_robustness.parquet") as rows:
        assert len(rows) == 2 * len(candidates) + 1
        assert rows.candidate_count.eq(len(candidates)).all()
        assert set(rows.candidate_id) == {"joint", *ledgers}
        assert not rows.reason.isin(
            ("test_excluded", "global_validation_selection_reuses_validation")
        ).any()
        baseline = rows.copy()
    with artifact.table("portfolio_group_comparison.parquet") as rows:
        assert len(rows) == 5
        assert rows.global_mean.notna().all() and rows.quarterly_mean.notna().all()
        assert rows.observation_count.eq(73).all()
    with artifact.table("portfolio_capacity.parquet") as rows:
        assert rows.empty
    with artifact.table("portfolio_capacity_summary.parquet") as rows:
        assert rows.status.eq("UNDEFINED").all()
        assert rows.reason.eq("no_rebalance_requests").all()

    if constant:
        with artifact.table("portfolio_performance.parquet") as rows:
            selected = rows.loc[
                rows.candidate_id.eq(qualified[0].candidate_id)
            ].set_index("metric_id")
            assert selected.loc["sharpe", "status"] == "UNDEFINED"
            assert selected.loc["annualized_volatility", "value"] == 0.0
        assert (
            baseline.loc[baseline.test_id.isin(("dsr", "cscv_pbo")), "status"]
            .eq("UNDEFINED")
            .all()
        )
        psr = baseline.loc[baseline.test_id.eq("psr")].set_index("candidate_id")
        assert (
            psr.loc[[qualified[0].candidate_id, blocked[0].candidate_id], "status"]
            .eq("UNDEFINED")
            .all()
        )
        assert (
            psr.loc[[*qualified_ids[1:], *blocked_ids[1:]], "status"]
            .eq("DEFINED")
            .all()
        )
        return

    # All explicitly supplied candidates contribute to DSR and CSCV, including
    # descriptive historical groups. A name alone cannot remove a trial.
    daily = owned["portfolio_daily"]
    matrix = daily.pivot(
        index=["entry_at", "end_at"], columns="candidate_id", values="net_return"
    ).reindex(columns=[candidate.candidate_id for candidate in candidates])
    values = matrix.to_numpy()
    sharpes = values.mean(axis=0) / values.std(axis=0, ddof=1)
    count = len(candidates)
    reference = sharpes.std(ddof=1) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / count)
        + np.euler_gamma * norm.ppf(1 - 1 / (count * np.e))
    )
    for column, candidate in enumerate(candidates):
        series, score = values[:, column], sharpes[column]
        se = np.sqrt(
            (
                1
                - skew(series, bias=False) * score
                + (kurtosis(series, fisher=False, bias=False) - 1) * score**2 / 4
            )
            / 72
        )
        actual = (
            baseline.loc[baseline.candidate_id.eq(candidate.candidate_id)]
            .set_index("test_id")
            .value
        )
        assert actual.psr == pytest.approx(norm.cdf(score / se))
        assert actual.dsr == pytest.approx(norm.cdf((score - reference) / se))
    blocks = np.array_split(np.arange(73), 10)
    outcomes = []
    for chosen in combinations(range(10), 5):
        train = values[np.concatenate([blocks[i] for i in chosen])]
        test = values[np.concatenate([blocks[i] for i in range(10) if i not in chosen])]
        winner = np.argmax(train.mean(axis=0) / train.std(axis=0, ddof=1))
        rank = rankdata(test.mean(axis=0) / test.std(axis=0, ddof=1))[winner]
        outcomes.append(rank / (count + 1) <= 0.5)
    assert len(outcomes) == 252
    assert baseline.loc[baseline.test_id.eq("cscv_pbo"), "value"].iloc[
        0
    ] == pytest.approx(np.mean(outcomes))

    for role, robustness in (
        ("test", False),
        ("validation", True),
        ("test", "training_weighting"),
    ):
        varied = produce(ledgers, sample_role=role, robustness=robustness)
        comparison = varied["portfolio_group_comparison"]
        assert comparison.status.eq("DEFINED").all()
        assert comparison.bootstrap_status.eq("DEFINED").all()
        assert comparison.fdr_method.eq("NOT_APPLICABLE" if robustness else "BY").all()
        assert comparison.fdr_adjusted_p_value.isna().eq(bool(robustness)).all()
        paired = comparison.set_index("portfolio_id")
        for portfolio_id in paired.index:
            expected = (
                matrix[f"quarterly_{portfolio_id}"].mean()
                - matrix[f"global_{portfolio_id}"].mean()
            )
            assert paired.loc[portfolio_id, "mean_difference"] == pytest.approx(
                expected, abs=1e-14
            )
        pd.testing.assert_frame_equal(
            varied["portfolio_robustness"].drop(columns="sample_role"),
            owned["portfolio_robustness"].drop(columns="sample_role"),
            check_exact=False,
            rtol=1e-14,
            atol=1e-15,
        )
        metadata_variant = {**metadata, "sample_role": role}
        published = write_statistics_artifact(
            tmp_path / f"role-{role}-{robustness}",
            metadata=metadata_variant,
            tables=varied,
            table_group="portfolio",
        )
        with read_statistics_artifact(published).table(
            "portfolio_group_comparison.parquet"
        ) as saved:
            np.testing.assert_allclose(
                saved.p_value, comparison.p_value, rtol=1e-14, atol=1e-15
            )

    altered = dict(ledgers)
    for column, candidate in enumerate(blocked):
        altered[candidate.candidate_id] = persisted_ledger(
            candidate,
            np.where(np.arange(73) % 2, -0.2, 0.5 + column / 10),
            tmp_path / "changed" / candidate.candidate_id,
        )
    changed = produce(altered)["portfolio_robustness"]
    baseline_own = baseline.loc[baseline.candidate_id.isin(qualified_ids)].set_index(
        ["candidate_id", "test_id"]
    )
    changed_own = changed.loc[changed.candidate_id.isin(qualified_ids)].set_index(
        ["candidate_id", "test_id"]
    )
    np.testing.assert_allclose(
        changed_own.xs("psr", level="test_id").value,
        baseline_own.xs("psr", level="test_id").value,
        rtol=1e-14,
        atol=1e-15,
    )
    assert not np.array_equal(
        changed_own.xs("dsr", level="test_id").value,
        baseline_own.xs("dsr", level="test_id").value,
    )


def test_public_portfolio_statistics_derive_missing_csi_opens_and_preserve_sources(
    tmp_path,
):
    import hashlib
    from fractions import Fraction

    import numpy as np

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.data import InterimDataSpec
    from quant_research_platform.contracts.portfolios import (
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.diagnostics import (
        build_portfolio_statistics_tables,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.paths import StorageLayout
    from tests.helpers.portfolio_inputs import _frames, _interim_specs, _processed_specs
    from tests.helpers.portfolio_ledger import (
        CONTINUOUS_EXECUTION_COLUMNS,
        write_test_sessions,
    )
    from tests.helpers.research_configuration import research_configuration

    dates = pd.bdate_range("2019-01-02", periods=3)
    frames = _frames(dates)
    originals = {name: frame.copy(deep=True) for name, frame in frames.items()}
    candidates, config = portfolio_statistics_config(
        configuration=research_configuration(name="pv25")
    )
    manifest = deepcopy(config)
    window = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "access": "validation",
    }

    def metadata(logical_id, *, metadata=None):
        return {
            "sample_role": window["access"],
            "window": {key: window[key] for key in ("start", "end")},
        }

    def tree_hash(root):
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    layout = StorageLayout(tmp_path)
    roots = {}
    for spec in (*_interim_specs().list(), *_processed_specs().list()):
        if spec.data_id in {"index_market", "a_share_equity_events"}:
            continue
        root = (
            layout.interim_data_root(spec.source_id, spec.data_id)
            if isinstance(spec, InterimDataSpec)
            else layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        )
        write_dataset(root, frames[spec.data_id], spec=spec)
        roots[spec.data_id] = root
    ledger_rows = []
    for asset, value in (("a", 1000.0), ("__cash__", 0.0)):
        row = {name: np.nan for name in CONTINUOUS_EXECUTION_COLUMNS}
        row.update(
            decision_timestamp=dates[0],
            execution_date=dates[1],
            asset_id=asset,
            pre_trade_notional=value,
            target_notional=value,
            executed_delta_notional=0.0,
            post_trade_notional=value,
            holding_return=0.04 if value else 0.0,
            end_notional=value * 1.04,
            decision_delta_weight=0.0,
            decision_reason="simple_rule" if not value else np.nan,
            adv20=10000.0,
        )
        row.update({name: 0.0 for name in PORTFOLIO_LEDGER_COST_COLUMNS})
        ledger_rows.append(row)
    ledger_frame = pd.DataFrame(ledger_rows, columns=CONTINUOUS_EXECUTION_COLUMNS)
    ledger_frame.attrs["end_timestamp"] = dates[2]
    ledgers = {}
    for candidate in candidates:
        capital = PortfolioCapital(candidate.candidate_id, 1000.0, "CNY", "cash_equity")
        contract = metadata(candidate.candidate_id)
        root = write_test_sessions(
            tmp_path / "ledgers" / candidate.candidate_id,
            metadata=contract,
            ledger_chunks=iter((ledger_frame,)),
            capital=capital,
        )
        ledgers[candidate.candidate_id] = read_portfolio_ledger_artifact(root)
    market = pd.DataFrame(
        [
            {
                "source_asset_id": code,
                "timestamp": date,
                "trade_date": date,
                "index_name": code,
                "open": opening,
                "close": closing,
                "period_return": np.nan,
            }
            for code, opens, closes in (
                ("000905", (50.0, 100.0, 102.0), (55.0, 110.0, 100.0)),
                ("H00905", (np.nan,) * 3, (110.0, 220.0, 210.0)),
                ("000852", (50.0, 100.0, 102.0), (55.0, 110.0, 100.0)),
                ("H00852", (110.0, 120.0, 126.0), (110.0, 220.0, 210.0)),
            )
            for date, opening, closing in zip(dates, opens, closes, strict=True)
        ]
    )
    cases = [
        ("derived_and_native", (), None),
        ("scale_small", (), None),
        ("scale_large", (), None),
    ]
    for code, field in (("000905", "open"), ("000905", "close"), ("H00905", "close")):
        for label, value in (
            ("missing", np.nan),
            ("negative", -1.0),
            ("infinite", np.inf),
        ):
            cases.append(
                (
                    f"{code}_{field}_{label}",
                    ((code, field, value),),
                    {
                        "missing": "BLOCKED",
                        "negative": "ERROR",
                        "infinite": "WRITE_ERROR",
                    }[label],
                )
            )
    cases.extend(
        (
            (
                "negative_ratio",
                (("000905", "open", -100.0), ("000905", "close", -110.0)),
                "ERROR",
            ),
            ("zero_denominator", (("000905", "close", 0.0),), "ERROR"),
            ("native_negative", (("H00905", "open", -1.0),), "ERROR"),
            ("native_infinite", (("H00905", "open", np.inf),), "WRITE_ERROR"),
            ("native_ignores_unused_close", (("000852", "close", 0.0),), None),
            ("unconsumed_invalid", (), None),
            ("missing_price_date", (), "BLOCKED"),
            ("missing_tr_date", (), "BLOCKED"),
        )
    )
    input_hashes = {name: tree_hash(root) for name, root in roots.items()}
    good_root = good_hash = None
    for case, mutations, expected_status in cases:
        source = market.copy(deep=True)
        if case in {"scale_small", "scale_large"}:
            source.loc[:, ["open", "close"]] *= 0.01 if case == "scale_small" else 100.0
        for code, field, value in mutations:
            source.loc[
                source.source_asset_id.eq(code) & source.timestamp.eq(dates[1]), field
            ] = value
        if case == "unconsumed_invalid":
            unused = source.timestamp.eq(dates[0])
            source.loc[unused & source.source_asset_id.eq("000905"), "open"] = -1.0
            source.loc[unused & source.source_asset_id.eq("000905"), "close"] = 0.0
            source.loc[unused & source.source_asset_id.eq("H00852"), "open"] = -1.0
        if case in {"missing_price_date", "missing_tr_date"}:
            code = "000905" if case == "missing_price_date" else "H00905"
            source = source.loc[
                ~(source.source_asset_id.eq(code) & source.timestamp.eq(dates[1]))
            ].copy()
        original = source.copy(deep=True)
        index_root = StorageLayout(tmp_path / "index" / case).interim_data_root(
            "csmar", "index_market"
        )
        if expected_status == "WRITE_ERROR":
            with pytest.raises(ValueError, match="^dataset_infinite_value$"):
                write_dataset(
                    index_root, source, spec=_interim_specs().get("index_market")
                )
            assert not index_root.exists()
            pd.testing.assert_frame_equal(source, original, check_exact=True)
            assert tree_hash(good_root) == good_hash
            continue
        write_dataset(index_root, source, spec=_interim_specs().get("index_market"))
        before = tree_hash(index_root)
        kwargs = dict(
            ledgers=ledgers,
            candidates=candidates,
            config=manifest,
            time_window=TimeWindow(dates[0], dates[-1]),
            unified_trading_sessions=lambda calendar: pd.DatetimeIndex(
                calendar.calendar_date
            ),
            inputs=_holding_paths(
                {
                    **roots,
                    "index_market": {"path": index_root, "columns": ["open", "close"]},
                }
            ),
        )
        if case == "derived_and_native":
            excluded = {
                **kwargs,
                "inputs": {
                    **kwargs["inputs"],
                    "index_market": {"path": index_root, "columns": ["close"]},
                },
            }
            with pytest.raises(
                ValueError, match="portfolio_input_business_columns_excluded"
            ):
                build_portfolio_statistics_tables(
                    iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
                    **excluded,
                )
        owned = build_portfolio_statistics_tables(
            iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
            **kwargs,
        )
        contract = {
            **metadata("pv_all_portfolio_statistics"),
            "execution_mode": "continuous",
            "candidate_ids": [c.candidate_id for c in candidates],
            "comparison_portfolio_ids": config["statistics"]["group_comparison"][
                "portfolio_ids"
            ],
        }
        root = write_statistics_artifact(
            tmp_path / "statistics" / case,
            metadata=contract,
            tables=owned,
            table_group="portfolio",
        )
        artifact = read_statistics_artifact(root)
        with artifact.table("portfolio_attribution.parquet") as attribution:
            for candidate in candidates:
                if candidate.benchmark_id is None:
                    continue
                row = attribution.loc[
                    attribution.candidate_id.eq(candidate.candidate_id)
                    & attribution.component_id.eq("benchmark")
                ].iloc[0]
                if candidate.benchmark_id == "csi500" and expected_status in {
                    "BLOCKED",
                    "ERROR",
                }:
                    assert row.status in {"BLOCKED", "UNDEFINED"}
                    assert row.reason and pd.isna(row.value)
                else:
                    expected = (
                        Fraction(71, 1000)
                        if candidate.benchmark_id == "csi500"
                        else Fraction(1, 20)
                    )
                    assert row.status == "DEFINED"
                    assert row.value == pytest.approx(float(expected), abs=1e-14)
        with artifact.table("portfolio_daily.parquet") as daily:
            assert daily.return_status.eq("DEFINED").all()
            assert daily.net_return.to_numpy() == pytest.approx(0.04)
        if case == "derived_and_native":
            good_root, good_hash = root, tree_hash(root)
        assert tree_hash(index_root) == before
        pd.testing.assert_frame_equal(source, original, check_exact=True)
        assert tree_hash(good_root) == good_hash
    assert {name: tree_hash(root) for name, root in roots.items()} == input_hashes
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, originals[name], check_exact=True)


@pytest.mark.parametrize(
    "kind", ("registration", "dividend", "stock_distribution", "zero_recovery")
)
def test_registration_bookkeeping_preserves_holding_attribution(tmp_path, kind):
    from dataclasses import replace
    from fractions import Fraction

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        PortfolioCandidateSpec,
        PortfolioCapital,
        PortfolioDiagnostic,
        PortfolioLedgerBatch,
        PortfolioValuation,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.accounting import (
        EquityEntitlements,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from quant_research_platform.research.portfolios.risk import (
        FactorReturnFit,
        HoldingFactorInputs,
    )

    dates = pd.date_range("2024-01-01", periods=3)
    events = pd.DataFrame(
        [
            dict(
                event_id="cash",
                asset_id="a",
                listing_date=pd.Timestamp("2000-01-01"),
                kind="stock_distribution"
                if kind == "stock_distribution"
                else "cash_dividend",
                record_at=dates[2] if kind == "registration" else dates[1],
                effective_at=pd.Timestamp("2024-01-05")
                if kind == "registration"
                else dates[2],
                settlement_at=pd.Timestamp("2024-01-05"),
                available_at=dates[0],
                cash_per_share=1.0,
                additional_shares_per_share=0.5
                if kind == "stock_distribution"
                else 0.0,
                evidence_identity="a" * 64,
                entitlement_proved=True,
            )
        ]
    )
    entitlements = EquityEntitlements()
    units = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    registration = entitlements.register(
        events,
        units,
        lives,
        dates[2] if kind == "registration" else dates[1],
        phase="end" if kind == "registration" else "after",
    )
    if kind in {"dividend", "stock_distribution"}:
        _units, _lives, _cash, effective = entitlements.process(
            events, pd.DataFrame(), units, lives, dates[2], phase="end"
        )
        registration += effective
    if kind == "zero_recovery":
        registration = ()
    vals = tuple(
        v
        for phase, at, price in [
            ("before", dates[1], 0.0 if kind == "zero_recovery" else 10.0),
            ("after", dates[1], 0.0 if kind == "zero_recovery" else 10.0),
            ("end", dates[2], 1.0 if kind == "zero_recovery" else 11.0),
        ]
        for v in (
            PortfolioValuation(
                at.value,
                "a",
                phase,
                Fraction(
                    150 if kind == "stock_distribution" and phase == "end" else 100
                ),
                (150 if kind == "stock_distribution" and phase == "end" else 100)
                * price
                + (100 if kind == "dividend" and phase == "end" else 0),
                price,
                lifecycle_id="2000-01-01",
                receivable=100 if kind == "dividend" and phase == "end" else 0,
                locked_quantity=Fraction(50)
                if kind == "stock_distribution" and phase == "end"
                else Fraction(),
            ),
            PortfolioValuation(
                at.value,
                "__cash__",
                phase,
                1000.0 if kind == "zero_recovery" else 0.0,
                1000.0 if kind == "zero_recovery" else 0.0,
            ),
        )
    )
    diags = tuple(
        PortfolioDiagnostic(asset, None, name, number=n)
        for asset, target in [
            ("a", 0.0 if kind == "zero_recovery" else 1000.0),
            ("__cash__", 1000.0 if kind == "zero_recovery" else 0.0),
        ]
        for name, n in [("target_notional", target), ("decision_delta_weight", 0.0)]
    )
    diags += (
        PortfolioDiagnostic("__cash__", None, "decision_reason", text="simple_rule"),
    )
    batch = PortfolioLedgerBatch(0, (), (), vals, diags, registration)
    fit = FactorReturnFit(
        pd.DataFrame({"style": [1.0]}, index=["a"]),
        pd.Series({"style": 0.1}),
        pd.Series({"a": 0.0}),
        pd.Series({"a": 0.1}),
    )
    inputs = HoldingFactorInputs(
        *dates, fit, pd.Series({"a": 10.0}), pd.Series({"a": 10.0}), pd.DataFrame()
    )
    candidate = PortfolioCandidateSpec(
        "proof", "global", "quantile_equal_weight", "explicit", "daily", 0.2, None
    )
    results = []
    cases = (
        (replace(batch, rights_adjustments=()), batch)
        if kind == "registration"
        else (batch,)
    )
    for index, b in enumerate(cases):
        root = write_portfolio_ledger_artifact(
            tmp_path / str(index),
            capital=PortfolioCapital("proof", 1000.0, "CNY", "cash_equity"),
            metadata={
                "window": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()}
            },
            produce=lambda emit, b=b: emit(b),
        )
        ledger = read_portfolio_ledger_artifact(root)
        tables = _portfolio_statistics_tables(
            ledgers={"proof": ledger},
            candidates=(candidate,),
            config={"analyses": ["attribution"], "sample_role": "exploratory"},
            time_window=TimeWindow(dates[0], dates[-1]),
            benchmark_opens=pd.DataFrame(),
            calendar=dates,
            unified_trading_sessions=lambda x: x,
            holding_inputs={(): [inputs]},
        )
        results.append(tables)
    if kind != "registration":
        style = (
            results[0]["portfolio_attribution"].set_index("component_id").loc["style"]
        )
        assert pd.isna(style["value"])
        assert style["status"] == "UNDEFINED"
        assert style["reason"] == (
            "zero_valuation_recovery"
            if kind == "zero_recovery"
            else "corporate_action_adjustment"
        )
        return
    for table in ("portfolio_attribution", "portfolio_exposures"):
        pd.testing.assert_frame_equal(results[0][table], results[1][table])
    style = results[1]["portfolio_attribution"].set_index("component_id").loc["style"]
    assert style["value"] == pytest.approx(0.1)
    assert style["status"] == "DEFINED"
    assert not results[1]["portfolio_exposures"].empty
