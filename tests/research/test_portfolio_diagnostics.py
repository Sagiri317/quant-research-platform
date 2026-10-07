from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.targets import unmeasured_operation_evidence


def test_extreme_valid_ledgers_keep_finite_group_bootstrap(tmp_path):
    from fractions import Fraction

    import numpy as np

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.portfolios import (
        CONTINUOUS_EXECUTION_COLUMNS,
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.portfolios.accounting import (
        portfolio_interval_economics,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _group_portfolio_comparison,
    )
    from tests.helpers.portfolio_ledger import write_test_sessions

    gross = [2.0**-53, *([2.0**1020, *([2.0**-51] * 20)] * 17)]
    dates = pd.date_range("2024-01-01", periods=len(gross) + 1)
    columns = {}
    for group in ("global", "quarterly"):
        candidate = f"{group}_extreme"
        envelope = build_business_manifest_envelope(
            logical_artifact_id=candidate,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window={
                "start": str(dates[0].date()),
                "end": str(dates[-1].date()),
                "access": "holdout",
            },
            upstream_identities={},
            semantic_metadata={
                "portfolio_capital": {
                    "candidate_id": candidate,
                    "initial_capital": 1e8,
                    "valuation_currency": "CNY",
                    "valuation_method": "adjusted_notional",
                }
            },
        )
        frames = []
        nav = 1e8
        for index, factor in enumerate(gross):
            invested = group == "quarterly"
            end = nav * factor if invested else nav
            records = []
            for asset in ("asset", "__cash__"):
                cash = asset == "__cash__"
                before = (
                    nav
                    if (cash and (not invested or index == 0))
                    or (not cash and invested and index > 0)
                    else 0.0
                )
                after = nav if cash != invested else 0.0
                row = dict.fromkeys(CONTINUOUS_EXECUTION_COLUMNS, np.nan)
                row.update(
                    decision_timestamp=dates[index],
                    execution_date=dates[index],
                    asset_id=asset,
                    adv20=1e10,
                    pre_trade_notional=before,
                    target_notional=after,
                    executed_delta_notional=nav
                    if invested and index == 0 and not cash
                    else 0.0,
                    post_trade_notional=after,
                    holding_return=factor - 1.0 if invested and not cash else 0.0,
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
            tmp_path / candidate, envelope=envelope, ledger_chunks=iter(frames)
        )
        artifact = read_portfolio_ledger_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with closing_iterator(artifact.iter_chunks()) as batches:
            intervals = pd.DataFrame(
                [portfolio_interval_economics(batch) for batch in batches]
            )
        assert len(intervals) == 358
        assert intervals.nav_end.gt(0).all() and np.isfinite(intervals.nav_end).all()
        columns[candidate] = intervals.set_index(["entry_at", "end_at"]).net_return
    matrix = pd.DataFrame(columns)
    expected = float(sum(map(Fraction, matrix.quarterly_extreme)) / len(matrix))
    assert expected == pytest.approx(5.335332278746398e305)
    result = _group_portfolio_comparison(
        matrix,
        {
            "access_window": "holdout",
            "statistics": {
                "hac_lag": 19,
                "bootstrap_block": 20,
                "bootstrap_repetitions": 10_000,
                "group_comparison": {
                    "portfolio_ids": ["extreme"],
                    "primary_portfolio_id": "extreme",
                    "fdr_method": "BY",
                    "alpha": 0.05,
                    "seed": 42,
                    "confidence": 0.95,
                },
            },
        },
    )
    row = result.iloc[0]
    assert row.status == row.bootstrap_status == "DEFINED"
    assert row.mean_difference == expected
    assert row.standard_error == pytest.approx(3.6692201368766146e304)
    assert np.isfinite([row.bootstrap_lower, row.bootstrap_upper]).all()
    assert row.bootstrap_lower <= row.bootstrap_upper


def test_simple_only_configuration_reads_its_persisted_risk_context(tmp_path):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.profiles.research import (
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        iter_holding_factor_inputs,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.research_configuration import research_configuration
    from tests.workflows.test_portfolio_workflow_integration import (
        _frames,
        _processed_specs,
    )

    config = research_configuration(name="pv_expanded")
    profile = bind_research_configuration(research_profile(), config)
    protocol = profile.protocol(config["scenario_id"])
    candidates = protocol.portfolio_candidates()
    assert candidates and all(
        candidate.benchmark_id is None for candidate in candidates
    )
    candidate = candidates[0]
    statistics_target = next(
        target
        for target in protocol.target_specs("research run")
        if target.target_id == "pv_expanded_portfolio_statistics"
    )
    assert set(candidate.risk_context_target_ids) <= set(statistics_target.dependencies)
    dates = pd.bdate_range("2020-01-02", periods=3)
    panel = _frames(dates)["market_panel_1d"]
    layout = StorageLayout(tmp_path)
    roots = {}
    specs = (
        _processed_specs().get("market_panel_1d"),
        *(
            profile.processed.get(target)
            for target in candidate.risk_context_target_ids
        ),
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
    source = iter_holding_factor_inputs(
        layout=layout.with_target_roots(roots),
        calendar=pd.DataFrame({"calendar_date": dates}),
        time_window=TimeWindow(dates[0], dates[-1]),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        style_columns=candidate.risk_style_columns,
        risk_context_target_ids=candidate.risk_context_target_ids,
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
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.portfolios import (
        CONTINUOUS_EXECUTION_COLUMNS,
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.profiles.research import (
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from tests.helpers.portfolio_ledger import write_test_sessions
    from tests.helpers.research_configuration import research_configuration

    selected_protocol = bind_research_configuration(
        research_profile(), research_configuration(name="pv25")
    ).protocol("pv_all")
    candidates = selected_protocol.portfolio_candidates()
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
        envelope = build_business_manifest_envelope(
            logical_artifact_id=candidate.candidate_id,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window={
                "start": "2023-01-02",
                "end": "2023-01-06",
                "access": "development",
            },
            upstream_identities={},
            semantic_metadata={
                "portfolio_capital": {
                    "candidate_id": candidate.candidate_id,
                    "initial_capital": 1000.0,
                    "valuation_currency": "CNY",
                    "valuation_method": "adjusted_notional",
                }
            },
        )
        root = write_test_sessions(
            tmp_path / candidate.candidate_id,
            envelope=envelope,
            ledger_chunks=iter(frames),
        )
        artifacts[candidate.candidate_id] = read_portfolio_ledger_artifact(
            root, expected_identity=envelope.instance_identity
        )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="pv_all_portfolio_statistics",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-02",
            "end": "2023-01-06",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )
    from quant_research_platform.research.portfolios.risk import (
        FactorRiskState,
        HoldingFactorInputs,
    )
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS

    rng = np.random.default_rng(179)
    assets = ["a", *[f"asset_{i:03d}" for i in range(1, 100)]]
    inputs = []
    for index, frame in enumerate(frames):
        entry = pd.Timestamp(frame.execution_date.iloc[0])
        cross = pd.DataFrame(
            {
                "timestamp": entry - pd.Timedelta(days=1),
                "asset_id": assets,
                "industry_standard_id": "CSRC2012",
                "industry_level1_code": ["A01"] * 50 + ["B01"] * 50,
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
    original_eq = pd.Series.eq

    def counted_eq(series, other, *args, **kwargs):
        if series.name == "benchmark_id" and len(series) == len(benchmark):
            benchmark_scans.append(len(series))
        return original_eq(series, other, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "eq", counted_eq)
    owned = _portfolio_statistics_tables(
        ledgers=artifacts,
        candidates=candidates,
        protocol_manifest=selected_protocol.manifest(),
        time_window=TimeWindow("2023-01-02", "2023-01-06"),
        benchmark_opens=benchmark,
        holding_inputs={
            candidate.risk_style_columns: iter(inputs) for candidate in candidates
        },
        calendar=pd.DataFrame(
            {"calendar_date": pd.bdate_range("2023-01-02", "2023-01-06")}
        ),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
    )
    assert benchmark_scans == [len(benchmark), len(benchmark)]
    assert owned["portfolio_daily"].candidate_id.tolist() == [
        candidate.candidate_id for _ in frames for candidate in candidates
    ]
    assert (
        not owned["portfolio_daily"]
        .duplicated(["candidate_id", "entry_at", "end_at"])
        .any()
    )
    for candidate in candidates:
        actual = owned["portfolio_daily"].loc[
            owned["portfolio_daily"].candidate_id.eq(candidate.candidate_id)
        ]
        if candidate.benchmark_id is not None:
            np.testing.assert_allclose(
                actual.relative_wealth,
                np.array([1.04, 1.005, 1.035]) / np.array([1.01, 1.03, 1.04]),
                rtol=1e-15,
                atol=0,
            )
        else:
            assert actual.relative_wealth.isna().all()
    participation = selected_protocol.manifest()["execution_assumptions"][
        "adv_sensitivities"
    ]
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
        envelope=envelope,
        tables=owned,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
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
            protocol_manifest=selected_protocol.manifest(),
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
            started.append(self.identity)
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
                closed.append(self.identity)

        with monkeypatch.context() as patch:
            patch.setattr(PortfolioLedgerArtifact, "iter_chunks", altered)
            if mode == "missing_adv":
                owned = produce()
                assert owned["portfolio_capacity"].status.eq("BLOCKED").all()
                assert owned["portfolio_capacity_summary"].status.eq("BLOCKED").all()
            else:
                expected_error = (
                    KeyboardInterrupt if mode == "interrupt" else ValueError
                )
                with pytest.raises(expected_error):
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
        envelope=envelope,
        tables=owned,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table("holding_factor_returns.parquet") as rows:
        assert rows.empty
    with artifact.table("portfolio_exposures.parquet") as rows:
        assert rows.empty
    with artifact.table("portfolio_performance.parquet") as rows:
        assert (
            rows.loc[rows.metric_id.eq("tracking_error"), "status"].eq("BLOCKED").all()
        )
        assert rows.loc[rows.metric_id.eq("net_pnl"), "status"].eq("DEFINED").all()
    from quant_research_platform.artifacts.research import load_a_share_run_evidence
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.targets import (
        ProducerKind,
        ResearchRunRecord,
        ResearchRunTargetResult,
        RunStatus,
        TargetStatus,
    )
    from quant_research_platform.reports.a_share import build_a_share_report
    from quant_research_platform.storage.layout import StorageLayout

    manifest = selected_protocol.manifest()
    record = ResearchRunRecord(
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
        run_id="run_portfolio_statistics",
        profile_id="fixture",
        scenario_id="pv_all",
        protocol=manifest,
        protocol_identity=canonical_json_hash(manifest),
        research_window=envelope.research_window,
        software_identity="c" * 64,
        terminal_targets=("pv_all_portfolio_statistics",),
        target_results=(
            ResearchRunTargetResult(
                target_id="pv_all_portfolio_statistics",
                producer_kind=ProducerKind.STATISTIC,
                status=TargetStatus.BUILT,
                instance_identity=envelope.instance_identity,
                schema_identity=envelope.schema_identity,
                artifact_root="data/artifacts/a_share/pv_all_portfolio_statistics",
            ),
        ),
        status=RunStatus.COMPLETE,
    )
    evidence = load_a_share_run_evidence(
        layout=StorageLayout(tmp_path), record=record, selections={}
    )
    report = build_a_share_report(record=record, evidence=evidence)
    assert {
        row["candidate_id"] for row in report["sections"]["portfolio"]["results"]
    } == {candidate.candidate_id for candidate in candidates}
    assert "artifact_status" in evidence["portfolio_statistics"], evidence[
        "portfolio_statistics"
    ]
    assert evidence["portfolio_statistics"]["artifact_status"] == "DEFINED"
    capacity_detail = report["sections"]["capacity"]["detail"]
    assert capacity_detail == {
        "artifact_identity": envelope.instance_identity,
        "table": "portfolio_capacity.parquet",
        "key_columns": ["candidate_id", "entry_at", "participation"],
        "value_column": "bottleneck_asset_id",
    }
    with artifact.table(capacity_detail["table"]) as capacity_table:
        assert set(capacity_detail["key_columns"]) | {
            capacity_detail["value_column"]
        } <= set(capacity_table.columns)
    assert report["sections"]["risk"]["status"] == "BLOCKED"
    exposures = report["sections"]["risk"]["exposures"]
    assert exposures["artifact_identity"] == envelope.instance_identity
    assert exposures["table"] == "portfolio_exposures.parquet"
    with artifact.table(exposures["table"]) as exposure_table:
        expected = exposure_table.loc[
            exposure_table.entry_at.eq(
                exposure_table.groupby("candidate_id").entry_at.transform("max")
            )
        ]
        actual = pd.DataFrame(exposures["latest"])
        assert len(actual) == len(expected)
        if not actual.empty:
            keys = ["candidate_id", "factor_id"]
            np.testing.assert_allclose(
                actual.sort_values(keys).value, expected.sort_values(keys).value
            )
            assert actual.factor_id.str.startswith("industry_").any()
            assert (~actual.factor_id.str.startswith("industry_")).any()
    assert report["source_run_status"] == "COMPLETE"
    blocker = "global_development_selection_reuses_validation"
    sections = report["sections"]
    assert sections["portfolio"]["development_robustness_groups"] == ["quarterly"]
    assert sections["OOS prediction"]["global"]["inference_qualification"] == {
        "status": "UNDEFINED",
        "reason": blocker,
    }
    assert sections["OOS prediction"]["quarterly"]["inference_qualification"] == {
        "status": "ELIGIBLE",
        "reason": None,
    }
    robustness = pd.DataFrame(evidence["portfolio_statistics"]["robustness"])
    blocked_ids = [
        item.candidate_id for item in candidates if item.selection_group == "global"
    ]
    global_rows = robustness.loc[robustness.candidate_id.isin(blocked_ids)]
    assert len(global_rows) == 10 and global_rows.status.eq("UNDEFINED").all()
    assert global_rows.reason.eq(blocker).all() and global_rows.value.isna().all()
    assert robustness.candidate_count.eq(5).all()
    assert sections["portfolio"]["paired_comparison"]
    assert all(
        row["reason"] == row["bootstrap_reason"] == blocker
        for row in sections["portfolio"]["paired_comparison"]
    )
    assert any(row.get("test_id") == "psr" for row in report["sections"]["robustness"])


def test_holding_input_reader_replays_real_history_and_freezes_decision_universe(
    tmp_path,
):
    import numpy as np

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.portfolios.diagnostics import (
        iter_holding_factor_inputs,
    )
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
    from quant_research_platform.storage.datasets import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.research_specs import make_demo_research_dataset_spec
    from tests.workflows.test_portfolio_workflow_integration import _processed_specs

    dates = pd.bdate_range("2017-01-02", periods=350)
    assets = [f"asset_{index:03d}" for index in range(105)]
    rng = np.random.default_rng(818)
    returns = rng.normal(scale=0.008, size=(350, 1)) * rng.uniform(
        0.5, 1.5, size=(1, 105)
    ) + rng.normal(size=(350, 105)) * np.linspace(0.003, 0.009, 105)
    prices = 100 * np.cumprod(1 + returns, axis=0)
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "timestamp": dates.repeat(105),
            "asset_id": assets * 350,
            "adj_open": prices.ravel(),
            "adj_close": prices.ravel(),
            "amount": 1000000.0,
            "market_type": 1,
            "period_return": returns.ravel(),
            "float_market_cap": np.tile(np.geomspace(100.0, 1000000.0, 105), 350),
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": np.tile(["A01"] * 52 + ["B01"] * 53, 350),
        }
    )
    panel["industry_level1_code"] = np.tile(["A01"] * 52 + ["B01"] * 53, 350)
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
    data = panel[["market_id", "timestamp", "asset_id"]].copy()
    data["eligible"] = True
    data["label"] = 0.0
    for name in styles:
        data[name] = rng.normal(size=len(data))
    data.loc[data.asset_id.eq(assets[-1]), styles[0]] = np.nan
    risk_data = data.loc[:, ["market_id", "timestamp", "asset_id", *styles]].copy()
    data = data.drop(columns=list(styles)).assign(signal=1.0)
    data = data.loc[data.timestamp.ne(dates[10])].copy()
    spec = make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=("signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=(),
    )
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
        "fixture": base_layout.research_dataset_root("a_share", "fixture"),
        "benchmark_weights": base_layout.processed_data_root(
            "a_share",
            _processed_specs().get("benchmark_weights").kind,
            "benchmark_weights",
        ),
    }
    write_dataset(
        roots["market_panel_1d"], panel, spec=_processed_specs().get("market_panel_1d")
    )
    write_dataset(roots["fixture"], data, spec=spec)
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
    layout = StorageLayout(tmp_path).with_target_roots(roots)
    calendar = pd.DataFrame({"calendar_date": dates})
    window = TimeWindow(dates[-4], dates[-1])
    source = iter_holding_factor_inputs(
        benchmark_required=True,
        layout=layout,
        calendar=calendar,
        time_window=window,
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        style_columns=RISK_STYLE_COLUMNS,
        risk_context_target_ids=risk_targets,
    )
    with closing_iterator(source):
        first = next(source)
        assert first.reason is None and first.fitted is not None
        assert len(first.fitted.asset_returns) == 104
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
        first = None

    def first_for_styles(selected_styles, selected_layout):
        stream = iter_holding_factor_inputs(
            benchmark_required=True,
            layout=selected_layout,
            calendar=calendar,
            time_window=window,
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
            style_columns=selected_styles,
            risk_context_target_ids=risk_targets
            if len(selected_styles) == 6
            else tuple(
                target for target in risk_targets if target != "fundamental_factors"
            ),
        )
        with closing_iterator(stream):
            return next(stream)

    five = first_for_styles(RISK_STYLE_COLUMNS[:-1], layout)
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
    missing_earnings_layout = missing_earnings_layout.with_target_roots(
        {**roots, "fundamental_factors": missing_earnings_root}
    )
    five_after = first_for_styles(RISK_STYLE_COLUMNS[:-1], missing_earnings_layout)
    pd.testing.assert_series_equal(
        five.fitted.factor_returns, five_after.fitted.factor_returns, check_exact=True
    )
    pd.testing.assert_frame_equal(
        five.fitted.exposures, five_after.fitted.exposures, check_exact=True
    )
    six = first_for_styles(RISK_STYLE_COLUMNS, missing_earnings_layout)
    assert six.fitted is None and six.reason == "holding_factor_fit_unidentified"

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
    missing_layout = missing_layout.with_target_roots(
        {**roots, "market_panel_1d": missing_root}
    )
    source = iter_holding_factor_inputs(
        benchmark_required=True,
        layout=missing_layout,
        calendar=calendar,
        time_window=window,
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        style_columns=RISK_STYLE_COLUMNS,
        risk_context_target_ids=risk_targets,
    )
    with closing_iterator(source):
        first = next(source)
        assert first.fitted is None
        assert first.reason == "holding_returns_incomplete"
        first = None


@pytest.mark.parametrize(
    ("constant", "swap_group_metadata"), ((False, False), (True, False), (False, True))
)
def test_mixed_group_ledgers_publish_qualified_universe_robustness(
    tmp_path, constant, swap_group_metadata
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
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.portfolios import (
        CONTINUOUS_EXECUTION_COLUMNS,
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.profiles.research import (
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
    from tests.helpers.portfolio_ledger import write_test_sessions
    from tests.helpers.research_configuration import research_configuration

    dates = pd.bdate_range("2023-01-02", periods=75)
    selected_protocol = bind_research_configuration(
        research_profile(), research_configuration(name="pv25")
    ).protocol("pv_all")
    candidates = selected_protocol.portfolio_candidates()
    if swap_group_metadata:
        targets = dict(
            zip(
                ("global", "quarterly"),
                selected_protocol.prediction_target_ids,
                strict=True,
            )
        )
        candidates = tuple(
            replace(
                candidate,
                selection_group=other,
                prediction_target_id=targets[other],
            )
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
    reason = "global_development_selection_reuses_validation"
    returns = np.random.default_rng(802).normal(0.0002, 0.008, (73, 5))
    if constant:
        returns[:, 0] = 0.1
    window = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "access": "development",
    }

    def persisted_ledger(candidate, series, destination):
        envelope = build_business_manifest_envelope(
            logical_artifact_id=candidate.candidate_id,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window=window,
            upstream_identities={},
            semantic_metadata={
                "fixture_returns": series.tolist(),
                "portfolio_capital": {
                    "candidate_id": candidate.candidate_id,
                    "initial_capital": 1000.0,
                    "valuation_currency": "CNY",
                    "valuation_method": "adjusted_notional",
                },
            },
        )
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
            destination, envelope=envelope, ledger_chunks=iter(frames)
        )
        return read_portfolio_ledger_artifact(
            root, expected_identity=envelope.instance_identity
        )

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
    envelope = build_business_manifest_envelope(
        logical_artifact_id="pv_all_portfolio_statistics",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window=window,
        upstream_identities={},
        semantic_metadata={},
    )

    def produce(selected_ledgers, *, robustness=False):
        manifest = selected_protocol.manifest()
        if robustness:
            manifest["expectations"] = {"freshness_days": 180, "role": "robustness"}
        return _portfolio_statistics_tables(
            ledgers=selected_ledgers,
            candidates=candidates,
            protocol_manifest=manifest,
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
        envelope=envelope,
        tables=owned,
        table_group="portfolio",
    )
    corrupt = dict(owned)
    corrupt["portfolio_daily"] = pd.concat(
        [corrupt["portfolio_daily"], corrupt["portfolio_daily"].iloc[:1]],
        ignore_index=True,
    )
    rejected = build_business_manifest_envelope(
        logical_artifact_id="pv_all_portfolio_statistics",
        schema_identity="d" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window=window,
        upstream_identities={},
        semantic_metadata={},
    )
    with pytest.raises(ValueError, match="statistics_table_duplicate_keys"):
        write_statistics_artifact(
            tmp_path / "statistics",
            envelope=rejected,
            tables=corrupt,
            table_group="portfolio",
        )
    assert not (root.parent / rejected.instance_identity).exists()
    corrupt = None
    artifact = read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table("portfolio_robustness.parquet") as rows:
        assert len(rows) == 21 and rows.candidate_count.eq(5).all()
        global_rows = rows.loc[rows.candidate_id.isin(blocked_ids)]
        assert len(global_rows) == 10
        assert global_rows.status.eq("UNDEFINED").all()
        assert global_rows.reason.eq(reason).all()
        assert global_rows.value.isna().all()
        baseline_qualified = (
            rows.loc[rows.candidate_id.isin([*qualified_ids, "joint"])]
            .reset_index(drop=True)
            .copy()
        )
    with artifact.table("portfolio_group_comparison.parquet") as rows:
        assert len(rows) == 5
        assert rows.status.eq("UNDEFINED").all()
        assert rows.reason.eq(reason).all()
        assert rows.bootstrap_status.eq("UNDEFINED").all()
        assert rows.bootstrap_reason.eq(reason).all()
        assert (
            rows[["p_value", "bootstrap_lower", "bootstrap_upper"]].isna().all().all()
        )
        assert rows.global_mean.notna().all() and rows.quarterly_mean.notna().all()
    if constant:
        with artifact.table("portfolio_performance.parquet") as rows:
            selected = rows.loc[
                rows.candidate_id.eq(qualified[0].candidate_id)
            ].set_index("metric_id")
            assert selected.loc["sharpe", "status"] == "UNDEFINED"
            assert selected.loc["annualized_volatility", "value"] == 0.0
        with artifact.table("portfolio_robustness.parquet") as rows:
            rows = rows.loc[rows.candidate_id.isin([*qualified_ids, "joint"])]
            assert len(rows) == 11
            assert rows.candidate_count.eq(5).all()
            assert rows.loc[rows.test_id.eq("dsr"), "status"].eq("UNDEFINED").all()
            assert rows.loc[rows.test_id.eq("cscv_pbo"), "status"].eq("UNDEFINED").all()
            psr = rows.loc[rows.test_id.eq("psr")].set_index("candidate_id")
            assert psr.loc[qualified[0].candidate_id, "status"] == "UNDEFINED"
            assert psr.loc[qualified_ids[1:], "status"].eq("DEFINED").all()
        return
    with artifact.table("portfolio_robustness.parquet") as rows:
        rows = rows.loc[rows.candidate_id.isin([*qualified_ids, "joint"])]
        assert len(rows) == 11 and rows.status.eq("DEFINED").all()
        sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
        reference = sharpes.std(ddof=1) * (
            (1 - np.euler_gamma) * norm.ppf(0.8)
            + np.euler_gamma * norm.ppf(1 - 1 / (5 * np.e))
        )
        for column, candidate in enumerate(qualified):
            series = returns[:, column]
            score = sharpes[column]
            se = np.sqrt(
                (
                    1
                    - skew(series, bias=False) * score
                    + (kurtosis(series, fisher=False, bias=False) - 1) * score**2 / 4
                )
                / 72
            )
            actual = (
                rows.loc[rows.candidate_id.eq(candidate.candidate_id)]
                .set_index("test_id")
                .value
            )
            assert actual.psr == pytest.approx(norm.cdf(score / se))
            assert actual.dsr == pytest.approx(norm.cdf((score - reference) / se))
        blocks = np.array_split(np.arange(73), 10)
        outcomes = []
        for chosen in combinations(range(10), 5):
            train = returns[np.concatenate([blocks[i] for i in chosen])]
            test = returns[
                np.concatenate([blocks[i] for i in range(10) if i not in chosen])
            ]
            winner = np.argmax(train.mean(axis=0) / train.std(axis=0, ddof=1))
            rank = rankdata(test.mean(axis=0) / test.std(axis=0, ddof=1))[winner]
            outcomes.append(rank / 6 <= 0.5)
        assert len(outcomes) == 252
        assert rows.loc[rows.test_id.eq("cscv_pbo"), "value"].iloc[0] == pytest.approx(
            np.mean(outcomes)
        )
    with artifact.table("portfolio_capacity.parquet") as rows:
        assert rows.empty
    with artifact.table("portfolio_capacity_summary.parquet") as rows:
        assert (
            rows.status.eq("UNDEFINED").all()
            and rows.reason.eq("no_rebalance_requests").all()
        )
    if not swap_group_metadata:
        robust = produce(ledgers, robustness=True)
        pd.testing.assert_frame_equal(
            robust["portfolio_performance"],
            owned["portfolio_performance"],
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            robust["portfolio_daily"], owned["portfolio_daily"], check_exact=True
        )
        pbo = robust["portfolio_robustness"].query("test_id == 'cscv_pbo'")
        assert pbo.status.eq("UNDEFINED").all()
        assert pbo.reason.eq("expectations_robustness_excluded").all()
        comparison = robust["portfolio_group_comparison"]
        assert comparison.reason.eq("expectations_robustness_excluded").all()
        assert comparison.global_mean.notna().all()
        robust_root = write_statistics_artifact(
            tmp_path / "robustness_statistics",
            envelope=envelope,
            tables=robust,
            table_group="portfolio",
        )
        with read_statistics_artifact(
            robust_root, expected_identity=envelope.instance_identity
        ).table("portfolio_robustness.parquet") as rows:
            assert (
                rows.loc[rows.test_id.eq("cscv_pbo"), "reason"]
                .eq("expectations_robustness_excluded")
                .all()
            )
        altered_ledgers = dict(ledgers)
        for column, candidate in enumerate(blocked):
            extreme = np.where(np.arange(73) % 2, -0.2, 0.5 + column / 10)
            altered_ledgers[candidate.candidate_id] = persisted_ledger(
                candidate, extreme, tmp_path / "changed_global" / candidate.candidate_id
            )
        changed = produce(altered_ledgers)
        changed_envelope = build_business_manifest_envelope(
            logical_artifact_id="pv_all_portfolio_statistics",
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window=window,
            upstream_identities={
                key: ledger.identity for key, ledger in altered_ledgers.items()
            },
            semantic_metadata={},
        )
        changed_root = write_statistics_artifact(
            tmp_path / "changed_statistics",
            envelope=changed_envelope,
            tables=changed,
            table_group="portfolio",
        )
        saved = read_statistics_artifact(
            changed_root, expected_identity=changed_envelope.instance_identity
        )
        with saved.table("portfolio_robustness.parquet") as changed_rows:
            pd.testing.assert_frame_equal(
                changed_rows.loc[
                    changed_rows.candidate_id.isin([*qualified_ids, "joint"])
                ].reset_index(drop=True),
                baseline_qualified,
                check_exact=True,
            )
        changed_rows = changed["portfolio_robustness"]
        global_rows = changed_rows.loc[changed_rows.candidate_id.isin(blocked_ids)]
        assert global_rows.status.eq("UNDEFINED").all()
        assert global_rows.reason.eq(reason).all()
        assert global_rows.value.isna().all()
        actual = changed["portfolio_daily"]
        for column, candidate in enumerate(blocked):
            selected = actual.loc[actual.candidate_id.eq(candidate.candidate_id)]
            np.testing.assert_allclose(
                selected.net_return,
                np.where(np.arange(73) % 2, -0.2, 0.5 + column / 10),
            )


def test_public_portfolio_statistics_derive_missing_csi_opens_and_preserve_sources(
    tmp_path,
):
    from fractions import Fraction

    import numpy as np

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.data import InterimDataSpec
    from quant_research_platform.contracts.portfolios import (
        CONTINUOUS_EXECUTION_COLUMNS,
        PORTFOLIO_LEDGER_COST_COLUMNS,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.profiles.research import (
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        build_portfolio_statistics_tables,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from quant_research_platform.storage.identity import (
        content_hash_for_files,
        filesystem_glob,
        filesystem_is_file,
    )
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.portfolio_ledger import write_test_sessions
    from tests.helpers.research_configuration import research_configuration
    from tests.workflows.test_portfolio_workflow_integration import (
        _frames,
        _interim_specs,
        _processed_specs,
    )

    dates = pd.bdate_range("2019-01-02", periods=3)
    frames = _frames(dates)
    originals = {name: frame.copy(deep=True) for name, frame in frames.items()}
    protocol = bind_research_configuration(
        research_profile(), research_configuration(name="pv25")
    ).protocol("pv_all")
    candidates = protocol.portfolio_candidates()
    manifest = protocol.manifest()
    window = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "access": "development",
    }

    def envelope(logical_id, *, metadata=None):
        return build_business_manifest_envelope(
            logical_artifact_id=logical_id,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window=window,
            upstream_identities={},
            semantic_metadata=metadata or {},
        )

    def tree_hash(root):
        return content_hash_for_files(
            (
                (path.relative_to(root).as_posix(), path)
                for path in filesystem_glob(root, "**/*")
                if filesystem_is_file(path)
            ),
            error="fixture_input_changed",
        )

    layout = StorageLayout(tmp_path)
    roots = {}
    for spec in (*_interim_specs().list(), *_processed_specs().list()):
        if spec.data_id == "index_market":
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
        contract = envelope(
            candidate.candidate_id,
            metadata={
                "portfolio_capital": {
                    "candidate_id": candidate.candidate_id,
                    "initial_capital": 1000.0,
                    "valuation_currency": "CNY",
                    "valuation_method": "adjusted_notional",
                }
            },
        )
        root = write_test_sessions(
            tmp_path / "ledgers" / candidate.candidate_id,
            envelope=contract,
            ledger_chunks=iter((ledger_frame,)),
        )
        ledgers[candidate.candidate_id] = read_portfolio_ledger_artifact(
            root, expected_identity=contract.instance_identity
        )
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
            source.loc[:, ["open", "close"]] *= (
                1e-200 if case == "scale_small" else 1e200
            )
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
            protocol_manifest=manifest,
            time_window=TimeWindow(dates[0], dates[-1]),
            unified_trading_sessions=lambda calendar: pd.DatetimeIndex(
                calendar.calendar_date
            ),
            layout=layout.with_target_roots({**roots, "index_market": index_root}),
        )
        if expected_status == "ERROR":
            with pytest.raises(ValueError, match="^benchmark_return_open_invalid$"):
                build_portfolio_statistics_tables(**kwargs)
        else:
            owned = build_portfolio_statistics_tables(**kwargs)
            contract = envelope(
                "pv_all_portfolio_statistics", metadata={"fixture_case": case}
            )
            root = write_statistics_artifact(
                tmp_path / "statistics",
                envelope=contract,
                tables=owned,
                table_group="portfolio",
            )
            artifact = read_statistics_artifact(
                root, expected_identity=contract.instance_identity
            )
            with artifact.table("portfolio_daily.parquet") as daily:
                for candidate in candidates:
                    if candidate.benchmark_id is None:
                        continue
                    row = daily.loc[daily.candidate_id.eq(candidate.candidate_id)].iloc[
                        0
                    ]
                    if (
                        candidate.benchmark_id == "csi500"
                        and expected_status == "BLOCKED"
                    ):
                        assert row.benchmark_status == "BLOCKED"
                        assert (
                            row.benchmark_reason
                            == "benchmark_total_return_open_missing"
                        )
                        assert pd.isna(row.benchmark_return) and pd.isna(
                            row.active_return
                        )
                    else:
                        expected = (
                            Fraction(71, 1000)
                            if candidate.benchmark_id == "csi500"
                            else Fraction(1, 20)
                        )
                        assert row.benchmark_status == "DEFINED"
                        assert row.benchmark_return == pytest.approx(
                            float(expected), abs=1e-14
                        )
                        assert row.active_return == pytest.approx(
                            float(Fraction(1, 25) - expected), abs=1e-14
                        )
                        assert row.relative_wealth == pytest.approx(
                            float(Fraction(26, 25) / (1 + expected)), abs=1e-14
                        )
            with artifact.table("portfolio_performance.parquet") as summary:
                for candidate in candidates:
                    if candidate.benchmark_id != "csi500":
                        continue
                    value = summary.loc[
                        summary.candidate_id.eq(candidate.candidate_id)
                        & summary.metric_id.eq("annualized_active_return")
                    ].iloc[0]
                    if expected_status == "BLOCKED":
                        assert value.status == "BLOCKED" and pd.isna(value.value)
                    else:
                        assert value.status == "DEFINED"
                        assert value.value == pytest.approx(
                            float(Fraction(-31, 1000) * 252), abs=1e-12
                        )
            if case == "derived_and_native":
                good_root, good_hash = (root, tree_hash(root))
        assert tree_hash(index_root) == before
        pd.testing.assert_frame_equal(source, original, check_exact=True)
        assert tree_hash(good_root) == good_hash
    assert {name: tree_hash(root) for name, root in roots.items()} == input_hashes
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, originals[name], check_exact=True)
