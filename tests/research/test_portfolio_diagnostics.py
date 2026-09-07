from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.targets import unmeasured_operation_evidence


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
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from quant_research_platform.research.protocols.a_share import protocol
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.portfolio_ledger import write_test_sessions

    candidates = protocol("pv_all").portfolio_candidates()
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
                block_reason="cash_pro_rata" if index == 0 and not cash else None,
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
                "industry_level1_code": ["A01"] * 50 + ["B01"] * 50,
                "float_market_cap": np.geomspace(100.0, 1e8, 100),
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
    with runtime_resource_policy_scope() as policy:
        owned = _portfolio_statistics_tables(
            ledgers=artifacts,
            candidates=candidates,
            protocol_manifest=protocol("pv_all").manifest(),
            time_window=TimeWindow("2023-01-02", "2023-01-06"),
            resource_policy=policy,
            benchmark_opens=benchmark,
            holding_inputs=iter(inputs),
            calendar=pd.DataFrame(
                {"calendar_date": pd.bdate_range("2023-01-02", "2023-01-06")}
            ),
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
        )
        try:
            root = write_statistics_artifact(
                tmp_path / "statistics",
                envelope=envelope,
                tables=owned.value,
                table_group="portfolio",
            )
        finally:
            owned.release(owner="research.portfolio.statistics.output")
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
            np.testing.assert_allclose(
                rows.gross_return - rows.cost_drag, rows.net_return
            )
            assert rows.nav_post.tolist() == [990.0, 1020.0, 1005.0]
            assert (
                (rows.nav_end - rows.nav_post).sum(),
                (rows.nav_end - rows.nav_before).sum(),
                (rows.cost_drag * rows.nav_before).sum(),
            ) == pytest.approx((65.0, 35.0, 30.0))
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
                detail["stock_residual_difference"]
                + detail["benchmark_replication_gap"]
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
        assert policy.used_memory_bytes == 0

    from dataclasses import replace

    from quant_research_platform.artifacts.portfolio_ledger import (
        PortfolioLedgerArtifact,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator

    def produce(policy, *, factor_inputs=inputs, benchmark_frame=benchmark):
        return _portfolio_statistics_tables(
            ledgers=artifacts,
            candidates=candidates,
            protocol_manifest=protocol("pv_all").manifest(),
            time_window=TimeWindow("2023-01-02", "2023-01-06"),
            resource_policy=policy,
            benchmark_opens=benchmark_frame,
            holding_inputs=iter(factor_inputs),
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

        def altered(self, mode=mode, closed=closed):
            try:
                with closing_iterator(original(self)) as source:
                    for index, item in enumerate(source):
                        if mode == "interrupt" and index == 1:
                            raise KeyboardInterrupt("fixture interruption")
                        if (mode == "missing_first" and index == 0) or (
                            mode == "missing_last" and index == 2
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

        with monkeypatch.context() as patch, runtime_resource_policy_scope() as policy:
            patch.setattr(PortfolioLedgerArtifact, "iter_chunks", altered)
            if mode == "missing_adv":
                owned = produce(policy)
                try:
                    assert owned.value["portfolio_capacity"].status.eq("BLOCKED").all()
                    assert (
                        owned.value["portfolio_capacity_summary"]
                        .status.eq("BLOCKED")
                        .all()
                    )
                finally:
                    owned.release(owner="research.portfolio.statistics.output")
            else:
                expected_error = (
                    KeyboardInterrupt if mode == "interrupt" else ValueError
                )
                with pytest.raises(expected_error):
                    produce(policy)
            assert closed and policy.used_memory_bytes == 0

    no_fit = [
        replace(item, fitted=None, reason="attribution_pit_history_missing")
        for item in inputs
    ]
    absent_benchmark = benchmark.assign(open=np.nan)
    with runtime_resource_policy_scope() as policy:
        owned = produce(policy, factor_inputs=no_fit, benchmark_frame=absent_benchmark)
        try:
            root = write_statistics_artifact(
                tmp_path / "data/artifacts/a_share/pv_all_portfolio_statistics",
                envelope=envelope,
                tables=owned.value,
                table_group="portfolio",
            )
        finally:
            owned.release(owner="research.portfolio.statistics.output")
        artifact = read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with artifact.table("holding_factor_returns.parquet") as rows:
            assert rows.empty
        with artifact.table("portfolio_exposures.parquet") as rows:
            assert rows.empty
        with artifact.table("portfolio_performance.parquet") as rows:
            assert (
                rows.loc[rows.metric_id.eq("tracking_error"), "status"]
                .eq("BLOCKED")
                .all()
            )
            assert rows.loc[rows.metric_id.eq("net_pnl"), "status"].eq("DEFINED").all()
        assert policy.used_memory_bytes == 0

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

    manifest = protocol("pv_all").manifest()
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
    from quant_research_platform.artifacts.research import report_payload_working_set

    with (
        runtime_resource_policy_scope() as report_policy,
        report_policy.lease(
            "reports.test_payload", report_payload_working_set(record.payload())
        ) as report_lease,
    ):
        evidence = load_a_share_run_evidence(
            layout=StorageLayout(tmp_path), record=record, report_lease=report_lease
        )
        report = build_a_share_report(record=record, evidence=evidence)
        assert "artifact_status" in evidence["portfolio_statistics"], evidence[
            "portfolio_statistics"
        ]
        assert evidence["portfolio_statistics"]["artifact_status"] == "DEFINED"
        assert report["sections"]["risk"]["status"] == "BLOCKED"
        assert report["source_run_status"] == "COMPLETE"
        assert any(
            row.get("test_id") == "psr" for row in report["sections"]["robustness"]
        )
    assert report_policy.used_memory_bytes == 0


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
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
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
            "amount": 1e6,
            "market_type": 1,
            "period_return": returns.ravel(),
            "float_market_cap": np.tile(np.geomspace(100.0, 1e6, 105), 350),
            "industry_level1_code": np.tile(["A01"] * 52 + ["B01"] * 53, 350),
        }
    )
    panel["industry_level1_code"] = np.tile(["A01"] * 52 + ["B01"] * 53, 350)
    panel["is_nonfinancial_industry"] = True
    styles = tuple(
        name
        for name in RISK_STYLE_COLUMNS
        if name not in {"beta_252d", "residual_volatility_60d"}
    )
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
    risk_targets = ("price_volume_facts", "fundamental_factors")
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
    with runtime_resource_policy_scope() as policy:
        source = iter_holding_factor_inputs(
            layout=layout,
            dataset_spec=spec,
            calendar=calendar,
            time_window=window,
            resource_policy=policy,
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
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
                and first.end_at == dates[-2]
            )
            expected = (
                panel.loc[panel.timestamp.eq(first.end_at)]
                .set_index("asset_id")
                .adj_open
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
        assert policy.used_memory_bytes == 0

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
    with runtime_resource_policy_scope() as policy:
        source = iter_holding_factor_inputs(
            layout=missing_layout,
            dataset_spec=spec,
            calendar=calendar,
            time_window=window,
            resource_policy=policy,
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
            style_columns=RISK_STYLE_COLUMNS,
            risk_context_target_ids=risk_targets,
        )
        with closing_iterator(source):
            first = next(source)
            assert first.fitted is None
            assert first.reason == "holding_returns_incomplete"
            first = None
        assert policy.used_memory_bytes == 0


def test_five_persisted_no_order_ledgers_publish_fixed_universe_robustness(tmp_path):
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
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )
    from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
    from quant_research_platform.research.protocols.a_share import protocol
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.portfolio_ledger import write_test_sessions

    dates = pd.bdate_range("2023-01-02", periods=75)
    candidates = protocol("pv_all").portfolio_candidates()
    returns = np.random.default_rng(802).normal(0.0002, 0.008, (73, 5))
    window = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "access": "development",
    }
    ledgers = {}
    for column, candidate in enumerate(candidates):
        envelope = build_business_manifest_envelope(
            logical_artifact_id=candidate.candidate_id,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window=window,
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
        frames = []
        nav = 1000.0
        for index in range(73):
            rows = []
            for asset in (f"asset_{column}", "__cash__"):
                cash = asset == "__cash__"
                value = 0.0 if cash else nav
                rate = 0.0 if cash else returns[index, column]
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
            nav *= 1 + returns[index, column]
        root = write_test_sessions(
            tmp_path / candidate.candidate_id,
            envelope=envelope,
            ledger_chunks=iter(frames),
        )
        ledgers[candidate.candidate_id] = read_portfolio_ledger_artifact(
            root, expected_identity=envelope.instance_identity
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
    with runtime_resource_policy_scope() as policy:
        owned = _portfolio_statistics_tables(
            ledgers=ledgers,
            candidates=candidates,
            protocol_manifest=protocol("pv_all").manifest(),
            time_window=TimeWindow(dates[0], dates[-1]),
            resource_policy=policy,
            benchmark_opens=pd.DataFrame(columns=("benchmark_id", "timestamp", "open")),
            calendar=pd.DataFrame({"calendar_date": dates}),
            unified_trading_sessions=lambda frame: pd.DatetimeIndex(
                frame.calendar_date
            ),
            holding_inputs=iter(inputs),
        )
        try:
            root = write_statistics_artifact(
                tmp_path / "statistics",
                envelope=envelope,
                tables=owned.value,
                table_group="portfolio",
            )
            corrupt = dict(owned.value)
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
        finally:
            owned.release(owner="research.portfolio.statistics.output")
        artifact = read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with artifact.table("portfolio_robustness.parquet") as rows:
            assert len(rows) == 11 and rows.status.eq("DEFINED").all()
            sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
            reference = sharpes.std(ddof=1) * (
                (1 - np.euler_gamma) * norm.ppf(0.8)
                + np.euler_gamma * norm.ppf(1 - 1 / (5 * np.e))
            )
            for column, candidate in enumerate(candidates):
                series = returns[:, column]
                score = sharpes[column]
                se = np.sqrt(
                    (
                        1
                        - skew(series, bias=False) * score
                        + (kurtosis(series, fisher=False, bias=False) - 1)
                        * score**2
                        / 4
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
            assert rows.loc[rows.test_id.eq("cscv_pbo"), "value"].iloc[
                0
            ] == pytest.approx(np.mean(outcomes))
        with artifact.table("portfolio_capacity.parquet") as rows:
            assert rows.empty
        with artifact.table("portfolio_capacity_summary.parquet") as rows:
            assert (
                rows.status.eq("UNDEFINED").all()
                and rows.reason.eq("no_rebalance_requests").all()
            )
        assert policy.used_memory_bytes == 0
