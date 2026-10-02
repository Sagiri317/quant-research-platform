from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.identity import content_hash_for_files
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


def test_interim_identity_receives_common_root_joined_with_source_id(
    tmp_path: Path,
) -> None:
    common_root = tmp_path / "raw"
    source_root = common_root / "csmar"
    source_root.mkdir(parents=True)
    marker = source_root / "market_bars.raw"
    marker.write_bytes(b"synthetic")
    observed: list[Path] = []
    data_id = "market_bars_1d"

    def source_identity(raw_root: Path, _data_id: str) -> str:
        observed.append(raw_root)
        return content_hash_for_files(
            ((data_id, raw_root / marker.name),),
            error="synthetic_source_identity_invalid",
        )

    profile = replace(
        research_profile(),
        producer_bindings={
            **research_profile().producer_bindings,
            data_id: ProducerBinding(
                ProducerKind.INTERIM,
                lambda **_: None,
                source_identity=source_identity,
            ),
        },
    )
    result = research.plan_targets(
        profile=profile,
        action="data materialize",
        terminal_targets=(data_id,),
        layout=StorageLayout(project_root=tmp_path),
        raw_root=common_root,
    )

    assert result["status"] == "COMPLETE"
    assert result["targets"][0]["decision"] == "BUILD"
    assert observed == [source_root]


def test_a_share_target_windows_are_shared_by_plan_build_and_reuse() -> None:
    profile = research_profile()
    protocol = profile.protocol("pv_all")
    requested = {
        "start": "2019-01-01",
        "end": "2019-12-31",
        "access": "development",
    }
    assert research._target_time_window(
        "research run",
        protocol,
        profile.interim.get("market_bars_1d"),
        requested,
    ) == TimeWindow("2011-01-01", "2023-12-31")
    assert research._target_time_window(
        "research run",
        protocol,
        profile.processed.get("market_panel_1d"),
        requested,
    ) == TimeWindow("2011-01-01", "2023-12-31")
    assert research._target_time_window(
        "research run",
        protocol,
        profile.research_datasets.get("pv_all"),
        requested,
    ) == TimeWindow("2014-01-01", "2023-12-31")
    requested_window = TimeWindow(requested["start"], requested["end"])
    assert (
        research._target_time_window(
            "research run",
            protocol,
            None,
            requested,
        )
        == requested_window
    )
    assert (
        research._target_time_window(
            "dataset build",
            protocol,
            profile.research_datasets.get("pv_all"),
            requested,
        )
        == requested_window
    )


def test_crypto_plan_preserves_source_system_io_failure(tmp_path, monkeypatch):
    import errno

    import pytest

    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_PANEL_DATA_ID,
    )
    from quant_research_platform.profiles.research import CRYPTO_PROFILE_ID
    from quant_research_platform.sources.binance import replay as binance_replay
    from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
    from tests.sources.binance.test_replay import _agg_trade, _write_run

    root = tmp_path / "native"
    _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    primary = OSError(errno.EIO, "controlled_io_failure")
    original = binance_replay.filesystem_stat

    def fail(path, *args, **kwargs):
        if path == root / "run.json":
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(binance_replay, "filesystem_stat", fail)
    with pytest.raises(OSError) as caught:
        research.plan_targets(
            profile=research_profile(CRYPTO_PROFILE_ID),
            action="data materialize",
            terminal_targets=(HFT_PANEL_DATA_ID,),
            layout=StorageLayout(tmp_path / "project"),
            raw_root=root,
            research_window={
                "start": "2026-08-27T00:00:00",
                "end": "2026-08-27T00:00:01",
                "access": "development",
            },
        )
    assert caught.value is primary


def test_plan_preserves_persisted_instance_system_io_failure(tmp_path, monkeypatch):
    import pytest

    from tests.sources.reference.test_interim import _write_reference_root

    raw = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(tmp_path)
    profile = research_profile()
    data_id = "a_share_fee_schedule"
    built = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw,
        start_date=None,
        end_date=None,
        data_ids=(data_id,),
    )
    assert built["target_results"][0]["status"] == "BUILT"
    arguments = dict(
        profile=profile,
        action="data materialize",
        terminal_targets=(data_id,),
        layout=layout,
        raw_root=raw,
    )
    planned = research.plan_targets(**arguments)
    assert planned["targets"][0]["decision"] == "REUSE"
    target = layout.instance_root(
        layout.interim_data_root("reference", data_id),
        planned["targets"][0]["requested_identity"],
    )
    primary = OSError(5, "synthetic_storage_io")
    original = research.filesystem_stat

    def stat(path, *args, **kwargs):
        if path == target:
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(research, "filesystem_stat", stat)
    with pytest.raises(OSError) as caught:
        research.plan_targets(**arguments)
    assert caught.value is primary


@pytest.mark.parametrize("target_kind", ("calibration", "ledger"))
@pytest.mark.parametrize("use_bound_root", (False, True))
def test_crypto_semantic_source_identity_and_producer_share_one_root(
    tmp_path, monkeypatch, target_kind, use_bound_root
):
    from quant_research_platform.profiles.research import bind_research_configuration
    from quant_research_platform.research.protocols.crypto_hft import (
        CryptoMakerCandidate,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.software_identity import (
        production_software_identity,
    )
    from tests.helpers.research_configuration import research_configuration

    common_root = tmp_path / "common"
    bound_root = tmp_path / "bound"
    for root, value in ((common_root, b"common"), (bound_root, b"bound")):
        root.mkdir()
        (root / "marker").write_bytes(value)
    expected_root = bound_root if use_bound_root else common_root
    configuration = research_configuration("crypto", "baseline")
    profile = bind_research_configuration(
        research_profile(configuration["profile_id"]),
        configuration,
        tardis_root=bound_root,
    )
    protocol = profile.protocol(configuration["scenario_id"], access_window="holdout")
    candidate = next(
        item
        for item in protocol.portfolio_candidates()
        if isinstance(item, CryptoMakerCandidate) and item.calibrated
    )
    calibration = profile.producer_bindings[candidate.calibration_target_id]
    panel = profile.producer_bindings["binance_futures_l2_panel_100ms"]
    assert calibration.raw_root == panel.raw_root == bound_root
    assert calibration.source_identity is panel.source_identity
    assert callable(calibration.source_identity)
    target_id = (
        candidate.calibration_target_id
        if target_kind == "calibration"
        else next(
            target.target_id
            for target in protocol.execution_targets()
            if target.candidate == candidate and target.execution_mode == "continuous"
        )
    )
    target = next(
        item
        for item in protocol.target_specs("research run")
        if item.target_id == target_id
    )
    seen = []
    primary = RuntimeError("controlled_producer_failure")

    def source_identity(raw_root, data_id):
        seen.append(("identity", raw_root))
        return content_hash_for_files(
            ((data_id, raw_root / "marker"),), error="synthetic_source_identity_invalid"
        )

    def produce(*, raw_root, **_):
        seen.append(("producer", raw_root))
        assert (raw_root / "marker").read_bytes() == (
            b"bound" if use_bound_root else b"common"
        )
        raise primary

    binding = replace(
        profile.producer_bindings[target_id],
        producer=produce,
        source_identity=source_identity,
        raw_root=bound_root if use_bound_root else None,
    )
    profile = replace(
        profile, producer_bindings={**profile.producer_bindings, target_id: binding}
    )
    if target_kind == "calibration":

        def write(_root, *, tables, **_):
            tables(on_table=lambda *_: None)

        monkeypatch.setattr(research, "write_statistics_artifact", write)
    else:

        def write(_root, *, produce, **_):
            produce(lambda *_: None)

        monkeypatch.setattr(research, "write_portfolio_ledger_artifact", write)
    layout = StorageLayout(tmp_path / "project")
    window = TimeWindow(**configuration["windows"]["holdout"])
    research_window = {**configuration["windows"]["holdout"], "access": "holdout"}
    upstream = {dependency: "1" * 64 for dependency in target.dependencies}
    software = production_software_identity()
    with runtime_resource_policy_scope() as policy:
        envelope = research._target_envelope_for_spec(
            profile=profile,
            protocol=protocol,
            target_spec=target,
            data_spec=None,
            research_window=research_window,
            upstream_identities=upstream,
            software_identity=software,
            raw_content_identity=source_identity(expected_root, target_id),
        )
        seen.clear()
        with pytest.raises(RuntimeError) as caught:
            research._build_semantic_target(
                profile=profile,
                protocol=protocol,
                layout=layout,
                spec=target,
                planned={
                    "requested_identity": envelope.instance_identity,
                    "upstream_identities": upstream,
                },
                target_layout=layout,
                policy=policy,
                time_window=window,
                research_window=research_window,
                software_identity=software,
                raw_root=common_root,
            )
        assert caught.value is primary
    assert seen == [("identity", expected_root), ("producer", expected_root)]
