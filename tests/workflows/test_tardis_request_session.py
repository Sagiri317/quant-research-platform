"""Native research publication keeps request-final cleanup failures visible."""

import inspect
from collections import Counter
from contextlib import contextmanager

import pytest

from quant_research_platform.artifacts.research import read_research_run_record
from quant_research_platform.sources.tardis import replay
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.workflows import operations
from tests.workflows.test_crypto_hft import _tardis_cross_midnight_case


@pytest.mark.parametrize("failure", [None, "close", "body_and_close"])
def test_native_request_shares_panel_and_thirty_takers_and_records_cleanup_failure(
    tmp_path, monkeypatch, failure
):
    raw, layout, profile, configuration = _tardis_cross_midnight_case(tmp_path, None)
    candidates = [c for c in configuration["portfolios"] if "model_id" in c][:10]
    assert len(candidates) == 10
    configuration["portfolios"] = [
        {
            **candidate,
            "candidate_id": f"shared_{index}_{latency}",
            "latency_ms": latency,
        }
        for index, candidate in enumerate(candidates)
        for latency in (0, 100, 200)
    ]
    developed = operations.research_run(
        profile=profile, layout=layout, raw_root=raw, research_plan=configuration
    )
    assert developed["status"] == "COMPLETE"
    original_open = replay.open_tardis_archive
    original_decode = replay.iter_native_slice
    original_temporary = replay.TemporaryFile
    leases = {}
    decode_counts = Counter()
    buffers = []
    primary = OSError("controlled_native_body_failure")
    cleanup = OSError("controlled_request_buffer_close_failure")

    @contextmanager
    def observed_open(*args, **kwargs):
        frame = inspect.currentframe().f_back
        target = None
        consumer = False
        while frame is not None:
            consumer |= frame.f_code.co_name in {
                "produce_hft_processed_data",
                "produce_taker_ledger",
            }
            if frame.f_code.co_name == "_build_target":
                target = frame.f_locals["spec"].target_id
                break
            frame = frame.f_back
        del frame
        with original_open(*args, **kwargs) as source:
            if target is not None and consumer:
                leases.setdefault(target, set()).add(
                    (kwargs.get("time_window"), tuple(id(a) for a in source.archives))
                )
            yield source

    def decoded(*args, **kwargs):
        decode_counts[str(args[0])] += 1
        with replay.closing_iterator(original_decode(*args, **kwargs)) as rows:
            for index, row in enumerate(rows):
                yield row
                if failure == "body_and_close" and index == 8:
                    raise primary

    class Buffer:
        def __init__(self):
            self.inner = original_temporary()
            self.closes = 0
            buffers.append(self)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            self.closes += 1
            self.inner.close()
            if failure is not None:
                raise cleanup

    monkeypatch.setattr(replay, "open_tardis_archive", observed_open)
    monkeypatch.setattr(replay, "iter_native_slice", decoded)
    monkeypatch.setattr(replay, "TemporaryFile", Buffer)
    run_parent = layout.research_run_path(developed["run_id"]).parent
    before_runs = set(run_parent.glob("*.json"))
    kwargs = dict(
        profile=profile,
        layout=layout,
        raw_root=raw,
        research_plan=configuration,
        holdout=True,
        development_run_id=developed["run_id"],
    )
    if failure is None:
        result = operations.research_run(**kwargs)
        assert result["status"] == "COMPLETE"
    else:
        with pytest.raises(OSError) as caught:
            operations.research_run(**kwargs)
        assert caught.value is (primary if failure == "body_and_close" else cleanup)
    new_runs = set(run_parent.glob("*.json")) - before_runs
    assert len(new_runs) == 1
    record = read_research_run_record(new_runs.pop())
    assert record.status.value == ("COMPLETE" if failure is None else "FAILED")
    assert buffers and all(b.inner.closed and b.closes == 1 for b in buffers)
    assert replay._REQUEST_ARCHIVES.get() is None
    assert set(decode_counts.values()) == {1}
    if failure == "body_and_close":
        assert any("close also failed" in note for note in primary.__notes__)
        return
    panel = leases["binance_futures_l2_panel_100ms"]
    ledgers = {k: v for k, v in leases.items() if k.endswith("_ledger")}
    assert len(ledgers) == 30
    assert all(value == panel for value in ledgers.values()), (panel, ledgers)
    assert len(panel) == 1
    payload = record.payload()
    published_ledgers = [
        t
        for t in payload["target_results"]
        if t["target_id"].endswith("_ledger") and t["status"] == "BUILT"
    ]
    assert len(published_ledgers) == 30
    panel_roots = list(
        (layout.project_root / "data/processed/crypto/panel").glob(
            "binance_futures_l2_panel_100ms/instances/*"
        )
    )
    assert len(panel_roots) == 2
    assert all(len(read_dataset(root)) > 0 for root in panel_roots)
