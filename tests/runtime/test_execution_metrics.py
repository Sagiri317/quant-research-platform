import pytest

from quant_research_platform.runtime.execution_metrics import record_run_observations


def test_runtime_observations_sample_boundaries_and_mark_unavailable_io(monkeypatch):
    import quant_research_platform.runtime.execution_metrics as owner

    class Process:
        def cpu_times(self):
            return (1.0, 0.5)

        def num_threads(self):
            return 2

        def children(self, recursive):
            assert recursive
            return []

        def io_counters(self):
            raise AttributeError("io_not_supported")

    monkeypatch.setattr(owner.psutil, "Process", Process)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    evidence = {}
    with record_run_observations(evidence) as sample:
        sample()
    assert evidence["sample_count"] == 3
    assert evidence["metrics"]["cpu_seconds"] == {"status": "DEFINED", "value": 0.0}
    for name in ("read_bytes", "write_bytes"):
        assert evidence["metrics"][name]["status"] == "UNDEFINED"
        assert "value" not in evidence["metrics"][name]
    assert evidence["status"] == "UNDEFINED"
    assert evidence["environment"]["native_thread_env"]["NUMEXPR_NUM_THREADS"] == {
        "status": "UNDEFINED",
        "reason": "environment_variable_unset",
    }


def test_runtime_observation_finalization_keeps_primary(monkeypatch):
    import quant_research_platform.runtime.execution_metrics as owner
    from quant_research_platform.contracts.targets import _operation_evidence

    primary = KeyboardInterrupt("primary")
    times = iter((0.0, 0.1))
    monkeypatch.setattr(owner, "perf_counter", lambda: next(times))
    evidence = {}
    with pytest.raises(KeyboardInterrupt) as raised:
        with record_run_observations(evidence):
            raise primary
    assert raised.value is primary
    assert any(
        "observation finalization also failed" in note for note in primary.__notes__
    )
    assert _operation_evidence(evidence)["reason"] == "operation_incomplete"


def test_runtime_finalization_surfaces_failure_inside_handled_outer(monkeypatch):
    import quant_research_platform.runtime.execution_metrics as owner

    times = iter((0.0, 0.1))
    monkeypatch.setattr(owner, "perf_counter", lambda: next(times))
    try:
        raise RuntimeError("handled_outer")
    except RuntimeError as outer:
        with pytest.raises(RuntimeError, match="generator raised StopIteration"):
            with record_run_observations({}):
                pass
        assert not getattr(outer, "__notes__", [])
