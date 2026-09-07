from types import SimpleNamespace

import pytest

from quant_research_platform.runtime.execution_metrics import record_run_observations
from quant_research_platform.runtime.resources import runtime_resource_policy_scope


def test_runtime_observations_sample_boundaries_and_mark_unavailable_io(monkeypatch):
    import quant_research_platform.runtime.execution_metrics as owner

    class Process:
        def memory_info(self):
            return SimpleNamespace(rss=4096)

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
    with runtime_resource_policy_scope() as policy:
        with record_run_observations(policy, evidence) as sample:
            with policy.lease("fixture.payload", 1024):
                sample()
        assert policy.used_memory_bytes == 0
    assert evidence["sample_count"] == 3
    assert evidence["metrics"]["sampled_peak_reserved_bytes"] == {
        "status": "DEFINED",
        "value": 1024,
    }
    assert evidence["metrics"]["sampled_peak_rss_bytes"] == {
        "status": "DEFINED",
        "value": 4096,
    }
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

    primary = KeyboardInterrupt("primary")
    times = iter((0.0, 0.1))
    monkeypatch.setattr(owner, "perf_counter", lambda: next(times))
    evidence = {}
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(KeyboardInterrupt) as raised:
            with record_run_observations(policy, evidence):
                raise primary
        assert raised.value is primary
        assert any(
            "observation finalization also failed" in note for note in primary.__notes__
        )
        assert policy.used_memory_bytes == 0
    from quant_research_platform.contracts.targets import _operation_evidence

    assert _operation_evidence(evidence)["reason"] == "operation_incomplete"
