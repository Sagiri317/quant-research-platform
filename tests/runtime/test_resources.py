import os
import sys
from pathlib import Path

import pytest

import quant_research_platform.runtime.resources as resources


@pytest.mark.parametrize(
    "quota,parent_quota,affinity,expected",
    [
        ("1400000 100000", "max 100000", 32, 14),
        ("max 100000", "250000 100000", 32, 2),
        ("1400000 100000", "max 100000", 3, 3),
        ("50000 100000", "max 100000", 32, 1),
        ("max 100000", "max 100000", 32, 16),
        ("invalid", "max 100000", 32, None),
        ("100000 0", "max 100000", 32, None),
    ],
)
def test_model_threads_bind_affinity_and_cgroup_bandwidth(
    tmp_path, monkeypatch, quota, parent_quota, affinity, expected
):
    import psutil

    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.supervised import (
        resolved_model_hyperparameters,
    )

    root = tmp_path / "cgroup"
    child = root / "job"
    child.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("memory cpu")
    (root / "cpu.max").write_text(parent_quota)
    (child / "cpu.max").write_text(quota)
    membership = tmp_path / "membership"
    membership.write_text("0::/job\n")
    original_path = Path
    monkeypatch.setattr(
        resources,
        "Path",
        lambda value: {"/sys/fs/cgroup": root, "/proc/self/cgroup": membership}.get(
            str(value), original_path(value)
        ),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _: set(range(affinity)), raising=False
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda **_: 16)
    spec = ModelSpec(
        model_id="capacity_model",
        model_algorithm="random_forest_regressor",
        input_feature_columns=("feature",),
        target_column="label",
        hyperparameters={"n_jobs": 16},
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="runtime_cpu_capacity_discovery_failed"):
            resolved_model_hyperparameters(spec)
    else:
        assert resources.model_cpu_capacity() == (16, expected)
        assert resolved_model_hyperparameters(spec)["n_jobs"] == expected


@pytest.mark.parametrize(
    "membership", ["0::/\n", "0::/missing\n", "0::/../escape\n", "1:cpu:/\n"]
)
def test_cgroup_membership_is_bound_to_visible_hierarchy(
    tmp_path, monkeypatch, membership
):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("memory cpu")
    member_file = tmp_path / "membership"
    member_file.write_text(membership)
    original_path = Path
    monkeypatch.setattr(
        resources,
        "Path",
        lambda value: {"/sys/fs/cgroup": root, "/proc/self/cgroup": member_file}.get(
            str(value), original_path(value)
        ),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    if membership == "0::/\n":
        assert list(resources._cgroup_v2_directories()) == [root]
        (root / "cgroup.controllers").unlink()
        assert list(resources._cgroup_v2_directories()) == []
    else:
        with pytest.raises((ValueError, RuntimeError), match="cgroup_membership"):
            list(resources._cgroup_v2_directories())


@pytest.mark.parametrize(
    "requested,expected",
    [(-1, 4), (-2, 3), (1, 1), (8, 4), (0, None), (-5, None), (17, None), (True, None)],
)
def test_model_thread_rule_uses_capacity_without_rewriting_request(requested, expected):
    from quant_research_platform.contracts.models import _resolved_model_thread_count

    if expected is None:
        with pytest.raises(ValueError):
            _resolved_model_thread_count(
                requested, physical_cores=16, available_cores=4
            )
    else:
        assert (
            _resolved_model_thread_count(
                requested, physical_cores=16, available_cores=4
            )
            == expected
        )


def test_windows_model_capacity_respects_process_affinity(monkeypatch):
    from types import SimpleNamespace

    import psutil

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(psutil, "cpu_count", lambda **_: 14)
    monkeypatch.setattr(
        psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: [7])
    )
    assert resources.model_cpu_capacity() == (14, 1)
