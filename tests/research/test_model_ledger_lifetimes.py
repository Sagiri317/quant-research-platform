import sys
from collections.abc import Mapping
from contextlib import contextmanager

import pytest

from quant_research_platform.contracts.experiments import (
    ExperimentPlan,
    TrialRecord,
)
from quant_research_platform.research.models import persisted
from quant_research_platform.runtime.resources import (
    MemoryLease,
    runtime_resource_policy_scope,
)
from tests.research.test_crypto_rank_persistence import crypto_rank_case
from tests.research.workflow_test_support import shared_model_case


@contextmanager
def observe_model_ledger_release():
    observations = []
    validators = {
        persisted._validate_a_share_model_payload.__code__,
        persisted._validate_crypto_model_payload.__code__,
    }

    def contains_typed(value):
        if isinstance(value, (ExperimentPlan, TrialRecord)):
            return True
        if isinstance(value, Mapping):
            return any(contains_typed(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains_typed(item) for item in value)
        return False

    def profile(frame, event, _arg):
        if event != "call" or frame.f_code is not MemoryLease.close.__code__:
            return
        lease = frame.f_locals["self"]
        if lease.owner != "storage.trial.ledger":
            return
        # A failed second __enter__ can unwind while the first ledger is active.
        if any(
            other is not lease and other.owner == "storage.trial.ledger"
            for other in lease._root._leases
        ):
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code in validators:
                retained = tuple(
                    name
                    for name, value in caller.f_locals.items()
                    if contains_typed(value)
                )
                observations.append((caller.f_code.co_name, retained))
            caller = caller.f_back

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield observations
    finally:
        sys.setprofile(previous)


def test_model_ledger_consumers_release_typed_graphs_before_lease(tmp_path):
    with observe_model_ledger_release() as observations:
        with (
            runtime_resource_policy_scope() as policy,
            policy.lease("test.models.metadata", 0) as lease,
        ):
            case = shared_model_case(tmp_path / "a_share", lease=lease)
            assert case.artifact.final_models
            case = None
        with crypto_rank_case(tmp_path / "crypto") as case:
            assert len(case.saved.final_models) == 3
    assert {name for name, _retained in observations} == {
        "_validate_a_share_model_payload",
        "_validate_crypto_model_payload",
    }
    assert all(not retained for _name, retained in observations), observations


@pytest.mark.parametrize("stage", ["selection", "refit"])
def test_a_share_ledger_failure_preserves_primary_and_old_good(
    tmp_path, monkeypatch, stage
):
    primary = RuntimeError("consumer_failure")
    validate = persisted._validate_a_share_model_parameters

    def fail(metadata, parameters):
        if ("fit_cutoff" in metadata) == (stage == "refit"):
            raise primary
        return validate(metadata, parameters)

    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("test.models.metadata", 0) as lease,
    ):
        case = shared_model_case(tmp_path, lease=lease)
        old_good = {
            path.relative_to(case.root): path.read_bytes()
            for path in case.root.rglob("*")
            if path.is_file()
        }
        monkeypatch.setattr(persisted, "_validate_a_share_model_parameters", fail)
        with observe_model_ledger_release() as observations:
            with pytest.raises(RuntimeError) as caught:
                persisted.read_quarterly_model_artifact(
                    case.root,
                    lease=lease,
                    trial_layout=case.layout,
                )
        assert caught.value is primary
        assert primary.__traceback__ is not None
        assert observations and all(not retained for _name, retained in observations)
        assert old_good == {
            path.relative_to(case.root): path.read_bytes()
            for path in case.root.rglob("*")
            if path.is_file()
        }


@pytest.mark.parametrize("stage", ["selection", "refit_enter"])
def test_crypto_ledger_failure_preserves_primary_and_old_good(
    tmp_path, monkeypatch, stage
):
    primary = RuntimeError("consumer_failure")
    with crypto_rank_case(tmp_path) as case:
        old_good = {
            path.relative_to(case.root): path.read_bytes()
            for path in case.root.rglob("*")
            if path.is_file()
        }
        if stage == "selection":

            def fail(*args, **kwargs):
                raise primary

            monkeypatch.setattr(persisted, "_crypto_validation_chunks", fail)
        else:
            read = persisted.read_model_trial_ledger_reference

            @contextmanager
            def fail_refit(layout, reference):
                with read(layout, reference) as (plan, records):
                    try:
                        if plan.stage == "refit":
                            raise primary
                        yield plan, records
                    finally:
                        plan = records = None

            monkeypatch.setattr(
                persisted, "read_model_trial_ledger_reference", fail_refit
            )
        with observe_model_ledger_release() as observations:
            with pytest.raises(RuntimeError) as caught:
                persisted.read_crypto_model_artifact(
                    case.root, trial_layout=case.layout, lease=case.lease
                )
        assert caught.value is primary
        assert primary.__traceback__ is not None
        assert observations and all(not retained for _name, retained in observations)
        assert old_good == {
            path.relative_to(case.root): path.read_bytes()
            for path in case.root.rglob("*")
            if path.is_file()
        }
