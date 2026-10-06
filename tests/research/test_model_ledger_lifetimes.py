import pytest

from quant_research_platform.research.models import persisted
from tests.research.test_crypto_rank_persistence import crypto_rank_case
from tests.research.workflow_test_support import shared_model_case


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

    case = shared_model_case(tmp_path)
    old_good = {
        path.relative_to(case.root): path.read_bytes()
        for path in case.root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(persisted, "_validate_a_share_model_parameters", fail)
    with pytest.raises(RuntimeError) as caught:
        persisted.read_quarterly_model_artifact(case.root, trial_layout=case.layout)
    assert caught.value is primary
    assert primary.__traceback__ is not None
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
    case = crypto_rank_case(tmp_path)
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

        def fail_refit(layout, reference):
            plan, records = read(layout, reference)
            if plan.stage == "refit":
                raise primary
            return (plan, records)

        monkeypatch.setattr(persisted, "read_model_trial_ledger_reference", fail_refit)
    with pytest.raises(RuntimeError) as caught:
        persisted.read_crypto_model_artifact(case.root, trial_layout=case.layout)
    assert caught.value is primary
    assert primary.__traceback__ is not None
    assert old_good == {
        path.relative_to(case.root): path.read_bytes()
        for path in case.root.rglob("*")
        if path.is_file()
    }
