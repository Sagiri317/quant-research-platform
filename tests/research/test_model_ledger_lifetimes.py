"""A secondary failure writing trial evidence must preserve the learning failure."""

import pytest

from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.models.production import fit
from tests.correctness.test_direct_model_operations import data, fit_config


def test_failure_record_does_not_mask_primary_or_publish_model(tmp_path, monkeypatch):
    import quant_research_platform.research.models.production as production

    config = fit_config(tmp_path, data(tmp_path))
    primary = KeyboardInterrupt()

    def fail(*a, **k):
        raise primary

    def record_fail(*a, **k):
        raise OSError("trial")

    monkeypatch.setattr(production, "fit_model", fail)
    monkeypatch.setattr(TrialRecorder, "finish", record_fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        fit(config)
    assert caught.value is primary
    assert not (tmp_path / "linear").exists()
    assert (tmp_path / "linear.trial" / "attempted.json").exists()


@pytest.mark.parametrize("failure", ["runtime", "interrupt", "publication"])
def test_native_selected_refit_failure_preserves_prior_selection_and_model(
    tmp_path, monkeypatch, failure
):
    import json
    from pathlib import Path

    import quant_research_platform.research.models.persisted as persisted
    import quant_research_platform.research.models.production as production
    from quant_research_platform.research.models.trial_selection import score, select
    from tests.research.test_prediction_runner_model_selection import candidates

    cases, scoring = candidates(tmp_path, algorithms=("linear_regression",))
    selection = select(
        {
            "scores": str(score(scoring)),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["linear_regression"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "refit_cutoff": "2020-01-06",
        }
    )
    request = {
        **cases[0].config,
        "inputs": scoring["labels"],
        "output": str(tmp_path / "final-good"),
        "selection": str(selection),
        "test_window": scoring["test_window"],
        "fit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
        "fit_cutoff": "2020-01-06",
    }
    good = fit(request)
    assert persisted.load_model(good)[0].metadata["rows_after_input_label_dropna"] == 15
    unchanged = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    existing_roots = set(tmp_path.iterdir())
    output = tmp_path / "final-failed"
    primary = (
        KeyboardInterrupt()
        if failure == "interrupt"
        else RuntimeError("synthetic refit failure")
    )
    real_fit = production.fit_model
    if failure == "publication":
        real_write = persisted.atomic_write_json

        def write(path, *args, **kwargs):
            if Path(path).name == "model.json":
                assert (Path(path).parent / "predictor.pkl").exists()
                raise primary
            return real_write(path, *args, **kwargs)

        monkeypatch.setattr(persisted, "atomic_write_json", write)
    else:

        def interrupted(*args, **kwargs):
            result = real_fit(*args, **kwargs)
            assert result.metadata["rows_after_input_label_dropna"] == 15
            raise primary

        monkeypatch.setattr(production, "fit_model", interrupted)
    with pytest.raises(type(primary)) as caught:
        fit({**request, "output": str(output)})
    assert caught.value is primary
    assert not output.exists()
    trial = output.with_name(output.name + ".trial")
    assert set(tmp_path.iterdir()) == existing_roots | {trial}
    assert all(path.read_bytes() == original for path, original in unchanged.items())
    terminal = json.loads((trial / "terminal.json").read_text())
    # A completed fit remains a completed learning attempt if later publication fails.
    assert (
        terminal["status"]
        == {"runtime": "failed", "interrupt": "abandoned", "publication": "completed"}[
            failure
        ]
    )
    assert persisted.load_model(good)[0].metadata["rows_after_input_label_dropna"] == 15
