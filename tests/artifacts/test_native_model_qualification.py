"""Native saved-model qualification binds actual learning facts and predictors."""

import json
import shutil

import pytest

from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from tests.correctness.test_direct_model_operations import data, fit_config


@pytest.mark.parametrize(
    "mutation",
    [
        "failed_trial",
        "attempted_trial",
        "wrong_trial",
        "wrong_predictor",
        "seed",
        "parameters",
        "fit_window",
    ],
)
def test_model_reader_rejects_wrong_learning_evidence_without_changing_good_model(
    tmp_path, mutation
):
    inputs = data(tmp_path)
    config = fit_config(tmp_path, inputs)
    good = fit(config)
    alternate = tmp_path / "alternate-input.parquet"
    shutil.copyfile(inputs, alternate)
    other = fit({**config, "inputs": str(alternate), "output": str(tmp_path / "other")})
    ridge = fit(
        {
            **config,
            "output": str(tmp_path / "ridge"),
            "model": {**config["model"], "model_algorithm": "ridge_regression"},
        }
    )
    protected = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    bad = tmp_path / "bad"
    shutil.copytree(good, bad)
    model = json.loads((bad / "model.json").read_text())
    if mutation in {"failed_trial", "attempted_trial", "wrong_trial"}:
        # Only the damaged model refers to a copied trial; originals stay intact.
        original = (
            json.loads((other / "model.json").read_text())
            if mutation == "wrong_trial"
            else model
        )
        from pathlib import Path

        trial = tmp_path / "bad.trial"
        shutil.copytree(Path(original["facts"]["trial"]), trial)
        model["facts"]["trial"] = str(trial)
        model["trial_id"] = original["trial_id"]
        if mutation != "wrong_trial":
            terminal_path = trial / "terminal.json"
            terminal = json.loads(terminal_path.read_text())
            if mutation == "attempted_trial":
                terminal_path.unlink()
            else:
                terminal["status"] = "failed"
                terminal_path.write_text(json.dumps(terminal))
    elif mutation == "wrong_predictor":
        shutil.copyfile(ridge / "predictor.pkl", bad / "predictor.pkl")
    elif mutation == "seed":
        model["facts"]["model"]["random_seed"] += 1
    elif mutation == "parameters":
        model["facts"]["candidate_parameters"] = {"fit_intercept": False}
    else:
        model["facts"]["fit_windows"][0]["start"] = "2020-01-02"
    (bad / "model.json").write_text(json.dumps(model))
    with pytest.raises((ValueError, TypeError)):
        load_model(bad)
    assert all(path.read_bytes() == original for path, original in protected.items())
    assert load_model(good)[0].metadata["rows_after_input_label_dropna"] == 9
