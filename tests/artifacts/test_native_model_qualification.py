"""Native saved-model qualification binds actual learning facts and predictors."""

import json
import shutil

import pytest

from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from tests.correctness.test_direct_model_operations import data, fit_config


@pytest.fixture(scope="module")
def native_models(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("native-models")
    inputs = data(tmp_path)
    config = fit_config(tmp_path, inputs)
    good = fit(config)
    ridge = fit(
        {
            **config,
            "output": str(tmp_path / "ridge"),
            "model": {**config["model"], "model_algorithm": "ridge_regression"},
        }
    )
    return good, ridge


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_predictor",
        "seed",
        "parameters",
        "fit_window",
        "input_unit_whitespace",
    ],
)
def test_model_reader_rejects_wrong_learning_evidence_without_changing_good_model(
    tmp_path, mutation, native_models
):
    good, ridge = native_models
    protected = {
        path: path.read_bytes() for path in good.parent.rglob("*") if path.is_file()
    }
    bad = tmp_path / "bad"
    shutil.copytree(good, bad)
    model = json.loads((bad / "model.json").read_text())
    if mutation == "wrong_predictor":
        shutil.copyfile(ridge / "predictor.pkl", bad / "predictor.pkl")
    elif mutation == "seed":
        model["facts"]["model"]["random_seed"] += 1
    elif mutation == "parameters":
        model["facts"]["candidate_parameters"] = {"fit_intercept": False}
    elif mutation == "input_unit_whitespace":
        model["facts"]["input_units"] = {"x": " "}
    else:
        model["facts"]["fit_windows"][0]["start"] = "2020-01-02"
    (bad / "model.json").write_text(json.dumps(model))
    with pytest.raises((ValueError, TypeError)):
        load_model(bad)
    assert all(path.read_bytes() == original for path, original in protected.items())
    assert load_model(good)[0].metadata["rows_after_input_label_dropna"] == 9


def test_model_load_is_self_contained_without_external_trial(tmp_path):
    inputs = data(tmp_path)
    model = fit(fit_config(tmp_path, inputs))
    shutil.rmtree(model.with_name(model.name + ".trial"))
    assert load_model(model)[0].metadata["rows_after_input_label_dropna"] == 9
