"""Saved models load independently of external trial records."""

import shutil

from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from tests.correctness.test_direct_model_operations import data, fit_config


def test_model_load_is_self_contained_without_external_trial(tmp_path):
    inputs = data(tmp_path)
    model = fit(fit_config(tmp_path, inputs))
    shutil.rmtree(model.with_name(model.name + ".trial"))
    assert load_model(model)[0].metadata["rows_after_input_label_dropna"] == 9
