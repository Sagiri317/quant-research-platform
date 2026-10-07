import hashlib
import json
import pickle
import shutil
from dataclasses import replace
from fractions import Fraction

import pytest

from quant_research_platform.contracts.models import (
    ModelSpec,
    validate_model_metadata_core,
)
from quant_research_platform.research.models import persisted
from quant_research_platform.storage.identity import native_filesystem_path
from tests.research.test_crypto_rank_persistence import crypto_rank_case
from tests.research.workflow_test_support import shared_model_case, shared_model_rows


@pytest.mark.parametrize("market", ["crypto", "a_share"])
@pytest.mark.parametrize("old_member", ["metadata", "pickle"])
def test_native_models_reject_old_float_scaler_offsets(tmp_path, market, old_member):
    if market == "crypto":
        case = crypto_rank_case(tmp_path / "good", algorithms=("linear_regression",))
        artifact = case.saved
        read = persisted.read_crypto_model_artifact
        for entry in artifact.final_models:
            predictor = persisted.read_crypto_model_member(artifact, entry)
            assert all(type(value) is Fraction for value in predictor.scaler.offset_)
    else:
        case = shared_model_case(
            tmp_path / "good",
            specs=(
                ModelSpec(
                    model_id="linear_x",
                    model_algorithm="linear_regression",
                    input_factor_columns=("x",),
                    label_column="label",
                ),
            ),
        )
        artifact = case.artifact
        read = persisted.read_quarterly_model_artifact
        for entry in artifact.final_models:
            predictor = persisted.read_quarterly_predictor(
                artifact, entry["selection_group"], entry["family_id"]
            )
            assert all(type(value) is Fraction for value in predictor.scaler.offset_)
    metadata = artifact.final_models[0]["model_metadata"]
    validate_model_metadata_core(metadata, quarterly=market == "a_share")
    for bad_ratio in [0.0, [0, 2], [1, 0], [1, -2], [True, 1], [1], [2, 4]]:
        changed = json.loads(json.dumps(metadata))
        changed["scaler"]["weighted_mean_offset"][0] = bad_ratio
        with pytest.raises(ValueError, match="^invalid_model_metadata$"):
            validate_model_metadata_core(changed, quarterly=market == "a_share")

    root = tmp_path / "old" / native_filesystem_path(case.root).name
    root.parent.mkdir()
    shutil.copytree(native_filesystem_path(case.root), root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())

    def old_offsets(value):
        if isinstance(value, dict):
            if "weighted_mean_offset" in value:
                value["weighted_mean_offset"] = [
                    float(Fraction(*pair)) for pair in value["weighted_mean_offset"]
                ]
            else:
                for child in value.values():
                    old_offsets(child)
        elif isinstance(value, list):
            for child in value:
                old_offsets(child)

    changed_count = 0
    for member in manifest["members"]:
        path = root / member["path"]
        payload = path.read_bytes()
        if old_member == "metadata" and path.suffix == ".json":
            document = json.loads(payload)
            old_offsets(document)
            changed = json.dumps(document).encode()
        elif old_member == "pickle" and path.suffix == ".pkl":
            predictor = pickle.loads(payload)
            predictor = replace(
                predictor,
                scaler=replace(
                    predictor.scaler,
                    offset_=tuple(map(float, predictor.scaler.offset_)),
                ),
            )
            changed = pickle.dumps(predictor)
        else:
            continue
        path.write_bytes(changed)
        member.update(size=len(changed), sha256=hashlib.sha256(changed).hexdigest())
        changed_count += 1
    assert changed_count
    manifest_path.write_text(json.dumps(manifest))
    error = (
        "invalid_model_metadata"
        if old_member == "metadata"
        else "model_state_metadata_mismatch"
    )
    with pytest.raises(ValueError, match=f"^{error}$"):
        fresh = read(
            root,
            expected_identity=artifact.envelope["instance_identity"],
            trial_layout=case.layout,
        )
        if market == "crypto":
            persisted.read_crypto_model_member(fresh, fresh.final_models[0])


def test_native_quarterly_refit_preserves_nonbinary_center(tmp_path):
    import numpy as np
    import pandas as pd

    specs = (
        ModelSpec(
            model_id="linear_x",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
    )
    frame = shared_model_rows(specs, assets=3)
    frame["x"] = frame.asset_id.map(
        {"asset_0": 1e200, "asset_1": 1.0, "asset_2": -1e200}
    )
    case = shared_model_case(tmp_path, specs=specs, frame=frame, chunk_size=2)
    for entry in case.artifact.final_models:
        predictor = persisted.read_quarterly_predictor(
            case.artifact, entry["selection_group"], entry["family_id"]
        )
        scaler = predictor.scaler
        assert Fraction(scaler.origin_[0]) + scaler.offset_[0] == Fraction(1, 3)
        assert (
            Fraction(*entry["model_metadata"]["scaler"]["weighted_mean_offset"][0])
            == scaler.offset_[0]
        )
        zero = pd.DataFrame({"x": [0.0]})
        assert scaler.transform(zero).x.iloc[0] == float(
            -Fraction(1, 3) / Fraction(scaler.scale_[0])
        )
        assert np.isfinite(predictor.predict(frame[["x"]])).all()
