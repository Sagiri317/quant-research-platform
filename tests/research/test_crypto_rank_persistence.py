from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.business_manifest import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.models import fitting, persisted, production
from quant_research_platform.research.models.supervised import preflight_model_specs
from quant_research_platform.research.predictions.runner import build_crypto_predictions
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.research import (
    _target_envelope_for_spec,
    production_software_identity,
)
from tests.helpers.binance_hft import hft_research_plan
from tests.helpers.research_configuration import research_configuration


def crypto_rank_case(output, *, missing_rank_input=False):
    plan = hft_research_plan()
    plan["folds"] = plan["folds"][:1]
    window = TimeWindow(**plan["folds"][0]["validation"])
    configuration = research_configuration(
        "crypto",
        "baseline",
        plan=plan,
        development=(window.start.isoformat(), window.end.isoformat()),
    )
    template = deepcopy(configuration["models"][0])
    models = []
    for algorithm in (
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
        "linear_regression",
    ):
        model = deepcopy(template)
        model.update(
            model_id="probe_" + algorithm,
            model_algorithm=algorithm,
            input_factor_columns=["imbalance_1"],
            input_feature_columns=[],
            hyperparameters={},
            training_asset_ids=["btc_usdt", "eth_usdt"],
            evaluation_asset_ids=["btc_usdt", "eth_usdt"],
        )
        model.pop("parameter_candidates", None)
        models.append(model)
    configuration.update(
        models=models, portfolios=[], comparison_model_id=models[0]["model_id"]
    )
    configuration["evaluation"].update(
        factors=["imbalance_1"],
        labels=[models[0]["label_column"]],
        primary_label=models[0]["label_column"],
    )
    profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID), configuration
    )
    preflight_model_specs(profile.models.list())
    dataset_id = configuration["scenario_id"]
    dataset = profile.research_datasets.get(dataset_id)
    protocol = profile.protocol(dataset_id)
    target_id = dataset_id + "_models"
    target = next(
        item
        for item in protocol.target_specs("research run")
        if item.target_id == target_id
    )
    dataset_identity = "a" * 64
    layout = StorageLayout(output)
    dataset_root = layout.instance_root(
        layout.research_dataset_root("crypto", dataset_id), dataset_identity
    )
    layout = layout.with_target_roots({dataset_id: dataset_root})
    envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=protocol,
        target_spec=target,
        data_spec=None,
        research_window={
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "access": "development",
        },
        upstream_identities={dataset_id: dataset_identity},
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    manifest = envelope.payload()["semantic_metadata"]["target_contract"]["protocol"]
    stamps = pd.date_range("1970-01-01", periods=91, freq="100ms")
    timestamps = stamps.repeat(2)
    count = len(timestamps)
    signs = np.tile([-1.0, 1.0], len(stamps))
    values = {}
    for column in dataset.required_columns:
        if column == "market_id":
            value = ["crypto"] * count
        elif column == "asset_id":
            value = np.tile(["btc_usdt", "eth_usdt"], len(stamps))
        elif column in ("timestamp", "observed_utc"):
            value = timestamps
        elif column.endswith("_available_at"):
            value = timestamps + pd.Timedelta(milliseconds=500)
        elif column == "eligible":
            value = np.ones(count, dtype=bool)
        elif column.endswith("_id"):
            value = ["synthetic"] * count
        elif column in dataset.label_columns:
            value = signs * 0.001
        elif column in dataset.factor_columns:
            value = signs
        else:
            value = np.ones(count)
        values[column] = value
    frame = (
        pd.DataFrame(values)
        .sort_values(["market_id", "asset_id", "timestamp"])
        .reset_index(drop=True)
    )
    if missing_rank_input:
        first = frame.index[
            (frame.timestamp == pd.Timestamp(window.start))
            & frame.asset_id.eq("btc_usdt")
        ][0]
        frame.loc[first, "imbalance_1"] = np.nan
    ids = tuple(model["model_id"] for model in models)
    write_dataset(dataset_root, frame, spec=dataset)
    session = open_dataset_session(
        dataset_root,
        columns=None,
        time_window=TimeWindow(
            plan["folds"][0]["train"]["start"], plan["folds"][0]["validation"]["end"]
        ),
    )

    def results(**callbacks):
        return production.build_crypto_fold_models(
            layout=layout,
            dataset_session=session,
            dataset_catalog=profile.research_datasets,
            model_catalog=profile.models,
            dataset_id=dataset_id,
            model_ids=ids,
            time_window=window,
            protocol_manifest=manifest,
            run_id="run_rank_roundtrip",
            research_protocol_identity=canonical_json_hash(manifest),
            model_target_id=target_id,
            backend_versions={"fixture": "1"},
            **callbacks,
        )

    root = persisted.write_crypto_model_artifact(
        output / "models",
        envelope=envelope,
        model_results=results,
        trial_layout=layout,
        verify_source=session.verify,
    )
    saved = persisted.read_crypto_model_artifact(
        root, expected_identity=envelope.instance_identity, trial_layout=layout
    )
    return SimpleNamespace(
        root=root,
        dataset=dataset,
        manifest=manifest,
        saved=saved,
        layout=layout,
        profile=profile,
        holdout=TimeWindow(**plan["test"]),
    )


@pytest.mark.parametrize("missing_rank_input", (False, True))
def test_crypto_rank_roundtrip_and_frozen_holdout(tmp_path, missing_rank_input):
    case = crypto_rank_case(tmp_path, missing_rank_input=missing_rank_input)
    for entry in case.saved.final_models:
        algorithm = entry["model_metadata"]["model_algorithm"]
        expected = (
            [-0.001, 0.001] if algorithm == "linear_regression" else [-0.25, 0.25]
        )
        model = persisted.read_crypto_model_member(case.saved, entry)
        probe = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(case.holdout.start)] * 2,
                "imbalance_1": [-1.0, 1.0],
            }
        )
        scores = fitting.prediction_callable(("imbalance_1",), model)(probe)
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)
        if algorithm != "linear_regression":
            parameter = (
                "directions" if algorithm == "equal_weight_rank_ensemble" else "weights"
            )
            assert entry["model_metadata"]["rank_parameters"] == {
                "columns": ["imbalance_1"],
                "label_calibration": pytest.approx([0.0, 0.004]),
                parameter: [1.0],
            }
            for changes in (
                {"columns": ("other",)},
                {parameter: (-1.0,)},
                {"label_calibration": (0.0, 1.0)},
            ):
                changed = replace(model, **changes)
                with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
                    fitting.validate_model_state(changed, entry["model_metadata"])
    observed = set()
    with closing_iterator(
        build_crypto_predictions(
            layout=case.layout,
            dataset_catalog=case.profile.research_datasets,
            persisted_model=case.saved,
            time_window=case.holdout,
            access_window="holdout",
        )
    ) as results:
        for result in results:
            observed.add(result.spec.model_id)
            rows = 0
            with result:
                for frame in result.output_chunks:
                    scale = (
                        0.001
                        if result.spec.model_id.endswith("linear_regression")
                        else 0.25
                    )
                    expected = frame.asset_id.map(
                        {"btc_usdt": -scale, "eth_usdt": scale}
                    )
                    np.testing.assert_allclose(
                        frame.prediction_score, expected, rtol=0, atol=1e-15
                    )
                    assert frame.timestamp.between(
                        pd.Timestamp(case.holdout.start),
                        pd.Timestamp(case.holdout.end),
                    ).all()
                    rows += len(frame)
                assert rows == 10
    assert observed == {entry["model_id"] for entry in case.saved.final_models}


def test_rank_economic_consumers_use_frozen_training_label_units(tmp_path):
    from quant_research_platform.artifacts.outputs import (
        read_prediction_output_bundle,
        write_prediction_outputs,
    )
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _prediction_rows,
        _validation_rows,
    )
    from quant_research_platform.workflows.research import _output_spec_for_target

    case = crypto_rank_case(tmp_path)
    prediction_root = tmp_path / "prediction_outputs"
    with closing_iterator(
        build_crypto_predictions(
            layout=case.layout,
            dataset_catalog=case.profile.research_datasets,
            persisted_model=case.saved,
            time_window=case.holdout,
            access_window="holdout",
        )
    ) as results:
        for result in results:
            with result:
                write_prediction_outputs(
                    spec=result.spec,
                    output_spec=_output_spec_for_target(
                        result.spec.prediction_id, "crypto", "prediction"
                    ),
                    layout=StorageLayout(prediction_root),
                    output_chunks=result.output_chunks,
                    model_metadata=result.model_metadata,
                    prediction_metadata_factory=result.prediction_metadata_factory,
                )
    layout = case.layout.with_target_roots(
        {case.dataset.data_id + "_predictions": prediction_root}
    )
    for entry in case.saved.final_models:
        for asset, expected in (("btc_usdt", -0.001), ("eth_usdt", 0.001)):
            values = list(
                _prediction_rows(
                    layout=layout,
                    dataset=case.dataset,
                    windows=(("final", case.holdout),),
                    model_id=entry["model_id"],
                    asset_id=asset,
                    protocol_manifest=case.profile.protocol(
                        case.dataset.data_id
                    ).manifest(),
                    read_prediction_bundle=read_prediction_output_bundle,
                    output_window=case.holdout,
                )
            )
            assert len(values) == 5
            assert [value["prediction_score"] for value in values] == pytest.approx(
                [expected] * 5
            )
            validation = list(
                _validation_rows(
                    case.saved,
                    fold_id=case.saved.models[0]["fold_id"],
                    model_id=entry["model_id"],
                    asset_id=asset,
                )
            )
            assert validation
            assert [row.prediction_score for row in validation] == pytest.approx(
                [expected] * len(validation)
            )
