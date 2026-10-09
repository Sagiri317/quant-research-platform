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


def crypto_rank_case(
    output,
    *,
    missing_rank_input=False,
    algorithms=(
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
        "linear_regression",
    ),
    training_assets=("btc_usdt", "eth_usdt"),
    sample_filter_mode="eligible",
    ineligible_holdout=False,
    time_series_inputs=False,
    omit_hyperparameters=False,
    model_updates=None,
):
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
    for algorithm in algorithms:
        model = deepcopy(template)
        model.update(
            model_id="probe_" + algorithm,
            model_algorithm=algorithm,
            input_factor_columns=["imbalance_1"],
            input_feature_columns=[],
            hyperparameters={},
            training_asset_ids=list(training_assets),
            evaluation_asset_ids=["btc_usdt", "eth_usdt"],
            sample_filter_mode=sample_filter_mode,
        )
        model.pop("parameter_candidates", None)
        if omit_hyperparameters:
            model.pop("hyperparameters")
        if model_updates is not None:
            model.update(model_updates.get(algorithm, {}))
        models.append(model)
    configuration.update(
        models=models, portfolios=[], comparison_model_id=models[0]["model_id"]
    )
    configuration["evaluation"].update(
        factors=["imbalance_1"],
        labels=[models[0]["label_column"]],
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
    if time_series_inputs:
        signs = np.repeat(np.arange(len(stamps), dtype="float64") + 1.0, 2)
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
    if ineligible_holdout:
        frame.loc[
            frame.timestamp.between(
                pd.Timestamp(plan["test"]["start"]), pd.Timestamp(plan["test"]["end"])
            )
            & frame.asset_id.eq("btc_usdt"),
            "eligible",
        ] = False
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
        frame=frame,
    )


@pytest.mark.parametrize("missing_rank_input", (False, True))
def test_crypto_rank_roundtrip_and_frozen_holdout(
    tmp_path, monkeypatch, missing_rank_input
):
    case = crypto_rank_case(tmp_path, missing_rank_input=missing_rank_input)
    from quant_research_platform.research.predictions import runner
    from quant_research_platform.storage.datasets import reading

    opened = []
    original_open = reading.open_dataset_session

    def open_session(*args, **kwargs):
        session = original_open(*args, **kwargs)
        opened.append(session)
        return session

    monkeypatch.setattr(reading, "open_dataset_session", open_session)
    monkeypatch.setattr(runner, "open_dataset_session", open_session)
    for entry in case.saved.final_models:
        algorithm = entry["model_metadata"]["model_algorithm"]
        expected = (
            [-0.001, 0.001] if algorithm == "linear_regression" else [-0.25, 0.25]
        )
        model = persisted.read_crypto_model_member(case.saved, entry)
        if algorithm == "linear_regression":
            from quant_research_platform.contracts.models import (
                validate_model_metadata_core,
            )

            assert entry["model_metadata"]["scaler"] == model.scaler.to_payload()
            validate_model_metadata_core(entry["model_metadata"])
            legacy = deepcopy(entry["model_metadata"])
            legacy["scaler"] = {
                "columns": list(model.scaler.columns),
                "weighted_mean": list(model.scaler.mean_),
                "weighted_population_scale": list(model.scaler.scale_),
            }
            with pytest.raises(ValueError, match="invalid_model_metadata"):
                validate_model_metadata_core(legacy)
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
    assert len(opened) == 1
    opened[0].verify()


def test_missing_hyperparameters_bind_and_roundtrip_models(tmp_path):
    case = crypto_rank_case(
        tmp_path,
        algorithms=("linear_regression",),
        omit_hyperparameters=True,
    )
    assert case.manifest["research_configuration"]["models"][0]["hyperparameters"] == {}
    for entry in case.saved.final_models:
        predictor = persisted.read_crypto_model_member(case.saved, entry)
        assert entry["model_metadata"]["hyperparameters"] == {}
        fitting.validate_model_state(predictor, entry["model_metadata"])


def test_crypto_rank_trial_mse_uses_frozen_training_label_calibration(tmp_path):
    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
        research_run_trial_references,
    )

    case = crypto_rank_case(tmp_path)
    references = {
        item["semantic_identity"]: item
        for item in research_run_trial_references(case.layout, "run_rank_roundtrip")
    }
    for entry in case.saved.models:
        _plan, records = read_model_trial_ledger_reference(
            case.layout, references[entry["trial_semantic_identity"]]
        )
        metrics = records[0].metrics
        assert metrics["validation_mse"] == pytest.approx(0.0, abs=1e-28)
        assert metrics["validation_ic"] == pytest.approx(1.0)
        assert metrics["validation_rankic"] == pytest.approx(1.0)
        with closing_iterator(
            case.saved.validation_chunks(
                fold_id=entry["fold_id"], model_id=entry["model_id"]
            )
        ) as chunks:
            frame = pd.concat(list(chunks), ignore_index=True)
        score = frame.prediction_score.to_numpy()
        if entry["model_metadata"]["model_algorithm"] != "linear_regression":
            assert set(score) == {-0.25, 0.25}
            intercept, slope = entry["model_metadata"]["rank_parameters"][
                "label_calibration"
            ]
            score = intercept + slope * score
        np.testing.assert_allclose(score, frame["label"], rtol=0, atol=1e-15)


def test_crypto_prediction_scope_rejects_native_source_change_between_models(tmp_path):
    from quant_research_platform.storage.identity import (
        filesystem_glob,
        open_filesystem_file,
    )

    case = crypto_rank_case(tmp_path)
    root = case.layout.target_root(
        case.dataset.data_id,
        case.layout.research_dataset_root("crypto", case.dataset.data_id),
    )
    with closing_iterator(
        build_crypto_predictions(
            layout=case.layout,
            dataset_catalog=case.profile.research_datasets,
            persisted_model=case.saved,
            time_window=case.holdout,
            access_window="holdout",
        )
    ) as results:
        first = next(results)
        with first:
            initial = pd.concat(list(first.output_chunks), ignore_index=True)
        assert len(initial) == 10
        part = next(filesystem_glob(root / "data", "**/part-*.parquet"))
        with open_filesystem_file(part, "ab") as handle:
            handle.write(b"x")
        second = next(results)
        with pytest.raises(ValueError, match="^dataset_file_identity_changed$"):
            with second:
                list(second.output_chunks)


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


def test_crypto_unfiltered_frozen_predictions_preserve_exact_holdout_domain(tmp_path):
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs import (
        read_prediction_output_bundle,
        write_prediction_outputs,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        build_hft_prediction_catalog,
    )
    from quant_research_platform.storage.identity import native_filesystem_path
    from quant_research_platform.workflows.research import _output_spec_for_target

    case = crypto_rank_case(
        tmp_path,
        algorithms=("linear_regression",),
        sample_filter_mode="unfiltered",
        ineligible_holdout=True,
    )
    entry = case.saved.final_models[0]
    assert entry["model_metadata"]["sample_filter_mode"] == "unfiltered"
    configuration = case.manifest["research_configuration"]
    configured = build_hft_prediction_catalog(
        configuration=configuration,
        train_end=case.saved.models[0]["training_window"]["end"],
        test_start=case.holdout.start.isoformat(),
        test_end=case.holdout.end.isoformat(),
    ).get(entry["model_id"] + "_prediction")
    assert configured.sample_filter_mode == "unfiltered"
    output = StorageLayout(tmp_path / "prediction_outputs")
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
            assert result.spec.sample_filter_mode == "unfiltered"
            with result:
                output_root = write_prediction_outputs(
                    spec=result.spec,
                    output_spec=_output_spec_for_target(
                        result.spec.prediction_id, "crypto", "prediction"
                    ),
                    layout=output,
                    output_chunks=result.output_chunks,
                    model_metadata=result.model_metadata,
                    prediction_metadata_factory=result.prediction_metadata_factory,
                )
    manifest, lookup = read_prediction_output_bundle(
        output_root, expected_output_id=configured.prediction_id
    )
    expected = case.frame.loc[
        case.frame.timestamp.between(
            pd.Timestamp(case.holdout.start), pd.Timestamp(case.holdout.end)
        ),
        ["timestamp", "asset_id"],
    ].sort_values(["timestamp", "asset_id"], ignore_index=True)
    expected["asset_id"] = expected.asset_id.astype("str")
    expected["timestamp"] = expected.timestamp.astype("datetime64[ns]")
    with pq.ParquetFile(
        native_filesystem_path(output_root / "prediction.parquet")
    ) as reader:
        predictions = reader.read().to_pandas()
    actual = predictions.loc[:, ["timestamp", "asset_id"]]
    pd.testing.assert_frame_equal(actual, expected)
    scores, found = lookup.values_for_frame(expected, mark_matched=True)
    assert list(found) == [1] * len(expected)
    assert list(scores) == predictions.prediction_score.tolist()
    assert lookup.coverage_counts() == (0, 0)
    assert len(actual) == 10
    assert (
        manifest["metadata"]["prediction_metadata"]["sample_filter_mode"]
        == "unfiltered"
    )


@pytest.mark.parametrize("training_assets", [("btc_usdt",), ("btc_usdt", "eth_usdt")])
def test_crypto_validation_ic_uses_declared_training_scope(tmp_path, training_assets):
    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
        research_run_trial_references,
    )

    case = crypto_rank_case(
        tmp_path,
        algorithms=("linear_regression",),
        training_assets=training_assets,
        time_series_inputs=True,
    )
    entry = case.saved.models[0]
    reference = next(
        item
        for item in research_run_trial_references(case.layout, "run_rank_roundtrip")
        if item["semantic_identity"] == entry["trial_semantic_identity"]
    )
    plan, records = read_model_trial_ledger_reference(case.layout, reference)
    metrics = records[0].metrics
    with closing_iterator(
        case.saved.validation_chunks(
            fold_id=entry["fold_id"], model_id=entry["model_id"]
        )
    ) as chunks:
        validation = pd.concat(list(chunks), ignore_index=True)
    assert validation.asset_id.unique().tolist() == list(training_assets)
    assert len(validation) == metrics["validation_rows"] == 6 * len(training_assets)
    assert plan.folds[0].validation.end == pd.Timestamp("1970-01-01T00:00:05")
    if len(training_assets) == 1:
        expected = np.corrcoef(validation["label"], validation.prediction_score)[0, 1]
        assert expected == pytest.approx(1.0)
        assert metrics["validation_ic"] == pytest.approx(expected)
        assert metrics["validation_ic_reason"] is None
        assert metrics["validation_rankic"] is None
        assert metrics["validation_rankic_reason"] == "not_applicable"
    else:
        assert metrics["validation_ic"] is None
        assert metrics["validation_ic_reason"] == "no_valid_observations"
        assert metrics["validation_rankic_reason"] == "no_valid_observations"
    assert metrics["validation_mse"] == pytest.approx(
        np.mean((validation.prediction_score - validation["label"]) ** 2), abs=1e-28
    )


@pytest.mark.parametrize("member_kind", ("validation", "model_state"))
def test_crypto_new_reader_preserves_immutable_and_lazy_acceptance_contract(
    tmp_path, member_kind
):
    import hashlib
    import json
    import pickle

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
        research_run_trial_references,
    )
    from quant_research_platform.storage.identity import (
        filesystem_read_bytes,
        filesystem_write_bytes,
        native_filesystem_path,
        open_filesystem_file,
    )
    from quant_research_platform.storage.manifests import read_qrp_json_file
    from quant_research_platform.storage.parquet import parquet_table_writer

    case = crypto_rank_case(
        tmp_path,
        algorithms=("linear_regression",),
        training_assets=("btc_usdt",),
        time_series_inputs=True,
    )
    original_entry = case.saved.models[0]
    reference = next(
        item
        for item in research_run_trial_references(case.layout, "run_rank_roundtrip")
        if item["semantic_identity"] == original_entry["trial_semantic_identity"]
    )
    _plan, records = read_model_trial_ledger_reference(case.layout, reference)
    assert records[0].metrics["validation_ic"] == pytest.approx(1.0)
    persisted.read_crypto_model_member(case.saved, case.saved.final_models[0])
    manifest_path = case.root / "manifest.json"
    _raw, manifest = read_qrp_json_file(manifest_path)
    original_manifest = deepcopy(manifest)
    if member_kind == "validation":
        from quant_research_platform.markets.implementations.crypto.portfolio import (
            _validation_rows,
        )

        relative = original_entry["validation_prediction_path"]
        with pq.ParquetFile(native_filesystem_path(case.root / relative)) as reader:
            schema = reader.schema_arrow
            frame = reader.read().to_pandas()
        frame["prediction_score"] = -frame.prediction_score
        with open_filesystem_file(case.root / relative, "wb") as handle:
            with parquet_table_writer(
                handle, compression="zstd", schema=schema
            ) as write:
                write(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
    else:
        relative = f"models/final/{case.saved.final_models[0]['model_id']}.pkl"
        filesystem_write_bytes(
            case.root / relative, pickle.dumps({"state": "rewritten"})
        )
    scientific_identity = case.saved.envelope["instance_identity"]
    with pytest.raises(
        ValueError, match="^semantic_artifact_member_integrity_invalid$"
    ):
        persisted.read_crypto_model_artifact(
            case.root, expected_identity=scientific_identity, trial_layout=case.layout
        )
    changed = filesystem_read_bytes(case.root / relative)
    for member in manifest["members"]:
        if member["path"] == relative:
            member.update(size=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    filesystem_write_bytes(manifest_path, (json.dumps(manifest) + "\n").encode("utf-8"))
    assert manifest["common_manifest"] == original_manifest["common_manifest"]
    assert manifest["operational_metadata"] == original_manifest["operational_metadata"]
    with pytest.raises(ValueError, match="^dataset_metadata_file_identity_changed$"):
        persisted.read_crypto_model_artifact(
            case.root,
            expected_identity=scientific_identity,
            trial_layout=case.layout,
            expected_member_identities=case.saved.member_identities,
        )
    fresh = persisted.read_crypto_model_artifact(
        case.root, expected_identity=scientific_identity, trial_layout=case.layout
    )
    if member_kind == "validation":
        with closing_iterator(
            fresh.validation_chunks(
                fold_id=original_entry["fold_id"], model_id=original_entry["model_id"]
            )
        ) as chunks:
            actual = pd.concat(list(chunks), ignore_index=True)
        pd.testing.assert_frame_equal(actual, frame, check_exact=True)
        with closing_iterator(
            _validation_rows(
                fresh,
                fold_id=original_entry["fold_id"],
                model_id=original_entry["model_id"],
                asset_id="btc_usdt",
            )
        ) as consumer:
            consumed = list(consumer)
        assert consumed == list(actual.itertuples(index=False, name=None))
        assert np.corrcoef(actual["label"], actual.prediction_score)[
            0, 1
        ] == pytest.approx(-1.0)
        _plan, unchanged = read_model_trial_ledger_reference(case.layout, reference)
        assert unchanged == records
    else:
        with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
            persisted.read_crypto_model_member(fresh, fresh.final_models[0])


@pytest.mark.parametrize(
    "difference",
    (
        {},
        {"training_asset_ids": ["btc_usdt"]},
        {"training_weighting": {"method": "equal_timestamp", "half_life": 8.0}},
        {"max_train_rows": 12},
        {"label_column": "future_mid_return_100ms"},
    ),
)
def test_crypto_prepares_each_training_domain_once_without_changing_fits(
    tmp_path,
    monkeypatch,
    difference,
):
    from quant_research_platform.research.models.scaler import ModelMatrixScaler

    counts = {"collect": 0, "scale": 0, "fit": 0}
    collect, scale, fit = (
        fitting._collect_training_frame,
        ModelMatrixScaler.fit,
        fitting._fit_model_result,
    )

    def collect_once(*args, **kwargs):
        counts["collect"] += 1
        return collect(*args, **kwargs)

    def scale_once(*args, **kwargs):
        counts["scale"] += 1
        return scale(*args, **kwargs)

    def fit_each(*args, **kwargs):
        counts["fit"] += 1
        return fit(*args, **kwargs)

    monkeypatch.setattr(fitting, "_collect_training_frame", collect_once)
    monkeypatch.setattr(ModelMatrixScaler, "fit", scale_once)
    monkeypatch.setattr(fitting, "_fit_model_result", fit_each)
    options = dict(
        algorithms=("linear_regression", "ridge_regression", "decision_tree_regressor"),
        model_updates={"ridge_regression": difference},
    )
    shared = crypto_rank_case(tmp_path / "shared", **options)
    domains = 2 if difference else 1
    assert counts == {"collect": 2 * domains, "scale": 2 * domains, "fit": 6}
    domain = production._training_domain
    monkeypatch.setattr(
        production, "_training_domain", lambda spec: (spec.model_id, *domain(spec))
    )
    counts.update(dict.fromkeys(counts, 0))
    isolated = crypto_rank_case(tmp_path / "isolated", **options)
    assert counts == {"collect": 6, "scale": 6, "fit": 6}
    assert shared.saved.envelope == isolated.saved.envelope
    expected = {entry["model_id"]: entry for entry in isolated.saved.final_models}
    for entry in shared.saved.final_models:
        other = expected[entry["model_id"]]
        assert entry == other
        model = persisted.read_crypto_model_member(shared.saved, entry)
        reference = persisted.read_crypto_model_member(isolated.saved, other)
        columns = tuple(entry["model_metadata"]["input_columns"])
        np.testing.assert_array_equal(
            fitting.prediction_callable(columns, model)(shared.frame),
            fitting.prediction_callable(columns, reference)(shared.frame),
        )
