from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.research.predictions import alignment
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.binance_hft import hft_research_plan


class _PredictionLookup:
    primary_key = ("timestamp", "asset_id")
    prediction_column = "prediction_score"

    def __init__(self, values: dict[tuple[pd.Timestamp, str], float]) -> None:
        self.values = values
        self.matched: set[tuple[pd.Timestamp, str]] = set()

    @contextmanager
    def values_for_frame(self, frame, *, mark_matched: bool):  # noqa: ANN001
        values: list[float] = []
        found = bytearray()
        for row in frame.itertuples(index=False):
            key = (row.timestamp, str(row.asset_id))
            value = self.values.get(key)
            if value is None:
                values.append(float("nan"))
                found.append(0)
            else:
                values.append(value)
                found.append(1)
                if mark_matched:
                    self.matched.add(key)
        yield values, found

    def coverage_counts(self) -> tuple[int, int]:
        return len(set(self.values).difference(self.matched)), 0


def test_hft_prediction_alignment_requires_usable_model_rows_and_rejects_extras(
    monkeypatch,
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto import portfolio
    from quant_research_platform.research.protocols.crypto_hft import (
        build_hft_research_dataset_catalog,
    )

    times = pd.to_datetime(
        [
            "1970-01-01T00:00:00.100",
            "1970-01-01T00:00:00.200",
            "1970-01-01T00:00:00.300",
        ]
    )
    frame = pd.DataFrame(
        {
            "market_id": ["crypto"] * 3,
            "timestamp": times,
            "asset_id": ["btc_usdt"] * 3,
            "segment_id": ["book-1:trade-1"] * 3,
            "book_segment_id": ["book-1"] * 3,
            "trade_connection_id": ["trade-1"] * 3,
            "ofi_100ms": [1.0, float("nan"), 2.0],
            "eligible": [True, True, False],
        }
    )
    manifest = {
        "metadata": {
            "prediction_metadata": {
                "input_columns": ["ofi_100ms"],
                "sample_filter_mode": "eligible",
            }
        }
    }

    @contextmanager
    def bundle(*_args, **_kwargs):
        yield manifest, lookup

    monkeypatch.setattr(
        portfolio,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: iter((frame,)),
    )
    dispatches = []

    def dispatch(*args, prediction_lookup, **kwargs):
        dispatches.append(prediction_lookup)
        yield from alignment._iter_prediction_joined_chunks(
            *args, prediction_lookup=prediction_lookup, **kwargs
        )

    monkeypatch.setattr(portfolio, "_iter_prediction_joined_chunks", dispatch)
    window = TimeWindow("1970-01-01T00:00:00.100", "1970-01-01T00:00:00.300")
    with runtime_resource_policy_scope() as policy:

        def rows():
            return list(
                portfolio._prediction_rows(
                    layout=StorageLayout(Path("fixture")),
                    dataset=build_hft_research_dataset_catalog().get(
                        HFT_RESEARCH_DATASET_ID
                    ),
                    windows=(("fold", window),),
                    model_id="model",
                    asset_id="btc_usdt",
                    protocol_manifest={
                        "model_scopes": {
                            "model": {"evaluation_asset_ids": ["btc_usdt"]}
                        }
                    },
                    resource_policy=policy,
                    read_prediction_bundle=bundle,
                    output_window=window,
                )
            )

        lookup = _PredictionLookup({})
        with pytest.raises(ValueError, match="missing_required_prediction_keys"):
            rows()
        lookup = _PredictionLookup({(times[0], "btc_usdt"): 0.5})
        actual = rows()
        assert [row["prediction_score"] for row in actual] == [0.5, None, None]
        lookup = _PredictionLookup(
            {(times[0], "btc_usdt"): 0.5, (times[2], "btc_usdt"): 0.7}
        )
        with pytest.raises(ValueError, match="misaligned_persisted_prediction_keys"):
            rows()
        assert len(dispatches) == 3 and policy.used_memory_bytes == 0


@pytest.mark.parametrize("selected_folds", [22, 21, 1])
def test_crypto_valid_selected_windows_reach_training(tmp_path, selected_folds):
    from copy import deepcopy
    from datetime import datetime, timedelta

    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.models import (
        CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    )
    from quant_research_platform.research.models.persisted import (
        write_crypto_model_artifact,
    )
    from quant_research_platform.research.protocols.crypto_hft import protocol

    plan = hft_research_plan()
    template = deepcopy(plan["folds"][0])
    plan["folds"] = []
    for index in range(22):
        fold = deepcopy(template)
        fold["fold_id"] = f"fold_{index + 1}"
        for window in ("train", "validation", "development_test"):
            fold[window] = {
                bound: (
                    datetime.fromisoformat(value) + timedelta(seconds=10 * index)
                ).isoformat()
                for bound, value in fold[window].items()
            }
        plan["folds"].append(fold)
    plan["final_holdout"] = {
        "start": "1970-01-01T00:03:36",
        "end": "1970-01-01T00:03:37",
    }
    manifest = protocol(HFT_RESEARCH_DATASET_ID, research_plan=plan).manifest()
    envelope = build_business_manifest_envelope(
        logical_artifact_id=f"{HFT_RESEARCH_DATASET_ID}_models",
        schema_identity=canonical_json_hash(
            {"schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID}
        ),
        protocol_or_spec_identity=canonical_json_hash(manifest),
        software_identity="0" * 64,
        research_window={
            "start": plan["folds"][0]["development_test"]["start"],
            "end": plan["folds"][selected_folds - 1]["development_test"]["end"],
            "access": "development",
        },
        upstream_identities={HFT_RESEARCH_DATASET_ID: "1" * 64},
        semantic_metadata={"target_contract": {"protocol": manifest}},
    )
    calls = []

    def model_results(*, on_validation):
        calls.append(True)
        raise RuntimeError("training_entry_sentinel")

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(
            RuntimeError,
            match="training_entry_sentinel",
        ):
            write_crypto_model_artifact(
                tmp_path / "models",
                envelope=envelope,
                model_results=model_results,
                trial_layout=None,
            )
        assert len(calls) == 1
        assert policy.used_memory_bytes == 0
    assert not tuple((tmp_path / "models" / "instances").glob("*"))


def test_crypto_explicit_folds_publish_nine_models_and_reload_predictions(
    tmp_path, monkeypatch
):
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        research_profile,
    )
    from quant_research_platform.research.models.persisted import (
        read_crypto_model_artifact,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_RESEARCH_DATASET_ID,
    )
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows import research
    from tests.helpers.research_acceptance import run_crypto_acceptance

    layout = StorageLayout(tmp_path / "project")
    raw = layout.project_root / "raw/recorder"
    profile = research_profile(CRYPTO_PROFILE_ID)
    plan = hft_research_plan()
    validation_rows = {}
    write_models = research.write_crypto_model_artifact
    from quant_research_platform.storage import parquet as parquet_storage
    from quant_research_platform.storage.parquet import (
        _parquet_row_group_footer_upper_bound,
    )

    table_write_state = parquet_storage._parquet_table_write_state
    expected_footer = 0
    multirow_callbacks = []

    def one_row_groups(table, *, previous_metadata, retained_memory):
        nonlocal expected_footer
        if previous_metadata is None:
            expected_footer = 0
        assert retained_memory[1] == expected_footer
        converted, state, _batch_rows, retained = table_write_state(
            table, previous_metadata=previous_metadata, retained_memory=retained_memory
        )
        expected_footer += table.num_rows * _parquet_row_group_footer_upper_bound(
            converted.schema, rows=0
        )
        multirow_callbacks.append(table.num_rows)
        return converted, state, 1, retained

    def write_fragmented_validation(*args, model_results, **kwargs):
        def fragmented_results(*, on_validation):
            def write_rows(fold, spec, frame):
                key = (fold.fold_id, spec.model_id)
                validation_rows.setdefault(key, []).append(
                    frame.rename(columns={spec.label_column: "label"}).copy()
                )
                for index in range(0, len(frame), 3):
                    on_validation(fold, spec, frame.iloc[index : index + 3])

            return model_results(on_validation=write_rows)

        with monkeypatch.context() as writing_patch:
            writing_patch.setattr(
                parquet_storage, "_parquet_table_write_state", one_row_groups
            )
            return write_models(*args, model_results=fragmented_results, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            research, "write_crypto_model_artifact", write_fragmented_validation
        )
        result, report = run_crypto_acceptance(layout.project_root)
    assert max(multirow_callbacks) == 3
    targets = {item["target_id"]: item for item in result["target_results"]}
    model_id = f"{HFT_RESEARCH_DATASET_ID}_models"
    prediction_id = f"{HFT_RESEARCH_DATASET_ID}_predictions"
    evaluation_ids = tuple(spec.evaluation_id for spec in profile.evaluations.list())
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        maker_candidates,
        taker_candidates,
    )

    calibration_target = f"{HFT_RESEARCH_DATASET_ID}_maker_calibration"
    statistics_target = f"{HFT_RESEARCH_DATASET_ID}_portfolio_statistics"
    root = layout.instance_root(
        layout.semantic_artifact_root("crypto", model_id),
        targets[model_id]["instance_identity"],
    )
    saved = read_crypto_model_artifact(
        root,
        expected_identity=targets[model_id]["instance_identity"],
        trial_layout=layout,
    )
    import pyarrow.parquet as pq

    from quant_research_platform.research.models.persisted import (
        _crypto_validation_chunks,
    )
    from quant_research_platform.storage.identity import native_filesystem_path

    with runtime_resource_policy_scope() as policy:
        for entry in saved.models:
            key = (entry["fold_id"], entry["model_id"])
            relative = entry["validation_prediction_path"]
            assert "validation_prediction_parts" not in entry
            if key not in validation_rows:
                assert relative is None
                continue
            expected_validation = pd.concat(validation_rows[key], ignore_index=True)
            expected_validation["label_available_at"] = expected_validation[
                "label_available_at"
            ].astype("datetime64[ns]")
            actual_validation = pd.concat(
                _crypto_validation_chunks(root, saved.member_records, entry),
                ignore_index=True,
            )
            pd.testing.assert_frame_equal(
                actual_validation, expected_validation.loc[:, actual_validation.columns]
            )
            assert len(expected_validation) > 1
            with pq.ParquetFile(native_filesystem_path(root / relative)) as parquet:
                assert parquet.metadata.num_row_groups == len(expected_validation)
            assert (
                sum(member["path"] == relative for member in saved.member_records) == 1
            )
        assert len(validation_rows) == 4
        assert policy.used_memory_bytes == 0
    validation_rows.clear()

    import numpy as np

    from quant_research_platform.artifacts.outputs import read_prediction_output_bundle
    from quant_research_platform.research.models import fitting
    from quant_research_platform.storage.datasets.reading import (
        DatasetReadMode,
        read_dataset,
    )

    dataset_root = layout.instance_root(
        layout.research_dataset_root("crypto", HFT_RESEARCH_DATASET_ID),
        targets[HFT_RESEARCH_DATASET_ID]["instance_identity"],
    )
    prediction_root = layout.instance_root(
        layout.semantic_artifact_root("crypto", prediction_id),
        targets[prediction_id]["instance_identity"],
    )
    with runtime_resource_policy_scope() as policy:
        frame = read_dataset(
            dataset_root, mode=DatasetReadMode.MATERIALIZED, resource_policy=policy
        )
        oos = frame.timestamp.between(
            pd.Timestamp("1970-01-01T00:00:04"), pd.Timestamp("1970-01-01T00:00:05")
        ) | frame.timestamp.between(
            pd.Timestamp("1970-01-01T00:00:06.500000"),
            pd.Timestamp("1970-01-01T00:00:08.500000"),
        )
        for spec in profile.models.list():
            entries = [
                entry for entry in saved.models if entry["model_id"] == spec.model_id
            ]
            expected = frame.loc[oos & frame.eligible].dropna(
                subset=list(spec.input_columns)
            )
            assets = entries[0]["evaluation_asset_ids"]
            if assets:
                expected = expected.loc[expected.asset_id.isin(assets)]
            output_id = f"{spec.model_id}_prediction"
            with read_prediction_output_bundle(
                prediction_root / "outputs" / output_id,
                expected_output_id=output_id,
                expected_run_id=f"run_{output_id}",
            ) as (manifest, lookup):
                assert manifest["metadata"][
                    "model_metadata"
                ] == saved.prediction_lineage(spec.model_id)
                assert lookup.row_count == len(expected)
                with lookup.values_for_frame(expected, mark_matched=True) as (
                    values,
                    found,
                ):
                    assert all(found) and np.isfinite(values).all()
                assert lookup.coverage_counts() == (0, 0)
        assert (
            frame.loc[
                frame.timestamp.eq(pd.Timestamp("1970-01-01T00:00:08.500000")),
                "future_mid_return_500ms",
            ]
            .isna()
            .all()
        )
        from tests.research.crypto_evaluation_test_support import (
            check_crypto_evaluation_readback,
        )

        check_crypto_evaluation_readback(
            layout=layout,
            profile=profile,
            targets=targets,
            frame=frame,
            plan=plan,
            prediction_root=prediction_root,
        )
        frame = expected = None
        assert policy.used_memory_bytes == 0
    before = tuple(
        sorted((layout.project_root / "data/runs/trials").rglob("records/*.json"))
    )

    def no_fit(*args, **kwargs):
        raise AssertionError("REUSE must not fit")

    with monkeypatch.context() as patch:
        patch.setattr(fitting, "_fit_model_result_owned_from_frames", no_fit)
        reused = research.research_run(
            profile=profile,
            layout=layout,
            scenario_id=HFT_RESEARCH_DATASET_ID,
            start_date="1970-01-01T00:00:04",
            end_date="1970-01-01T00:00:08.500000",
            raw_root=raw,
            research_plan=plan,
        )
    observed = {item["target_id"]: item for item in reused["target_results"]}
    assert observed[model_id]["status"] == observed[prediction_id]["status"] == "REUSED"
    assert all(observed[target]["status"] == "REUSED" for target in evaluation_ids)
    assert all(
        observed[candidate.target_id]["status"] == "REUSED"
        for candidate in (*taker_candidates(), *maker_candidates())
    )
    assert observed[calibration_target]["status"] == "REUSED"
    assert observed[statistics_target]["status"] == "REUSED"
    from tests.interfaces.published_research_test_support import (
        check_published_evidence_views,
    )

    check_published_evidence_views(
        layout=layout,
        profile=profile,
        run_id=result["run_id"],
        other_run_id=reused["run_id"],
        report_identity=report["target_result"]["instance_identity"],
        monkeypatch=monkeypatch,
    )
    missing_execution = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id=HFT_RESEARCH_DATASET_ID,
        start_date="1970-01-01T00:00:04",
        end_date="1970-01-01T00:00:08.500000",
        raw_root=raw,
        research_plan={key: value for key, value in plan.items() if key != "execution"},
        plan=True,
    )
    missing_targets = {row["target_id"]: row for row in missing_execution["targets"]}
    assert all(
        missing_targets[target]["decision"] != "BLOCKED"
        for target in (model_id, prediction_id, *evaluation_ids)
    )
    assert all(
        missing_targets[candidate.target_id]["blocker"]
        == "crypto_execution_plan_missing"
        for candidate in taker_candidates()
    )
    eth_metadata = raw.parent / "funding/ETHUSDT/response.metadata.json"
    saved_metadata = eth_metadata.read_bytes()
    eth_metadata.unlink()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(fitting, "_fit_model_result_owned_from_frames", no_fit)
            missing_eth = research.research_run(
                profile=profile,
                layout=layout,
                scenario_id=HFT_RESEARCH_DATASET_ID,
                start_date="1970-01-01T00:00:04",
                end_date="1970-01-01T00:00:08.500000",
                raw_root=raw,
                research_plan=plan,
            )
        missing_targets = {
            row["target_id"]: row for row in missing_eth["target_results"]
        }
        assert all(
            missing_targets[target]["status"] == "REUSED"
            for target in (model_id, prediction_id, *evaluation_ids)
        )
        for candidate in taker_candidates():
            assert missing_targets[candidate.target_id]["status"] == (
                "BLOCKED" if candidate.asset_id == "eth_usdt" else "REUSED"
            )
    finally:
        eth_metadata.write_bytes(saved_metadata)
    btc_metadata = raw.parent / "funding/BTCUSDT/response.metadata.json"
    saved_metadata = btc_metadata.read_bytes()
    incomplete = json.loads(saved_metadata)
    incomplete["metadata"]["coverage"]["end_ns"] = 8_000_000_000
    btc_metadata.write_text(json.dumps(incomplete), encoding="utf-8")
    old_good = {
        candidate.target_id: (
            layout.instance_root(
                layout.semantic_artifact_root("crypto", candidate.target_id),
                targets[candidate.target_id]["instance_identity"],
            )
            / "manifest.json"
        ).read_bytes()
        for candidate in taker_candidates()
    }
    try:
        with monkeypatch.context() as patch:
            patch.setattr(fitting, "_fit_model_result_owned_from_frames", no_fit)
            incomplete_run = research.research_run(
                profile=profile,
                layout=layout,
                scenario_id=HFT_RESEARCH_DATASET_ID,
                start_date="1970-01-01T00:00:04",
                end_date="1970-01-01T00:00:08.500000",
                raw_root=raw,
                research_plan=plan,
            )
        incomplete_targets = {
            row["target_id"]: row for row in incomplete_run["target_results"]
        }
        assert all(
            incomplete_targets[target]["status"] == "REUSED"
            for target in (model_id, prediction_id, *evaluation_ids)
        )
        with runtime_resource_policy_scope() as policy:
            for candidate in taker_candidates():
                incomplete_result = incomplete_targets[candidate.target_id]
                assert incomplete_result["status"] == (
                    "BLOCKED" if candidate.asset_id == "btc_usdt" else "REUSED"
                )
                if candidate.asset_id == "btc_usdt":
                    assert (
                        incomplete_result["reason"]
                        == "binance_funding_coverage_incomplete"
                    )
                old_identity = targets[candidate.target_id]["instance_identity"]
                old_root = layout.instance_root(
                    layout.semantic_artifact_root("crypto", candidate.target_id),
                    old_identity,
                )
                assert (old_root / "manifest.json").read_bytes() == old_good[
                    candidate.target_id
                ]
                read_portfolio_ledger_artifact(old_root, expected_identity=old_identity)
            assert policy.used_memory_bytes == 0
    finally:
        btc_metadata.write_bytes(saved_metadata)
    from tests.research.crypto_report_test_support import check_crypto_report_lifecycle

    check_crypto_report_lifecycle(
        layout=layout,
        profile=profile,
        complete_run=result,
        reused_run=reused,
        blocked_run=incomplete_run,
        targets=targets,
        monkeypatch=monkeypatch,
    )
    assert (
        tuple(
            sorted((layout.project_root / "data/runs/trials").rglob("records/*.json"))
        )
        == before
    )
    from tests.research.crypto_model_test_support import (
        check_crypto_fold_boundaries,
        check_crypto_model_staging,
        check_crypto_prediction_staging,
        check_crypto_wide_fold_artifact,
    )

    check_crypto_fold_boundaries(
        layout=layout,
        dataset_root=dataset_root,
        profile=profile,
        saved=saved,
        monkeypatch=monkeypatch,
    )
    check_crypto_wide_fold_artifact(
        layout=layout,
        dataset_root=dataset_root,
        profile=profile,
        saved=saved,
        monkeypatch=monkeypatch,
    )
    check_crypto_model_staging(layout=layout, profile=profile, saved=saved)
    check_crypto_prediction_staging(
        layout=layout, prediction_root=prediction_root, saved=saved
    )
    from tests.research.crypto_evaluation_test_support import (
        check_crypto_evaluation_input_boundaries,
        check_crypto_evaluation_old_good,
    )

    check_crypto_evaluation_input_boundaries(
        layout=layout,
        profile=profile,
        targets=targets,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        plan=plan,
        monkeypatch=monkeypatch,
    )
    check_crypto_evaluation_old_good(
        layout=layout,
        profile=profile,
        targets=targets,
        raw=raw,
        plan=plan,
    )
    insufficient_plan = {
        **plan,
        "execution": {
            **plan["execution"],
            "maker": {**plan["execution"]["maker"], "min_calibration_samples": 100000},
        },
    }
    insufficient = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id=HFT_RESEARCH_DATASET_ID,
        start_date="1970-01-01T00:00:04",
        end_date="1970-01-01T00:00:08.500000",
        raw_root=raw,
        research_plan=insufficient_plan,
    )
    insufficient_targets = {
        row["target_id"]: row for row in insufficient["target_results"]
    }
    assert insufficient["status"] == "BLOCKED"
    assert insufficient_targets[calibration_target]["status"] == "BUILT"
    assert insufficient_targets[maker_candidates()[1].target_id]["reason"] == (
        "maker_calibration_insufficient_samples"
    )
    assert insufficient_targets[statistics_target]["status"] == "BLOCKED"
    assert all(
        insufficient_targets[candidate.target_id]["status"] in {"BUILT", "REUSED"}
        for candidate in (*taker_candidates(), maker_candidates()[0])
    )
    assert all(
        insufficient_targets[target]["status"] == "BUILT"
        for target in (model_id, prediction_id, *evaluation_ids)
    )
    assert (
        insufficient_targets[model_id]["instance_identity"]
        != targets[model_id]["instance_identity"]
    )
