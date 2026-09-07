"""Independent fold-maturity counts and publication failure checks."""

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.contracts.targets import ProducerBinding
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.evaluations import runner
from quant_research_platform.research.evaluations.engine import EvaluationResult
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage._scratch_ownership import _windows_extended_path
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact,
    read_semantic_artifact_record,
)
from quant_research_platform.workflows import research


def check_crypto_evaluation_readback(
    *, layout, profile, targets, frame, plan, prediction_root
):
    def read_parquet(path):
        return pd.read_parquet(
            _windows_extended_path(path) if os.name == "nt" else path
        )

    oos = pd.Series(False, index=frame.index)
    cutoff = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for fold in plan["folds"]:
        window = fold["development_test"]
        mask = frame.timestamp.between(
            pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        )
        oos |= mask
        cutoff.loc[mask] = pd.Timestamp(window["end"])
    for spec in profile.evaluations.list():
        identity = targets[spec.evaluation_id]["instance_identity"]
        root = layout.instance_root(
            layout.semantic_artifact_root("crypto", spec.evaluation_id), identity
        )
        read_semantic_artifact(root, expected_identity=identity)
        bundle = root / "outputs" / spec.evaluation_id
        manifest, _ = validate_output_bundle(
            bundle,
            expected_output_id=spec.evaluation_id,
            expected_run_id=f"run_{spec.evaluation_id}",
        )
        metrics = read_parquet(bundle / "metrics.parquet")
        work = frame.copy()
        assets = ()
        if spec.signals[0].source_kind == "prediction":
            model_id = spec.signals[0].signal_id
            predictions = read_parquet(
                prediction_root
                / "outputs"
                / f"{model_id}_prediction"
                / "prediction.parquet"
            )
            work = work.merge(
                predictions,
                on=["timestamp", "asset_id"],
                how="left",
                validate="one_to_one",
            )
            scopes = profile.protocol(research_plan=plan).manifest()["model_scopes"]
            assets = tuple(scopes[model_id]["evaluation_asset_ids"])
        scope = oos & work.eligible
        if assets:
            scope &= work.asset_id.isin(assets)
        for signal in spec.signals:
            for label, availability in spec.label_availability_columns.items():
                mask = (
                    scope
                    & work[availability].le(cutoff)
                    & work[label].notna()
                    & work[signal.source_column].notna()
                )
                pair = metrics.loc[
                    metrics.signal.eq(signal.signal_id) & metrics.label.eq(label)
                ]
                assert int(pair.n_obs.sum()) == int(mask.sum()), (
                    spec.evaluation_id,
                    label,
                )
                if (
                    spec.evaluation_axis == "single_asset_time_series"
                    and mask.sum() > 1
                ):
                    values = work.loc[mask, [signal.source_column, label]].to_numpy(
                        dtype=float
                    )
                    if np.std(values[:, 0]) > 0 and np.std(values[:, 1]) > 0:
                        assert float(pair.ic.iloc[0]) == pytest.approx(
                            float(np.corrcoef(values.T)[0, 1]), abs=1e-10
                        )
        diagnostics = manifest["metadata"]["evaluation_diagnostics"]
        assert (
            diagnostics["min_timestamp"]
            >= plan["folds"][0]["development_test"]["start"]
        )


def check_crypto_evaluation_old_good(*, layout, profile, targets, raw, plan):
    target = next(
        spec.evaluation_id
        for spec in profile.evaluations.list()
        if spec.signals[0].source_kind == "factor"
    )
    logical = layout.semantic_artifact_root("crypto", target)
    root = layout.instance_root(logical, targets[target]["instance_identity"])

    original = read_semantic_artifact_record(root)
    binding = profile.producer_bindings[target]
    primary = KeyboardInterrupt("evaluation_interrupted_after_output")

    def interrupted(**kwargs):
        result = binding.producer(**kwargs)

        def chunks():
            with result:
                for chunk in result.output_chunks:
                    yield chunk
                    raise primary

        source = chunks()

        def close():
            source.close()
            raise RuntimeError("evaluation_cleanup_failed")

        return EvaluationResult(source, result.summaries, close)

    altered = replace(
        profile,
        producer_bindings={
            **profile.producer_bindings,
            target: ProducerBinding(binding.kind, interrupted),
        },
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        research.research_run(
            profile=altered,
            layout=layout,
            scenario_id=profile.research_datasets.list()[0].data_id,
            start_date=plan["folds"][0]["development_test"]["start"],
            end_date=plan["folds"][-1]["development_test"]["end"],
            research_plan=plan,
            raw_root=raw,
            rebuild=True,
        )
    assert caught.value is primary
    assert any("RuntimeError" in note for note in primary.__notes__)
    assert read_semantic_artifact_record(root) == original
    read_semantic_artifact(root, expected_identity=targets[target]["instance_identity"])


def check_crypto_evaluation_input_boundaries(
    *, layout, profile, targets, dataset_root, prediction_root, plan, monkeypatch
):
    dataset_id = profile.research_datasets.list()[0].data_id
    target_layout = layout.with_target_roots(
        {dataset_id: dataset_root, f"{dataset_id}_predictions": prediction_root}
    )
    protocol = profile.protocol(research_plan=plan).manifest()
    window = TimeWindow(
        plan["folds"][0]["development_test"]["start"],
        plan["folds"][-1]["development_test"]["end"],
    )
    original_reader = runner.iter_research_dataset_chunks
    mode = "late_labels"

    def changed_reader(*args, **kwargs):
        # Controlled read faults over real published data; the prediction lookup
        # always reads and validates the actual immutable on-disk bundle.
        for frame in original_reader(*args, **kwargs):
            changed = frame.copy()
            if mode == "late_labels":
                for availability in changed.columns:
                    if availability.endswith("_available_at"):
                        label = availability.removesuffix("_available_at")
                        if label in changed:
                            changed.loc[
                                changed[availability].gt(
                                    pd.Timestamp(kwargs["time_window"].end)
                                ),
                                label,
                            ] = 999.0
            elif mode == "extra_prediction":
                changed = changed.iloc[1:].copy()
            else:
                changed.loc[changed.index[0], "timestamp"] += pd.Timedelta(
                    microseconds=1
                )
            yield changed

    with monkeypatch.context() as patch, runtime_resource_policy_scope() as policy:
        patch.setattr(runner, "iter_research_dataset_chunks", changed_reader)

        def compute(spec):
            with profile.producer_bindings[spec.evaluation_id].producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                evaluation_catalog=profile.evaluations,
                evaluation_id=spec.evaluation_id,
                time_window=window,
                resource_policy=policy,
                protocol_manifest=protocol,
            ) as result:
                return pd.concat(
                    [chunk.value[0].copy() for chunk in result.output_chunks],
                    ignore_index=True,
                )

        for spec in profile.evaluations.list():
            actual = compute(spec)
            root = layout.instance_root(
                layout.semantic_artifact_root("crypto", spec.evaluation_id),
                targets[spec.evaluation_id]["instance_identity"],
            )
            path = root / "outputs" / spec.evaluation_id / "metrics.parquet"
            expected = pd.read_parquet(
                _windows_extended_path(path) if os.name == "nt" else path
            )
            pd.testing.assert_frame_equal(actual, expected)
            assert policy.used_memory_bytes == 0
        spec = profile.evaluations.get("crypto_hft_linear_regression_oos_evaluation")
        for fault in ("extra_prediction", "missing_prediction"):
            mode = fault
            with pytest.raises(
                ValueError, match="misaligned_persisted_prediction_keys"
            ):
                compute(spec)
            assert policy.used_memory_bytes == 0
