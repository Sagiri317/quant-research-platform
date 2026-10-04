from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs import read_prediction_output_bundle
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research import statistics_production
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
    canonical_feature_transforms,
    model_input_validity_columns,
)
from quant_research_platform.research.portfolios.production import (
    _DATASET_COLUMNS,
    _prediction_alpha,
    _session_frame,
)
from quant_research_platform.research.predictions import runner
from quant_research_platform.research.protocols import a_share
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.prediction_output_helpers import write_prediction_result
from tests.research.workflow_test_support import shared_model_case, shared_model_rows


@pytest.mark.parametrize(
    "algorithm,input_column",
    [
        ("equal_weight_rank_ensemble", "x"),
        ("equal_weight_rank_ensemble", "feature_x"),
        ("linear_regression", "feature_x"),
        ("linear_regression", "feature_missing_x"),
        ("linear_regression", "x"),
    ],
)
@pytest.mark.parametrize("missing_required", [False, True])
def test_transform_prediction_publication_and_consumers_share_exact_validity(
    tmp_path, monkeypatch, algorithm, input_column, missing_required
):
    count = 106
    raw = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{index:03d}" for index in range(count)],
            "timestamp": pd.Timestamp("2019-02-01"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "industry",
            "log_float_market_cap": 20 + np.arange(count) * 0.01,
            "x": np.sin(np.arange(count)),
            "float_market_cap": 100.0,
            "forward_return_20d": np.arange(count) * 0.001,
            "forward_return_20d_available_at": pd.Timestamp("2019-03-01"),
        }
    )
    raw.loc[count - 2, "x"] = np.nan
    raw.loc[count - 1, "log_float_market_cap"] = np.nan
    frame = (
        apply_feature_transforms(raw, canonical_feature_transforms(("x",)))
        .iloc[-4:]
        .copy()
    )
    assert frame.feature_x.notna().sum() == 3
    expected = (
        frame.asset_id.tolist()
        if algorithm == "equal_weight_rank_ensemble" and input_column == "x"
        else frame.loc[frame.x.notna(), "asset_id"].tolist()
        if input_column == "x"
        else frame.asset_id.iloc[:3].tolist()
    )
    spec = ModelSpec(
        model_id="fixture_model",
        model_algorithm=algorithm,
        input_factor_columns=(input_column,) if input_column == "x" else (),
        input_feature_columns=() if input_column == "x" else (input_column,),
        label_column="forward_return_20d",
    )
    dataset = make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=("x",),
        feature_columns=("feature_x", "feature_missing_x", "feature_observed_x"),
        feature_transforms=canonical_feature_transforms(("x",)),
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    frame = frame.loc[:, list(dataset.required_columns)]
    dates = sorted(
        set(shared_model_rows((spec,), development_end="2019-03-31").timestamp)
    )
    history = (
        pd.concat(
            [
                frame.assign(
                    timestamp=date,
                    forward_return_20d_available_at=date + pd.offsets.BDay(1),
                )
                for date in dates
            ]
            + [
                frame.loc[frame.asset_id.ne(expected[-1])]
                if missing_required
                else frame
            ],
            ignore_index=True,
        )
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    window = TimeWindow("2019-01-01", "2019-03-31")
    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("test.artifact.metadata", 0) as artifact_lease,
    ):
        case = shared_model_case(
            tmp_path,
            specs=(spec,),
            frame=history,
            dataset=dataset,
            development_end="2019-03-31",
            lease=artifact_lease,
        )
        artifact, dataset = case.artifact, case.dataset
        dataset_root = case.layout.instance_root(
            case.layout.research_dataset_root(dataset.market_id, dataset.data_id),
            "d" * 64,
        )
        layout = case.layout.with_target_roots({dataset.data_id: dataset_root})
        write_dataset(
            dataset_root,
            frame,
            spec=dataset,
            resource_policy=policy,
            borrowed_input=True,
        )
        prediction_spec = PredictionSpec(
            prediction_id="prediction",
            model_id="model-artifact",
            research_dataset_id=dataset.data_id,
            train_end=None,
            test_start="2019-02-01",
            test_end="2019-02-01",
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        )
        result = runner.build_prediction_result_from_persisted_model(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((dataset,)),
            prediction_spec=prediction_spec,
            persisted_model=artifact,
            resource_policy=policy,
            selection_group="global",
        )
        bundle = write_prediction_result(layout, result)
        with read_prediction_output_bundle(
            bundle,
            expected_output_id="prediction",
            expected_run_id="test_run",
            family_selection="champion",
        ) as (_manifest, lookup):
            assert lookup.row_count == len(expected) - int(missing_required)
            validity = model_input_validity_columns(spec.input_columns, algorithm)
            session = _session_frame(
                frame, tuple(dict.fromkeys((*_DATASET_COLUMNS, *validity)))
            )
            if missing_required:
                with pytest.raises(
                    ValueError, match="^missing_required_prediction_keys$"
                ):
                    _prediction_alpha(session, lookup, validity)
            else:
                alpha = _prediction_alpha(session, lookup, validity)
                assert alpha.dropna().index.tolist() == expected
                assert alpha.index.tolist() == frame.asset_id.tolist()
                assert lookup.coverage_counts() == (0, 0)

        statistics_protocol = deepcopy(a_share.protocol("pv_all").manifest())
        statistics_protocol.update(
            selection_group="global",
            formal_dataset_window=case.protocol["formal_dataset_window"],
            development_window=case.protocol["development_window"],
        )
        statistics_protocol["statistics"].update(
            factor_fdr_families={"fixture": ["x"]},
            fama_macbeth_core_factors=["x"],
        )
        statistics_protocol["research_configuration"] = {
            "evaluation": {"factors": ["x"], "primary_label": "forward_return_20d"},
            "comparison_model_id": spec.model_id,
        }
        with read_prediction_output_bundle(
            bundle,
            expected_output_id="prediction",
            expected_run_id="test_run",
        ) as (_manifest, lookup):

            def statistics():
                return statistics_production.build_a_share_statistics_tables(
                    layout=layout,
                    dataset_catalog=ResearchDatasetCatalog((dataset,)),
                    dataset_id=dataset.data_id,
                    prediction_id="prediction",
                    persisted_model=artifact,
                    protocol_manifest=statistics_protocol,
                    time_window=window,
                    prediction_lookup=lookup,
                    resource_policy=policy,
                    global_statistics=None,
                )

            if missing_required:
                with pytest.raises(
                    ValueError, match="^missing_required_prediction_keys$"
                ):
                    statistics()
            else:
                tables = statistics()
                try:
                    assert lookup.coverage_counts() == (0, 0)
                    assert tables.value["model_losses"].observation_count.tolist() == [
                        len(expected)
                    ]
                    envelope = build_business_manifest_envelope(
                        logical_artifact_id="statistics",
                        schema_identity="a" * 64,
                        protocol_or_spec_identity="b" * 64,
                        software_identity="c" * 64,
                        research_window={
                            "start": "2019-01-01",
                            "end": "2019-03-31",
                            "access": "development",
                        },
                        upstream_identities={},
                        semantic_metadata={},
                    )
                    root = write_statistics_artifact(
                        tmp_path / "statistics", envelope=envelope, tables=tables.value
                    )
                finally:
                    tables.release(owner="research.statistics.output")
                restored = read_statistics_artifact(
                    root,
                    expected_identity=envelope.instance_identity,
                    lease=artifact_lease,
                )
                with restored.table("model_losses.parquet") as saved:
                    assert saved.observation_count.tolist() == [len(expected)]
        assert policy.used_memory_bytes == artifact_lease.reserved_bytes


def test_frozen_prediction_alpha_is_permutation_invariant_and_reader_uses_alpha(
    tmp_path,
):
    from quant_research_platform.research.predictions.alignment import (
        _iter_prediction_joined_chunks,
    )

    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("test.artifact.metadata", 0) as artifact_lease,
    ):
        case = shared_model_case(
            tmp_path,
            dataset=make_demo_research_dataset_spec(
                data_id="fixture",
                input_data_ids=("processed_demo",),
                base_input_data_id="processed_demo",
                factor_columns=("x",),
                label_columns=("label", "label_available_at"),
                filter_columns=("eligible",),
                context_columns=(),
            ),
            lease=artifact_lease,
        )
        frame = case.frame.iloc[:3].copy()
        frame["timestamp"] = pd.Timestamp("2024-01-02")
        frame["label_available_at"] = pd.Timestamp("2024-01-03")
        prediction_spec = PredictionSpec(
            prediction_id="prediction",
            model_id="model-artifact",
            research_dataset_id=case.dataset.data_id,
            train_end=None,
            test_start="2024-01-02",
            test_end="2024-01-02",
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        )
        tables = []
        for index, order in enumerate(([0, 1, 2], [2, 0, 1])):
            from quant_research_platform.storage.layout import StorageLayout

            layout = StorageLayout(tmp_path / f"order_{index}")
            dataset_root = layout.instance_root(
                layout.research_dataset_root(
                    case.dataset.market_id, case.dataset.data_id
                ),
                "d" * 64,
            )
            layout = layout.with_target_roots({case.dataset.data_id: dataset_root})
            selected = frame.iloc[order].reset_index(drop=True)
            write_dataset(
                dataset_root,
                selected,
                spec=case.dataset,
                resource_policy=policy,
                borrowed_input=True,
            )
            result = runner.build_prediction_result_from_persisted_model(
                layout=layout,
                dataset_catalog=ResearchDatasetCatalog((case.dataset,)),
                prediction_spec=prediction_spec,
                persisted_model=case.artifact,
                resource_policy=policy,
                selection_group="global",
                access_window="holdout",
            )
            bundle = write_prediction_result(layout, result)
            tables.append(
                pd.read_parquet(bundle / "prediction.parquet")
                .sort_values("asset_id")
                .reset_index(drop=True)
            )
        pd.testing.assert_frame_equal(*tables)
        assert tables[0].alpha_score.tolist() == pytest.approx([-1 / 3, 0, 1 / 3])
        requested = frame.iloc[[2, 0, 1]].reset_index(drop=True)
        for mutation in (None, "asset", "timestamp"):
            selected = requested.copy()
            if mutation == "asset":
                selected.loc[0, "asset_id"] = "missing_asset"
            if mutation == "timestamp":
                selected.loc[0, "timestamp"] += pd.Timedelta(days=1)
            with read_prediction_output_bundle(
                bundle,
                expected_output_id="prediction",
                expected_run_id="test_run",
                family_selection="champion",
            ) as (_manifest, lookup):
                assert lookup.prediction_column == "alpha_score"
                joined = _iter_prediction_joined_chunks(
                    (selected.iloc[:1], selected.iloc[1:]),
                    prediction_lookup=lookup,
                    input_columns=("x",),
                    model_algorithm="linear_regression",
                    sample_filter_mode=prediction_spec.sample_filter_mode,
                    test_start=prediction_spec.test_start,
                    test_end=prediction_spec.test_end,
                )
                if mutation is None:
                    matched = pd.concat(joined, ignore_index=True)
                    assert matched.asset_id.tolist() == requested.asset_id.tolist()
                    assert matched.alpha_score.tolist() == pytest.approx(
                        [1 / 3, -1 / 3, 0]
                    )
                    assert lookup.coverage_counts() == (0, 0)
                else:
                    with pytest.raises(
                        ValueError, match="misaligned_persisted_prediction_keys"
                    ):
                        list(joined)
        assert policy.used_memory_bytes == artifact_lease.reserved_bytes


def test_mixed_rank_and_linear_trials_share_score_keys_but_keep_own_predictions(
    tmp_path,
):
    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
    )

    specs = (
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
        ModelSpec(
            model_id="rank",
            model_algorithm="equal_weight_rank_ensemble",
            input_factor_columns=("x", "y"),
            label_column="label",
        ),
    )
    frame = shared_model_rows(specs)
    frame.loc[frame.asset_id.eq("asset_0"), "x"] = np.nan
    frame.loc[frame.asset_id.eq("asset_1"), "y"] = np.nan
    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("test.artifact.metadata", 0) as artifact_lease,
    ):
        case = shared_model_case(
            tmp_path, specs=specs, frame=frame, lease=artifact_lease
        )
        scoring = []
        for family, count in (("linear", 7), ("rank", 8)):
            entry = next(
                item
                for item in case.artifact.candidate_models
                if item["quarter_id"] == "2019Q1" and item["family_id"] == family
            )
            saved = pd.concat(case.artifact.validation_chunks(entry), ignore_index=True)
            assert saved.groupby("timestamp").size().tolist() == [count, count]
            # Only the first session's labels mature by this fold's cutoff.
            assert (
                saved.score_eligible
                & saved.label_available_at.le(saved.timestamp.max())
            ).sum() == 7
            scoring.append(
                saved.loc[saved.score_eligible, ["timestamp", "asset_id"]]
                .sort_values(["timestamp", "asset_id"])
                .reset_index(drop=True)
            )
        pd.testing.assert_frame_equal(*scoring)
        assert len(scoring[0]) == 14
        # Raw-rank missing values are neutral; linear x excludes asset_0.
        assert (
            scoring[0].asset_id.tolist()
            == [f"asset_{index}" for index in range(1, 8)] * 2
        )
        for reference in case.artifact.trial_ledgers:
            with read_model_trial_ledger_reference(case.layout, reference) as (
                plan,
                records,
            ):
                if plan.stage == "selection":
                    assert records[0].metrics["validation_rows"] == 7
        assert policy.used_memory_bytes == artifact_lease.reserved_bytes


def test_persisted_ic_rank_keeps_only_defined_weight_columns(tmp_path):
    spec = ModelSpec(
        model_id="ic_rank",
        model_algorithm="ic_weighted_rank_ensemble",
        input_factor_columns=("x", "y"),
        label_column="label",
    )
    frame = shared_model_rows((spec,))
    frame["y"] = 1.0
    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("test.artifact.metadata", 0) as artifact_lease,
    ):
        case = shared_model_case(
            tmp_path, specs=(spec,), frame=frame, lease=artifact_lease
        )
        for entry in (*case.artifact.candidate_models, *case.artifact.final_models):
            assert entry["model_metadata"]["input_columns"] == ["x", "y"]
            metadata = entry["model_metadata"]
            parameters = metadata["rank_parameters"]
            assert set(parameters) == {"columns", "weights", "label_calibration"}
            assert parameters["columns"] == ["x"]
            assert parameters["weights"] == [1.0]
            trained = frame.loc[
                frame.timestamp.between(
                    pd.Timestamp(metadata["train_start"]),
                    pd.Timestamp(metadata["train_end"]),
                )
            ]
            # Eight centered asset ranks have slope 8 * 0.03; their mean is zero.
            expected_intercept = trained.groupby("timestamp").label.mean().mean()
            np.testing.assert_allclose(
                parameters["label_calibration"],
                [expected_intercept, 0.24],
                rtol=0,
                atol=1e-14,
            )
        assert policy.used_memory_bytes == artifact_lease.reserved_bytes
