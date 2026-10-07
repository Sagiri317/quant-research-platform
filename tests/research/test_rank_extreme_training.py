"""Extreme decay is preserved through native model publication and reload."""

from fractions import Fraction
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.models import fitting, persisted
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)
from tests.helpers.research_configuration import research_configuration
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.workflow_test_support import shared_model_case


def _permutation(count, correlation):
    return next(
        values
        for values in permutations(range(1, count + 1))
        if 1
        - Fraction(
            6 * sum((i - v) ** 2 for i, v in enumerate(values, 1)),
            count * (count * count - 1),
        )
        == correlation
    )


@pytest.mark.parametrize("negative", (False, True))
@pytest.mark.parametrize("ordinary", (False, True))
def test_sparse_decay_rank_factories_publish_exact_direction_and_weights(
    tmp_path, negative, ordinary
):
    count, assets = (1082, 6) if negative else (1083, 4)
    if ordinary:
        count = 32
    factors = ("earnings_yield",) if negative else ("earnings_yield", "book_to_market")
    correlations = (Fraction(-1, 35),) if negative else (Fraction(1, 5), Fraction(4, 5))
    config = research_configuration("a_share", "fusion_nf_core")
    config["models"] = [
        dict(
            model,
            input_factor_columns=list(factors),
            input_feature_columns=[],
            label_column="forward_return_5d",
            training_weighting={"method": "equal_timestamp", "half_life": 1},
        )
        for model in config["models"]
        if model["model_algorithm"]
        in {"equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"}
    ]
    config["evaluation"] = {
        "factors": list(factors),
        "labels": ["forward_return_5d"],
        "primary_label": "forward_return_5d",
    }
    config["comparison_model_id"] = config["models"][0]["model_id"]
    config["portfolios"] = [
        item
        for item in config["portfolios"]
        if item["prediction_selector"] == "champion"
    ]
    specs = bind_research_configuration(research_profile(), config).models.list()
    all_dates = pd.bdate_range("2014-01-02", "2019-03-20")
    dates = all_dates[np.linspace(0, len(all_dates) - 1, count, dtype=int)]
    if ordinary:
        dates = [
            pd.Timestamp("2014-01-02"),
            pd.Timestamp("2016-01-04"),
            pd.Timestamp("2017-12-29"),
        ]
        for quarter in pd.period_range("2018Q1", "2019Q1", freq="Q"):
            dates.extend(
                (
                    quarter.start_time + pd.offsets.BDay(1),
                    quarter.start_time + pd.offsets.BDay(8),
                    min(
                        quarter.end_time.normalize() - pd.offsets.BDay(1),
                        pd.Timestamp("2019-03-20"),
                    ),
                )
            )
        dates.extend(pd.bdate_range("2014-01-03", periods=14))
        dates = pd.DatetimeIndex(sorted(dates))
        assert len(dates) == 32 and dates.is_unique
    rows = []
    patterns = [_permutation(assets, correlation) for correlation in correlations]
    for index, timestamp in enumerate(dates):
        for asset in range(assets if index == 0 else 1):
            rows.append(
                dict(
                    market_id="a_share",
                    asset_id=f"asset_{asset}",
                    timestamp=timestamp,
                    eligible=True,
                    forward_return_5d=float(asset + 1) if index == 0 else 0.0,
                    forward_return_5d_available_at=timestamp + pd.offsets.BDay(6),
                    **{
                        factor: float(patterns[j][asset]) if index == 0 else 1.0
                        for j, factor in enumerate(factors)
                    },
                )
            )
    frame = pd.DataFrame(rows)
    weights = equal_timestamp_sample_weights(
        frame.timestamp, specs[0].training_weighting
    )
    assert (weights > 0).all() and np.isfinite(weights).all()
    if not ordinary:
        assert weights[:assets].max() <= np.nextafter(0.0, 1.0) * 4
    dataset = make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=factors,
        label_columns=("forward_return_5d", "forward_return_5d_available_at"),
        filter_columns=("eligible",),
        context_columns=(),
    )
    case = shared_model_case(
        tmp_path,
        specs=specs,
        frame=frame,
        dataset=dataset,
        development_end="2019-03-31",
        chunk_size=4096,
    )
    probe = frame.iloc[:assets].copy()
    ranks = np.column_stack(
        [
            (np.asarray(pattern, dtype=float) - (assets + 1) / 2) / assets
            for pattern in patterns
        ]
    )
    for group in ("global", "quarterly"):
        for spec in specs:
            model = persisted.read_quarterly_predictor(
                case.artifact, group, spec.model_id
            )
            if spec.model_algorithm == "equal_weight_rank_ensemble":
                expected = np.array([-1.0] if negative else [1.0, 1.0])
                assert model.directions == tuple(expected)
                scores = ranks @ expected / len(factors)
            else:
                expected = np.array([-1.0] if negative else [0.2, 0.8])
                np.testing.assert_array_equal(model.weights, expected)
                scores = ranks @ expected
            np.testing.assert_array_equal(
                fitting.prediction_callable(factors, model)(probe), scores
            )
            entry = next(
                item
                for item in case.artifact.final_models
                if item["selection_group"] == group
                and item["family_id"] == spec.model_id
            )
            fitting.validate_model_state(model, entry["model_metadata"])
            assert entry["model_metadata"]["rows_after_input_label_dropna"] == len(
                frame
            )


@pytest.mark.parametrize("period_count", (32, 1083))
def test_native_crypto_sparse_decay_calibration_survives_model_publication(
    tmp_path, period_count
):
    from functools import partial

    from quant_research_platform.contracts.business_manifest import canonical_json_hash
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_factor_frames,
        iter_label_frames,
        iter_panel_frames,
    )
    from quant_research_platform.research.models.production import (
        build_crypto_fold_models,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows.research import (
        _target_envelope_for_spec,
        production_software_identity,
    )
    from tests.helpers.binance_hft import _coverage, coverage_lookup, hft_research_plan
    from tests.markets.crypto.test_hft import _state

    def iso(milliseconds):
        return pd.Timestamp(milliseconds, unit="ms").isoformat()

    last_train = (period_count - 2) * 100
    train_end = last_train + 500
    validation_start = train_end + 600
    validation_end = validation_start + 500
    plan = hft_research_plan()
    plan["folds"] = [
        {
            "fold_id": "fold_1",
            "train": {"start": iso(0), "end": iso(train_end)},
            "validation": {"start": iso(validation_start), "end": iso(validation_end)},
        }
    ]
    plan["test"] = {
        "start": iso(validation_end + 300),
        "end": iso(validation_end + 800),
    }
    config = research_configuration(
        "crypto",
        "baseline",
        plan=plan,
        development=(iso(validation_start), iso(validation_end)),
    )
    template = config["models"][0]
    config["models"] = [
        dict(
            template,
            model_id="probe_" + algorithm,
            model_algorithm=algorithm,
            input_factor_columns=["imbalance_1"],
            input_feature_columns=[],
            hyperparameters={},
            training_asset_ids=["btc_usdt", "eth_usdt"],
            evaluation_asset_ids=["btc_usdt", "eth_usdt"],
            training_weighting={"method": "equal_timestamp", "half_life": 1},
        )
        for algorithm in ("equal_weight_rank_ensemble", "ic_weighted_rank_ensemble")
    ]
    config.update(portfolios=[], comparison_model_id=config["models"][0]["model_id"])
    config["evaluation"] = {
        "factors": ["imbalance_1"],
        "labels": ["future_mid_return_500ms"],
    }
    profile = bind_research_configuration(research_profile(CRYPTO_PROFILE_ID), config)
    dataset_id = config["scenario_id"]
    dataset = profile.research_datasets.get(dataset_id)
    protocol = profile.protocol(dataset_id)
    target = next(
        target
        for target in protocol.target_specs("research run")
        if target.target_id == dataset_id + "_models"
    )
    frames = []
    for asset, sign in (("btc_usdt", -1), ("eth_usdt", 1)):
        end = validation_end if sign < 0 else 500
        states = [
            _state(0, bid_qty=1 if sign < 0 else 3, ask_qty=3 if sign < 0 else 1),
            _state(
                500,
                bid_index=995 + sign,
                ask_index=1005 + sign,
                bid_qty=1 if sign < 0 else 3,
                ask_qty=3 if sign < 0 else 1,
            ),
        ]
        panel = pd.concat(
            list(
                iter_panel_frames(
                    states,
                    [],
                    market_id="crypto",
                    asset_id=asset,
                    start_ms=0,
                    end_ms=end,
                    book_coverage={"s1": _coverage(0, end + 100)},
                    trade_coverage=coverage_lookup(
                        {"trade-connection-1": _coverage(0, end + 100)}
                    ),
                    batch_rows=211,
                )
            ),
            ignore_index=True,
        )
        factors = pd.concat(
            list(iter_factor_frames((panel,), columns=("imbalance_1",))),
            ignore_index=True,
        )
        labels = pd.concat(
            list(iter_label_frames((panel,), horizons_ms=(100, 500, 1000, 5000))),
            ignore_index=True,
        )
        keys = ["market_id", "asset_id", "timestamp"]
        frame = panel.merge(factors, on=keys, validate="one_to_one").merge(
            labels, on=keys, validate="one_to_one"
        )
        selected = (
            frame.timestamp.eq(pd.Timestamp(0))
            if sign > 0
            else (
                frame.timestamp.le(pd.Timestamp(last_train, unit="ms"))
                | frame.timestamp.eq(pd.Timestamp(validation_start, unit="ms"))
            )
        )
        frame = frame.loc[selected].copy()
        frame["eligible"] = True
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["market_id", "asset_id", "timestamp"], ignore_index=True
    )
    frame = frame.loc[:, list(dataset.required_columns)]
    assert frame.timestamp.nunique() == period_count and len(frame) == period_count + 1
    assert frame.future_mid_return_500ms_available_at.eq(
        frame.timestamp + pd.Timedelta(milliseconds=500)
    ).all()
    layout = StorageLayout(tmp_path)
    native = layout.instance_root(
        layout.research_dataset_root("crypto", dataset_id), "a" * 64
    )
    layout = layout.with_target_roots({dataset_id: native})
    write_dataset(native, frame, spec=dataset)
    window = TimeWindow(**plan["folds"][0]["validation"])
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
        upstream_identities={dataset_id: "a" * 64},
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    manifest = envelope.payload()["semantic_metadata"]["target_contract"]["protocol"]
    session = open_dataset_session(
        native, columns=None, time_window=TimeWindow(iso(0), iso(validation_end))
    )
    results = partial(
        build_crypto_fold_models,
        layout=layout,
        dataset_session=session,
        dataset_catalog=profile.research_datasets,
        model_catalog=profile.models,
        dataset_id=dataset_id,
        model_ids=tuple(spec.model_id for spec in profile.models.list()),
        time_window=window,
        protocol_manifest=manifest,
        run_id="run_extreme_rank",
        research_protocol_identity=canonical_json_hash(manifest),
        model_target_id=target.target_id,
        backend_versions={"internal": "test"},
    )
    root = persisted.write_crypto_model_artifact(
        tmp_path / "models",
        envelope=envelope,
        model_results=results,
        trial_layout=layout,
        verify_source=session.verify,
    )
    saved = persisted.read_crypto_model_artifact(
        root, expected_identity=envelope.instance_identity, trial_layout=layout
    )
    ordered = frame.sort_values(["timestamp", "asset_id"], ignore_index=True)
    weights = equal_timestamp_sample_weights(
        ordered.timestamp, profile.models.list()[0].training_weighting
    )
    scores = np.zeros(len(ordered))
    scores[:2] = [-0.25, 0.25]
    mass = list(map(lambda x: Fraction(float(x)), weights))
    xs = list(map(Fraction, scores))
    ys = list(map(Fraction, ordered.future_mid_return_500ms))
    total = sum(mass)
    sx = sum(w * x for w, x in zip(mass, xs, strict=True))
    sy = sum(w * y for w, y in zip(mass, ys, strict=True))
    xx = sum(w * x * x for w, x in zip(mass, xs, strict=True))
    xy = sum(w * x * y for w, x, y in zip(mass, xs, ys, strict=True))
    slope = (total * xy - sx * sy) / (total * xx - sx * sx)
    intercept = (sy - slope * sx) / total
    assert float(slope) == pytest.approx(0.004, abs=1e-15)
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _score_in_label_units,
    )
    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
        research_run_trial_references,
    )

    references = research_run_trial_references(layout, "run_extreme_rank")
    assert len(references) == 4
    for reference in references:
        _plan, records = read_model_trial_ledger_reference(layout, reference)
        assert records
    for entry in saved.final_models:
        model = persisted.read_crypto_model_member(saved, entry)
        if period_count == 1083:
            assert model.label_calibration == (float(intercept), float(slope))
        else:
            np.testing.assert_allclose(
                model.label_calibration,
                (float(intercept), float(slope)),
                rtol=1e-14,
                atol=0,
            )
        probe = ordered.iloc[:2]
        actual = fitting.prediction_callable(("imbalance_1",), model)(probe)
        np.testing.assert_array_equal(actual, [-0.25, 0.25])
        np.testing.assert_allclose(
            model.label_calibration[0] + model.label_calibration[1] * actual,
            float(intercept) + float(slope) * np.array([-0.25, 0.25]),
        )
        assert (
            _score_in_label_units(
                0.25, entry["model_metadata"]["rank_parameters"]["label_calibration"]
            )
            == float(intercept) + float(slope) * 0.25
        )
