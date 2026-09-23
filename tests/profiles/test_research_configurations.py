"""Explicit selections resolve into the same production catalogs."""

import copy

import pytest

from quant_research_platform.contracts.models import model_spec_payload
from quant_research_platform.markets.implementations.crypto import ids as crypto_ids
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.protocols import crypto_hft
from tests.helpers.binance_hft import hft_research_plan


def configuration(profile_id, scenario, model=None):
    return {
        "schema_id": "qrp-research-configuration",
        "windows": {
            "development": {"start": "2019-01-01", "end": "2023-12-31"},
            "holdout": {"start": "2024-01-01", "end": "2025-12-31"},
        }
        if profile_id == "a_share_csmar_choice"
        else {
            "development": {
                "start": "1970-01-01T00:00:04",
                "end": "1970-01-01T00:00:08.500000",
            },
            "holdout": {
                "start": "1970-01-01T00:00:08.600000",
                "end": "1970-01-01T00:00:09",
            },
        },
        "profile_id": profile_id,
        "scenario_id": scenario,
        "stage": "evaluation" if model is None else "research",
        "evaluation": {
            "primary_label": "forward_return_20d"
            if profile_id == "a_share_csmar_choice"
            else "future_mid_return_500ms",
            "factors": ["share_price"]
            if profile_id == "a_share_csmar_choice"
            else ["ofi_500ms"],
            "labels": ["forward_return_20d"]
            if profile_id == "a_share_csmar_choice"
            else ["future_mid_return_500ms"],
        },
        "models": [] if model is None else [model_spec_payload(model)],
        "portfolios": [],
        "comparison_model_id": None if model is None else model.model_id,
    }


def test_factor_only_selection_has_no_model_or_financial_dependency():
    profile = research_profile("a_share_csmar_choice")
    config = configuration(profile.profile_id, "pv_expanded")
    selected = bind_research_configuration(profile, config)
    dataset = selected.research_datasets.get("pv_expanded")
    assert dataset.factor_columns == ("share_price",)
    assert dataset.input_data_ids == (
        "market_panel_1d",
        "price_volume_facts",
        "forward_returns",
    )
    assert selected.models.list() == ()
    assert dataset.feature_columns == ()
    assert dataset.feature_transforms == ()
    protocol = selected.protocol("pv_expanded")
    assert protocol.terminal_targets("research run") == ("pv_expanded_evaluation",)
    config["evaluation"]["factors"].append("beta_252d")
    assert protocol.manifest()["research_configuration"]["evaluation"]["factors"] == [
        "share_price"
    ]


def test_model_input_and_evaluation_selection_are_independent():
    from quant_research_platform.contracts.models import ModelSpec

    model = ModelSpec(
        model_id="pv_expanded_xgboost_regressor",
        model_algorithm="xgboost_regressor",
        input_feature_columns=("feature_turnover_250d",),
        label_column="model_target",
        hyperparameters={"n_estimators": 17},
    )
    config = configuration("a_share_csmar_choice", "pv_expanded", model)
    selected = bind_research_configuration(research_profile(), config)
    assert selected.models.get(model.model_id).input_columns == (
        "feature_turnover_250d",
    )
    assert selected.models.get(model.model_id).hyperparameters["n_estimators"] == 17
    assert selected.research_datasets.get("pv_expanded").factor_columns == (
        "share_price",
        "turnover_250d",
        "log_float_market_cap",
    )
    dataset = selected.research_datasets.get("pv_expanded")
    assert dataset.feature_transforms[0].source_columns == ("turnover_250d",)
    assert "feature_log_float_market_cap" not in dataset.feature_columns
    assert selected.protocol("pv_expanded").portfolio_candidates() == ()
    assert selected.protocol("pv_expanded").model_ids_for_target(
        "pv_expanded_model_selection_refit"
    ) == (model.model_id,)


def test_crypto_selection_keeps_execution_assets_and_can_evaluate_only():
    profile = research_profile("crypto_binance_hft")
    config = configuration(profile.profile_id, crypto_ids.HFT_RESEARCH_DATASET_ID)
    config["research_plan"] = hft_research_plan()
    selected = bind_research_configuration(profile, config)
    assert selected.research_datasets.list()[0].factor_columns == ("ofi_500ms",)
    protocol = selected.protocol(crypto_ids.HFT_RESEARCH_DATASET_ID)
    assert protocol.terminal_targets("research run") == (
        crypto_ids.HFT_RESEARCH_DATASET_ID + "_evaluation",
    )
    assert selected.models.list() == ()
    assert set(protocol.manifest()["research_plan"]["exposure_limits"]) == {
        "btc_usdt",
        "eth_usdt",
    }


def test_bad_selection_fails_without_silent_column_removal():
    profile = research_profile()
    config = configuration(profile.profile_id, "pv_expanded")
    invalid = copy.deepcopy(config)
    invalid["evaluation"]["factors"] = ["not_a_factor"]
    with pytest.raises(ValueError, match="factor_not_available"):
        bind_research_configuration(profile, invalid)
    invalid = copy.deepcopy(config)
    invalid["evaluation"]["factors"] = ["share_price", "share_price"]
    with pytest.raises(ValueError, match="factor_duplicate"):
        bind_research_configuration(profile, invalid)
    invalid = copy.deepcopy(config)
    invalid["portfolios"] = ["champion_top20_equal_weight"]
    with pytest.raises(ValueError, match="evaluation_only"):
        bind_research_configuration(profile, invalid)


def test_crypto_model_assets_and_inputs_follow_configuration_not_model_name():
    from quant_research_platform.research.protocols.crypto_hft import (
        build_hft_prediction_catalog,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    config["models"] = [config["models"][0]]
    model = config["models"][0]
    model["model_id"] = "chosen_model"
    model["training_asset_ids"] = ["eth_usdt"]
    model["evaluation_asset_ids"] = ["btc_usdt"]
    model["input_factor_columns"] = ["ofi_500ms"]
    config["comparison_model_id"] = model["model_id"]
    config["portfolios"] = []
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    assert selected.models.get("chosen_model").input_columns == ("ofi_500ms",)
    bound = selected.protocol(config["scenario_id"])
    scopes = bound.manifest()["model_scopes"]["chosen_model"]
    assert scopes["training_asset_ids"] == ["eth_usdt"]
    assert scopes["evaluation_asset_ids"] == ["btc_usdt"]
    predictions = build_hft_prediction_catalog(
        configuration=config,
        train_end="2026-08-01T08:00:00",
        test_start="2026-08-01T12:00:01",
        test_end="2026-08-01T20:00:00",
    )
    assert predictions.get("chosen_model_prediction").evaluation_asset_ids == (
        "btc_usdt",
    )
    assert (
        selected.evaluations.get("chosen_model_oos_evaluation").evaluation_axis
        == "single_asset_time_series"
    )
    model["training_asset_ids"] = ["not_authorized"]
    with pytest.raises(ValueError, match="asset_scope_invalid"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_portfolio_parameters_flow_through_existing_candidate_contracts():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name="pv_expanded")
    config["portfolios"] = [
        {
            "candidate_id": "champion_top20_equal_weight",
            "construction": "top20_equal_weight",
            "benchmark_id": None,
            "prediction_selector": "champion",
            "top_quantile": 0.1,
            "rebalance_schedule": "weekly-last-trading-day",
        }
    ]
    bound = bind_research_configuration(research_profile(), config)
    candidate = bound.protocol("pv_expanded").portfolio_candidates()[0]
    assert candidate.top_quantile == 0.1
    assert candidate.prediction_selector == "champion"
    assert candidate.risk_context_target_ids == ("price_volume_facts",)
    assert candidate.risk_style_columns
    ledger = next(
        target
        for target in bound.protocol("pv_expanded").target_specs("research run")
        if target.target_id == "pv_expanded_champion_top20_equal_weight_ledger"
    )
    assert set(candidate.risk_context_target_ids) <= set(ledger.dependencies)
    config["portfolios"][0]["top_quantile"] = 0
    with pytest.raises(ValueError, match="quantile_invalid"):
        bind_research_configuration(research_profile(), config)


def test_auxiliary_source_dependency_and_age_context_are_selected_locally(tmp_path):
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_DERIVATIVE_DATA_ID,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    baseline = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    assert (
        HFT_DERIVATIVE_DATA_ID
        not in baseline.research_datasets.list()[0].input_data_ids
    )
    config["evaluation"]["factors"] = ["log_open_interest"]
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    dataset = selected.research_datasets.list()[0]
    assert HFT_DERIVATIVE_DATA_ID in dataset.input_data_ids
    assert "openInterest_age_ms" in dataset.context_columns
    assert "openInterest_event_at" in dataset.context_columns
    assert "openInterest_observed_at" in dataset.context_columns
    assert HFT_DERIVATIVE_DATA_ID not in selected.producer_bindings
    config = research_configuration("crypto", "derivatives_state")
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    binding = selected.producer_bindings[HFT_DERIVATIVE_DATA_ID]
    assert (
        binding.raw_root
        == tmp_path / config["auxiliary_sources"]["derivatives_state"]["archive"]
    )
    config["auxiliary_sources"]["derivatives_state"]["max_age_ms"]["openInterest"] = 0
    with pytest.raises(ValueError, match="state_age_invalid"):
        bind_research_configuration(
            research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
        )


def test_primary_statistics_label_is_explicit_and_cannot_select_availability_column():
    config = configuration("a_share_csmar_choice", "pv_expanded")
    config["evaluation"]["labels"] = ["forward_return_5d", "forward_return_60d"]
    config["evaluation"]["primary_label"] = "forward_return_5d"
    selected = bind_research_configuration(research_profile(), config)
    manifest = selected.protocol("pv_expanded").manifest()
    assert (
        manifest["research_configuration"]["evaluation"]["primary_label"]
        == "forward_return_5d"
    )
    assert (
        selected.labels.get("forward_return_5d").available_at_column
        == "forward_return_5d_available_at"
    )
    assert selected.evaluations.list()[0].label_columns == (
        "forward_return_5d",
        "forward_return_60d",
    )
    config["evaluation"]["labels"] = ["forward_return_5d_available_at"]
    config["evaluation"]["primary_label"] = "forward_return_5d_available_at"
    with pytest.raises(ValueError, match="evaluation_label_invalid"):
        bind_research_configuration(research_profile(), config)


def test_auxiliary_age_changes_processed_identity_without_changing_schema(tmp_path):
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_DERIVATIVE_DATA_ID,
    )
    from quant_research_platform.storage.datasets.schema import (
        data_spec_payload,
        validate_dataset_spec_payload,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "derivatives_state")
    first = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    before = data_spec_payload(first.processed.get(HFT_DERIVATIVE_DATA_ID))
    validate_dataset_spec_payload(before)
    config["auxiliary_sources"]["derivatives_state"]["max_age_ms"]["openInterest"] = (
        30000
    )
    second = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    after = data_spec_payload(second.processed.get(HFT_DERIVATIVE_DATA_ID))
    validate_dataset_spec_payload(after)
    assert before["column_roles"] == after["column_roles"]
    assert canonical_json_hash(before) != canonical_json_hash(after)
    assert before["parameters"]["max_age_ms"]["openInterest"] == 15000


def test_research_windows_allow_continuous_crypto_across_midnight():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    config["windows"]["development"]["end"] = "2024-01-01"
    with pytest.raises(ValueError, match="outside_access_boundary"):
        bind_research_configuration(research_profile(), config)
    config = research_configuration("crypto", "baseline")
    config["research_plan"]["final_holdout"]["end"] = "2026-08-02T00:00:00"
    config["windows"]["holdout"]["end"] = "2026-08-02T00:00:00"
    bound = bind_research_configuration(research_profile("crypto_binance_hft"), config)
    assert (
        bound.protocol(
            config["scenario_id"]
        ).research_plan.final_holdout.end.isoformat()
        == "2026-08-02T00:00:00"
    )


def test_spot_uses_shared_native_owner_and_only_selected_graph_dependencies(tmp_path):
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_SPOT_CONTEXT_DATA_ID,
        HFT_SPOT_PANEL_DATA_ID,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer as crypto_producer,
    )
    from quant_research_platform.workflows.research import (
        _target_specs,
        resolve_target_graph,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "spot_context")
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    binding = selected.producer_bindings[HFT_SPOT_PANEL_DATA_ID]
    assert binding.producer.func is crypto_producer.produce_hft_processed_data
    assert binding.raw_root == tmp_path / config["auxiliary_sources"]["spot"]["archive"]
    assert binding.producer.keywords["market_type"] == "spot"
    protocol = selected.protocol(config["scenario_id"])
    graph = resolve_target_graph(
        _target_specs(selected, action="research run", protocol=protocol),
        protocol.terminal_targets("research run"),
    )
    ordered = [target.target_id for target in graph]
    assert (
        ordered.index(HFT_SPOT_PANEL_DATA_ID)
        < ordered.index(HFT_SPOT_CONTEXT_DATA_ID)
        < ordered.index(config["scenario_id"])
    )
    assert set(candidate.asset_id for candidate in protocol.portfolio_candidates()) <= {
        "btc_usdt",
        "eth_usdt",
    }
    plain = research_configuration("crypto", "baseline")
    baseline = bind_research_configuration(
        research_profile("crypto_binance_hft"), plain
    )
    base_protocol = baseline.protocol(plain["scenario_id"])
    graph = resolve_target_graph(
        _target_specs(baseline, action="research run", protocol=base_protocol),
        base_protocol.terminal_targets("research run"),
    )
    assert HFT_SPOT_CONTEXT_DATA_ID not in {target.target_id for target in graph}
    assert HFT_SPOT_PANEL_DATA_ID not in {target.target_id for target in graph}


def test_sol_read_and_training_scopes_do_not_expand_execution(tmp_path):
    import json

    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_FACTOR_DATA_ID,
        HFT_LABEL_DATA_ID,
    )
    from quant_research_platform.workflows.research import (
        _target_specs,
        resolve_target_graph,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "cross_asset")
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    from quant_research_platform.storage.datasets.schema import (
        data_spec_payload,
        validate_dataset_spec_payload,
    )

    for item in selected.processed.list():
        validate_dataset_spec_payload(data_spec_payload(item))
    dataset = selected.research_datasets.list()[0]
    assert dataset.base_input_data_id == crypto_ids.HFT_INFORMATION_PANEL_DATA_ID
    assert "sol_grid_age_ms" in dataset.context_columns
    assert "sol_grid_at" in dataset.context_columns
    for key in (HFT_FACTOR_DATA_ID, HFT_LABEL_DATA_ID):
        spec = selected.processed.get(key)
        assert spec.processed_input_data_ids == (
            crypto_ids.HFT_INFORMATION_PANEL_DATA_ID,
        )
        assert json.loads(spec.parameters_json)["asset_ids"] == [
            "btc_usdt",
            "eth_usdt",
            "sol_usdt",
        ]
    protocol = selected.protocol(config["scenario_id"])
    assert protocol.manifest()["model_scopes"]["crypto_hft_sol_to_btc"][
        "training_asset_ids"
    ] == ["sol_usdt"]
    assert protocol.manifest()["model_scopes"]["crypto_hft_btc_eth_to_sol"][
        "evaluation_asset_ids"
    ] == ["sol_usdt"]
    assert set(candidate.asset_id for candidate in protocol.portfolio_candidates()) <= {
        "btc_usdt",
        "eth_usdt",
    }
    graph = resolve_target_graph(
        _target_specs(selected, action="research run", protocol=protocol),
        protocol.terminal_targets("research run"),
    )
    assert crypto_ids.HFT_SOL_PANEL_DATA_ID in {target.target_id for target in graph}
    del config["auxiliary_sources"]["sol"]
    missing = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    assert crypto_ids.HFT_SOL_PANEL_DATA_ID not in missing.producer_bindings
    plain = research_configuration("crypto", "cross_btc_eth")
    independent = bind_research_configuration(
        research_profile("crypto_binance_hft"), plain
    )
    protocol = independent.protocol(plain["scenario_id"])
    graph = resolve_target_graph(
        _target_specs(independent, action="research run", protocol=protocol),
        protocol.terminal_targets("research run"),
    )
    assert crypto_ids.HFT_SOL_PANEL_DATA_ID not in {
        target.target_id for target in graph
    }


def test_execution_candidate_requires_predictions_for_its_asset():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    for item in config["models"]:
        if item["model_id"] == "crypto_hft_linear_regression":
            item["evaluation_asset_ids"] = ["sol_usdt"]
    config["portfolios"] = [
        item
        for item in config["portfolios"]
        if item["candidate_id"] == "crypto_hft_maker"
    ]
    with pytest.raises(ValueError, match="maker_model_asset_mismatch"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_external_sources_have_independent_owners_units_and_sol_nulls(tmp_path):
    import json

    from quant_research_platform.markets.implementations.crypto.processed import (
        producer as crypto_producer,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "external_context")
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"),
        config,
        tardis_root=tmp_path,
    )
    dataset = selected.research_datasets.list()[0]
    for venue, data_id in crypto_ids.HFT_EXTERNAL_DATA_IDS.items():
        assert data_id in dataset.input_data_ids
        assert venue + "_quote_observed_at" in dataset.context_columns
        binding = selected.producer_bindings[data_id]
        assert binding.producer.func is crypto_producer.produce_normalized_context
        params = json.loads(selected.processed.get(data_id).parameters_json)
        assert params["base_units_per_amount"]["btc_usdt"] == (
            1.0 if venue == "bybit" else 0.01
        )
    del config["auxiliary_sources"]["okx"]
    missing = bind_research_configuration(
        research_profile("crypto_binance_hft"),
        config,
        tardis_root=tmp_path,
    )
    assert crypto_ids.HFT_EXTERNAL_DATA_IDS["bybit"] in missing.producer_bindings
    assert crypto_ids.HFT_EXTERNAL_DATA_IDS["okx"] not in missing.producer_bindings
    transfer = research_configuration("crypto", "cross_asset")
    transfer["evaluation"]["factors"].append("bybit_price_difference")
    transfer["auxiliary_sources"]["bybit"] = config["auxiliary_sources"]["bybit"]
    mixed = bind_research_configuration(
        research_profile("crypto_binance_hft"),
        transfer,
        tardis_root=tmp_path,
    )
    assert (
        mixed.research_datasets.list()[0].key_coverage[
            crypto_ids.HFT_EXTERNAL_DATA_IDS["bybit"]
        ]
        == "sparse"
    )
    config["auxiliary_sources"]["bybit"]["archives"]["BTCUSDT"]["quotes"] = "../escape"
    with pytest.raises(ValueError, match="external_archive_invalid"):
        bind_research_configuration(
            research_profile("crypto_binance_hft"),
            config,
            tardis_root=tmp_path,
        )


def test_a_share_portfolio_configuration_owns_names_and_construction():
    from quant_research_platform.workflows.research import (
        _target_specs,
        resolve_target_graph,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name="pv_expanded")
    original = config["portfolios"][0]
    selected_item = {
        **original,
        "candidate_id": "custom_monthly_top_decile",
        "construction": "top20_float_cap",
        "top_quantile": 0.1,
        "rebalance_schedule": "monthly-last-trading-day",
    }
    config["portfolios"] = [selected_item]
    selected = bind_research_configuration(research_profile(), config)
    protocol = selected.protocol(config["scenario_id"])
    (candidate,) = protocol.portfolio_candidates()
    assert candidate.candidate_id == "custom_monthly_top_decile"
    assert candidate.top_quantile == 0.1
    assert candidate.construction == "top20_float_cap"
    assert candidate.rebalance_schedule == "monthly-last-trading-day"
    target = config["scenario_id"] + "_custom_monthly_top_decile_ledger"
    assert target in selected.producer_bindings
    graph = resolve_target_graph(
        _target_specs(selected, action="research run", protocol=protocol),
        protocol.terminal_targets("research run"),
    )
    assert target in {item.target_id for item in graph}
    assert "fundamental_factors" not in {item.target_id for item in graph}
    assert not any("champion_csi" in item.target_id for item in graph)
    baseline = research_configuration(name="core6")
    formal = bind_research_configuration(research_profile(), baseline)
    candidates = formal.protocol(baseline["scenario_id"]).portfolio_candidates()
    assert len(candidates) == 5
    assert [item.benchmark_id for item in candidates[-2:]] == ["csi500", "csi1000"]


def test_crypto_named_selections_drive_ledger_statistics_and_asset_calibration(
    tmp_path,
):
    from quant_research_platform.workflows.research import (
        _target_specs,
        resolve_target_graph,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    for model in config["models"]:
        model["evaluation_asset_ids"] = ["btc_usdt", "eth_usdt"]
        model["training_asset_ids"] = ["btc_usdt", "eth_usdt"]
    config["portfolios"] = [
        {
            "candidate_id": "eth_price_taker",
            "asset_id": "eth_usdt",
            "model_id": "crypto_hft_linear_regression",
            "latency_ms": 17,
        },
        {
            "candidate_id": "eth_calibrated_maker",
            "asset_id": "eth_usdt",
            "price_model_id": "crypto_hft_linear_regression",
            "proxy_model_id": "crypto_hft_maker_bid_sell_pressure_proxy",
        },
    ]
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    protocol = selected.protocol(config["scenario_id"])
    candidates = protocol.portfolio_candidates()
    assert [item.candidate_id for item in candidates] == [
        "eth_price_taker",
        "eth_calibrated_maker",
    ]
    assert candidates[0].latency_ms == 17
    calibration = candidates[1].calibration_target_id
    binding = selected.producer_bindings[calibration]
    assert binding.producer.keywords["asset_id"] == "eth_usdt"
    assert binding.producer.keywords["symbol"] == "ETHUSDT"
    graph = resolve_target_graph(
        _target_specs(selected, action="research run", protocol=protocol),
        protocol.terminal_targets("research run"),
    )
    targets = {item.target_id: item for item in graph}
    descriptors = protocol.execution_targets()
    assert calibration in targets[descriptors[1].target_id].dependencies
    for descriptor in descriptors:
        assert targets[descriptor.statistics_target_id].dependencies == (
            descriptor.target_id,
        )
    assert not any("latency_30" in key for key in targets)
    config["portfolios"][0]["asset_id"] = "sol_usdt"
    with pytest.raises(ValueError, match="portfolio_configuration_invalid"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_maker_model_roles_follow_configuration_and_validate_units():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    selected = next(item for item in config["portfolios"] if item.get("price_model_id"))
    old_price, old_proxy = selected["price_model_id"], selected["proxy_model_id"]
    config["portfolios"] = [selected]
    selected["price_model_id"], selected["proxy_model_id"] = "my_price", "my_pressure"
    for model in config["models"]:
        if model["model_id"] == old_price:
            model["model_id"] = "my_price"
        elif model["model_id"] == old_proxy:
            model["model_id"] = "my_pressure"
    if config["comparison_model_id"] == old_price:
        config["comparison_model_id"] = "my_price"
    if config["comparison_model_id"] == old_proxy:
        config["comparison_model_id"] = "my_pressure"
    profile = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    (candidate,) = profile.protocol(config["scenario_id"]).portfolio_candidates()
    assert candidate.model_ids == ("my_price", "my_pressure")
    assert (
        profile.producer_bindings[candidate.calibration_target_id].producer.keywords[
            "model_ids"
        ]
        == candidate.model_ids
    )
    selected["proxy_model_id"] = "my_price"
    with pytest.raises(ValueError, match="maker_model_label_mismatch"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_model_names_and_latency_scenarios_do_not_select_hidden_baselines():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name="pv_expanded")
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "ridge_regression"
    )
    model["model_id"] = "my_ridge_trial"
    config["models"] = [model]
    config["portfolios"] = []
    config["comparison_model_id"] = "my_ridge_trial"
    selected = bind_research_configuration(research_profile(), config)
    assert [item.model_id for item in selected.models.list()] == ["my_ridge_trial"]
    assert selected.protocol(config["scenario_id"]).model_ids_for_target(
        config["scenario_id"] + "_model_selection_refit"
    ) == ("my_ridge_trial",)

    config = research_configuration("crypto", "baseline")
    config["research_plan"]["execution"]["latency_scenarios_ms"] = [17, 55, 250]
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    assert selected.protocol(
        config["scenario_id"]
    ).research_plan.execution.latency_scenarios_ms == (
        17,
        55,
        250,
    )
    config["research_plan"]["execution"]["latency_scenarios_ms"] = [17, 17]
    with pytest.raises(ValueError, match="execution_parameters_invalid"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_crypto_common_comparisons_reuse_evaluation_and_require_individual_coverage():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    comparisons = [
        item
        for item in selected.evaluations.list()
        if item.signals[0].source_kind == "prediction" and len(item.signals) == 2
    ]
    reference = config["comparison_model_id"]
    models = {item["model_id"]: item for item in config["models"]}
    assert len(comparisons) == sum(
        item["model_id"] != reference
        and item["label_column"] == models[reference]["label_column"]
        for item in config["models"]
    )
    protocol = selected.protocol(config["scenario_id"])
    targets = {item.target_id: item for item in protocol.target_specs("research run")}
    for spec in comparisons:
        assert spec.signals[0].signal_id == reference
        assert spec.evaluation_id in protocol.terminal_targets("research run")
        assert spec.evaluation_id in selected.producer_bindings
        assert all(
            signal.signal_id + "_oos_evaluation"
            in targets[spec.evaluation_id].dependencies
            for signal in spec.signals
        )


def test_crypto_model_horizon_follows_selected_label_and_taker_requires_returns():
    from tests.helpers.research_configuration import research_configuration

    plan = hft_research_plan()
    plan["folds"] = [plan["folds"][0]]
    plan["folds"][0]["train"] = {
        "start": "1970-01-01T00:00:00",
        "end": "1970-01-01T00:00:00.200000",
    }
    plan["folds"][0]["validation"] = {
        "start": "1970-01-01T00:00:00.300000",
        "end": "1970-01-01T00:00:00.500000",
    }
    config = research_configuration(
        "crypto",
        "baseline",
        development=(
            plan["folds"][0]["development_test"]["start"],
            plan["folds"][0]["development_test"]["end"],
        ),
        plan=plan,
    )
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "linear_regression"
    )
    model["model_id"] = "short_return"
    model["label_column"] = "future_mid_return_100ms"
    config["models"] = [model]
    config["comparison_model_id"] = model["model_id"]
    config["portfolios"] = [
        {
            "candidate_id": "short_taker",
            "asset_id": "btc_usdt",
            "model_id": model["model_id"],
            "latency_ms": None,
        }
    ]
    profile = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    manifest = profile.protocol(config["scenario_id"]).manifest()
    assert manifest["model_scopes"]["short_return"]["label_horizon_ms"] == 100
    assert crypto_hft.CryptoMakerPlan.quote_horizon_ms == 500
    model["label_column"] = "future_mid_direction_100ms"
    with pytest.raises(ValueError, match="taker_model_label_mismatch"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)
    model["label_column"] = "future_mid_return_5000ms"
    with pytest.raises(ValueError, match="label_window_too_short"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_configuration_rejects_fields_that_have_no_market_consumer():
    from tests.helpers.research_configuration import research_configuration

    crypto = research_configuration("crypto", "baseline")
    crypto["models"][0]["parameter_candidates"] = [{"alpha": 0.01}, {"alpha": 0.1}]
    with pytest.raises(ValueError, match="crypto_parameter_candidates_not_supported"):
        bind_research_configuration(research_profile("crypto_binance_hft"), crypto)
    del crypto["models"][0]["parameter_candidates"]
    del crypto["research_plan"]
    with pytest.raises(ValueError, match="crypto_research_plan_required"):
        bind_research_configuration(research_profile("crypto_binance_hft"), crypto)
    a_share = research_configuration()
    a_share["research_plan"] = {"unused": True}
    with pytest.raises(ValueError, match="a_share_research_plan_not_supported"):
        bind_research_configuration(research_profile(), a_share)


def test_rebinding_drops_previous_auxiliary_context(tmp_path):
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        bind_research_configuration,
        research_profile,
    )
    from tests.helpers.research_configuration import research_configuration

    baseline = research_configuration("crypto", "baseline")
    fresh = bind_research_configuration(research_profile(CRYPTO_PROFILE_ID), baseline)
    for name in (
        "cross_asset",
        "cross_btc_eth",
        "derivatives_state",
        "external_context",
        "spot_context",
    ):
        expanded = bind_research_configuration(
            research_profile(CRYPTO_PROFILE_ID),
            research_configuration("crypto", name),
            tardis_root=tmp_path,
        )
        rebound = bind_research_configuration(expanded, baseline, tardis_root=tmp_path)
        assert rebound.research_datasets.list() == fresh.research_datasets.list()
        assert rebound.processed.list() == fresh.processed.list()
        assert (
            rebound.protocol(baseline["scenario_id"]).manifest()
            == fresh.protocol(baseline["scenario_id"]).manifest()
        )


@pytest.mark.parametrize("enhanced", [False, True])
def test_benchmark_manifest_follows_candidates_instead_of_scenario(enhanced):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name="core6")
    if enhanced:
        config["scenario_id"] = "pv_expanded"
        config["portfolios"] = [
            item
            for item in config["portfolios"]
            if item["construction"] == "benchmark_enhanced"
        ]
    else:
        config["portfolios"] = []
    selected = bind_research_configuration(research_profile(), config)
    protocol = selected.protocol(config["scenario_id"])
    expected = {"csi500": "H00905", "csi1000": "H00852"} if enhanced else {}
    assert protocol.manifest()["benchmark_total_return_indices"] == expected
    targets = protocol.target_specs("research run")
    assert (
        any("benchmark_weights" in target.dependencies for target in targets)
        == enhanced
    )


def test_crypto_modes_share_models_and_isolate_execution_targets():
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    profile = research_profile("crypto_binance_hft")
    base = bind_research_configuration(profile, config).protocol(config["scenario_id"])
    changed = copy.deepcopy(config)
    changed["research_plan"]["execution"]["backtest_modes"] = [
        "independent_segments",
        "continuous",
    ]
    changed["research_plan"]["execution"]["taker_fee_bps"] += 1
    altered = bind_research_configuration(profile, changed).protocol(
        config["scenario_id"]
    )
    model_id = config["scenario_id"] + "_models"
    assert base.target_manifest(model_id, "MODEL") == altered.target_manifest(
        model_id, "MODEL"
    )
    targets = altered.execution_targets()
    assert len(targets) == 2 * len(base.execution_targets())
    assert [item.execution_mode for item in targets[:2]] == [
        "continuous",
        "independent_segments",
    ]
    specs = {
        target.target_id: target for target in altered.target_specs("research run")
    }
    for target in targets:
        assert specs[target.statistics_target_id].dependencies == (target.target_id,)
    taker = next(
        item
        for item in targets
        if isinstance(item.candidate, crypto_hft.CryptoTakerCandidate)
        and item.execution_mode == "continuous"
    )
    assert canonical_json_hash(
        base.target_manifest(taker.target_id, "PORTFOLIO")
    ) != canonical_json_hash(altered.target_manifest(taker.target_id, "PORTFOLIO"))
    changed["models"][0]["input_factor_columns"] = ["ofi_500ms"]
    input_changed = bind_research_configuration(profile, changed).protocol(
        config["scenario_id"]
    )
    assert input_changed.target_manifest(model_id, "MODEL") != altered.target_manifest(
        model_id, "MODEL"
    )


def test_crypto_factor_evaluation_manifest_ignores_model_configuration():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    profile = research_profile("crypto_binance_hft")
    base = bind_research_configuration(profile, config).protocol(config["scenario_id"])
    changed = copy.deepcopy(config)
    changed["models"][0]["model_algorithm"] = "linear_regression"
    changed["models"][0]["hyperparameters"] = {}
    altered = bind_research_configuration(profile, changed).protocol(
        config["scenario_id"]
    )

    factor_evaluation = f"{config['scenario_id']}_evaluation"
    changed_evaluation = f"{config['models'][0]['model_id']}_oos_evaluation"
    targets = {item.target_id: item for item in base.target_specs("research run")}
    assert targets[factor_evaluation].dependencies == (config["scenario_id"],)
    assert base.target_manifest(factor_evaluation, "EVALUATION") == (
        altered.target_manifest(factor_evaluation, "EVALUATION")
    )
    assert base.target_manifest(changed_evaluation, "EVALUATION") != (
        altered.target_manifest(changed_evaluation, "EVALUATION")
    )


@pytest.mark.parametrize(
    "modes", [[], ["continuous", "continuous"], ["unknown"], "continuous"]
)
def test_crypto_mode_configuration_rejects_invalid_values(modes):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    config["research_plan"]["execution"]["backtest_modes"] = modes
    with pytest.raises(
        ValueError, match="crypto_(backtest_modes|execution_scenarios)_invalid"
    ):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


@pytest.mark.parametrize("inventory,valid", [(0.5, True), (1.1, False)])
def test_crypto_maker_inventory_uses_candidate_asset_limit(inventory, valid):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    candidate = next(
        item
        for item in config["portfolios"]
        if item["candidate_id"] == "crypto_hft_maker_baseline"
    )
    candidate["asset_id"] = "eth_usdt"
    config["portfolios"] = [candidate]
    config["research_plan"]["execution"]["maker"]["inventory_target"] = inventory
    profile = research_profile("crypto_binance_hft")
    if valid:
        bound = bind_research_configuration(profile, config)
        protocol = bound.protocol(config["scenario_id"])
        assert protocol.execution_targets()[0].candidate.asset_id == "eth_usdt"
    else:
        with pytest.raises(ValueError, match="crypto_maker_inventory_target_invalid"):
            bind_research_configuration(profile, config)
