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
            **(
                {"primary_label": "forward_return_20d"}
                if profile_id == "a_share_csmar_choice"
                else {}
            ),
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
        **(
            {"primary_portfolio_id": None}
            if profile_id == "a_share_csmar_choice"
            else {}
        ),
    }


@pytest.mark.parametrize(
    "model_id",
    ["timestamp", "asset_id", "market_id", "renamed_model", "prediction_score"],
)
def test_crypto_model_names_cannot_overwrite_evaluation_structure(model_id):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "linear_regression"
    )
    model["model_id"] = model_id
    config.update(models=[model], portfolios=[], comparison_model_id=model_id)
    base = research_profile("crypto_binance_hft")
    if model_id in {"timestamp", "asset_id", "market_id"}:
        with pytest.raises(
            ValueError,
            match="^evaluation_signal_structural_column_overlap: " + model_id + "$",
        ):
            bind_research_configuration(base, config)
    else:
        selected = bind_research_configuration(base, config)
        signal = selected.evaluations.get(model_id + "_oos_evaluation").signals[0]
        assert signal.signal_id == model_id
        assert signal.source_column == "prediction_score"


@pytest.mark.parametrize("market", ["a_share", "crypto"])
@pytest.mark.parametrize(
    "model_id", ["champion", "winner", "Champion", "champion__config_0", "named_model"]
)
def test_champion_model_name_is_reserved_only_for_a_share_families(market, model_id):
    from quant_research_platform.catalog.models import ModelCatalog
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(
        market, "core6" if market == "a_share" else "baseline"
    )
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "linear_regression"
    )
    model["model_id"] = model_id
    config.update(models=[model], portfolios=[], comparison_model_id=model_id)
    assert (
        ModelCatalog.from_configuration(config["models"]).get(model_id).model_id
        == model_id
    )
    profile = research_profile(
        "a_share_csmar_choice" if market == "a_share" else "crypto_binance_hft"
    )
    if market == "a_share":
        config["primary_portfolio_id"] = None
    if market == "a_share" and model_id == "champion":
        with pytest.raises(ValueError, match="^a_share_model_id_reserved$"):
            bind_research_configuration(profile, config)
    elif market == "crypto" and model_id == "Champion":
        with pytest.raises(ValueError, match="^invalid_run_id$"):
            bind_research_configuration(profile, config)
    else:
        selected = bind_research_configuration(profile, config)
        assert selected.models.get(model_id).model_id == model_id
        assert (
            selected.protocol(config["scenario_id"]).configuration()["models"][0][
                "model_id"
            ]
            == model_id
        )


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
    assert protocol.terminal_targets("research run") == (
        "pv_expanded_evaluation",
        "pv_expanded_factor_statistics",
    )
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


@pytest.mark.parametrize("market,name", (("a_share", "core6"), ("crypto", "baseline")))
@pytest.mark.parametrize(
    "algorithm",
    ("linear_regression", "equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"),
)
def test_fixed_algorithm_configuration_rejects_unconsumed_parameters(
    market, name, algorithm
):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(market, name)
    model = config["models"][0]
    model["model_algorithm"] = algorithm
    model["hyperparameters"] = {"fit_intercept": False}
    profile = research_profile(config["profile_id"])
    with pytest.raises(ValueError, match="^model_hyperparameters_not_supported$"):
        bind_research_configuration(profile, config)
    if market == "a_share":
        model["hyperparameters"] = {}
        model["parameter_candidates"] = [{}, {"fit_intercept": False}]
        with pytest.raises(ValueError, match="^model_hyperparameters_not_supported$"):
            bind_research_configuration(profile, config)


def test_crypto_selection_keeps_execution_assets_and_can_evaluate_only():
    profile = research_profile("crypto_binance_hft")
    config = configuration(profile.profile_id, crypto_ids.HFT_RESEARCH_DATASET_ID)
    config["research_plan"] = hft_research_plan()
    config["source_archives"] = None
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
    config["primary_portfolio_id"] = config["portfolios"][0]["candidate_id"]
    bound = bind_research_configuration(research_profile(), config)
    candidate = bound.protocol("pv_expanded").portfolio_candidates()[0]
    assert candidate.top_quantile == 0.1
    assert candidate.prediction_selector == "champion"
    assert candidate.risk_context_target_ids == (
        "price_volume_facts",
        "market_model_factors",
    )
    assert candidate.risk_style_columns
    ledger = next(
        target
        for target in bound.protocol("pv_expanded").target_specs("research run")
        if target.target_id == "pv_expanded_global_champion_top20_equal_weight_ledger"
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
    assert binding.raw_root == tmp_path
    archives = tuple(config["auxiliary_sources"]["derivatives_state"]["archives"])
    assert (
        binding.producer.keywords["open_archive"].keywords["archive_paths"] == archives
    )
    assert binding.source_identity.keywords["archive_paths"] == archives
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
    config["research_plan"]["test"]["end"] = "2026-10-02T00:00:00"
    config["windows"]["holdout"]["end"] = "2026-10-02T00:00:00"
    bound = bind_research_configuration(research_profile("crypto_binance_hft"), config)
    assert (
        bound.protocol(config["scenario_id"]).research_plan.test.end.isoformat()
        == "2026-10-02T00:00:00"
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
    assert binding.raw_root == tmp_path
    archives = tuple(config["auxiliary_sources"]["spot"]["archives"])
    assert binding.producer.keywords["open_run"].keywords["archive_paths"] == archives
    assert binding.source_identity.keywords["archive_paths"] == archives
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
        unit = params["base_units_per_amount"]["btc_usdt"]
        assert (unit if venue == "bybit" else unit["after"]) == (
            1.0 if venue == "bybit" else 0.01
        )
        if venue == "okx":
            assert unit["before"] == 0.0001
            assert unit["transition_start"] == "2020-03-20T08:00:00"
            assert unit["transition_end"] == "2020-03-20T08:30:00"
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
    config["auxiliary_sources"]["bybit"]["archives"]["BTCUSDT"]["quotes"] = [
        "../escape"
    ]
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
    config["primary_portfolio_id"] = selected_item["candidate_id"]
    selected = bind_research_configuration(research_profile(), config)
    protocol = selected.protocol(config["scenario_id"])
    candidates = protocol.portfolio_candidates()
    assert [item.selection_group for item in candidates] == ["global", "quarterly"]
    targets = []
    for candidate in candidates:
        assert (
            candidate.candidate_id
            == candidate.selection_group + "_custom_monthly_top_decile"
        )
        assert candidate.top_quantile == 0.1
        assert candidate.construction == "top20_float_cap"
        assert candidate.rebalance_schedule == "monthly-last-trading-day"
        target = config["scenario_id"] + "_" + candidate.candidate_id + "_ledger"
        assert target in selected.producer_bindings
        targets.append(target)
    graph = resolve_target_graph(
        _target_specs(selected, action="research run", protocol=protocol),
        protocol.terminal_targets("research run"),
    )
    assert set(targets).issubset({item.target_id for item in graph})
    assert "fundamental_factors" not in {item.target_id for item in graph}
    assert not any("champion_csi" in item.target_id for item in graph)
    baseline = research_configuration(name="core6")
    formal = bind_research_configuration(research_profile(), baseline)
    candidates = formal.protocol(baseline["scenario_id"]).portfolio_candidates()
    assert len(candidates) == 10
    for group in ("global", "quarterly"):
        selected_group = [item for item in candidates if item.selection_group == group]
        assert [item.benchmark_id for item in selected_group[-2:]] == [
            "csi500",
            "csi1000",
        ]


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
    development = selected.protocol(config["scenario_id"])
    assert not any(
        target.endswith("_ledger")
        for target in development.terminal_targets("research run")
    )
    protocol = selected.protocol(config["scenario_id"], access_window="holdout")
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
    config["primary_portfolio_id"] = None
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
    development = selected.protocol(config["scenario_id"])
    assert not any(
        spec.evaluation_id in development.terminal_targets("research run")
        for spec in comparisons
    )
    protocol = selected.protocol(config["scenario_id"], access_window="holdout")
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
            plan["folds"][0]["validation"]["start"],
            plan["folds"][0]["validation"]["end"],
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
    from dataclasses import replace

    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        bind_research_configuration,
        research_profile,
    )
    from tests.helpers.research_configuration import research_configuration

    baseline = research_configuration("crypto", "baseline")
    base = research_profile(CRYPTO_PROFILE_ID)
    custom = next(iter(base.producer_bindings.values()))
    base = replace(
        base, producer_bindings={**base.producer_bindings, "custom_source": custom}
    )
    fresh = bind_research_configuration(base, baseline, tardis_root=tmp_path)
    for name in (
        "cross_asset",
        "cross_btc_eth",
        "derivatives_state",
        "external_context",
        "spot_context",
    ):
        expanded = bind_research_configuration(
            base,
            research_configuration("crypto", name),
            tardis_root=tmp_path,
        )
        rebound = bind_research_configuration(expanded, baseline, tardis_root=tmp_path)
        assert rebound.research_datasets.list() == fresh.research_datasets.list()
        assert rebound.processed.list() == fresh.processed.list()
        assert set(rebound.producer_bindings) == set(fresh.producer_bindings)
        assert rebound.producer_bindings["custom_source"] is custom
        assert {
            key: binding.raw_root for key, binding in rebound.producer_bindings.items()
        } == {key: binding.raw_root for key, binding in fresh.producer_bindings.items()}
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
    config["primary_portfolio_id"] = (
        config["portfolios"][0]["candidate_id"] if enhanced else None
    )
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
    base = bind_research_configuration(profile, config).protocol(
        config["scenario_id"], access_window="holdout"
    )
    changed = copy.deepcopy(config)
    changed["research_plan"]["execution"]["backtest_modes"] = [
        "independent_segments",
        "continuous",
    ]
    changed["research_plan"]["execution"]["taker_fee_bps"] += 1
    altered = bind_research_configuration(profile, changed).protocol(
        config["scenario_id"], access_window="holdout"
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
        config["scenario_id"], access_window="holdout"
    )
    assert input_changed.target_manifest(model_id, "MODEL") != altered.target_manifest(
        model_id, "MODEL"
    )


def test_crypto_factor_evaluation_manifest_ignores_model_configuration():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    profile = research_profile("crypto_binance_hft")
    base = bind_research_configuration(profile, config).protocol(
        config["scenario_id"], access_window="holdout"
    )
    changed = copy.deepcopy(config)
    changed["models"][0]["model_algorithm"] = "linear_regression"
    changed["models"][0]["hyperparameters"] = {}
    altered = bind_research_configuration(profile, changed).protocol(
        config["scenario_id"], access_window="holdout"
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


@pytest.mark.parametrize(
    "name",
    [
        "baseline",
        "cross_asset",
        "cross_btc_eth",
        "derivatives_state",
        "external_context",
        "multiscale",
        "spot_context",
    ],
)
def test_crypto_configurations_preserve_market_signal_semantics(tmp_path, name):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", name)
    profile = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    target_id = config["scenario_id"] + "_evaluation"
    actual = profile.evaluations.get(target_id)
    assert [signal.signal_id for signal in actual.signals] == config["evaluation"][
        "factors"
    ]
    assert [signal.direction for signal in actual.signals] == [
        "lower_is_better"
        if name in {"spread", "relative_spread"}
        else "higher_is_better"
        for name in config["evaluation"]["factors"]
    ]
    assert all(signal.transform == "identity" for signal in actual.signals)
    assert actual.label_columns == tuple(config["evaluation"]["labels"])
    assert actual == crypto_hft.build_crypto_evaluation_catalog(
        configuration=config
    ).get(target_id)
    assert all(
        signal.direction == "higher_is_better"
        for spec in profile.evaluations.list()
        for signal in spec.signals
        if signal.source_kind == "prediction"
    )
    protocol = profile.protocol(config["scenario_id"])
    assert target_id in protocol.terminal_targets("research run")
    target = next(
        spec
        for spec in protocol.target_specs("research run")
        if spec.target_id == target_id
    )
    assert target.dependencies == (config["scenario_id"],)


@pytest.mark.parametrize(
    "name",
    [
        "baseline",
        "multiscale",
        "derivatives_state",
        "spot_context",
        "external_context",
        "cross_btc_eth",
        "cross_asset",
    ],
)
def test_crypto_month_start_chronology_and_source_closure(tmp_path, name):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", name)
    plan = config["research_plan"]
    assert plan["folds"] == [
        {
            "fold_id": "development",
            "train": {
                "start": "2026-05-01T00:00:00",
                "end": "2026-07-01T23:59:59.900000",
            },
            "validation": {
                "start": "2026-08-01T00:00:00",
                "end": "2026-09-01T23:59:59.900000",
            },
        }
    ]
    assert plan["test"] == {
        "start": "2026-10-01T00:00:00",
        "end": "2026-10-01T23:59:59.900000",
    }
    assert config["windows"] == {
        "development": plan["folds"][0]["validation"],
        "holdout": plan["test"],
    }
    assert plan["execution"]["backtest_modes"] == ["continuous"]
    days = [f"2026-{month:02d}-01" for month in range(5, 11)]
    assert config["source_archives"] == [
        f"binance-futures/{day}-btcusdt-ethusdt-native" for day in days
    ]
    auxiliary = config.get("auxiliary_sources", {})
    expected_sources = {
        "baseline": set(),
        "multiscale": set(),
        "derivatives_state": {"derivatives_state"},
        "spot_context": {"spot"},
        "external_context": {"bybit", "okx"},
        "cross_btc_eth": {"cross_asset"},
        "cross_asset": {"cross_asset", "sol"},
    }
    assert set(auxiliary) == expected_sources[name]
    for key, prefix, suffix in (
        ("derivatives_state", "binance-futures", "btcusdt-ethusdt-state"),
        ("spot", "binance", "btcusdt-ethusdt-native"),
        ("sol", "binance-futures", "solusdt-native"),
    ):
        if key in auxiliary:
            assert auxiliary[key]["archives"] == [
                f"{prefix}/{day}-{suffix}" for day in days
            ]
    for venue, symbols in (
        ("bybit", ("BTCUSDT", "ETHUSDT")),
        ("okx", ("BTC-USDT-SWAP", "ETH-USDT-SWAP")),
    ):
        if venue in auxiliary:
            assert auxiliary[venue]["archives"] == {
                symbol: {
                    kind: [f"{day}-{symbol}-{kind}" for day in days]
                    for kind in ("quotes", "trades")
                }
                for symbol in symbols
            }
    profile = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    assert profile.protocol(config["scenario_id"]).manifest()["research_plan"] == plan


@pytest.mark.parametrize(
    "factors",
    [["spread"], ["relative_spread", "mark_mid_basis", "spread"]],
)
def test_crypto_factor_and_label_subsets_keep_order_and_catalog_identity(
    tmp_path, factors
):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "derivatives_state")
    config.update(
        stage="evaluation", models=[], portfolios=[], comparison_model_id=None
    )
    config["evaluation"] = {
        "factors": factors,
        "labels": ["future_mid_return_500ms"],
    }
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    (actual,) = selected.evaluations.list()
    (expected,) = crypto_hft.build_crypto_evaluation_catalog(
        configuration=config
    ).list()
    assert actual == expected
    assert [signal.signal_id for signal in actual.signals] == factors
    assert actual.label_columns == ("future_mid_return_500ms",)
    assert dict(actual.label_availability_columns) == {
        "future_mid_return_500ms": "future_mid_return_500ms_available_at"
    }


@pytest.mark.parametrize(
    "name", ["core6", "fusion_nf_core", "fusion_nf_expectations", "pv25", "pv_expanded"]
)
def test_a_share_configurations_keep_natural_signal_direction(name):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("a_share", name)
    selected = bind_research_configuration(research_profile(), config)
    assert all(
        signal.direction == "higher_is_better"
        for spec in selected.evaluations.list()
        for signal in spec.signals
    )


@pytest.mark.parametrize(
    "primary", [None, "unknown_portfolio", "global_champion_top20_equal_weight"]
)
def test_a_share_primary_portfolio_requires_declared_candidate_identity(primary):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    config["primary_portfolio_id"] = primary
    with pytest.raises(ValueError, match="research_primary_portfolio_invalid"):
        bind_research_configuration(research_profile(), config)


def test_crypto_latency_scenarios_use_existing_execution_dag_and_precise_identity():
    from quant_research_platform.workflows.research import (
        _target_specs,
        resolve_target_graph,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    config["portfolios"] = [
        {
            "candidate_id": "base",
            "asset_id": "btc_usdt",
            "model_id": "crypto_hft_linear_regression",
            "latency_ms": None,
        },
        {
            "candidate_id": "explicit",
            "asset_id": "btc_usdt",
            "model_id": "crypto_hft_linear_regression",
            "latency_ms": 100,
        },
    ]
    config["research_plan"]["execution"]["latency_ms"] = 17
    config["research_plan"]["execution"]["latency_scenarios_ms"] = [0, 17, 100, 200]
    profile = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    protocol = profile.protocol(config["scenario_id"], access_window="holdout")
    assert [
        (item.candidate_id, item.latency_ms) for item in protocol.portfolio_candidates()
    ] == [
        ("base", None),
        ("explicit", 100),
        ("base_latency_0ms", 0),
        ("base_latency_200ms", 200),
    ]
    graph = {
        item.target_id: item
        for item in resolve_target_graph(
            _target_specs(profile, action="research run", protocol=protocol),
            protocol.terminal_targets("research run"),
        )
    }
    for target in protocol.execution_targets():
        assert target.target_id in profile.producer_bindings
        assert graph[target.statistics_target_id].dependencies == (target.target_id,)
        actual = protocol.target_manifest(target.target_id, "PORTFOLIO")
        assert (
            actual["portfolio_candidates"][0]["latency_ms"]
            == target.candidate.latency_ms
        )
    base_before = protocol.target_manifest("base_continuous_statistics", "STATISTIC")
    config["research_plan"]["execution"]["latency_scenarios_ms"] = [0, 17, 100, 250]
    changed = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    changed_protocol = changed.protocol(config["scenario_id"], access_window="holdout")
    assert (
        changed_protocol.target_manifest("base_continuous_statistics", "STATISTIC")
        == base_before
    )
    assert "base_latency_250ms_ledger" in {
        item.target_id for item in changed_protocol.execution_targets()
    }
    assert "base_latency_200ms_ledger" not in {
        item.target_id for item in changed_protocol.execution_targets()
    }
    config["portfolios"].append(
        {
            "candidate_id": "base_latency_250ms",
            "asset_id": "btc_usdt",
            "model_id": "crypto_hft_lightgbm_regressor",
            "latency_ms": 100,
        }
    )
    with pytest.raises(ValueError, match="research_portfolio_candidate_duplicate"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_formal_seed_normalizes_native_duplicate_before_derived_selection():
    from quant_research_platform.research.models.production import (
        _selection_model_specs,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "random_forest_regressor"
    )
    model["random_seed"] = 42
    model.setdefault("hyperparameters", {})["random_state"] = 42
    for candidate in model.get("parameter_candidates", []):
        candidate["random_state"] = 42
    bound = bind_research_configuration(research_profile(), config)
    canonical = next(
        item
        for item in bound.protocol(config["scenario_id"]).manifest()[
            "research_configuration"
        ]["models"]
        if item["model_id"] == model["model_id"]
    )
    spec = bound.models.get(model["model_id"])
    assert spec.random_seed == 42
    assert "random_state" not in spec.hyperparameters
    assert all(
        "random_state" not in value
        for value in canonical.get("parameter_candidates", [])
    )
    selected = _selection_model_specs(
        spec,
        (dict(spec.hyperparameters),),
        n_train=100,
        root_seed=42,
        quarter_id="2023Q1",
    )
    assert len(selected) == 1
    assert selected[0].random_seed != 42
    model["random_seed"] = 123
    model["hyperparameters"].pop("random_state")
    for candidate in model.get("parameter_candidates", []):
        candidate.pop("random_state", None)
    with pytest.raises(ValueError, match="research_root_seed_must_be_42"):
        bind_research_configuration(research_profile(), config)


@pytest.mark.parametrize(
    "section,field,value,affects_calibration",
    [
        ("maker", "maker_fee_bps", 3.0, False),
        ("execution", "markout_horizon_ms", 600, False),
        ("maker", "inventory_skew", 0.5, False),
        ("maker", "inventory_target", 0.01, False),
        ("maker", "latency_ms", 210, True),
        ("maker", "min_calibration_samples", 101, True),
        ("limits", "max_order_quantity", 0.02, True),
        ("limits", "max_position_quantity", 0.004, True),
    ],
)
def test_maker_calibration_identity_tracks_reference_parameters(
    section, field, value, affects_calibration
):
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "baseline")
    profile = research_profile(config["profile_id"])
    initial = bind_research_configuration(profile, config).protocol(
        config["scenario_id"]
    )
    candidate = next(
        candidate
        for candidate in initial.portfolio_candidates()
        if isinstance(candidate, crypto_hft.CryptoMakerCandidate)
        and candidate.calibrated
    )
    config["portfolios"] = [
        item
        for item in config["portfolios"]
        if item["candidate_id"] == candidate.candidate_id
    ]
    base = bind_research_configuration(profile, config).protocol(config["scenario_id"])
    changed = copy.deepcopy(config)
    plan = changed["research_plan"]
    target = (
        plan["exposure_limits"][candidate.asset_id]
        if section == "limits"
        else plan["execution"]["maker"]
        if section == "maker"
        else plan["execution"]
    )
    target[field] = value
    altered = bind_research_configuration(profile, changed).protocol(
        config["scenario_id"]
    )
    before = base.target_manifest(candidate.calibration_target_id, "STATISTIC")
    after = altered.target_manifest(candidate.calibration_target_id, "STATISTIC")
    assert (
        canonical_json_hash(before) != canonical_json_hash(after)
    ) is affects_calibration
    model_id = config["scenario_id"] + "_models"
    assert base.target_manifest(model_id, "MODEL") == altered.target_manifest(
        model_id, "MODEL"
    )
    execution_id = base.execution_targets()[0].target_id
    affects_execution = not (
        section == "maker" and field in {"inventory_target", "inventory_skew"}
    )
    assert (
        base.target_manifest(execution_id, "PORTFOLIO")
        != altered.target_manifest(execution_id, "PORTFOLIO")
    ) is affects_execution
    assert base.configuration() != altered.configuration()
    reference = before["research_plan"]["execution"]["maker"]
    assert set(reference) == {"order_quantity", "latency_ms", "min_calibration_samples"}
    assert reference == {
        "order_quantity": 0.01,
        "latency_ms": 100,
        "min_calibration_samples": 100,
    }


def test_crypto_parallel_labels_have_no_unused_primary_label_configuration():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "multiscale")
    assert "primary_label" not in config["evaluation"]
    profile = bind_research_configuration(
        research_profile("crypto_binance_hft"), config
    )
    evaluation = profile.evaluations.get(config["scenario_id"] + "_evaluation")
    assert evaluation.label_columns == tuple(config["evaluation"]["labels"])
    assert {item.label_column for item in profile.models.list()} == {
        item["label_column"] for item in config["models"]
    }
    config["evaluation"]["primary_label"] = config["evaluation"]["labels"][0]
    with pytest.raises(ValueError, match="^research_evaluation_configuration_invalid$"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


def test_a_share_still_requires_effective_primary_label():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    del config["evaluation"]["primary_label"]
    with pytest.raises(ValueError, match="^research_evaluation_configuration_invalid$"):
        bind_research_configuration(research_profile(), config)


def test_forecast_robustness_requires_a_real_expectations_consumer():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    assert "earnings_yield" not in config["evaluation"]["factors"]
    config["evaluation"]["forecast_freshness_days"] = 180
    with pytest.raises(
        ValueError, match="^research_forecast_freshness_not_applicable$"
    ):
        bind_research_configuration(research_profile(), config)


@pytest.mark.parametrize("scenario", ("pv_all", "pv_expanded"))
@pytest.mark.parametrize(
    "entrypoint",
    ("evaluation", "factor", "feature", "feature_missing", "feature_observed"),
)
def test_a_share_scenario_rejects_all_outside_factor_entrypoints(scenario, entrypoint):
    from quant_research_platform.contracts.models import ModelSpec

    column = "earnings_yield"
    model = None
    if entrypoint != "evaluation":
        model = ModelSpec(
            model_id="selected_ols",
            model_algorithm="linear_regression",
            input_factor_columns=(column,) if entrypoint == "factor" else (),
            input_feature_columns=(entrypoint + "_" + column,)
            if entrypoint != "factor"
            else (),
            label_column="model_target",
        )
    config = configuration("a_share_csmar_choice", scenario, model)
    if entrypoint == "evaluation":
        config["evaluation"]["factors"] = [column]
    with pytest.raises(ValueError, match="^research_factor_outside_scenario$"):
        bind_research_configuration(research_profile(), config)


@pytest.mark.parametrize("scenario", ("fusion_nf_core", "fusion_nf_expectations"))
def test_a_share_financial_scenarios_keep_allowed_model_feature_sources(scenario):
    from quant_research_platform.contracts.models import ModelSpec

    model = ModelSpec(
        model_id="selected_ols",
        model_algorithm="linear_regression",
        input_feature_columns=("feature_earnings_yield",),
        label_column="model_target",
    )
    config = configuration("a_share_csmar_choice", scenario, model)
    selected = bind_research_configuration(research_profile(), config)
    dataset = selected.research_datasets.get(scenario)
    assert dataset.factor_columns == (
        "share_price",
        "earnings_yield",
        "log_float_market_cap",
    )
    assert "fundamental_factors" in dataset.input_data_ids
    assert {"is_nonfinancial_industry", "has_legal_fundamental_pit"} <= set(
        dataset.context_columns
    )
    assert selected.models.get(model.model_id).input_columns == model.input_columns


@pytest.mark.parametrize("market", ("a_share", "crypto"))
@pytest.mark.parametrize("bad", ([], {}), ids=("array", "object"))
def test_configuration_stage_types_reject_at_binding_and_research_plan_boundary(
    tmp_path, market, bad
):
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows.operations import research_run
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(
        market, "core6" if market == "a_share" else "baseline"
    )
    before = copy.deepcopy(config)
    profile = research_profile(
        "a_share_csmar_choice" if market == "a_share" else "crypto_binance_hft"
    )
    selected = bind_research_configuration(profile, config)
    assert selected.profile_id == profile.profile_id
    assert config == before
    changed = copy.deepcopy(config)
    changed["stage"] = bad
    with pytest.raises(ValueError, match="^research_configuration_invalid$"):
        bind_research_configuration(profile, changed)
    with pytest.raises(ValueError, match="^research_configuration_invalid$"):
        research_run(
            profile=profile,
            layout=StorageLayout(tmp_path / "project"),
            research_plan=changed,
            plan=True,
        )
    assert not (tmp_path / "project").exists()
    assert config == before


def test_a_share_factor_statistics_are_a_separate_model_free_target():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    selected = bind_research_configuration(research_profile(), config)
    protocol = selected.protocol(
        scenario_id=config["scenario_id"], access_window="development"
    )
    targets = {
        target.target_id: target for target in protocol.target_specs("research run")
    }
    factor_target = protocol.factor_statistics_target_id
    assert targets[factor_target].dependencies == (protocol.scenario_id,)
    assert factor_target in protocol.terminal_targets("research run")
    assert (
        targets[protocol.statistics_target_ids[1]].dependencies[-1]
        == protocol.statistics_target_ids[0]
    )
    factor_manifest = protocol.target_manifest(factor_target, "STATISTIC")
    assert "selection_group" not in factor_manifest
    assert set(factor_manifest["research_configuration"]) == {"evaluation"}
    assert factor_manifest["training_weighting"]["role"] == "primary"


def test_named_feature_recipe_is_bound_to_dataset_and_inference_qualification():
    from quant_research_platform.research.protocols.a_share import _inference_blocker
    from quant_research_platform.storage.datasets.schema import data_spec_payload
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name="core6")
    primary = bind_research_configuration(research_profile(), config)
    config["feature_recipe"] = "a_share_quantile_1_99"
    robust = bind_research_configuration(research_profile(), config)
    first = primary.research_datasets.get(config["scenario_id"])
    second = robust.research_datasets.get(config["scenario_id"])
    assert first.feature_columns == second.feature_columns
    assert first.feature_columns
    assert first.feature_transforms[0].recipe_id == "a_share_canonical"
    assert second.feature_transforms[0].recipe_id == "a_share_quantile_1_99"
    assert data_spec_payload(first) != data_spec_payload(second)
    protocol = robust.protocol(
        scenario_id=config["scenario_id"], access_window="development"
    )
    manifest = protocol.target_manifest(protocol.statistics_target_ids[0], "STATISTIC")
    assert _inference_blocker(manifest) == "feature_recipe_robustness_excluded"
    assert (
        _inference_blocker(
            primary.protocol(
                scenario_id=config["scenario_id"], access_window="development"
            ).manifest()
        )
        is None
    )
    config["feature_recipe"] = "custom_winsorization"
    with pytest.raises(ValueError, match="unsupported_feature_recipe"):
        bind_research_configuration(research_profile(), config)


@pytest.mark.parametrize("market", ["a_share", "crypto"])
@pytest.mark.parametrize("stage", ["dataset", "evaluation", "models"])
def test_stage_terminals_stop_at_requested_boundary(market, stage):
    from quant_research_platform.workflows.research import (
        _target_specs,
        resolve_target_graph,
    )
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(
        market, "core6" if market == "a_share" else "baseline"
    )
    config["stage"] = stage
    config["portfolios"] = []
    if market == "a_share":
        config["primary_portfolio_id"] = None
    if stage in {"dataset", "evaluation"}:
        config["models"] = []
        config["comparison_model_id"] = None
    profile = bind_research_configuration(
        research_profile(config["profile_id"]), config
    )
    protocol = profile.protocol(
        scenario_id=config["scenario_id"], access_window="development"
    )
    terminals = protocol.terminal_targets("research run")
    if stage == "dataset":
        assert terminals == (config["scenario_id"],)
    else:
        targets = resolve_target_graph(
            _target_specs(profile, action="research run", protocol=protocol), terminals
        )
        names = {target.target_id for target in targets}
        assert not any(
            name.endswith(("_ledger", "_portfolio_statistics", "_maker_calibration"))
            for name in names
        )
        assert any("model" in name for name in names) == (stage == "models")
        if market == "a_share":
            assert protocol.factor_statistics_target_id in names


def test_model_stage_rejects_portfolio_execution():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    config["stage"] = "models"
    assert config["portfolios"]
    with pytest.raises(ValueError, match="model_stage_has_portfolios"):
        bind_research_configuration(research_profile(config["profile_id"]), config)


def test_nonfinancial_expectations_has_no_financial_pit_dependency():
    import pandas as pd

    from quant_research_platform.research.filtering import a_share_eligibility
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("a_share", "pv_nf_expectations")
    selected = bind_research_configuration(research_profile(), config)
    dataset = selected.research_datasets.get("pv_nf_expectations")
    assert "expectations_factors" in dataset.input_data_ids
    assert "fundamental_factors" not in dataset.input_data_ids
    assert "has_legal_fundamental_pit" not in dataset.context_columns
    assert "is_nonfinancial_industry" in dataset.context_columns
    frame = pd.DataFrame(
        {
            "is_a_share_scope": [True] * 4,
            "is_trading": [True] * 4,
            "is_st": [False] * 4,
            "is_pt": [False] * 4,
            "has_252_trading_sessions": [True] * 4,
            "close": [10.0] * 4,
            "adj_close": [10.0] * 4,
            "volume": [100.0] * 4,
            "amount": [1000.0] * 4,
            "turnover": [0.01] * 4,
            "is_nonfinancial_industry": [True, False, True, None],
            "industry_standard_id": ["csrc"] * 4,
            "industry_level1_code": ["C", "J", "", "C"],
            "industry_level1_name": ["manufacturing", "finance", "", "manufacturing"],
        }
    )
    assert a_share_eligibility(frame, "pv_nf_expectations").tolist() == [
        True,
        False,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="has_legal_fundamental_pit"):
        a_share_eligibility(frame, "fusion_nf_expectations")


@pytest.mark.parametrize(
    "factor,sources",
    [
        ("book_to_market", {"financial_balance"}),
        ("earnings_yield", {"financial_income"}),
        ("roe", {"financial_balance", "financial_income"}),
        ("accruals", {"financial_balance", "financial_income", "financial_cashflow"}),
    ],
)
def test_financial_factor_selection_requires_only_consumed_statements(factor, sources):
    config = configuration("a_share_csmar_choice", "fusion_nf_core")
    config["evaluation"]["factors"] = [factor]
    selected = bind_research_configuration(research_profile(), config)
    assert set(
        selected.processed.get("fundamental_pit").source_input_data_ids
    ) == sources | {"trading_calendar", "financial_first_disclosure"}
    columns = selected.processed.get("fundamental_factors").required_columns
    assert set(columns) == {
        "market_id",
        "asset_id",
        "timestamp",
        "has_legal_fundamental_pit",
        factor,
    }


def test_financial_pruning_keeps_enhanced_portfolio_risk_inputs():
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("a_share", "core6")
    selected = bind_research_configuration(research_profile(), config)
    assert (
        "earnings_yield"
        in selected.processed.get("fundamental_factors").required_columns
    )
    assert (
        "financial_income"
        in selected.processed.get("fundamental_pit").source_input_data_ids
    )


@pytest.mark.parametrize("delay", (-1, True, 0.5))
def test_auxiliary_input_delay_rejects_invalid_values(delay):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", "cross_btc_eth")
    config["auxiliary_sources"]["cross_asset"]["input_delay_ms"] = delay
    with pytest.raises(ValueError, match="research_input_delay_invalid"):
        bind_research_configuration(research_profile("crypto_binance_hft"), config)


@pytest.mark.parametrize(
    "name", ("cross_btc_eth", "spot_context", "external_context", "derivatives_state")
)
def test_auxiliary_input_delay_is_bound_into_producer_identity(tmp_path, name):
    import json

    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("crypto", name)
    sources = [
        source for key, source in config["auxiliary_sources"].items() if key != "sol"
    ]
    for source in sources:
        source["input_delay_ms"] = 100
    selected = bind_research_configuration(
        research_profile("crypto_binance_hft"), config, tardis_root=tmp_path
    )
    delayed = [
        spec
        for spec in selected.processed.list()
        if json.loads(spec.parameters_json).get("input_delay_ms") == 100
    ]
    assert len(delayed) == len(sources)


def test_a_share_binding_preserves_supplied_processed_dependencies():
    from dataclasses import replace

    from quant_research_platform.catalog.processed import ProcessedDataCatalog
    from tests.helpers.research_configuration import research_configuration

    base = research_profile()
    with pytest.raises(ValueError, match="optional_source_input_not_declared"):
        replace(base.processed.get("market_panel_1d"), source_input_data_ids=())
    selected = replace(
        base.processed.get("market_panel_1d"),
        source_input_data_ids=(),
        optional_source_input_data_ids=(),
    )
    profile = replace(
        base,
        processed=ProcessedDataCatalog(
            selected if spec.data_id == selected.data_id else spec
            for spec in base.processed.list()
        ),
    )
    bound = bind_research_configuration(profile, research_configuration(name="pv25"))
    assert bound.processed.get(selected.data_id) == selected


@pytest.mark.parametrize(
    "name",
    (
        "core6",
        "pv25",
        "pv_expanded",
        "pv_nf_expectations",
        "fusion_nf_core",
        "fusion_nf_expectations",
    ),
)
def test_a_share_candidates_are_complete_without_shadow_base_parameters(name):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(name=name)
    shadow = copy.deepcopy(config)
    covered = [model for model in config["models"] if model.get("parameter_candidates")]
    assert len(covered) == 5 and all(
        "hyperparameters" not in model for model in covered
    )
    for model in shadow["models"]:
        if model.get("parameter_candidates"):
            model["hyperparameters"] = {"unused_shadow_parameter": True}
    base = bind_research_configuration(research_profile(), config)
    altered = bind_research_configuration(research_profile(), shadow)
    assert [model_spec_payload(item) for item in base.models.list()] == [
        model_spec_payload(item) for item in altered.models.list()
    ]
    left, right = (
        base.protocol(config["scenario_id"]),
        altered.protocol(config["scenario_id"]),
    )
    for target in left.target_specs("research run"):
        kind = base.producer_bindings[target.target_id].kind
        if str(kind) != "REPORT":
            assert left.target_manifest(
                target.target_id, kind
            ) == right.target_manifest(target.target_id, kind)


@pytest.mark.parametrize("market", ["a_share", "crypto"])
def test_dataset_stage_preserves_declared_data_contracts(market, tmp_path):
    from quant_research_platform.storage.datasets.schema import data_spec_payload
    from quant_research_platform.workflows import research
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration(
        market, "pv_nf_expectations" if market == "a_share" else "cross_asset"
    )
    full = bind_research_configuration(
        research_profile(config["profile_id"]), config, tardis_root=tmp_path
    )
    config["stage"] = "dataset"
    staged = bind_research_configuration(
        research_profile(config["profile_id"]), config, tardis_root=tmp_path
    )
    full_protocol = full.protocol(scenario_id=config["scenario_id"])
    staged_protocol = staged.protocol(scenario_id=config["scenario_id"])
    graph = research.resolve_target_graph(
        research._target_specs(full, action="research run", protocol=full_protocol),
        (config["scenario_id"],),
    )
    assert staged_protocol.terminal_targets("research run") == (config["scenario_id"],)
    window = {**config["windows"]["development"], "access": "development"}
    for target in graph:
        left = research._catalog_data_spec(full, target.target_id)
        right = research._catalog_data_spec(staged, target.target_id)
        assert data_spec_payload(left) == data_spec_payload(right)
        assert research._semantic_target_time_window(
            "research run", full_protocol, target, left, window
        ) == research._semantic_target_time_window(
            "research run", staged_protocol, target, right, window
        )
    assert staged.models.list() == full.models.list()
    assert (
        staged_protocol.portfolio_candidates() == full_protocol.portfolio_candidates()
    )
    config["comparison_model_id"] = None
    bind_research_configuration(
        research_profile(config["profile_id"]), config, tardis_root=tmp_path
    )
    config["comparison_model_id"] = "not_declared"
    with pytest.raises(ValueError, match="research_comparison_model_invalid"):
        bind_research_configuration(
            research_profile(config["profile_id"]), config, tardis_root=tmp_path
        )


@pytest.mark.parametrize("field", ["model", "portfolio", "primary", "evaluation"])
def test_dataset_stage_keeps_declaration_validation(field):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration("a_share", "core6")
    config["stage"] = "dataset"
    if field == "model":
        config["models"][0]["model_algorithm"] = "not_an_algorithm"
    elif field == "portfolio":
        config["portfolios"][0]["not_a_field"] = 1
    elif field == "primary":
        config["primary_portfolio_id"] = "not_declared"
    else:
        config["stage"] = "evaluation"
    with pytest.raises((TypeError, ValueError)):
        bind_research_configuration(research_profile(), config)
