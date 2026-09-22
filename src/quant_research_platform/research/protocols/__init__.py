"""Executable research selection and market protocol owners."""

from quant_research_platform.markets.implementations.crypto import ids as crypto_ids


def bind_research_selection(profile, library, configuration):
    """Validate formal market selections before composing source producers."""
    import json
    from dataclasses import replace

    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.contracts.json_codec import (
        bounded_qrp_json_object_snapshot,
    )
    from quant_research_platform.research.features.transforms import (
        canonical_feature_transforms,
    )
    from quant_research_platform.research.protocols import a_share, crypto_hft

    config = bounded_qrp_json_object_snapshot(configuration, max_bytes=1024 * 1024)
    required = {
        "schema_id",
        "profile_id",
        "scenario_id",
        "stage",
        "evaluation",
        "models",
        "portfolios",
        "comparison_model_id",
        "windows",
    }
    if set(config) - {"research_plan", "auxiliary_sources"} != required or (
        config["schema_id"] != "qrp-research-configuration"
        or config["profile_id"] != profile.profile_id
        or config["stage"] not in {"evaluation", "research"}
    ):
        raise ValueError("research_configuration_invalid")
    from quant_research_platform.contracts.time import TimeWindow

    if str(profile.market_id) == "crypto" and "research_plan" not in config:
        raise ValueError("crypto_research_plan_required")
    if str(profile.market_id) == "a_share" and "research_plan" in config:
        raise ValueError("a_share_research_plan_not_supported")
    windows = config["windows"]
    if type(windows) is not dict or set(windows) != {"development", "holdout"}:
        raise ValueError("research_windows_invalid")
    for window in windows.values():
        if (
            type(window) is not dict
            or set(window) != {"start", "end"}
            or any(type(value) is not str for value in window.values())
        ):
            raise ValueError("research_window_invalid")
        TimeWindow(**window)
    scenario = config["scenario_id"]
    if type(scenario) is not str:
        raise ValueError("research_configuration_scenario_invalid")
    try:
        base = library.research_datasets.get(scenario)
    except KeyError:
        raise ValueError("research_configuration_scenario_invalid") from None
    evaluation = config["evaluation"]
    if type(evaluation) is not dict or set(evaluation) != {
        "factors",
        "labels",
        "primary_label",
    }:
        raise ValueError("research_evaluation_configuration_invalid")
    if any(
        type(evaluation[key]) is not list
        or not evaluation[key]
        or any(type(value) is not str for value in evaluation[key])
        for key in ("factors", "labels")
    ):
        raise ValueError("research_evaluation_selection_required")
    available_labels = {label.output_column for label in profile.labels.list()}
    if (
        any(
            type(label) is not str or label not in available_labels
            for label in evaluation["labels"]
        )
        or len(set(evaluation["labels"])) != len(evaluation["labels"])
        or type(evaluation["primary_label"]) is not str
        or evaluation["primary_label"] not in evaluation["labels"]
    ):
        raise ValueError("research_evaluation_label_invalid")
    if type(config["models"]) is not list or type(config["portfolios"]) is not list:
        raise ValueError("research_model_portfolio_selection_invalid")
    if config["stage"] == "evaluation" and (config["models"] or config["portfolios"]):
        raise ValueError("evaluation_only_has_model_or_portfolio")
    if config["stage"] == "research" and not config["models"]:
        raise ValueError("research_models_required")
    factors_by_column = {item.output_column: item for item in profile.factors.list()}
    feature_sources = (
        {
            prefix + column: column
            for column in factors_by_column
            for prefix in ("feature_", "feature_missing_", "feature_observed_")
        }
        if str(profile.market_id) == "a_share"
        else {}
    )
    selected_factors = list(evaluation["factors"])
    for item in config["models"]:
        if type(item) is not dict:
            raise ValueError("research_model_configuration_invalid")
        if str(profile.market_id) == "crypto" and "parameter_candidates" in item:
            raise ValueError("crypto_parameter_candidates_not_supported")
        for key in ("training_asset_ids", "evaluation_asset_ids"):
            if str(profile.market_id) == "crypto":
                values = item.get(key)
                if (
                    type(values) is not list
                    or not values
                    or any(
                        type(value) is not str
                        or value not in {"btc_usdt", "eth_usdt", "sol_usdt"}
                        for value in values
                    )
                    or len(set(values)) != len(values)
                ):
                    raise ValueError("research_model_asset_scope_invalid")
            elif key in item:
                raise ValueError("research_model_asset_scope_not_supported")
    model_catalog = ModelCatalog.from_configuration(config["models"])
    models = model_catalog.list()
    for item in config["models"]:
        item["training_weighting"] = model_catalog.get(
            item["model_id"]
        ).training_weighting.payload()
    model_configurations = {item["model_id"]: item for item in config["models"]}
    for model in models:
        selected_factors.extend(model.input_factor_columns)
        for column in model.input_feature_columns:
            if column not in feature_sources:
                raise ValueError("research_feature_not_available")
            selected_factors.append(feature_sources[column])
    uses_sol_rows = str(profile.market_id) == "crypto" and any(
        "sol_usdt" in item[key]
        for item in config["models"]
        for key in ("training_asset_ids", "evaluation_asset_ids")
    )
    if uses_sol_rows:
        base = replace(
            base,
            base_input_data_id=crypto_ids.HFT_INFORMATION_PANEL_DATA_ID,
            input_data_ids=tuple(
                crypto_ids.HFT_INFORMATION_PANEL_DATA_ID
                if key == base.base_input_data_id
                else key
                for key in base.input_data_ids
            ),
            key_coverage={
                (
                    crypto_ids.HFT_INFORMATION_PANEL_DATA_ID
                    if key == base.base_input_data_id
                    else key
                ): value
                for key, value in base.key_coverage.items()
            },
        )
    columns = tuple(dict.fromkeys(selected_factors))
    if any(
        type(column) is not str or column not in factors_by_column for column in columns
    ):
        raise ValueError("research_factor_not_available")
    if len(set(evaluation["factors"])) != len(evaluation["factors"]):
        raise ValueError("research_evaluation_factor_duplicate")
    # Feature transformations retain their existing semantics for selected factors.
    transforms = (
        canonical_feature_transforms(columns)
        if str(profile.market_id) == "a_share"
        else ()
    )
    features = tuple(
        name for transform in transforms for name in transform.output_columns
    )
    label_values = tuple(
        dict.fromkeys(
            (*evaluation["labels"], *(model.label_column for model in models))
        )
    )
    if any(
        type(label) is not str or label not in base.label_columns
        for label in label_values
    ):
        raise ValueError("research_label_not_available")
    availability = {
        label: (
            a_share.forward_return_available_at_column(20)
            if label == "model_target"
            else label + "_available_at"
        )
        for label in label_values
    }
    labels = tuple(
        dict.fromkeys(
            (
                *label_values,
                *(name for name in availability.values() if name in base.label_columns),
            )
        )
    )
    if "model_target" in label_values:
        labels = tuple(dict.fromkeys((*labels, "forward_return_20d")))
    context_columns = list(base.context_columns)
    dependencies = [base.base_input_data_id]
    needed = {*columns, *labels, *base.context_columns}
    covered = set(library.processed.get(base.base_input_data_id).required_columns)
    for column in columns:
        owners = [
            spec
            for spec in library.processed.list()
            if spec.column_roles.get(column) is ColumnRole.FACTOR
        ]
        if len(owners) != 1:
            raise ValueError("research_factor_owner_not_unique")
        dependencies.append(owners[0].data_id)
        covered.update(owners[0].required_columns)
        context_columns.extend(
            key
            for key, role in owners[0].column_roles.items()
            if role is ColumnRole.CONTEXT
        )
    for data_id in base.input_data_ids:
        spec = library.processed.get(data_id)
        if (needed - covered).intersection(spec.required_columns):
            dependencies.append(data_id)
            covered.update(spec.required_columns)
    dependencies = tuple(dict.fromkeys(dependencies))
    context_columns = tuple(dict.fromkeys(context_columns))
    roles = {
        **{key: base.column_roles[key] for key in base.primary_key},
        **{key: ColumnRole.FACTOR for key in columns},
        **{key: ColumnRole.FEATURE for key in features},
        **{key: ColumnRole.LABEL for key in labels},
        **{key: base.column_roles[key] for key in base.filter_columns},
        **{key: ColumnRole.CONTEXT for key in context_columns},
    }
    dataset = replace(
        base,
        factor_columns=columns,
        context_columns=context_columns,
        feature_columns=features,
        feature_transforms=transforms,
        label_columns=labels,
        column_roles=roles,
        input_data_ids=dependencies,
        key_coverage={
            key: (
                "sparse"
                if uses_sol_rows
                and key
                in {
                    crypto_ids.HFT_DERIVATIVE_DATA_ID,
                    crypto_ids.HFT_SPOT_CONTEXT_DATA_ID,
                    *crypto_ids.HFT_EXTERNAL_DATA_IDS.values(),
                }
                else "strict"
            )
            for key in dependencies
        },
    )
    if str(profile.market_id) == "crypto":
        bound = crypto_hft.protocol(
            scenario_id=scenario, research_plan=config["research_plan"]
        )
    else:
        bound = a_share.protocol(scenario_id=scenario)
    manifest = bound.manifest()
    for role, values in windows.items():
        requested = TimeWindow(**values)
        allowed = TimeWindow(**manifest[role + "_window"])
        if requested.start < allowed.start or requested.end > allowed.end:
            raise ValueError("research_window_outside_access_boundary")
    if str(profile.market_id) == "crypto":
        for model in models:
            horizon = crypto_hft.label_horizon_ms(model.label_column)
            if any(
                (part.end - part.start).total_seconds() * 1000 < horizon
                for fold in bound.research_plan.folds
                for part in (fold.train, fold.validation)
            ):
                raise ValueError("crypto_research_plan_label_window_too_short")
    bound = replace(
        bound,
        configuration_json=json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )
    candidates = bound.portfolio_candidates()
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("research_portfolio_duplicate")
    selected_model_ids = {model.model_id for model in models}
    if (
        config["stage"] == "research"
        and config["comparison_model_id"] not in selected_model_ids
    ) or (
        config["stage"] == "evaluation" and config["comparison_model_id"] is not None
    ):
        raise ValueError("research_comparison_model_invalid")
    for candidate in candidates:
        if str(profile.market_id) == "crypto":
            if (
                isinstance(candidate, crypto_hft.CryptoTakerCandidate)
                and candidate.model_id not in selected_model_ids
            ):
                raise ValueError("research_portfolio_model_not_selected")
            if isinstance(
                candidate, crypto_hft.CryptoTakerCandidate
            ) and not model_catalog.get(candidate.model_id).label_column.startswith(
                "future_mid_return_"
            ):
                raise ValueError("research_taker_model_label_mismatch")
            if (
                isinstance(candidate, crypto_hft.CryptoTakerCandidate)
                and candidate.asset_id
                not in model_configurations[candidate.model_id]["evaluation_asset_ids"]
            ):
                raise ValueError("research_portfolio_model_asset_mismatch")
            if (
                isinstance(candidate, crypto_hft.CryptoMakerCandidate)
                and candidate.calibrated
                and not set(candidate.model_ids) <= selected_model_ids
            ):
                raise ValueError("research_maker_models_not_selected")
            if (
                isinstance(candidate, crypto_hft.CryptoMakerCandidate)
                and candidate.calibrated
                and any(
                    candidate.asset_id
                    not in model_configurations[key]["evaluation_asset_ids"]
                    for key in candidate.model_ids
                )
            ):
                raise ValueError("research_maker_model_asset_mismatch")
            if (
                isinstance(candidate, crypto_hft.CryptoMakerCandidate)
                and candidate.calibrated
            ):
                if any(
                    candidate.asset_id
                    not in model_configurations[key]["training_asset_ids"]
                    for key in candidate.model_ids
                ):
                    raise ValueError("research_maker_validation_asset_mismatch")
                if (
                    model_catalog.get(candidate.price_model_id).label_column
                    != "future_mid_return_500ms"
                    or model_catalog.get(candidate.proxy_model_id).label_column
                    != crypto_hft.HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN
                ):
                    raise ValueError("research_maker_model_label_mismatch")
        elif (
            candidate.prediction_selector != "champion"
            and candidate.prediction_selector not in selected_model_ids
        ):
            raise ValueError("research_portfolio_model_not_selected")
    return bound, dataset, model_catalog
