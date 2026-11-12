"""Publish daily portfolio backtests and frozen alpha calibration from explicit
inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.risk import FactorRiskConfig, FactorRiskModel
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.calibration import (
    alpha_calibration_intervals,
    alpha_interval_semantics,
    calibrate_expected_alpha,
)
from quant_research_platform.research.portfolios.daily import (
    _CALENDAR_COLUMNS,
    _INDUSTRY_NEUTRALITY_DOMAIN,
    _NEUTRALITY_REFERENCE_DOMAIN,
    _read_materialized,
    backtest_a_share,
)
from quant_research_platform.research.portfolios.optimizer import (
    OptimizerConfig,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


def backtest(config: Mapping) -> Path:
    """Publish an A-share account from explicit JSON market and prediction inputs."""
    config = qrp_json_mapping_snapshot(config)
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )
    from quant_research_platform.artifacts.outputs.prediction import (
        _iter_prediction_member,
        _PredictionLookup,
    )
    from quant_research_platform.artifacts.portfolio_ledger import (
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioCapital
    from quant_research_platform.markets.implementations.a_share import (
        portfolio_sessions,
    )
    from quant_research_platform.research.portfolios.calibration import (
        frozen_alpha_calibration,
    )
    from quant_research_platform.storage.manifests import read_qrp_json_file

    required = {
        "candidate",
        "inputs",
        "predictions",
        "output",
        "output_window",
        "input_window",
        "formal_dataset_window",
        "prediction_window",
        "assumptions",
    }
    optional = {
        "risk_config",
        "risk_model",
        "optimizer_config",
        "calibration",
        "sample_role",
        "selection_mode",
        "selection",
    }
    if (
        not isinstance(config, Mapping)
        or not required <= set(config)
        or set(config) - required - optional
    ):
        raise ValueError("portfolio_config_fields_invalid")
    if "sample_role" in config and (
        not isinstance(config["sample_role"], str) or not config["sample_role"].strip()
    ):
        raise ValueError("portfolio_sample_role_invalid")
    selection_mode = config.get("selection_mode", "causal")
    if selection_mode not in ("causal", "retrospective"):
        raise ValueError("portfolio_selection_mode_invalid")
    selection = None
    if "selection" in config:
        from quant_research_platform.research.models.trial_selection import (
            read_selection,
        )

        selection = read_selection(Path(config["selection"]))
    if selection_mode == "retrospective" and selection is None:
        raise ValueError("portfolio_retrospective_selection_required")
    declared_risk = (
        FactorRiskModel.from_mapping(config["risk_model"])
        if "risk_model" in config
        else None
    )
    candidate_values = dict(config["candidate"])
    if declared_risk is not None:
        candidate_values.setdefault("risk_style_columns", declared_risk.style_columns)
    if "risk_style_columns" in candidate_values:
        candidate_values["risk_style_columns"] = tuple(
            candidate_values["risk_style_columns"]
        )
    candidate = PortfolioCandidateSpec(**candidate_values)
    inputs = dict(config["inputs"])
    base_inputs = {
        "dataset",
        "panel",
        "execution",
        "calendar",
        "fees",
        "market_bars",
        "stock_master",
        "corporate_actions",
    }
    if not base_inputs <= set(inputs) or set(inputs) - base_inputs - {
        "benchmark",
        "risk_context",
        "equity_events",
    }:
        raise ValueError("portfolio_input_fields_invalid")
    if "risk_context" in inputs and (
        type(inputs["risk_context"]) is not list or not inputs["risk_context"]
    ):
        raise ValueError("portfolio_risk_context_invalid")
    for key, value in inputs.items():
        for source in value if key == "risk_context" else (value,):
            _dataset_input_options(source)
    enhanced = candidate.construction == "benchmark_enhanced"
    if (candidate.benchmark_id is not None) != ("benchmark" in inputs) or enhanced != (
        "risk_context" in inputs
    ):
        raise ValueError("portfolio_enhanced_inputs_invalid")
    if not enhanced and (
        "risk_config" in config
        or "risk_model" in config
        or "optimizer_config" in config
        or "calibration" in config
    ):
        raise ValueError("portfolio_unused_configuration")
    risk_model = None
    if enhanced:
        risk_model = declared_risk or FactorRiskModel(
            candidate.risk_style_columns,
            FactorRiskConfig(**config.get("risk_config", {})),
        )
        if risk_model.style_columns != candidate.risk_style_columns or (
            declared_risk is not None
            and "risk_config" in config
            and risk_model.config != FactorRiskConfig(**config["risk_config"])
        ):
            raise ValueError("portfolio_risk_model_conflict")
    inherited_risk = False
    assumptions_values = dict(config["assumptions"])
    assumptions_values["adv_sensitivities"] = tuple(
        assumptions_values["adv_sensitivities"]
    )
    assumptions = ExecutionAssumptions(**assumptions_values)
    windows = {
        name: TimeWindow(**config[name])
        for name in (
            "output_window",
            "input_window",
            "formal_dataset_window",
            "prediction_window",
        )
    }
    domains = []
    prediction_facts = []
    index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
    if not isinstance(config["predictions"], list) or not config["predictions"]:
        raise ValueError("portfolio_prediction_inputs_required")
    for item in config["predictions"]:
        if not isinstance(item, dict) or set(item) != {"path", "window"}:
            raise ValueError("portfolio_prediction_input_invalid")
        path = Path(item["path"])
        metadata = validate_output_bundle(
            path, expected_definition_id="prediction_outputs"
        )[0]["metadata"]
        facts = metadata["model_facts"]
        target_definition = facts.get("target_definition") or {}
        if enhanced and target_definition.get("target_transform") == "risk_residual":
            label_risk = FactorRiskModel.from_mapping(
                target_definition.get("risk_model", {})
            )
            if declared_risk is not None or "risk_config" in config or inherited_risk:
                if risk_model != label_risk:
                    raise ValueError("portfolio_label_risk_model_conflict")
            elif candidate.risk_style_columns != label_risk.style_columns:
                raise ValueError("portfolio_label_risk_model_conflict")
            risk_model = label_risk
            inherited_risk = True
        window = TimeWindow(**item["window"])
        lower, upper, closed = time_window_timestamp_interval(window)
        if any(
            (lower < old_upper or lower == old_upper and old_closed)
            and (old_lower < upper or old_lower == upper and closed)
            for old_lower, old_upper, old_closed in domains
        ):
            raise ValueError("portfolio_prediction_domains_overlap")
        saved_window = PredictionWindow(**metadata["prediction_window"])
        outside = lower < pd.Timestamp(saved_window.start)
        if saved_window.end is not None:
            _, saved_upper, saved_closed = time_window_timestamp_interval(saved_window)
            outside |= upper > saved_upper or (
                upper == saved_upper and closed and not saved_closed
            )
        if outside:
            raise ValueError("portfolio_prediction_window_invalid")
        if facts["market_id"] != "a_share":
            raise ValueError("portfolio_model_market_mismatch")
        if candidate.prediction_selector not in {"champion", metadata["family_id"]}:
            raise ValueError("portfolio_prediction_selector_mismatch")
        if selection is not None:
            from quant_research_platform.contracts.models import compare_model_facts

            selected_id = (
                selection["selected_model_id"]
                if candidate.prediction_selector == "champion"
                else next(
                    (
                        row["selected_config_id"]
                        for row in selection["families"]
                        if row["family_id"] == candidate.prediction_selector
                    ),
                    None,
                )
            )
            expected = next(
                (
                    row
                    for row in selection["candidates"]
                    if row["model_id"] == selected_id
                ),
                None,
            )
            if expected is None:
                raise ValueError("portfolio_prediction_selection_mismatch")
            compare_model_facts(facts, expected)
        cutoffs = [facts["fit_cutoff"]]
        if selection_mode == "causal":
            cutoffs.append(facts.get("selection_cutoff"))
            if selection is not None:
                cutoffs.append(selection["cutoff"])
        for cutoff in cutoffs:
            if cutoff is not None and pd.Timestamp(cutoff) >= pd.Timestamp(
                window.start
            ):
                raise ValueError("portfolio_prediction_not_causal")
        prediction_facts.append({"window": item["window"], "model_facts": facts})
        with closing_iterator(
            _iter_prediction_member(
                path,
                metadata,
                columns=("timestamp", "asset_id", "prediction_rank_score"),
                time_window=window,
            )
        ) as batches:
            for frame in batches:
                if index.add_float_value_table(
                    pa.Table.from_pandas(frame, preserve_index=False),
                    namespace="prediction",
                    value_column="prediction_rank_score",
                    retain_values=True,
                ):
                    raise ValueError("portfolio_prediction_keys_invalid")
                del frame
        domains.append((lower, upper, closed))
    lookup = _PredictionLookup(
        index=index,
        primary_key=("timestamp", "asset_id"),
        prediction_column="prediction_rank_score",
    )
    frozen = None
    if "calibration" in config:
        payload = read_qrp_json_file(Path(config["calibration"]))[1]
        if pd.Timestamp(payload["cutoff"]) >= pd.Timestamp(
            windows["output_window"].start
        ):
            raise ValueError("portfolio_calibration_not_frozen")
        frozen, _ = frozen_alpha_calibration(
            payload,
            cutoff=payload["cutoff"],
            selection_group=candidate.selection_group,
            rebalance_schedule=candidate.rebalance_schedule,
        )
    final = []
    risk_config = risk_model.config if enhanced else None
    optimizer_config = (
        OptimizerConfig(**config.get("optimizer_config", {})) if enhanced else None
    )

    def produce(emit):
        with closing_iterator(
            backtest_a_share(
                candidate=candidate,
                inputs=inputs,
                **windows,
                prediction_lookup=lookup,
                assumptions=assumptions,
                risk_config=risk_config,
                optimizer_config=optimizer_config,
                frozen_calibration=frozen,
                on_final_calibration=final.append,
            )
        ) as batches:
            for batch in batches:
                emit(batch)

    metadata = {
        **dict(config),
        "selection_mode": selection_mode,
        "prediction_facts": prediction_facts,
        **({"selection_cutoff": selection["cutoff"]} if selection is not None else {}),
        "candidate": asdict(candidate),
        **(
            {
                "risk_config": asdict(risk_config),
                "risk_model": risk_model.to_dict(),
                "optimizer_config": asdict(optimizer_config),
            }
            if enhanced
            else {}
        ),
        "window": dict(config["output_window"]),
        "corporate_action_policy": portfolio_sessions.CORPORATE_ACTION_POLICY,
        "initial_holding_convention": "cash",
        "initial_turnover_policy": "first_actual_buy_exempt_fees_and_capacity_apply",
        **(
            {"neutrality_reference_domain": _NEUTRALITY_REFERENCE_DOMAIN}
            if candidate.neutrality != "none" and candidate.benchmark_id is None
            else {}
        ),
        **(
            {"industry_neutrality_domain": _INDUSTRY_NEUTRALITY_DOMAIN}
            if candidate.neutrality in {"industry", "industry_size"}
            else {}
        ),
    }
    metadata.pop("output")
    return write_portfolio_ledger_artifact(
        Path(config["output"]),
        capital=PortfolioCapital(
            candidate.candidate_id, assumptions.initial_nav, "CNY", "cash_equity"
        ),
        metadata=metadata,
        produce=produce,
        final_calibration=(lambda: final[0]) if enhanced and frozen is None else None,
    )


def calibrate(config: Mapping) -> Path:
    """Freeze mature A-share alpha observations from an explicit JSON configuration."""
    config = qrp_json_mapping_snapshot(config)
    from quant_research_platform.storage.atomic import atomic_write_json

    required = {
        "observations",
        "scheduled_intervals",
        "rebalance_schedule",
        "calendar",
        "cutoff",
        "selection_group",
        "output",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError("alpha_calibration_config_invalid")
    if config["selection_group"] not in {"global", "quarterly"}:
        raise ValueError("portfolio_selection_group_invalid")
    from quant_research_platform.storage.datasets.reading import (
        _selection_filters,
        _validated_years,
    )
    from quant_research_platform.storage.parquet import (
        inspect_dataset_part,
        iter_dataset_part_batches,
    )

    frames = {}
    for name, columns in (
        ("observations", ("timestamp", "prediction_rank_score", "holding_return")),
        ("scheduled_intervals", ("timestamp", "entry_at", "available_at")),
    ):
        options = _dataset_input_options(config[name])
        business_columns = columns[1:] if name == "observations" else ()
        if options.get("columns") is not None and not set(business_columns) <= set(
            options["columns"]
        ):
            raise ValueError("portfolio_input_business_columns_excluded")
        path = options["dataset_root"]
        if path.is_file() and options.get("spec") is None:
            if options.get("files") is not None:
                raise ValueError("dataset_files_require_directory")
            schema = inspect_dataset_part(path, require_logical_dtypes=False).schema
            filters = _selection_filters(
                schema,
                timestamp="timestamp",
                time_window=options.get("time_window"),
                years=_validated_years(options.get("years")),
                row_filters=options.get("row_filters", ()),
            )
            projection = tuple(
                dict.fromkeys((*columns, *(column for column, _, _ in filters)))
            )
            with closing_iterator(
                iter_dataset_part_batches(
                    path, columns=projection, parquet_filters=filters
                )
            ) as batches:
                selected = [batch.select(columns).to_pandas() for batch in batches]
            frames[name] = (
                pd.concat(selected, ignore_index=True)
                if selected
                else pd.DataFrame(columns=columns)
            )
        else:
            frames[name] = _read_materialized(config[name], columns)
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )

    semantics = alpha_interval_semantics(config["rebalance_schedule"])
    calendar = _read_materialized(config["calendar"], _CALENDAR_COLUMNS)
    trading_sessions = unified_trading_sessions(calendar)
    expected = alpha_calibration_intervals(
        trading_sessions, config["rebalance_schedule"]
    )
    supplied = frames["scheduled_intervals"].apply(pd.to_datetime, errors="raise")
    if supplied.timestamp.duplicated().any():
        raise ValueError("alpha_calibration_schedule_invalid")
    matched = supplied.merge(expected, on=list(expected), how="left", indicator=True)
    if not matched["_merge"].eq("both").all():
        raise ValueError("alpha_calibration_interval_semantics_mismatch")
    # The selected market calendar owns planned dates. A missing declared interval
    # makes its observations unavailable; it cannot shrink the coverage denominator.
    observations = frames["observations"]
    declared = pd.to_datetime(observations.timestamp, errors="raise").dt.normalize()
    observations = observations.loc[declared.isin(supplied.timestamp)]
    result = calibrate_expected_alpha(
        observations,
        scheduled_intervals=expected,
        trading_sessions=trading_sessions,
        as_of=pd.Timestamp(config["cutoff"]),
    )
    return atomic_write_json(
        Path(config["output"]),
        {
            "schema_id": "qrp-final-alpha-calibration",
            "selection_group": config["selection_group"],
            "interval_semantics": semantics,
            "cutoff": pd.Timestamp(config["cutoff"]).isoformat(),
            "inputs": {
                key: config[key]
                for key in ("observations", "scheduled_intervals", "calendar")
            },
            "calibration": asdict(result),
        },
    )
