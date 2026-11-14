"""A small daily research task using native factors, models and share accounting."""

from dataclasses import asdict

import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.portfolio_inputs import closed_market_inputs


def daily_research_config(root):
    inputs, _, _, panel = closed_market_inputs(root / "market", fault="normal_open")
    frame = panel[
        ["market_id", "timestamp", "asset_id", "float_market_cap", "total_market_cap"]
    ].copy()
    assets = sorted(frame.asset_id.unique())
    frame["x"] = frame.asset_id.map(
        {asset: float(len(assets) - index) for index, asset in enumerate(assets)}
    )
    frame["label"] = frame.x * 0.01
    frame.loc[frame.timestamp.ge("2023-01-04"), "label"] *= -1
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(hours=1)
    frame["eligible"] = True
    source = root / "source"
    write_dataset(
        source,
        frame,
        spec=DataSpec(
            data_id="agent_daily_fixture",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("market_id", "timestamp", "asset_id"),
            partition_columns=(),
            column_roles={
                "market_id": "key",
                "timestamp": "timestamp",
                "asset_id": "key",
                "float_market_cap": "context",
                "total_market_cap": "context",
                "x": "factor",
                "label": "label",
                "label_available_at": "context",
                "eligible": "mask",
            },
            column_units={"label": "decimal_return"},
            label_availability_columns={"label": "label_available_at"},
        ),
    )
    train = {"start": "2022-12-01", "end": "2022-12-05"}
    validation = {"start": "2022-12-07", "end": "2022-12-08"}
    historical = {"start": "2022-12-29", "end": "2023-01-03"}
    prediction = {"start": "2023-01-04", "end": "2023-01-06"}
    costs = asdict(
        ExecutionAssumptions(100000.0, 0.0, 0.0, 1.0, (1.0,), minimum_commission=0.0)
    )
    costs["adv_sensitivities"] = list(costs["adv_sensitivities"])

    def tool(
        operation, description, config, *, parameters=None, bindings=None, metrics=None
    ):
        return {
            "operation": operation,
            "description": description,
            "config": config,
            "parameters": parameters or {},
            "bindings": bindings or {},
            "metrics": metrics or {},
        }

    def metric(column, member="", filters=None):
        result = {"member": member, "column": column, "reduction": "mean"}
        if filters is not None:
            result["filters"] = filters
        return result

    tools = {}
    tools["search"] = tool(
        "search-factors",
        (
            "Propose causal expressions using x; compare the signs and "
            "select a discovery candidate. Inputs are dimensionless x and "
            "decimal-return label."
        ),
        {
            "dataset": str(source),
            "columns": ["x"],
            "baseline_columns": [],
            "label": "label",
            "train_window": train,
            "validation_window": validation,
            "train_cutoff": "2022-12-06",
            "selection_cutoff": "2022-12-09",
            "method": "random",
            "budget": 2,
            "max_depth": 1,
            "windows": [2],
            "seed": 1,
            "output": "assigned",
        },
        parameters={
            "expressions": (
                "One or two causal formula strings, e.g. rank(x), -x. "
                "Operators: +,-,*,/,abs,sign,logabs,rank,delay,mean,std."
            )
        },
        metrics={"discovery_rank_ic": metric("validation_score")},
    )
    tools["search"]["facts"] = {
        "selected_expression": {"member": "", "path": "selected_expression"},
        "candidates": {"member": "", "path": "candidates"},
    }
    tools["apply"] = tool(
        "apply-formula",
        (
            "Apply the frozen discovery formula to later dates. "
            "Bind search to the search evidence."
        ),
        {
            "search": "bound",
            "dataset": str(source),
            "input_window": {"start": "2022-12-01", "end": "2023-01-06"},
            "output_window": {"start": "2022-12-12", "end": "2023-01-06"},
            "output": "assigned",
        },
        bindings={"search": ""},
    )
    tools["assemble"] = tool(
        "assemble",
        (
            "Join the candidate factor with the baseline x, labels, "
            "maturity and portfolio context. Bind inputs.formula.path to "
            "apply evidence."
        ),
        {
            "market_id": "a_share",
            "dataset_id": "agent_candidate",
            "base_input_data_id": "source",
            "inputs": {
                "source": str(source),
                "formula": {"path": "bound", "columns": ["discovered_factor"]},
            },
            "factor_columns": ["x", "discovered_factor"],
            "feature_columns": [],
            "feature_recipe": None,
            "label_columns": ["label"],
            "filter_columns": ["eligible"],
            "context_columns": ["float_market_cap", "total_market_cap"],
            "time_window": {"start": "2022-12-12", "end": "2023-01-06"},
            "output_root": "assigned",
        },
        bindings={"inputs.formula.path": ""},
    )
    tools["evaluate"] = tool(
        "evaluate",
        (
            "Compare candidate and baseline on historical observation days; "
            "inspect their separate IC and coverage. Bind inputs to "
            "assembled evidence."
        ),
        {
            "inputs": "bound",
            "output": "assigned",
            "window": historical,
            "cutoff": "2023-01-03T18:00:00",
            "sample_role": "evaluation",
            "evaluation": {
                "evaluation_id": "agent_factors",
                "market_id": "a_share",
                "research_dataset_id": "agent_candidate",
                "signals": [
                    {
                        "signal_id": name,
                        "source_kind": "factor",
                        "source_column": column,
                    }
                    for name, column in (
                        ("candidate", "discovered_factor"),
                        ("baseline", "x"),
                    )
                ],
                "label_columns": ["label"],
                "n_groups": 2,
                "filter_mode": "eligible",
            },
        },
        bindings={"inputs": ""},
        parameters={
            "window": "Explicit evaluation window.",
            "cutoff": "Actual label maturity cutoff.",
        },
        metrics={
            "candidate_ic": metric(
                "ic", "metrics.parquet", [["signal", "==", "candidate"]]
            ),
            "baseline_ic": metric(
                "ic", "metrics.parquet", [["signal", "==", "baseline"]]
            ),
            "paired_count": metric(
                "n_obs", "metrics.parquet", [["signal", "==", "candidate"]]
            ),
            "eligible_count": metric("eligible_rows", "daily_coverage.parquet"),
            "mature_count": metric("mature_rows", "daily_coverage.parquet"),
            "candidate_rankic_std": metric(
                "std_rankic", "ic_summary.parquet", [["signal", "==", "candidate"]]
            ),
            "candidate_positive_rankic_rate": metric(
                "positive_rankic_rate",
                "ic_summary.parquet",
                [["signal", "==", "candidate"]],
            ),
        },
    )
    tools["statistics"] = tool(
        "statistics-factors",
        (
            "Calculate native factor inference; undefined small-sample "
            "inference must remain undefined. Bind dataset to assembled "
            "evidence."
        ),
        {
            "dataset": "bound",
            "output": "assigned",
            "window": historical,
            "label_cutoff": "2023-01-03T18:00:00",
            "analyses": ["rank_ic"],
            "evaluation": {
                "factors": ["x", "discovered_factor"],
                "primary_label": "label",
            },
            "statistics": {
                "hac_lag": 0,
                "factor_fdr_families": {"proposals": ["x", "discovered_factor"]},
            },
        },
        bindings={"dataset": ""},
        metrics={
            name: metric(
                column,
                "factor_diagnostics.parquet",
                [["factor_id", "==", "discovered_factor"], ["fdr_method", "==", "BY"]],
            )
            for name, column in (
                ("candidate_hac_standard_error", "standard_error"),
                ("candidate_hac_p_value", "p_value"),
                ("candidate_fdr_p_value", "fdr_adjusted_p_value"),
                ("inference_date_count", "observation_count"),
            )
        },
    )
    tools["fit"] = tool(
        "fit",
        (
            "Fit a candidate on Dec 12-23 only, after formula discovery. "
            "Bind inputs to assembled evidence; later prediction dates remain "
            "excluded."
        ),
        {
            "market_id": "a_share",
            "inputs": "bound",
            "output": "assigned",
            "fit_windows": [{"start": "2022-12-12", "end": "2022-12-23"}],
            "fit_cutoff": "2022-12-26",
            "label_unit": "decimal_return",
            "model": {
                "model_id": "agent_linear",
                "model_algorithm": "linear_regression",
                "input_factor_columns": ["discovered_factor"],
                "input_feature_columns": [],
                "sample_filter_mode": "eligible",
                "hyperparameters": {},
                "random_seed": 1,
                "training_scaling": True,
                "target_column": "label",
            },
        },
        bindings={"inputs": ""},
        parameters={
            "model.input_factor_columns": (
                "A list chosen from x and discovered_factor; use x for the "
                "baseline and discovered_factor for the candidate."
            )
        },
    )
    tools["predict"] = tool(
        "predict",
        (
            "Predict on the later dates. Bind model to "
            "fitted-model evidence and inputs to assembled evidence."
        ),
        {
            "model": "bound",
            "inputs": "bound",
            "output": "assigned",
            "prediction_window": prediction,
            "sample_role": "evaluation",
        },
        bindings={"model": "", "inputs": ""},
    )
    market = {key: str(value) for key, value in inputs.items() if key != "risk_context"}
    market["dataset"] = str(source)
    tools["backtest"] = tool(
        "backtest",
        (
            "Run the native A-share share-account backtest including fees. "
            "Bind predictions.0.path to prediction evidence. Keep initial "
            "cash, costs and selection rule fixed."
        ),
        {
            "candidate": {
                "candidate_id": "agent_candidate",
                "selection_group": "agent_candidate",
                "construction": "quantile_equal_weight",
                "prediction_selector": "champion",
                "top_quantile": 0.5,
                "rebalance_schedule": "daily",
                "benchmark_id": None,
            },
            "inputs": market,
            "predictions": [{"path": "bound", "window": prediction}],
            "output": "assigned",
            "output_window": prediction,
            "input_window": {"start": "2022-12-01", "end": "2023-01-06"},
            "formal_dataset_window": {"start": "2022-12-01", "end": "2023-01-06"},
            "prediction_window": prediction,
            "assumptions": costs,
            "sample_role": "evaluation",
        },
        bindings={"predictions.0.path": ""},
        metrics={name: metric(name) for name in ("net_pnl", "fees", "nav")},
    )
    tools["curve"] = tool(
        "plot",
        "Render the native account curve.",
        {
            "kind": "curve",
            "inputs": "bound",
            "output": "curve.png",
            "timezone": "Asia/Shanghai",
        },
        bindings={"inputs": ""},
    )
    return {
        "question": (
            "Compare formula and baseline ranks, native inference and "
            "later fee-adjusted account results."
        ),
        "tools": tools,
        "budget": {"max_requests": 20, "max_experiments": 20, "wall_seconds": 180},
        "output": str(root / "agent"),
        "llm": {},
    }
