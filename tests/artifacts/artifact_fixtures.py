from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.display_manifest import build_display_manifest


def write_evaluation_set(
    output_dir: Path,
    *,
    evaluation_id: str = "run_a",
    run_id: str | None = None,
    project: str = "quant-research-platform",
    kind: str = "factor",
    domain: str = "evaluation_synthetic",
    data_stage: str = "synthetic_only",
    uses_external_data: bool = False,
    ic_values: list[float] | None = None,
    rankic_values: list[float] | None = None,
    net_returns: list[float] | None = None,
    turnover: list[float] | None = None,
    cost: list[float] | None = None,
    n_rows_evaluated: int = 12,
) -> None:
    """写入供 tests 使用的最小有效 evaluation-set artifact。"""

    evaluation_id = evaluation_id if run_id is None else run_id
    ic_values = [0.4] if ic_values is None else ic_values
    rankic_values = [0.5] if rankic_values is None else rankic_values
    net_returns = [0.029] if net_returns is None else net_returns
    turnover = [2.0] if turnover is None else turnover
    cost = [0.001] if cost is None else cost

    (output_dir / "dimensions").mkdir(parents=True)
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "portfolio").mkdir(parents=True)
    (output_dir / "diagnostics").mkdir(parents=True)
    (output_dir / "input_quality").mkdir(parents=True)
    (output_dir / "summaries").mkdir(parents=True)
    metadata = {
        "run_id": evaluation_id,
        "project": project,
        "kind": kind,
        "domain": domain,
        "data_stage": data_stage,
        "uses_external_data": uses_external_data,
    }
    _write_json(output_dir / "manifest.json", _manifest(metadata=metadata))
    _write_json(
        output_dir / "config.json",
        {
            "metadata": metadata,
            "n_groups": 2,
            "kernel": "dataframe",
            "weight_configs": [
                {
                    "method": "equal_weight_quantile",
                    "portfolio_mode": "long_short",
                }
            ],
            "cost_configs": [{"slippage_bps": 0.0}],
        },
    )
    _write_json(
        output_dir / "lineage.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "input_reference": "fixture",
            "input_fingerprint": "sha256:fixture",
        },
    )
    _write_json(
        output_dir / "summary.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "evaluation_set",
            "evaluation_id": evaluation_id,
            "project": project,
            "kind": kind,
            "domain": domain,
            "data_stage": data_stage,
            "uses_external_data": uses_external_data,
            "n_rows_evaluated": n_rows_evaluated,
            "n_combinations": 1,
            "schema": {
                "timestamp_column": "timestamp",
                "asset_column": "asset",
                "feature_columns": ["factor"],
                "label_columns": ["forward_return"],
                "return_source_column": "forward_return_1d",
            },
            "kernel": "dataframe",
        },
    )
    _combinations_frame().to_csv(
        output_dir / "dimensions" / "combinations.csv",
        index=False,
    )
    _with_dims(_ic_frame(ic_values=ic_values, rankic_values=rankic_values)).to_csv(
        output_dir / "metrics" / "ic.csv",
        index=False,
    )
    ic_summary = _with_dims(
        _ic_summary_frame(ic_values=ic_values, rankic_values=rankic_values)
    )
    ic_summary.to_csv(
        output_dir / "metrics" / "ic_summary.csv",
        index=False,
    )
    _with_dims(_grouped_returns_frame()).to_csv(
        output_dir / "metrics" / "grouped_returns.csv",
        index=False,
    )
    _with_dims(_group_spread_summary_frame()).to_csv(
        output_dir / "metrics" / "group_spread_summary.csv",
        index=False,
    )
    returns = _with_dims(
        _returns_frame(net_returns=net_returns, turnover=turnover, cost=cost)
    )
    returns.to_csv(
        output_dir / "portfolio" / "returns.csv",
        index=False,
    )
    _with_dims(_cumulative_returns_frame(net_returns=net_returns, cost=cost)).to_csv(
        output_dir / "portfolio" / "cumulative_returns.csv",
        index=False,
    )
    _with_dims(_weights_frame()).to_parquet(
        output_dir / "portfolio" / "weights.parquet",
        index=False,
    )
    _with_dims(_performance_summary_frame(
        net_returns=net_returns,
        turnover=turnover,
        cost=cost,
    )).to_csv(
        output_dir / "metrics" / "performance_summary.csv",
        index=False,
    )
    _with_dims(_tradability_diagnostics_frame()).to_csv(
        output_dir / "diagnostics" / "tradability_diagnostics.csv",
        index=False,
    )
    _label_availability_frame().to_csv(
        output_dir / "input_quality" / "label_availability.csv",
        index=False,
    )
    _factor_coverage_frame().to_csv(
        output_dir / "input_quality" / "factor_coverage.csv",
        index=False,
    )
    _tradability_coverage_frame().to_csv(
        output_dir / "input_quality" / "tradability_coverage.csv",
        index=False,
    )
    _evaluation_summary_frame(
        net_returns=net_returns,
        turnover=turnover,
        cost=cost,
    ).to_csv(
        output_dir / "summaries" / "evaluation_summary.csv",
        index=False,
    )
    (output_dir / "report.md").write_text("# Report\n", encoding="utf-8")


def _manifest(*, metadata: dict[str, object]) -> dict[str, object]:
    artifacts = {
        "manifest": "manifest.json",
        "config": "config.json",
        "lineage": "lineage.json",
        "summary": "summary.json",
        "dimensions": {
            "combinations": "dimensions/combinations.csv",
        },
        "metrics": {
            "ic": "metrics/ic.csv",
            "ic_summary": "metrics/ic_summary.csv",
            "grouped_returns": "metrics/grouped_returns.csv",
            "group_spread_summary": "metrics/group_spread_summary.csv",
            "performance_summary": "metrics/performance_summary.csv",
        },
        "portfolio": {
            "returns": "portfolio/returns.csv",
            "cumulative_returns": "portfolio/cumulative_returns.csv",
            "weights": "portfolio/weights.parquet",
        },
        "diagnostics": {
            "tradability_diagnostics": (
                "diagnostics/tradability_diagnostics.csv"
            ),
        },
        "input_quality": {
            "label_availability": "input_quality/label_availability.csv",
            "factor_coverage": "input_quality/factor_coverage.csv",
            "tradability_coverage": "input_quality/tradability_coverage.csv",
        },
        "summaries": {
            "evaluation_summary": "summaries/evaluation_summary.csv",
        },
        "report": "report.md",
    }
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "evaluation_set",
        "evaluation_id": metadata["run_id"],
        "project": metadata["project"],
        "kind": metadata["kind"],
        "domain": metadata["domain"],
        "data_stage": metadata["data_stage"],
        "uses_external_data": metadata["uses_external_data"],
        "schema": {
            "timestamp_column": "timestamp",
            "asset_column": "asset",
            "feature_columns": ["factor"],
            "label_columns": ["forward_return"],
            "return_source_column": "forward_return_1d",
        },
        "kernel": "dataframe",
        "n_combinations": 1,
        "segment_columns": [],
        "artifacts": artifacts,
        "display": build_display_manifest(
            artifacts,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            default_selection={
                "feature": "factor",
                "label": "forward_return",
                "kernel": "dataframe",
                "portfolio_mode": "long_short",
            },
        ),
    }


def _combinations_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "combination_id": ["factor__forward_return__dataframe__long_short"],
            "feature": ["factor"],
            "label": ["forward_return"],
            "kernel": ["dataframe"],
            "portfolio_mode": ["long_short"],
        }
    )


def _with_dims(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["combination_id"] = "factor__forward_return__dataframe__long_short"
    dimension_columns = ["combination_id"]
    remaining_columns = [
        column for column in result.columns if column not in dimension_columns
    ]
    return result[dimension_columns + remaining_columns]


def _ic_frame(*, ic_values: list[float], rankic_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(len(ic_values)),
            "feature": ["factor"] * len(ic_values),
            "label": ["forward_return"] * len(ic_values),
            "ic": ic_values,
            "rankic": rankic_values,
            "n_obs": [5] * len(ic_values),
        }
    )


def _ic_summary_frame(
    *,
    ic_values: list[float],
    rankic_values: list[float],
) -> pd.DataFrame:
    ic = pd.Series(ic_values)
    rankic = pd.Series(rankic_values)
    std_ic = float(ic.std(ddof=0))
    std_rankic = float(rankic.std(ddof=0))
    return pd.DataFrame(
        {
            "feature": ["factor"],
            "label": ["forward_return"],
            "mean_ic": [float(ic.mean())],
            "std_ic": [std_ic],
            "icir": [float(ic.mean() / std_ic) if std_ic > 0 else 0.0],
            "positive_ic_rate": [float((ic > 0).mean())],
            "mean_rankic": [float(rankic.mean())],
            "std_rankic": [std_rankic],
            "rankicir": [float(rankic.mean() / std_rankic) if std_rankic > 0 else 0.0],
            "positive_rankic_rate": [float((rankic > 0).mean())],
        }
    )


def _grouped_returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "feature": ["factor"],
            "label": ["forward_return"],
            "group": [1],
            "mean_forward_return": [0.01],
            "n_obs": [3],
        }
    )


def _group_spread_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["factor"],
            "label": ["forward_return"],
            "top_group_mean_return": [0.02],
            "bottom_group_mean_return": [0.01],
            "top_bottom_spread_mean": [0.01],
            "top_bottom_spread_hit_rate": [1.0],
            "group_monotonicity_score": [1.0],
        }
    )


def _weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "asset": ["asset_001"],
            "target_weight": [1.0],
        }
    )


def _turnover_frame(turnover: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(len(turnover)),
            "feature": ["factor"] * len(turnover),
            "turnover": turnover,
        }
    )


def _returns_frame(
    *,
    net_returns: list[float],
    turnover: list[float],
    cost: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(len(net_returns)),
            "feature": ["factor"] * len(net_returns),
            "label": ["forward_return"] * len(net_returns),
            "portfolio_mode": ["long_short"] * len(net_returns),
            "long_return": [0.02] * len(net_returns),
            "short_return": [-0.01] * len(net_returns),
            "gross_portfolio_return": [0.03] * len(net_returns),
            "turnover": turnover,
            "cost": cost,
            "net_portfolio_return": net_returns,
            "long_count": [1] * len(net_returns),
            "short_count": [1] * len(net_returns),
            "return_source_column": ["forward_return_1d"] * len(net_returns),
        }
    )


def _cumulative_returns_frame(
    *,
    net_returns: list[float],
    cost: list[float] | None = None,
) -> pd.DataFrame:
    cost = [0.0] * len(net_returns) if cost is None else cost
    frame = pd.DataFrame(
        {
            "timestamp": _timestamps(len(net_returns)),
            "feature": ["factor"] * len(net_returns),
            "gross_portfolio_return": [
                net_return + cost_value
                for net_return, cost_value in zip(net_returns, cost, strict=True)
            ],
            "net_return": net_returns,
            "cost": cost,
        }
    )
    frame["gross_return"] = frame["gross_portfolio_return"]
    frame["cumulative_gross_return"] = frame["gross_return"].cumsum()
    frame["cumulative_net_return"] = frame["net_return"].cumsum()
    frame["cumulative_cost"] = frame["cost"].cumsum()
    frame["high_water_mark"] = frame["cumulative_net_return"].cummax()
    frame["drawdown"] = frame["cumulative_net_return"] - frame["high_water_mark"]
    return frame[
        [
            "timestamp",
            "gross_return",
            "net_return",
            "cost",
            "cumulative_gross_return",
            "cumulative_net_return",
            "cumulative_cost",
            "high_water_mark",
            "drawdown",
        ]
    ]


def _cost_breakdown_frame(
    *,
    turnover: list[float],
    cost: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(len(turnover)),
            "feature": ["factor"] * len(turnover),
            "buy_turnover": [value / 2 for value in turnover],
            "sell_turnover": [value / 2 for value in turnover],
            "turnover": turnover,
            "buy_cost": [0.0] * len(turnover),
            "sell_cost": [0.0] * len(turnover),
            "slippage_cost": [0.0] * len(turnover),
            "spread_cost": [0.0] * len(turnover),
            "impact_cost": [0.0] * len(turnover),
            "total_cost": cost,
            "cost_model": ["scalar_cost_bps"] * len(turnover),
        }
    )


def _performance_summary_frame(
    *,
    net_returns: list[float],
    turnover: list[float],
    cost: list[float],
) -> pd.DataFrame:
    returns = pd.Series(net_returns)
    costs = pd.Series(cost)
    turnover_values = pd.Series(turnover)
    gross_returns = returns + costs
    volatility = float(returns.std(ddof=0))
    downside_volatility = float(returns.clip(upper=0).std(ddof=0))
    sharpe_like = float(returns.mean() / volatility) if volatility > 0 else 0.0
    sortino_like = (
        float(returns.mean() / downside_volatility)
        if downside_volatility > 0
        else 0.0
    )
    cumulative = _cumulative_returns_frame(net_returns=net_returns, cost=cost)
    max_drawdown = float(cumulative["drawdown"].min())
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    var_95 = float(returns.quantile(0.05))
    q95 = float(returns.quantile(0.95))
    return pd.DataFrame(
        {
            "feature": ["factor"],
            "periods": [len(net_returns)],
            "final_cumulative_gross_return": [float(gross_returns.sum())],
            "final_cumulative_net_return": [float(returns.sum())],
            "mean_net_portfolio_return": [float(returns.mean())],
            "volatility_net_portfolio_return": [volatility],
            "downside_volatility": [downside_volatility],
            "sharpe_like": [sharpe_like],
            "sortino_like": [sortino_like],
            "calmar_like": [
                float(returns.sum() / abs(max_drawdown)) if max_drawdown < 0 else 0.0
            ],
            "best_period_return": [float(returns.max())],
            "worst_period_return": [float(returns.min())],
            "skew": [float(returns.skew()) if len(returns) > 2 else 0.0],
            "kurtosis": [float(returns.kurt()) if len(returns) > 3 else 0.0],
            "var_95": [var_95],
            "cvar_95": [float(returns[returns <= var_95].mean())],
            "tail_ratio": [float(abs(q95) / abs(var_95)) if var_95 != 0 else 0.0],
            "positive_return_rate": [float((returns > 0).mean())],
            "negative_return_rate": [float((returns < 0).mean())],
            "win_rate": [float((returns > 0).mean())],
            "loss_rate": [float((returns < 0).mean())],
            "profit_factor": [
                float(wins.sum() / abs(losses.sum()))
                if float(losses.sum()) != 0
                else 0.0
            ],
            "payoff_ratio": [
                float(wins.mean() / abs(losses.mean())) if len(losses) else 0.0
            ],
            "average_win": [float(wins.mean()) if len(wins) else 0.0],
            "average_loss": [float(losses.mean()) if len(losses) else 0.0],
            "max_consecutive_wins": [_max_consecutive(returns > 0)],
            "max_consecutive_losses": [_max_consecutive(returns < 0)],
            "max_drawdown": [max_drawdown],
            "drawdown_duration": [1 if max_drawdown < 0 else 0],
            "max_drawdown_start": [""],
            "max_drawdown_valley": [""],
            "max_drawdown_recovery": [""],
            "mean_turnover": [float(turnover_values.mean())],
            "max_turnover": [float(turnover_values.max())],
            "total_cost": [float(costs.sum())],
            "mean_cost": [float(costs.mean())],
            "cost_to_gross_return": [
                float(costs.sum() / abs(gross_returns.sum()))
                if float(gross_returns.sum()) != 0
                else 0.0
            ],
            "cost_to_net_return": [
                float(costs.sum() / abs(returns.sum()))
                if float(returns.sum()) != 0
                else 0.0
            ],
        }
    )


def _exposure_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["factor"],
            "mean_gross_exposure": [1.0],
            "mean_net_exposure": [0.0],
            "max_gross_exposure": [1.0],
            "max_abs_net_exposure": [0.0],
            "mean_long_count": [1.0],
            "mean_short_count": [1.0],
            "mean_position_count": [2.0],
        }
    )


def _diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "feature": ["factor"],
            "method": ["equal_weight_quantile"],
            "direction": ["long_high"],
            "universe_count": [100],
            "eligible_count": [90],
            "can_enter_count": [85],
            "can_exit_count": [95],
            "entry_blocked_count": [5],
            "exit_blocked_count": [5],
            "long_count": [1],
            "short_count": [1],
            "fallback_reason": [""],
        }
    )


def _neutralization_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "feature": ["factor"],
            "method": ["none"],
            "n_obs": [5],
            "group_columns": [""],
            "exposure_columns": [""],
        }
    )


def _tradability_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "feature": ["factor"],
            "method": ["equal_weight_quantile"],
            "direction": ["long_high"],
            "universe_count": [100],
            "eligible_count": [90],
            "can_enter_count": [85],
            "can_exit_count": [95],
            "entry_blocked_count": [5],
            "exit_blocked_count": [5],
            "long_count": [1],
            "short_count": [1],
            "eligible_rate": [0.9],
            "can_enter_rate": [85 / 90],
            "can_exit_rate": [0.95],
            "entry_blocked_rate": [5 / 90],
            "exit_blocked_rate": [0.05],
            "fallback_reason": [""],
        }
    )


def _label_availability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["forward_return"],
            "available_rows": [12],
            "missing_rows": [0],
            "coverage_rate": [1.0],
        }
    )


def _factor_coverage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["factor"],
            "available_rows": [12],
            "missing_rows": [0],
            "coverage_rate": [1.0],
        }
    )


def _tradability_coverage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "field": ["can_enter"],
            "available_rows": [12],
            "missing_rows": [0],
            "coverage_rate": [1.0],
        }
    )


def _evaluation_summary_frame(
    *,
    net_returns: list[float],
    turnover: list[float],
    cost: list[float],
) -> pd.DataFrame:
    performance = _performance_summary_frame(
        net_returns=net_returns,
        turnover=turnover,
        cost=cost,
    )
    return pd.DataFrame(
        {
            "combination_id": ["factor__forward_return__dataframe__long_short"],
            "feature": ["factor"],
            "label": ["forward_return"],
            "kernel": ["dataframe"],
            "portfolio_mode": ["long_short"],
            "mean_ic": [0.4],
            "final_cumulative_net_return": performance[
                "final_cumulative_net_return"
            ].tolist(),
            "mean_turnover": performance["mean_turnover"].tolist(),
            "total_cost": performance["total_cost"].tolist(),
        }
    )


def _timestamps(count: int) -> list[str]:
    return [f"2026-01-{day:02d}" for day in range(1, count + 1)]


def _max_consecutive(mask: pd.Series) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
