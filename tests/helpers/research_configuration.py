"""Small explicit research inputs, independent of documentation and experiments."""

CORE6_FACTORS = (
    "momentum_252_20",
    "reversal_5d",
    "volatility_20d",
    "amihud_20d",
    "turnover_20d",
    "log_float_market_cap",
)
PV25_FACTORS = (
    "reversal_1d",
    "reversal_5d",
    "reversal_20d",
    "momentum_60_20",
    "momentum_120_20",
    "momentum_252_20",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_20d",
    "downside_volatility_60d",
    "max_return_20d",
    "skewness_60d",
    "drawdown_60d",
    "amplitude_20d",
    "log_avg_amount_20d",
    "turnover_20d",
    "turnover_60d",
    "turnover_volatility_20d",
    "amihud_20d",
    "amihud_60d",
    "amount_shock_20d",
    "return_amount_corr_20d",
    "trend_efficiency_60d",
    "signed_amount_flow_20d",
    "log_float_market_cap",
)

_MODEL_PARAMETERS = {
    "linear_regression": {},
    "lasso_regression": {"alpha": 0.03, "max_iter": 20000, "tol": 1e-5},
    "xgboost_regressor": {"device": "cuda", "n_estimators": 3, "max_depth": 2},
    "lstm_regressor": {
        "device": "cuda",
        "sequence_length": 3,
        "hidden_size": 8,
        "epochs": 1,
    },
    "transformer_regressor": {
        "device": "cuda",
        "sequence_length": 3,
        "hidden_size": 8,
        "epochs": 1,
        "n_heads": 2,
    },
}
A_SHARE_MODEL_ALGORITHMS = tuple(_MODEL_PARAMETERS)


def research_configuration(*, factors=CORE6_FACTORS, enhanced=True):
    portfolios = [
        {
            "candidate_id": identifier,
            "prediction_selector": selector,
            "top_quantile": None if benchmark else 0.2,
            "rebalance_schedule": "weekly-last-trading-day",
            "construction": construction,
            "benchmark_id": benchmark,
        }
        for identifier, selector, construction, benchmark in (
            (
                "ols_quantile_equal_weight",
                "pv_all_linear_regression",
                "quantile_equal_weight",
                None,
            ),
            (
                "champion_quantile_equal_weight",
                "champion",
                "quantile_equal_weight",
                None,
            ),
            ("champion_quantile_float_cap", "champion", "quantile_float_cap", None),
            *(
                (
                    (
                        "champion_csi500_enhanced",
                        "champion",
                        "benchmark_enhanced",
                        "csi500",
                    ),
                    (
                        "champion_csi1000_enhanced",
                        "champion",
                        "benchmark_enhanced",
                        "csi1000",
                    ),
                )
                if enhanced
                else ()
            ),
        )
    ]
    return {
        "evaluation": {
            "factors": list(factors),
            "labels": ["forward_return_20d"],
            "primary_label": "forward_return_20d",
        },
        "models": [
            {
                "model_id": "pv_all_" + algorithm,
                "model_algorithm": algorithm,
                "input_feature_columns": [
                    prefix + factor for factor in factors for prefix in ("feature_",)
                ],
                "sample_filter_mode": "eligible",
                "hyperparameters": dict(parameters),
                "random_seed": 42,
                "target_column": "model_target",
            }
            for algorithm, parameters in _MODEL_PARAMETERS.items()
        ],
        "portfolios": portfolios,
        "comparison_model_id": "pv_all_linear_regression",
        "primary_portfolio_id": "champion_quantile_equal_weight",
        "market_id": "a_share",
        "dataset_id": "pv_all",
    }
