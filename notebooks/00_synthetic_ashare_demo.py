# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## A-share Shared-Core Demo
#
# 这个 notebook 演示当前 `a-share-alpha-lab` 的 shared-core workflow：
# synthetic A-share panel -> factor / label -> `alpha-research-core` metrics and
# portfolio artifacts -> train-only Ridge baseline checks。
#
# 当前只使用 synthetic/toy data。不使用真实股票数据、vendor API、benchmark 数据、
# 外部下载、凭据、paper trading、live trading 或投资建议。

# %% [markdown]
# ### 1. Prepare Imports
#
# Notebook 可以从项目根目录或 `notebooks/` 目录运行。这里同时加入本项目
# `src/` 和同级 `alpha-research-core/src/`。

# %%
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

project_root = Path.cwd().resolve()
if project_root.name == "notebooks":
    project_root = project_root.parent
workspace_projects = project_root.parent

for import_path in (
    project_root / "src",
    workspace_projects / "alpha-research-core" / "src",
):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

# %% [markdown]
# ### 2. Generate Synthetic A-share Panel

# %%
from ashare_alpha_lab.data import (  # noqa: E402
    EQUITY_PANEL_SPEC,
    make_synthetic_equity_panel,
)

panel = make_synthetic_equity_panel(seed=42, n_assets=12, n_periods=80)
panel.head()

# %%
panel.groupby("asset_id").agg(
    rows=("date", "size"),
    start=("date", "min"),
    end=("date", "max"),
    mean_factor=("factor_value", "mean"),
    mean_return=("return_1d", "mean"),
)

# %% [markdown]
# ### 3. Evaluate Factor Metrics With Core
#
# A-share 项目负责生成 domain panel；shared core 负责 IC / RankIC、分组收益、
# positions、turnover 和 long-short。

# %%
from alpha_research_core import WeightConfig, validate_panel  # noqa: E402
from alpha_research_core.metrics import (  # noqa: E402
    analyze_grouped_returns,
    evaluate_ic_by_timestamp,
)
from alpha_research_core.portfolio import (  # noqa: E402
    simulate_weighted_long_short_returns,
)

evaluation_panel = validate_panel(
    panel.dropna(subset=[EQUITY_PANEL_SPEC.label_column]).reset_index(drop=True),
    EQUITY_PANEL_SPEC,
)
ic_results = evaluate_ic_by_timestamp(evaluation_panel, EQUITY_PANEL_SPEC)
ic_results.head()

# %%
ic_results.groupby("feature").agg(
    mean_ic=("ic", "mean"),
    mean_rankic=("rankic", "mean"),
    timestamps=("timestamp", "nunique"),
)

# %%
grouped_returns = analyze_grouped_returns(
    evaluation_panel,
    EQUITY_PANEL_SPEC,
    n_groups=4,
)
grouped_returns.head()

# %%
grouped_returns.groupby(["feature", "group"]).agg(
    mean_forward_return=("mean_forward_return", "mean"),
    timestamps=("timestamp", "nunique"),
)

# %% [markdown]
# ### 4. Core Weighting, Positions, Turnover, And Long-Short

# %%
weight_config = WeightConfig(
    method="equal_weight_quantile",
    direction="long_high",
    top_quantile=0.25,
    bottom_quantile=0.25,
    cost_bps=5.0,
)
positions, turnover, long_short, diagnostics = simulate_weighted_long_short_returns(
    evaluation_panel,
    EQUITY_PANEL_SPEC,
    weight_config,
)
long_short.head()

# %%
long_short.groupby("feature").agg(
    mean_gross_return=("gross_long_short_return", "mean"),
    mean_net_return=("net_long_short_return", "mean"),
    mean_turnover=("turnover", "mean"),
    mean_cost=("cost", "mean"),
    periods=("timestamp", "nunique"),
)

# %%
positions.head()

# %%
turnover.head()

# %%
diagnostics.head()

# %% [markdown]
# ### 5. Train-Only Ridge Baseline

# %%
from ashare_alpha_lab.modeling import (  # noqa: E402
    FEATURE_COLUMNS,
    build_ml_features,
    split_by_date,
)

featured = build_ml_features(panel)
split_panel = split_by_date(featured, train_ratio=0.6, validation_ratio=0.2)
split_panel.groupby("split").agg(
    rows=("date", "size"),
    start=("date", "min"),
    end=("date", "max"),
)

# %%
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)
train = split_panel[split_panel["split"] == "train"]
pipeline.fit(train[list(FEATURE_COLUMNS)], train["forward_return_1d"])

predictions = split_panel[
    ["date", "asset_id", "split", "forward_return_1d", *FEATURE_COLUMNS]
].copy()
predictions["prediction"] = pipeline.predict(predictions[list(FEATURE_COLUMNS)])
predictions[["date", "asset_id", "split", "prediction", "forward_return_1d"]].head()

# %%
prediction_metrics: list[dict[str, object]] = []
for split, split_frame in predictions.groupby("split", sort=True):
    prediction_metrics.append(
        {
            "split": split,
            "n_obs": int(len(split_frame)),
            "mean_prediction": float(split_frame["prediction"].mean()),
            "mean_actual": float(split_frame["forward_return_1d"].mean()),
            "rankic": float(
                split_frame["prediction"].corr(
                    split_frame["forward_return_1d"],
                    method="spearman",
                )
            ),
        }
    )

pd.DataFrame.from_records(prediction_metrics)

# %% [markdown]
# ### 6. Prediction Ranked Portfolio Through Core

# %%
from ashare_alpha_lab.modeling import PREDICTION_PANEL_SPEC  # noqa: E402

prediction_panel = predictions[
    ["date", "asset_id", "split", "prediction", "forward_return_1d"]
].copy()
_, _, prediction_long_short, _ = simulate_weighted_long_short_returns(
    prediction_panel,
    PREDICTION_PANEL_SPEC,
    WeightConfig(
        method="equal_weight_quantile",
        direction="long_high",
        top_quantile=0.25,
        bottom_quantile=0.25,
        cost_bps=5.0,
    ),
    segment_columns=("split",),
    portfolio_features=("prediction",),
)
prediction_long_short.groupby("split").agg(
    mean_net_return=("net_long_short_return", "mean"),
    mean_turnover=("turnover", "mean"),
    periods=("timestamp", "nunique"),
)

# %% [markdown]
# ### 7. Boundary
#
# This notebook is a synthetic/toy research demo only. It does not use or imply
# real A-share data, benchmark data, vendor data, external downloads, API
# connections, credentials, paper trading, live trading, production automation,
# or investment advice.
