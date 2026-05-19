# a-share-alpha-lab 研究报告

## 摘要

本项目构建一条本地私有 A 股因子研究流程：用户从 CSMAR 手工导出数据，项目在本地生成 interim、research panel、factor table，并通过 `alpha-research-core` 进行第一版 IC、RankIC 和分组评估。

当前报告是流程和数据资产说明，不构成投资建议，也不代表最终因子结论。

## 数据

| 项目 | 内容 |
|---|---|
| 数据源 | CSMAR，用户手工导出 |
| 数据范围 | 2014-2025 interim / panel / factor 已生成 |
| 评估范围 | 2024-2025 full-sample evaluation；core 年度 breakdown |
| universe | 第一版 A 股范围，排除 B 股和北交所 |
| 私有边界 | raw/interim/processed 和 private reports 不公开、不提交 |

## 方法

```text
CSMAR raw exports
  -> interim CSV
  -> research_panel_2014_2025.parquet
  -> price_volume_factors_2014_2025.parquet
  -> alpha-research-core evaluation
```

第一批 factor set 是 `price_volume`，包括：

- momentum
- reversal
- volatility
- amount liquidity
- turnover
- Amihud illiquidity
- amplitude

默认窗口为 `20, 60, 120`，forward labels 为 `1, 5, 10, 20, 60, 120` 个交易日。

## 产物

```text
data/processed_private/csmar/
  panel_full/research_panel_2014_2025.parquet
  price_volume_factors_full/price_volume_factors_2014_2025.parquet

reports/private_real/factor_evaluation/
  full_2024_2025/
    label_availability.csv
    evaluation_manifest.json
    factor_summary.csv
    factor_summary.md
    core_runs/
  comparisons/full_vs_yearly_current_2024_2025/
```

当前规模：

| 产物 | 数量 | 规模 |
|---|---:|---:|
| panel Parquet | 1 file | 0.586 GB |
| factor Parquet | 1 file | 2.318 GB |
| factor evaluation reports | 1 main report root | 1.159 GB |

2024-2025 evaluation 使用 21 个 price-volume factors、`forward_return_20d` label、5 组分组设置。

## Evaluation 口径

正式主口径采用 `full_2024_2025`：一次性把 2024-2025 输入传给 `alpha-research-core`，用 `forward_return_20d` 做预测标签、`forward_return_1d` 做 PnL 日收益列，并启用 core 的年度 breakdown。

年度 breakdown 不在年底清仓，也不在次年重新初始化组合状态。它先保留连续回测产生的组合路径，再把 `long_short` 日收益按自然年切分，并在每个年度内部重新统计 IC、RankIC、分组收益、PnL、Sharpe、drawdown 和 turnover。因此，2024 年底无法退出的持仓可以自然带入 2025，并按 `can_exit` 约束继续影响 2025 年初 PnL。

这个口径与“每年空仓重启”的 yearly independent evaluation 不同。后者只能作为诊断对照，不作为主报告口径。当前同口径校验显示，full 年度 breakdown 与重新生成的 yearly diagnostic baseline 在 IC、RankIC 和分组收益上完全一致；路径型指标只剩很小差异，主要来自 full run 在 2025 年初保留了 2024 年底延续持仓产生的 82 条 `long_short` 记录。

## 当前观察

年度结果里，`rev_20d`、`rev_60d`、`rev_120d` 是第一版最稳定的候选信号。2024 年样本里 `rev_120d` 的 `mean_rankic` 约 0.114，2025 年样本里 `rev_60d` 的 `mean_rankic` 约 0.095。

早期 924 前后事件切片曾显示 regime 差异：924 前反转类因子仍然靠前；924 后 `amihud_60d` 的 `mean_rankic` 约 0.160，`sharpe_like` 约 9.075。对应临时报告目录已清理，这条只作为历史观察保留，后续若要纳入正式报告，需要重新生成事件切片或改为 regime 稳健性分析。

## 限制

- 当前只覆盖第一批 price-volume factors。
- 当前 evaluation 是 2024-2025 full-sample 第一版诊断。
- 未完成成本、行业中性化、风险暴露、鲁棒性和正式组合约束分析。
- CSMAR 明细数据和可还原 vendor 数据的报告不公开。
