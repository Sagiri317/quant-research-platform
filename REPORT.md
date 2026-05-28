# Quant Research Platform 研究报告

## 摘要

本项目已经收口为本地多市场量化研究平台。第一版完整 market module 是 A 股：平台把用户已批准的本地 CSMAR 数据整理为 research panel、factor / label table、pipeline snapshot 和 evaluation-layer-backed evaluation artifacts，并通过本地 API 与 React evidence workbench 展示。

当前报告是流程、数据资产和研究证据说明，不构成投资建议，不代表最终因子结论，也不表示 paper trading、live trading 或生产部署已经获批。

## 数据

| 项目 | 内容 |
|---|---|
| 第一市场模块 | A 股 |
| 数据源 | CSMAR，用户手工导出 |
| 数据范围 | 2014-2025 interim / panel / factor 已有历史产物 |
| 评估范围 | 历史 2024-2025 long-only full-sample evaluation；当前保留 2023-2025 schema `0.7` canonical sample、`rev_120d` kernel / portfolio parity shard 和 Factor Factory smoke；N1/N2/N3/N4/N7/N8 已合入 `main` |
| universe | 第一版 A 股范围，排除 B 股和北交所 |
| 私有边界 | raw/interim/processed 和 private reports 不公开、不提交 |

## 方法

```text
CSMAR raw exports
  -> A-share market module
  -> research_panel_2014_2025.parquet
  -> price_volume_factors_2014_2025.parquet
  -> EvaluationSpec
  -> quant_research_platform.run_evaluation
  -> schema 0.7 artifacts
  -> API / downloads / evidence workbench
```

第一批 factor set 是 `price_volume`，包括 momentum、reversal、volatility、amount liquidity、turnover、Amihud illiquidity 和 amplitude。默认窗口为 `20, 60, 120`，forward labels 为 `1, 5, 10, 20, 60, 120` 个交易日。

## 历史 A 股产物

```text
data/processed_private/csmar/
  panel_full/research_panel_2014_2025.parquet
  price_volume_factors_full/price_volume_factors_2014_2025.parquet

reports/private_real/factor_evaluation/
  full_2024_2025/
    label_availability.csv
    manifest.json
    factor_summary.csv
    factor_summary.md
  n7_eval_2023_2025_v1/
  n7_eval_2023_2025_rev120_v1/
  comparisons/

reports/private_real/ff/
  20260527_rev120_factory_smoke_v1/
```

当前标准 evaluation run 应输出 schema `0.7` artifacts，并通过 schema registry 和 `manifest.display` 暴露可浏览表格、下载项、下载 preset 和图表配置。没有进入 `manifest.display` 的文件不应成为前端或下载接口的隐式展示项。

## Evaluation 口径

正式研究口径继续区分 diagnostics label 与 return source：

- `label_columns` 用于 IC、RankIC、grouped returns 和 spread diagnostics。
- `return_source_column` 用于 portfolio returns、cumulative returns、performance summary 和 time breakdown。
- A 股主报告在 long-only 场景下必须显式选择 `portfolio_mode=long_only`。

历史 `full_2024_2025` 是一次性把 2024-2025 输入传给 evaluation layer，用 `forward_return_20d` 做预测标签、`forward_return_1d` 做 return source 日收益列，并启用年度 breakdown。年度 breakdown 是连续回测路径上的自然年切片，不是每年空仓重启。

## 当前观察

迁移前历史 long-only 年度结果里，`rev_20d`、`rev_60d`、`rev_120d` 是第一版最主要的候选信号。2024 年 `rev_120d` 的 `mean_rankic` 约 0.114，`mean_net_portfolio_return` 约 0.003086；2025 年 `rev_60d` 的 `mean_rankic` 约 0.095，`mean_net_portfolio_return` 约 0.001717。

这些数字只作为历史统计诊断，不是最终投资结论。当前 2023-2025 canonical sample 已包含 `mom_20d`、`vol_20d`、`avg_amount_20d` 和 `rev_120d` 的 8-combination dataframe long-only 样例；`rev_120d` parity shard 覆盖 2 labels、2 kernels 和 2 portfolio modes；Factor Factory smoke 已能从受控模板生成 `rev_120d` 候选并汇总到 evaluation results。Phase 1、N1/N2/N3/N4/N7/N8 的 pipeline runner、Factor Factory 本地入口、artifact/API/frontend 合同和 contract consumer convergence 已合入 `main`。Evidence workbench 通过 `manifest.display` 消费图表、CSV preview 和下载，图表使用 bounded preview pagination 拼接当前组合时间轴；前端不重新计算 eligibility、tradability、IC、returns、weights 或 turnover。后续仍需要继续补充正式成本敏感性、regime、中性化、风险暴露和 ML prediction handoff 分析。

## 限制

- 当前完整 market module 只有 A 股；crypto 和其他市场只是 planned / disabled 能力表述，不保留空实现包。
- 当前只覆盖第一批 price-volume factors。
- ML prediction、风险暴露、regime、factor mining 和组合优化仍需单独设计和验证。
- 模拟盘不属于当前范围；未来如果做，必须保持与真实交易接口隔离，并单独审批。
- CSMAR 明细数据和可还原 vendor 数据的报告不公开。
