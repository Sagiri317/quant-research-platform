# Quant Research Platform 研究证据卡

本卡只保留面试、复盘或项目交接时最需要的证据链索引；当前迁移状态看 [PROJECT_STATUS.md](PROJECT_STATUS.md)，正式研究说明看 [REPORT.md](REPORT.md)。

## 核心证据

| 字段 | 内容 |
|---|---|
| 记录名称 | `quant_research_platform_a_share_module` |
| 证据状态 | 2023-2025 canonical sample、`rev_120d` parity shard、Factor Factory `rev_120d` smoke、N5 private aggregate smoke 和 pipeline failed-run reliability MVS 是当前有效证据引用 |
| 日期 | 2026-05-30 |
| 项目定位 | 本地多市场量化研究平台；A 股为第一版完整 market module |
| 数据 | CSMAR 手工导出；2014-2025 raw -> interim -> full panel -> full factor |
| 主要产物 | `panel_full/research_panel_2014_2025.parquet`；`price_volume_factors_full/price_volume_factors_2014_2025.parquet`；2023-2025 canonical / `rev_120d` evaluation samples；Factor Factory `rev_120d` smoke；N5 private aggregate smoke run id `n5_private_smoke_2025h2_rev120_v1` |
| evaluation | schema `0.7` evaluation-layer-backed evidence；引用 aggregate tables、manifest metadata、图表配置、下载索引和 public-safe aggregate smoke 口径 |
| 公开边界 | CSMAR 明细数据、股票样本、row-level private data、完整 private paths、private reports 和可还原 vendor 数据不公开 |

## 方法证据

- `src/quant_research_platform/evaluations/`：`EvaluationSpec`、panel validation、`run_evaluation`。
- `src/quant_research_platform/artifacts/`：schema `0.7` registry / writer / loader / display contract。
- `src/quant_research_platform/interfaces/api/`：本地 evidence API、CSV preview、direct download、ZIP download。
- `src/quant_research_platform/markets/implementations/a_share/`：CSMAR-backed A-share data、factor、validation、evaluation adapter。
- `src/quant_research_platform/pipelines/a_share/`：A 股 workflow runner、checkpoint 和 pipeline contracts。
- `web/`：本地 React/Vite evidence workbench。
- `docs/code-map/`：当前平台全局地图、A 股数据/pipeline 地图、evaluation/artifact/API/frontend 合同地图。

## 研究证据索引

- 第一批 factor set 是 `price_volume`：momentum、reversal、volatility、amount liquidity、turnover、Amihud illiquidity、amplitude。
- 默认窗口为 `20, 60, 120`；forward labels 为 `1, 5, 10, 20, 60, 120` 个交易日。
- 历史 2024-2025 long-only 第一版诊断中，反转类因子是主要候选。
- 历史 long-only 不变量曾检查通过：`portfolio_mode=long_only`、`short_count=0`、`short_return=0`，0 成本下 `net_portfolio_return == gross_portfolio_return == long_return`。
- 当前 2023-2025 canonical sample 覆盖 `mom_20d`、`vol_20d`、`avg_amount_20d` 和 `rev_120d`；`rev_120d` parity shard 覆盖 dataframe/tensor 与 long-only/long-short。
- 当前证据路径：`reports/private_real/factor_evaluation/schema07_eval_2023_2025_v1`、`reports/private_real/factor_evaluation/schema07_eval_2023_2025_rev120_v1`、`reports/private_real/ff/20260527_rev120_factory_smoke_v1`。
- N5 private aggregate smoke 已通过：run id `n5_private_smoke_2025h2_rev120_v1`；只记录 2025H2 aggregate smoke、schema `0.7` validation、diagnostics tables、`manifest.display`、API preview 和 selected ZIP smoke。
- Pipeline failed-run reliability MVS 已完成 persisted status fallback、failed checkpoints consistency、stable `failure_code`、sanitized failed payload 和 partial failed evaluation guard；retry/resume/overwrite 与 global API error envelope migration 尚未实现。

## 复现与引用

- 复现 A 股 evaluation 时，从 `README.md` 的 A 股私有 evaluation 命令和 `EVALUATION_INTERFACE.md` 的 evaluation contract 开始。
- 引用研究结论时，优先指向 `REPORT.md`、上述 current evidence paths，以及 artifact 内的 `manifest.json` / `summary.json` / aggregate CSV。
- 公开展示只使用 aggregate evidence 和 public-safe summaries；不要复制 CSMAR 明细、股票样本、row-level private data、完整 private paths、private reports 或可还原 vendor 数据。
