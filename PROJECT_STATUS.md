# a-share-alpha-lab 项目状态

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 项目定位 | 本地私有 CSMAR A 股因子研究项目 |
| 代码结构 | workflow-first：主代码在 `code/`，不再使用 `src/` + `scripts/` 主结构 |
| Conda 环境 | `analysis` |
| 数据范围 | 2014-2025 interim / full panel / full factor 已重建 |
| 报告范围 | 2024-2025 long-only full-sample factor evaluation 已生成；历史 long-short 结果已用于迁移对比后清理 |
| 阶段位置 | 已完成 CSMAR 数据层、第一批 price-volume factors、core-backed long-only evaluation；下一阶段是在 long-only 口径下重跑成本/regime 稳健性，并建设行业/市值/风险暴露列 |
| 私有边界 | CSMAR raw/interim/processed 和 `reports/private_real/` 不提交、不公开 |

## 本轮完成项

1. 完成 `code/` workflow-first 结构重构。
2. 删除旧 `src/` 和 `scripts/` 主结构。
3. 接入 interim 列裁剪和 `config/private_column_overrides.json`。
4. 生成 `data/interim_private/csmar/_manifests/column_pruning_report.md`。
5. 重建 2014-2025：
   - `data/interim_private/csmar/`
   - `data/processed_private/csmar/panel_full/research_panel_2014_2025.parquet`
   - `data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet`
6. 只对 2024-2025 生成：
   - `reports/private_real/factor_evaluation/full_2024_2025/label_availability.csv`
   - `reports/private_real/factor_evaluation/full_2024_2025/evaluation_manifest.json`
   - `reports/private_real/factor_evaluation/full_2024_2025/core_runs/`
   - `reports/private_real/factor_evaluation/full_2024_2025/factor_summary.csv`
   - `reports/private_real/factor_evaluation/full_2024_2025/factor_summary.md`
7. 旧 2024-2025 yearly evaluation 产物已确认是旧 PnL 口径，不再作为 full breakdown 校验 baseline；对应目录已清理。
8. 增加 evaluation 日期过滤入口，支持在内存中按 `trade_date` 截取，不额外生成 subperiod Parquet。
9. 曾生成 2024 年 924 前后事件切片；对应临时报告目录已清理。
10. 删除旧 processed 年度文件后，主代码路径已统一为 full Parquet；年份只作为筛选和报告维度保留。
11. 旧 `yearly_2024_2025` baseline 的 full vs yearly 指标对比已作废；对应旧目录已清理。
12. 重新生成当前口径 yearly 诊断 baseline，并完成 full breakdown 对齐验证；诊断 baseline 目录已清理，保留轻量 comparison 结论：
   - `reports/private_real/factor_evaluation/comparisons/full_vs_yearly_current_2024_2025/`
13. 参考 `alpha-research-core` 的 code map 风格补充源码阅读地图：
   - `docs/code-map/README.md`
   - `docs/code-map/data-pipeline.md`
   - `docs/code-map/factors.md`
   - `docs/code-map/evaluation.md`
   - `docs/code-map/validation-runtime.md`
14. 修复 full panel 构造的跨年连续性：`build_panel_full_from_interim()` 先合并目标年份的 daily interim，再一次性构造 full panel，避免 `listing_age_trading_days` 年初重置和年末 `can_enter/can_exit` 截断。
15. 集中 processed 路径合同到 `code/runtime/paths.py`，并同步 `build_panel.py`、`build_factors.py`、`core_bridge.py`、`storage.py`。
16. 更新 full vs yearly diagnostic comparison 默认口径，不再默认指向已作废的旧 `yearly_2024_2025`。
17. 精简项目文档：core handoff 专题说明已并入 evaluation code map，阶段路线已并入本状态文件；保留并压缩 `RESEARCH_CARD.md`。
18. 生成过第一轮 long-short 稳健性评估；在 core 支持 `long_only` 后，该批历史结果已改名用于迁移对比，并在 long-only 主结果验证后清理，不再作为 A 股可执行口径结论。
19. A-share evaluation CLI 已支持 `--portfolio-mode long_short|long_only`，并透传到 core `WeightConfig.portfolio_mode`；`factor_summary.csv/md` 已兼容 core 新版 `mean_net_portfolio_return` 字段。
20. 已重新生成 2024-2025 long-only 主结果：
   - `reports/private_real/factor_evaluation/full_2024_2025/label_availability.csv`
   - `reports/private_real/factor_evaluation/full_2024_2025/evaluation_manifest.json`
   - `reports/private_real/factor_evaluation/full_2024_2025/core_runs/`
   - `reports/private_real/factor_evaluation/full_2024_2025/factor_summary.csv`
   - `reports/private_real/factor_evaluation/full_2024_2025/factor_summary.md`
   - `reports/private_real/factor_evaluation/comparisons/`

## 产物规模

| 目录 | 当前规模 |
|---|---:|
| `data/raw_private/` | 4.002 GB |
| `data/interim_private/` | 2.041 GB |
| `data/processed_private/` | 2.904 GB |
| `reports/private_real/` | 0.584 GB |
| `panel_full/` | 1 file, 0.586 GB |
| `price_volume_factors_full/` | 1 file, 2.318 GB |
| `full_2024_2025/` | 25 files, 0.584 GB |
| `comparisons/` | 5 files, 0.128 MB |

## 当前已验证

- `conda run -n analysis python -m pytest`：53 passed。
- `conda run -n analysis python -m ruff check .`：通过。
- raw 检查：所有 CSMAR 原始导出 metadata 和必需字段通过。
- interim 检查：无 blocking issue；指数 OHLC 异常只作为 warning 标记。
- panel 检查：2014-2025 无 blocking issue。
- factor 检查：2014-2025 无 blocking issue；每年 21 个 factors、6 个 forward labels。
- 总体 panel 核对：年度切片行数与旧年度产物一致。
- 总体 factor 核对：年度切片行数与旧年度产物一致；浮点数值使用 `atol=1e-7` 容差处理滚动标准差的微小舍入差异。
- processed 空间：删除旧年度产物后，当前 processed 私有数据约 2.904 GB。
- full evaluation 检查：`full_2024_2025` 已生成 1 个合并样本 long-only core run，`summary.json` 显示 `breakdown_enabled=true`，包含 `metrics_by_year/` 年度 breakdown，`weight_configs[0].portfolio_mode="long_only"`。
- full vs 旧 yearly 对比已作废：旧 `yearly_2024_2025` 没有 `pnl_return_column="forward_return_1d"`，不能作为当前 PnL 口径 baseline。
- long-only 不变量检查：`portfolio/returns.csv` 共 5,532 行，`portfolio_mode` 全为 `long_only`；`short_count` 最大值 0，`short_return` 最大绝对值 0；0 成本下 `gross_portfolio_return - long_return` 最大绝对值 0，`net_portfolio_return - gross_portfolio_return` 最大绝对值 0；`portfolio/positions.csv` 共 5,257,773 行，负持仓行数 0。
- long-only vs 历史 long-short 对比：`mean_ic`、`mean_rankic`、`top_bottom_spread_mean` 完全一致；差异集中在组合路径指标，符合“同一信号诊断、不同组合约束”的预期。对比报告保留在 `reports/private_real/factor_evaluation/comparisons/long_only_vs_long_short_2024_2025/`。
- 历史 long-short 成本敏感性和 924 regime 结果已清理，不再作为 A 股可执行口径结论；需要在 long-only 口径下重新生成。

## 当前研究观察

- 2024-2025 long-only full-sample evaluation 使用 21 个 price-volume factors、`forward_return_20d` label、5 组分组设置，并输出年度 breakdown。
- 2024 年和 2025 年里，`rev_20d`、`rev_60d`、`rev_120d` 的 RankIC 仍排名靠前，说明第一版样本里反转信号是主要候选。
- long-only PnL 下，2024 年 `rev_120d` 的 `mean_rankic` 约 0.114，`mean_net_portfolio_return` 约 0.003086；2025 年 `rev_60d` 的 `mean_rankic` 约 0.095，`mean_net_portfolio_return` 约 0.001717。
- 以上仍是统计诊断，不是最终投资结论；还没有在 long-only 口径下完成成本敏感性、924 regime、行业/市值中性化、风险暴露和正式组合约束分析。

## Core 接口适配缺口

当前已能从本项目调用 `alpha-research-core` 并把标准 core artifacts 写入 `reports/private_real/factor_evaluation/.../core_runs/<run_id>/`。已适配的关键项包括：`EvaluationSpec`、`run_evaluation(spec)`、`pnl_return_column="forward_return_1d"`、`eligible`、`can_enter`、`can_exit`。

仍未完全适配的点：

1. 当前主 evaluation 已支持跨年份 full factor 读取后形成单个 core run；legacy yearly run 仅作为诊断 baseline，不作为主报告口径。
2. `code/evaluation/run_factor_evaluation.py` 已开放 `--pnl-return-column`、`--portfolio-mode`、`--cost-bps`、`--engine`、`--segment-columns`、`--risk-column`、`--neutralization-method`、`--neutralization-group-columns` 和 `--neutralization-exposure-columns`。profiling、time breakdown 选择、详细 cost config 和 tensor config 仍未暴露。
3. 当前已能把 segment / risk / neutralization / exposure 参数传给 core，但本项目侧还没有稳定生成行业、市值、风险暴露等输入列；这些列仍需要后续数据层建设后再正式使用。
4. `reports/.../evaluation_manifest.json` 是 a-share 自己生成的汇总索引，不是 core 标准 `manifest.json`。它记录每个 core run 的输入配置和输出目录，不重新计算 IC、PnL、收益、回撤、换手等指标；真正的评价指标以各 `core_runs/<run_id>/` 下的 core artifacts 为准。

Full panel / factor Parquet 迁移：

- `code/datasets/build_panel.py` 默认写出总体 panel：`panel_full/research_panel_2014_2025.parquet`。
- `code/factors/build_factors.py` 默认写出总体 factor：`price_volume_factors_full/price_volume_factors_2014_2025.parquet`。
- `code/evaluation/core_bridge.py` 从 full factor 文件读取必要列，在 a-share 侧按年份或日期筛选后传入 core `EvaluationSpec`；core 不需要适配新的存储结构。
- `inspect_panel.py`、`inspect_factors.py` 和 `runtime/storage.py` 已改为 full 文件口径。

## 下一步

1. 在 long-only 口径下重跑 5 / 10 / 20 bps 成本敏感性和 924 前后 regime 切片。
2. 评估并生成行业、市值、风险暴露等列，正式启用 segment / risk / neutralization / exposure 诊断。
3. 基于 long-only 稳健性结果选择候选因子，优先复查 `rev_60d`、`rev_120d` 和 `rev_20d`。
4. 更新私有研究报告，并为未来脱敏公开材料保留聚合证据。

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-05-15 | structure | 子项目主结构从 `src/` + `scripts/` 改为 `code/` workflow-first。 | `WORKSPACE_STATUS.md`、`control/PROJECT_REGISTRY.md`、root planning records | closed 2026-05-15 |
| 2026-05-15 | milestone | 项目进入 CSMAR 私有真实数据重建和 2024-2025 factor evaluation 阶段。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-15 |
| 2026-05-18 | milestone | 2014-2025 数据层已重建；2024-2025 yearly evaluation 和 2024 年 924 前后事件切片已生成。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-18 |
| 2026-05-19 | planning | 已记录 core 接口适配缺口；已生成 full panel/factor Parquet；旧年度 processed 文件已删除，主代码路径已改为 full factor 读取后传 DataFrame 给 core；`full_2024_2025` 合并样本 evaluation 已生成并带年度 breakdown；当前口径 yearly baseline 已验证 full breakdown 基本对齐，旧 yearly baseline 已作废；阶段性重报告目录已清理，仅保留正式 full 结果和轻量 comparison；子项目 README/REPORT/AGENTS/RESEARCH_CARD/data README 已记录连续 full run + 年度 breakdown 主口径，CLI 已开放 PnL、cost、engine、segment/risk/neutralization 参数；已补充 `docs/code-map/` 源码阅读地图；已修复 full panel 跨年连续性并集中 processed 路径合同到 `code/runtime/paths.py`；已完成第一轮 5/10/20 bps 成本敏感性和 924 前后 regime 稳健性。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-19 |
| 2026-05-20 | milestone | core 支持 long-only 后，子项目 CLI 已新增 `--portfolio-mode` 并透传到 `WeightConfig.portfolio_mode`；`factor_summary` 已改用/兼容 `mean_net_portfolio_return`；2024-2025 long-only full run 已重新生成并通过 short leg / position 不变量检查；历史 long-short full 和 robustness 结果目录已清理，仅保留 long-only 主结果和轻量对比报告。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |
