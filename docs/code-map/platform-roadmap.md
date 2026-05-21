# Platform Roadmap

本文吸收临时 A 股平台蓝图中的可复用内容，作为 `quant-research-platform` 的路线说明。它不是当前已全部实现的功能清单，也不是实盘、模拟盘或部署审批记录。

## 1. 平台定位

平台的日常入口是本项目本身，而不是拆分成多个并行项目。A 股是第一版完整 market module，core evaluation 是平台内部引擎。

```text
approved local data
-> audit-friendly interim tables
-> research panel
-> factors / labels / predictions
-> risk / regime / liquidity context
-> core evaluation
-> schema 0.5 artifacts
-> evidence workbench / reports
```

平台边界：

- 只处理用户已批准的本地私有数据和后续明确批准的数据范围。
- 不登录 CSMAR，不自动下载，不处理账号、密码、cookie、token 或 `.env`。
- 不接券商、交易所或任何真实交易接口。
- 不把研究诊断写成投资建议、真实收益承诺或可交易策略声明。
- 未来模拟盘必须单独审批，并与真实交易接口保持隔离。

## 2. Core 作为内部引擎

Core 继续负责：

- `EvaluationSpec` 和 `run_evaluation(spec)`。
- IC / RankIC。
- grouped returns。
- long-only / long-short portfolio path。
- turnover、cost、PnL、performance summary。
- schema `0.5` artifacts。
- `manifest.display` 表格、下载、图表展示契约。
- FastAPI / React evidence workbench。

Market module 负责：

- 数据许可和私有边界。
- universe、eligible、can_enter、can_exit。
- factor、label、prediction 和 risk input 的生成。
- 研究结论、报告、公开 / 私有边界。

核心原则：

```text
market module 决定研究口径。
core 只评价显式传入的列。
所有可展示 artifacts 通过 manifest.display 暴露。
```

## 3. 数据层路线

当前 A 股数据层：

- `data/raw_private/csmar/`：保留原始导出。
- `data/interim_private/csmar/`：规范化、裁剪、可人工审计。
- `data/processed_private/csmar/`：研究 panel 和 factor Parquet。
- `reports/private_real/`：私有真实研究产物。

后续增强方向：

1. 补稳定行业、市值、流动性和风险暴露列。
2. 建立 data contract manifest，记录每次 panel / factor 输出的字段、日期范围、行数、过滤条件。
3. 增加 leakage checks：确认 factor 使用当时可知信息，label 和 PnL return 不反向污染 feature。
4. 对全样本运行增加空间估算，尤其关注 positions、CSV、ZIP 和前端加载规模。

## 4. 因子与 Label 层路线

当前主线是 price-volume factors + forward labels。平台化后，因子层应从“生成一批固定列”升级为“可追踪的因子生产系统”。

因子列命名建议：

```text
raw factor:        rev_20d
standardized:      rev_20d_zscore_cs
neutralized:       rev_20d_zscore_cs_neut_industry_size
orthogonalized:    rev_20d_orth_to_mom_20d
composite:         composite_reversal_v1
```

关键规则：

- 标准化、中性化、正交化都在 market module 完成，不让 core 静默改 factor。
- 每个 transform 都要有 metadata，记录输入列、方法、窗口、分组和输出列。
- longer-horizon label 用于 IC / RankIC / grouped diagnostics。
- `pnl_return_column` 继续显式使用 `forward_return_1d` 或平台明确命名的 realized return 列。

## 5. ML Prediction 层路线

ML 不进 core，放在 market module 或共享 research contract 内。

第一版目标：

- 使用已有 factor table。
- 按时间切分 train / validation / test。
- 支持 walk-forward validation。
- 输出 prediction Parquet。
- 输出 model manifest，记录 label、features、split、embargo、模型参数、训练日期和产物路径。
- 把 `prediction_*` 列交给 core 评价。

注意点：

- 禁止随机打散时间序列样本。
- 必须记录 embargo 或至少说明为什么不需要。
- 超参搜索结果必须保留失败和低表现候选，避免幸存者叙事。
- XGBoost / LightGBM 属于新依赖，必须单独审批后才能加入。

## 6. 自动挖因子路线

自动挖因子应作为平台内的可审计研究工具，而不是黑箱搜索器。

第一版只做保守范围：

- 使用已批准的 processed panel。
- 候选表达式来自显式 registry。
- 参数网格小而可解释。
- 每个候选有公式、输入字段、窗口、样本范围。
- 调用 core 批量评价。
- 产出 compact candidate summary 和 failed candidate log。

完成标准：能回答“这个因子怎么来的、用了哪些字段、什么时候可知、为什么失败或保留”。

## 7. 风险与组合层路线

风险模型先在 market module 做 domain version，core 只在输入合同稳定后消费。

第一阶段：

- 生成 industry、market_cap、volatility、beta、liquidity 等 exposure columns。
- 用这些列启用 segment / risk / neutralization / exposure 参数。
- 做 long-only 口径下成本敏感性和 regime 切片。

第二阶段：

- 构建 covariance 或 shrinkage covariance summary。
- 设计 beta neutral / exposure neutral / risk parity 的输入合同。
- 再决定是否扩 core 的 optimizer。

组合优化必须输出约束检查和未满足约束，不只输出收益。

## 8. Regime 与事件层路线

Regime 是研究假设，先在 market module 定义。

可支持：

- bull / bear；
- high-vol / low-vol；
- trend / mean-reversion；
- 事件窗口，例如 2024 年 924 前后；
- 宽基指数、成交额、波动率等 regime inputs。

Regime 必须在当时可知，不能用未来收益定义。事件切片是诊断，不是稳定策略证明。

## 9. Evidence Workbench 路线

Workbench 应展示：

- data coverage；
- factor summary；
- candidate factor cards；
- ML prediction runs；
- risk exposure diagnostics；
- long-only / long-short comparison；
- cost grid；
- regime breakdown；
- core artifact browser。

实现原则：

- 前端只消费 API 和 artifacts，不重新计算 IC / PnL / turnover。
- 新增可下载表必须同步进入 `manifest.display`。
- 私有真实报告默认不公开，不进入 Git。

## 10. 模拟盘边界

模拟盘不属于当前已批准范围。未来如果做，应作为相邻子系统或单独项目，并保持以下边界：

- 不接真实券商接口。
- 不创建、读取或保存 credentials。
- 不自动下单。
- 不输出投资建议。
- 只消费明确导出的 signal snapshot。
- 输出 paper orders、paper fills、portfolio state 和 risk report。

正式进入模拟盘前，应先完成 A 股 factor / ML prediction pipeline、core-backed evaluation、风险/成本/流动性诊断，以及明确审批范围和安全规则。
