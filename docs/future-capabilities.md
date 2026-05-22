# Future Capabilities / 后续能力池

本文是 `quant-research-platform` 的正式后续功能池。它收敛了历史迁移设计、平台路线图和旧 core 后续能力总结中的长期方向。

本文不是当前已实现功能清单，也不是部署、模拟盘、实盘或外部 API 审批记录。当前可执行接口以 [README.md](../README.md)、[ARCHITECTURE.md](../ARCHITECTURE.md)、[EVALUATION_INTERFACE.md](../EVALUATION_INTERFACE.md) 和 [docs/code-map/](code-map/) 为准。

## 1. 当前边界

当前平台定位：

```text
approved local data
-> market module
-> factor / label / prediction / risk inputs
-> EvaluationSpec
-> core evaluation
-> schema 0.5 artifacts
-> API / evidence workbench / downloads
```

当前已实现或已保留的核心能力：

- IC / RankIC。
- grouped return。
- long-only / long-short。
- transaction cost 基础口径。
- slippage / spread 字段级成本拆分。
- 因子中性化：group demean 和 cross-sectional OLS。
- DataFrame / Parquet / CSV input adapter。
- A 股 CSMAR-backed market module。
- schema `0.5` artifacts、`manifest.display`、API、下载和 React evidence workbench。
- 多市场接口骨架；`crypto` 当前只是 placeholder。

硬边界：

- 不处理账号、密码、token、cookie、`.env` 或凭据。
- 不连接券商、交易所、vendor API、外部行情 API 或真实交易系统。
- 不自动下载 CSMAR，不自动登录网站。
- 不把研究诊断写成投资建议、真实收益承诺或可交易策略声明。
- 网站部署、认证、数据库、模拟盘和实盘必须单独设计并审批。

## 2. 模块化目标结构

后续扩展应优先沿下面的职责边界推进：

```text
markets/
  a_share/        # 数据许可、domain rules、factor/label/risk/regime input
  crypto/         # 未来多频/高频 market module

research/
  factor_factory/ # 自动候选因子生成、筛选、失败记录
  models/         # ML prediction、walk-forward、model manifest
  portfolio/      # 多因子组合、optimizer、risk parity
  risk/           # exposure、covariance、risk attribution
  regime/         # regime definitions and diagnostics

core/
  contracts       # EvaluationSpec and related configs
  evaluation      # market-neutral evaluation runner
  results         # in-memory evaluation result contract

artifacts/
  contract        # schema and validation contract
  display         # manifest.display source of truth
  writer/loader   # persistence boundary
  comparison      # run comparison

api/
  routes          # markets, pipeline, runs, artifacts, downloads, comparison

web/
  evidence        # result browsing
  pipeline        # controlled run configuration
  factor_factory  # future candidate generation and review
```

原则：

- market module 决定研究口径和数据可用性。
- research 层生产候选信号、预测、风险输入和组合方案。
- core 只评价显式传入的 panel 和配置。
- artifacts 是 API、下载和前端展示的唯一结果合同。
- 前端只消费 API 和 artifacts，不重新计算 IC、PnL、turnover 或 weights。

## 3. 因子研究层

### 当前能力

- IC / RankIC。
- grouped return。
- factor / label 多组合评价。
- 因子中性化。
- 受控因子表达式解析和 pipeline snapshot。

### 后续能力

- 因子正交化。
- 多因子组合。
- 自动批量生成候选因子并验证的 Factor Factory。

### 推荐落点

- A 股字段、模板、表达式依赖和可知时间：`markets/a_share/factors/`。
- 通用候选生成、参数网格、候选记录、失败日志：`research/factor_factory/`。
- 评价仍走 `EvaluationSpec -> run_evaluation -> artifacts`。

### Factor Factory 第一阶段范围

- 使用已批准的本地 processed panel。
- 候选表达式来自显式 registry 或受控参数网格。
- 不使用 Python `eval`，不执行任意用户代码。
- 每个候选保存公式、输入字段、窗口、样本范围、依赖字段和失败原因。
- 调用 core 批量评价，输出 compact candidate summary 和 failed candidate log。
- 所有用户可见输出进入 `manifest.display`。

完成标准：能回答“这个因子怎么来的、用了哪些字段、什么时候可知、为什么失败或保留”。

## 4. 风险模型层

### 当前能力

- `segment_columns`。
- `risk_column`。
- neutralization exposure columns。
- `portfolio/exposure_summary.csv`。

### 后续能力

- 风险暴露分析。
- 协方差矩阵。
- Barra-like 风险模型。
- 风险归因。
- Tail Risk 分析。

### 推荐落点

- A 股行业、市值、流动性、波动率、beta 等输入列：`markets/a_share/`。
- 通用风险合同、协方差、风险归因：`research/risk/`。
- core 只在输入合同稳定后消费 exposure / risk model 输出。

第一阶段先做 exposure diagnostics，不做 Barra-like 模型承诺。

## 5. 组合构建层

### 当前能力

- long-only。
- long-short。
- quantile portfolio weights。
- cost-aware PnL path。

### 后续能力

- beta neutral。
- optimizer。
- risk parity。
- 自动多因子组合搜索。

### 推荐落点

- 组合搜索和约束配置：`research/portfolio/`。
- 最终组合列或权重输入再交给 core 评价。
- core 如需扩展 optimizer，应先稳定 `PortfolioSpec` 或等价合同，不直接把优化器塞进 `EvaluationSpec`。

所有组合优化必须输出约束检查、不可满足约束和失败原因，不只输出收益。

## 6. 执行、成本、流动性和容量

### 当前能力

- buy / sell cost。
- slippage bps。
- spread bps。
- cost breakdown。
- turnover。

### 后续能力

- market impact。
- liquidity diagnostics。
- capacity analysis。

### 推荐落点

- 日频 A 股的成交额、换手率、停牌、涨跌停和容量 proxy：`markets/a_share/`。
- 通用成本模型和容量诊断：`research/portfolio/` 或 `research/risk/`。
- core 只消费已定义好的 cost config 和输入列。

容量结果必须写成诊断和约束，不写成真实可交易规模承诺。

## 7. 机器学习层

### 当前能力

- label / PnL return 分离。
- prediction 列可作为 feature 交给 core 评价。

### 后续能力

- label pipeline。
- walk-forward validation。
- XGBoost / LightGBM。
- ensemble model。

### 推荐落点

- 通用模型训练和验证合同：`research/models/`。
- A 股特定 feature availability、样本过滤和 label 规则：`markets/a_share/`。
- 预测输出为 `prediction_*` 列或 prediction Parquet，再交给 core 评价。

模型 manifest 必须记录 label、features、split、embargo、参数、训练日期、产物路径和失败候选。禁止随机打散时间序列样本。XGBoost / LightGBM 属于新增依赖，必须单独审批。

## 8. Regime 和事件层

### 当前能力

- 可以通过 segment / time breakdown 做部分切片。

### 后续能力

- bull / bear split。
- high-vol / low-vol。
- trend / mean-reversion。
- 事件窗口，例如 2024 年 924 前后。
- regime detection。

### 推荐落点

- regime input 和事件定义：`markets/a_share/` 或未来对应 market module。
- 通用 regime 合同、切片和诊断：`research/regime/`。
- core 优先消费外部传入的 regime column，不自动检测 regime。

Regime 必须在当时可知，不能用未来收益定义。事件切片是诊断，不是稳定策略证明。

## 9. 数据基础设施层

### 当前能力

- Parquet。
- 多市场接口骨架。
- A 股日频 full panel 和 slim factor table。

### 后续能力

- 多频率支持。
- tick 数据支持。
- crypto 高频数据。
- data contract manifest。
- leakage checks。

### 推荐落点

- 多市场公共输入合同保留在 `core` / `io` / `markets` 边界。
- 高频或 tick 不应硬塞进当前日频 `EvaluationSpec`。
- 未来可新增 `IntradayEvaluationSpec` 或 `BacktestSpec`，但 artifact/display/API 结果合同尽量复用。

## 10. API 和前端扩展

当前前端应继续分清：

- Evidence Workbench：浏览结果。
- Pipeline Builder：配置和快照。
- CSV upload：本地调试或兼容入口，不是主流程。

后续新增：

- Factor Factory：候选生成、候选筛选、失败原因、提交评价。
- Pipeline Runner：从配置执行 A 股 build/evaluate。
- Model Lab：训练、walk-forward、prediction run。
- Risk Lab：风险暴露、组合约束、优化诊断。

新增可下载文件必须进入 `manifest.display`。不要添加只在前端知道、后端不知道，或只在后端下载、前端不可见的 artifact。

## 11. 阶段优先级

### P0: 合同和模块边界

- 拆清 `EvaluationRunPayload` 或等价中间结果合同。
- 把 evaluation 和 artifact persistence 解耦。
- 拆分 API route 边界。
- 完成 A 股 pipeline runner 设计。

### P1: Factor Factory

- 候选表达式 registry。
- 小参数网格。
- 批量评价。
- candidate summary。
- failed candidate log。
- workbench 浏览入口。

### P2: A 股 pipeline runner

- 配置驱动的 build/evaluate 编排。
- checkpoint 状态。
- 失败恢复和不覆盖规则。
- 小数据 smoke。

### P3: 风险、regime、成本扩展

- exposure columns。
- risk / regime diagnostics。
- cost grid。
- liquidity / capacity diagnostics。

### P4: ML prediction

- walk-forward。
- model manifest。
- prediction Parquet。
- core-backed prediction evaluation。

### P5: 多市场、多频率和部署

- crypto 高频设计。
- intraday / tick contract。
- 网站部署、认证、数据隔离和发布边界。

部署必须单独设计；它会改变当前 local-only、安全、私有数据、API 暴露和认证边界。
