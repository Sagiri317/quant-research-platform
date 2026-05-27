# N5 Research Diagnostics Framework 设计

状态：设计草案，等待用户 review 后再写 implementation plan
日期：2026-05-27
范围：N5 风险、regime、成本、容量和基础组合假设诊断；不包含 N6 ML prediction handoff、组合优化器、事件驱动仿真、paper/live trading、外部 API 或部署能力。

## 1. 核心目标

N5 的目标不是增加一组临时 CSV，也不是把 evaluation runner 膨胀成完整回测系统，而是建立一个可扩展的研究诊断框架：

```text
EvaluationSpec
  -> 生成基础 evaluation artifact
  -> combinations / weights / returns / turnover / cost / summaries

DiagnosticsSpec
  -> 读取 evaluation artifact + optional processed panel + assumptions
  -> 生成 risk / regime / cost / capacity diagnostics extension
```

用户心智应是：

```text
一个 evaluation run
  -> 基础 signal / portfolio 评价
  -> N5 diagnostics 解释收益来源、风险暴露、成本侵蚀、容量压力和 A 股假设边界
```

工程心智应是：

```text
DiagnosticsSpec
  source evaluation run
  input bundle
  diagnostic policies
  artifact specs
  manifest.display exposure
```

N5 应支持《打开量化投资的黑箱》第 1-6 章提炼出的平台要求：signal evaluation、risk diagnostics、cost diagnostics、capacity diagnostics、基础 portfolio construction assumptions、A 股约束和 artifact/API/workbench 合同化。

## 2. 非目标

N5 不做：

- 投资建议、真实收益承诺或真实可交易规模承诺；
- ML model training、prediction handoff 或 model manifest；这些属于 N6；
- mean-variance、convex optimization 或 transaction-cost-aware optimizer；
- 事件驱动撮合、订单生命周期、真实成交回放、market replay、paper trading 或 live trading；
- broker / exchange / vendor 外部 API 连接、自动下载、自动登录或 API key 处理；
- 部署服务、repo-level `apps/api`、多用户权限、租户、审计或商业化能力；
- private CSMAR raw/interim/processed data 的公开、复制、提交或路径泄露。

## 3. 关键设计决定

### 3.1 N5 是 Diagnostics Framework，不是单纯风险层

N5 的主干是风险诊断，但必须同时覆盖成本、容量、regime 和基础组合假设。否则平台无法解释收益到底来自 signal、组合构建、风险暴露、成本假设、容量压力还是执行假设。

N5 第一版应实现可复用框架和核心 post-run diagnostics；后续版本再把诊断升级为研究 gates、规则型 portfolio policy 和 optimizer bridge。

### 3.2 `EvaluationSpec` 与 `DiagnosticsSpec` 的边界

判断规则：

```text
改变 weights / returns / cumulative returns / performance_summary / tradable universe
  -> EvaluationSpec 或未来 PortfolioSpec

解释、切片、归因、敏感性分析已有结果
  -> DiagnosticsSpec
```

应留在 `EvaluationSpec`：

- `panel`、timestamp / asset columns、feature columns、label columns；
- `return_source_column`；
- 会生成 target weights 的 `weight_configs`；
- 会改变收益路径的 `portfolio_modes`；
- baseline `cost_config`，因为基础 `net_portfolio_return` 依赖它；
- eligibility / can_enter / can_exit；
- `n_groups`、kernel、metadata、output dir；
- 当前基础 time breakdowns。

应放入 `DiagnosticsSpec`：

- regime definitions 和 event windows；
- exposure columns 和 risk exposure policies；
- neutralized IC / RankIC diagnostics；
- cost profile grid 和 break-even sensitivity；
- capacity profile、ADV participation、liquidity bucket 和 capacity proxy；
- stress scenario definitions；
- crowding proxy definitions；
- A 股执行假设 metadata；
- diagnostic display/report grouping。

灰区规则：

| 配置 | N5 v1 归属 | 未来如果改变基础路径 |
|---|---|---|
| `risk_column` for volatility scaling | `EvaluationSpec` | 继续属于 weight policy |
| industry exposure diagnostics | `DiagnosticsSpec` | 不改变权重 |
| industry neutral weight | v1 不做；v3 `PortfolioPolicySpec` | 进入 weight policy identity |
| ADV cap diagnostics | `DiagnosticsSpec` | 如果限制持仓/成交，进入 portfolio/simulation policy |
| turnover cap diagnostics | v1 不做 | 如果改变交易路径，进入 portfolio/simulation policy |
| cost sensitivity grid | `DiagnosticsSpec` | 如果基础收益路径重跑，进入 evaluation identity |

### 3.3 解释型 policy 不膨胀 `combination_id`

N7 已定义：只有同一个 run 内同时存在多个取值，且这些取值改变结果语义时，字段才进入 `dimensions/combinations.csv`。

N5 延续该原则：

| Policy | N5 v1 是否进入 `combination_id` |
|---|---:|
| regime split 只是切片分析 | 否 |
| cost profile 只是 post-run sensitivity | 否 |
| risk exposure 只是解释已有权重 | 否 |
| neutralized IC 只是诊断 | 否 |
| capacity proxy 只是压力诊断 | 否 |
| industry neutral weight 改变 target weights | 是，后续作为 `weight_policy_id` 或新 evaluation run |
| ADV cap 改变持仓/收益路径 | 是，后续作为 portfolio policy |
| optimizer 生成新权重 | 是，后续作为 optimizer/portfolio run identity |
| execution simulation 改变 realized returns | 不塞进 evaluation schema；进入 simulation run identity |

### 3.4 N5 v1 复用 schema `0.6`

N5 v1 新增 diagnostics extension 表，仍写入当前 schema `0.6` evaluation-set run 目录，并通过 `manifest.display` 暴露。

不在 v1 新增独立 `artifact_type="diagnostics_set"`，原因：

- 当前 API-visible identity 是 `(source_alias, run_id)`；
- evidence workbench、download packaging 和 preview route 都已围绕 evaluation-set display contract 收敛；
- 独立 diagnostics artifact 会引入跨 artifact identity、run linking 和 stale extension 管理问题。

未来如果 diagnostics 需要多次追加、跨 evaluation run 复用或独立发布，可以设计独立 diagnostics artifact type，但不属于 N5 v1。

## 4. 版本路线

### 4.1 N5 v1: Post-Run Diagnostics Framework

目标：建立 `DiagnosticsSpec / DiagnosticPolicy / DiagnosticRunner`，并生成第一批 post-run diagnostics artifacts。

做：

- `DiagnosticsSpec`；
- `DiagnosticInputBundle`；
- `DiagnosticPolicy`；
- `DiagnosticRunner`；
- diagnostic artifact specs；
- `regime_definitions.csv`；
- `regime_performance.csv`；
- `cost_profiles.csv`；
- `cost_sensitivity.csv`；
- `risk_exposure.csv`；
- `capacity_diagnostics.csv`；
- optional `neutralized_signal_diagnostics.csv`；
- A 股假设 metadata 和 report 解释；
- generic diagnostic table display。

不做：

- 不改写 `portfolio/weights.parquet`；
- 不改写 `portfolio/returns.csv`；
- 不改写 `portfolio/cumulative_returns.csv`；
- 不改写基础 `metrics/performance_summary.csv`；
- 不实现 optimizer、execution simulation 或 paper/live trading。

### 4.2 N5 v2: Diagnostics-As-Constraints

目标：让 diagnostics 成为研究流程 gates，而不只是展示表。

做：

- risk / cost / capacity / neutralization policy ids；
- liquidity filter policy；
- ADV participation cap；
- single-name cap；
- industry cap；
- turnover cap 的研究合同；
- Factor Factory candidate risk/cost/capacity gates；
- diagnostic summary warning levels；
- policy validation：缺字段、非法阈值、覆盖率不足、约束过严。

不做：

- 不做优化求解器；
- 不做成交回放；
- 不写真实可交易规模承诺。

### 4.3 N5 v3: Rule-Based Portfolio Policy Layer

目标：把规则型组合构建从 evaluation runner 中抽象成可复用 policy，但仍不是优化器。

做：

- `PortfolioPolicySpec`；
- equal weight；
- top/bottom quantile；
- long-only top N；
- volatility scaling；
- industry neutral weight；
- beta neutral adjustment 的规则版；
- liquidity filter / ADV cap 影响 target weights；
- single-name cap / industry cap 影响 target weights；
- 简化 turnover cap；
- policy-generated weights 进入新的 evaluation run identity；
- 明确 `weight_policy_id`、`capacity_policy_id`、`risk_policy_id` 何时进入 combinations。

### 4.4 N5 v4: Optimizer Bridge

目标：为后续 optimizer 打接口，不要求在 N5 内落完整优化器。

做：

- optimizer input contract；
- objective / constraints schema；
- risk model input contract；
- expected return input contract；
- covariance input contract；
- transaction cost model input contract；
- constraint validation；
- optimizer output as target weights；
- optimizer diagnostics：binding constraints、turnover、exposure、cost estimate。

不做：

- 不接真实交易；
- 不接 broker / exchange；
- 不做订单级执行。

### 4.5 Out of N5

最终研究到交易链路应是：

```text
Signal / Prediction
  -> Evaluation
  -> Diagnostics
  -> PortfolioPolicy / Optimizer
  -> Simulation
  -> Paper Trading
  -> Live Trading
```

N5 长期职责停在 diagnostics、constraints、portfolio policy inputs 和 optimizer bridge。Simulation、paper/live trading、broker integration、portfolio accounting、controls 和 reconciliation 属于后续独立平台域。

## 5. N5 v1 架构组件

### 5.1 `DiagnosticsSpec`

`DiagnosticsSpec` 是 N5 v1 的主输入合同。

建议字段：

```text
source_run_dir
source_run_id
source_alias optional runtime-only
processed_panel_ref optional
timestamp_column
asset_column
policies
output_mode
metadata
```

规则：

- `source_run_dir` 应使用 repo-relative 或 runtime-resolved path，不写 public metadata；
- `source_alias` 是 API runtime identity，不写入 portable `manifest.json`；
- `processed_panel_ref` 可用于读取行业、市值、流动性、ADV、beta proxy 等列，但不得把本机绝对路径写入 public-downloadable metadata；
- 缺少 optional exposure/capacity columns 时，应输出 unavailable/warning，而不是静默填补。

### 5.2 `DiagnosticPolicy`

`DiagnosticPolicy` 是可注册、可版本化的诊断配置。

建议通用字段：

```text
policy_id
policy_family
policy_version
description
required_artifact_tables
optional_panel_columns
required_panel_columns
uses_future_return
changes_weights
changes_returns
output_tables
```

N5 v1 中 `changes_weights=false` 且 `changes_returns=false`。如果某个 policy 需要改变 weights 或 returns，它不应作为 v1 post-run diagnostics 执行。

### 5.3 `DiagnosticInputBundle`

`DiagnosticInputBundle` 统一读取：

- `manifest.json`；
- `config.json`；
- `lineage.json`；
- `summary.json`；
- `dimensions/combinations.csv`；
- `metrics/ic.csv`；
- `metrics/ic_summary.csv`；
- `metrics/grouped_returns.csv`；
- `metrics/group_spread_summary.csv`；
- `metrics/performance_summary.csv`；
- `portfolio/weights.parquet`；
- `portfolio/returns.csv`；
- `portfolio/cumulative_returns.csv`；
- `portfolio/cost_breakdown.csv` when available；
- `diagnostics/tradability_diagnostics.csv` when available；
- selected processed panel columns when approved and available.

低内存规则：

- 不为 metadata-only diagnostics 全量读取 `weights.parquet`；
- 对 weights、returns 和 panel join 应列裁剪；
- chart/table preview 继续通过 API bounded preview，不要求前端拉全表。

### 5.4 `DiagnosticRunner`

`DiagnosticRunner` 负责：

- 校验 source run 是 schema `0.6` evaluation-set；
- 校验 `manifest.display` 当前有效；
- 解析 policies；
- 建立 input bundle；
- 执行 post-run diagnostics；
- 写出新增 tables；
- 更新 `manifest.display`；
- 更新 report / summary metadata；
- 保持 config / lineage 脱敏。

失败规则：

- source run stale 或缺少 required artifacts：fail fast；
- required panel columns 缺失：对应 policy fail 或输出 unavailable，取决于 policy 约束；
- optional columns 缺失：输出 warning，不静默补字段；
- path 超出 run dir 或 configured run roots：拒绝；
- private absolute path 不写入 public-downloadable metadata。

## 6. N5 v1 Artifact 合同

所有组合相关表必须带 `combination_id`。所有用户可见表必须进入 `manifest.display.tables`，并声明 preview、download、role 和 combination filtering 语义。

### 6.1 `dimensions/regime_definitions.csv`

粒度：每个 regime rule 一行。

建议字段：

```text
regime_id
regime_family
rule_name
input_column
operator
threshold
lookback
source
uses_future_return
description
status
warning
```

输入依赖：

- processed panel 或 returns-derived historical state；
- 不允许使用未来收益定义 regime。

验收：

- `uses_future_return` 对正式规则必须为 `false`；
- `regime_id` 唯一；
- 缺少输入列时该 regime rule 标记 unavailable 或 fail fast。

### 6.2 `breakdowns/regime_performance.csv`

粒度：`combination_id x regime_id`。

建议字段：

```text
combination_id
regime_id
n_days
n_assets_observed
mean_ic
mean_rankic
long_return
short_return
long_short_return
gross_return
net_return
mean_turnover
total_cost
max_drawdown
warning
```

输入依赖：

- `dimensions/combinations.csv`；
- IC / grouped returns / portfolio returns / turnover / cost tables；
- regime definitions。

验收：

- 每行必须匹配有效 `combination_id` 和 `regime_id`；
- long-only run 中 short-side 字段必须为 0、空值或明确 unavailable，不得伪造真实做空表现。

### 6.3 `dimensions/cost_profiles.csv`

粒度：每个 cost profile 一行。

建议字段：

```text
cost_profile_id
model_type
flat_bps
linear_bps
piecewise_thresholds
piecewise_bps
quadratic_lambda
commission_bps
stamp_tax_bps
transfer_fee_bps
slippage_bps
spread_bps
impact_lambda
side_asymmetry
description
```

输入依赖：

- user-approved local config 或 built-in baseline profiles；
- A 股费用假设 metadata。

验收：

- 负成本拒绝；
- `model_type` 必须是受控枚举；
- 买卖方向不对称费用必须明确表达。

### 6.4 `diagnostics/cost_sensitivity.csv`

粒度：`combination_id x cost_profile_id`。

建议字段：

```text
combination_id
cost_profile_id
gross_return
baseline_net_return
diagnostic_net_return
total_cost
incremental_cost
mean_turnover
break_even_cost_bps
cost_adjusted_sharpe_like
warning
```

输入依赖：

- portfolio returns；
- turnover；
- cost breakdown when available；
- cost profiles。

验收：

- 不改写基础 `portfolio/returns.csv`；
- sensitivity 结果必须标记为 diagnostic；
- 成本增加导致 diagnostic net return 反向改善时必须输出 warning 或解释原因。

### 6.5 `diagnostics/risk_exposure.csv`

粒度：`combination_id x timestamp x exposure_name`，大样本可先输出 summary 粒度并记录 aggregation method。

建议字段：

```text
combination_id
timestamp
exposure_name
exposure_type
long_exposure
short_exposure
net_exposure
gross_exposure
coverage_rate
aggregation_method
status
warning
```

输入依赖：

- `portfolio/weights.parquet`；
- processed panel exposure columns，例如 industry、market beta proxy、size、style、volatility、liquidity。

验收：

- long-only run 不应产生真实 short exposure；
- 缺少行业、市值、beta 或流动性列时输出 unavailable/warning；
- exposure diagnostics 不覆盖原始 returns 或 IC。

### 6.6 `diagnostics/neutralized_signal_diagnostics.csv` optional

粒度：`combination_id x neutralization_policy_id`。

建议字段：

```text
combination_id
neutralization_policy_id
neutralized_by
n_obs
coverage_rate
mean_ic
mean_rankic
delta_mean_ic
delta_mean_rankic
status
warning
```

输入依赖：

- feature / label panel；
- industry、size、beta、volatility 或 liquidity columns；
- original IC summary。

验收：

- 只作为 signal diagnostics；
- 不覆盖原始 IC / RankIC；
- v1 不要求完整 neutralized return path。

### 6.7 `diagnostics/capacity_diagnostics.csv`

粒度：`combination_id x timestamp` 或 `combination_id x liquidity_bucket` summary。

建议字段：

```text
combination_id
timestamp
liquidity_bucket
adv_proxy
target_notional_proxy
participation_rate_proxy
adv_cap
over_adv_cap_weight
blocked_weight_rate
capacity_warning
status
```

输入依赖：

- `portfolio/weights.parquet`；
- processed panel liquidity columns，例如 amount、avg_amount_20d、turnover、market cap；
- tradability diagnostics when available。

验收：

- 字段名不得暗示真实管理规模；
- 输出解释为 capacity proxy，不是成交承诺；
- 缺少 ADV/liquidity columns 时输出 unavailable/warning。

### 6.8 Deferred v1 artifacts

这些进入 N5 路线图，但不作为 v1 必做：

```text
diagnostics/stress_test_summary.csv
diagnostics/capacity_curve.csv
diagnostics/crowding_proxy.csv
```

原因：

- stress scenarios 需要事件窗口和 regime definition 更稳定；
- capacity curve 容易被误读为真实容量承诺；
- crowding proxy 需要更明确数据来源和解释边界。

## 7. `manifest.display` 合同

N5 v1 不让前端写死每张诊断表。后端应在 `manifest.display.tables[]` 中声明诊断 role。

建议 display table metadata：

```text
key
label
path
format
role
preview
download_item
join_key
supports_combination_filter
```

建议 `role` 枚举：

```text
regime_dimension
regime_breakdown
cost_dimension
cost_diagnostic
risk_diagnostic
neutralization_diagnostic
capacity_diagnostic
```

前端规则：

- generic diagnostic table sections 读取 role 分组；
- 组合切换时，只刷新声明 `supports_combination_filter=true` 的表；
- CSV/detail table 使用 display preview row limit；
- chart table 才允许 bounded preview pagination 拼接完整当前组合时间轴；
- 前端不得重新计算 risk、cost、capacity、neutralization 或 regime results。

## 8. A 股假设边界

N5 v1 必须记录但不仿真：

- 涨跌停；
- 停牌；
- T+1；
- 印花税；
- 过户费或费用假设；
- 收盘成交假设；
- 小盘流动性；
- 无法成交样本处理；
- `long_short` 只是因子分组诊断口径。

这些应进入：

- diagnostics report text；
- sanitized `config.json` / `lineage.json` extension metadata；
- cost and capacity diagnostics warnings；
- workbench copy。

不得进入：

- broker/exchange API；
- order model；
- fill model；
- paper/live trading；
- secrets or credentials。

## 9. Verification Strategy

### 9.1 Unit and contract tests

Python tests:

- `DiagnosticsSpec` validation；
- `DiagnosticPolicy` validation；
- source run schema `0.6` validation；
- stale run rejection；
- display manifest validation for new diagnostic roles；
- cost profile negative / invalid model rejection；
- regime `uses_future_return=true` rejection for formal rules；
- missing optional exposure columns produce warnings；
- missing required columns fail or mark unavailable by policy；
- private absolute paths not written to public metadata。

API tests:

- generic table preview exposes N5 tables through `manifest.display`；
- direct downloads reject non-display items；
- ZIP presets include only display-approved items；
- `(source_alias, run_id)` identity preserved。

Frontend tests:

- generic diagnostic sections render by role；
- combination switching refreshes combination-aware diagnostic previews；
- frontend does not infer latest rows from non-exact preview counts；
- download picker uses display metadata only。

### 9.2 Private acceptance

N5 v1 private acceptance should use current small, stable evidence:

- `reports/private_real/factor_evaluation/n7_eval_2023_2025_v1`；
- `reports/private_real/factor_evaluation/n7_eval_2023_2025_rev120_v1`。

Do not expand to a single 32-combination 2023-2025 full run until low-memory weights/remap writing is designed.

### 9.3 Verification commands

Narrow first:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api
conda run -n analysis python -m pytest -q tests/evaluations tests/markets/implementations/a_share/evaluation
conda run -n analysis python -m ruff check src tests
Push-Location web; try { npm test -- --run --maxWorkers=1; npm run lint; npm run build } finally { Pop-Location }
git diff --check
```

Browser acceptance is required only after workbench behavior changes. API-backed workbench acceptance must run both FastAPI and Vite; an API-offline state is not acceptance evidence for N5.

## 10. Implementation Plan Outline

The implementation plan should split the work into reviewable increments:

1. Add diagnostics contracts and validation.
2. Add source evaluation run input bundle.
3. Add cost profile and cost sensitivity diagnostics.
4. Add regime definitions and regime performance diagnostics.
5. Add risk exposure diagnostics.
6. Add capacity diagnostics.
7. Update artifact writer / manifest display / loader validation.
8. Update API preview/download tests.
9. Add generic frontend diagnostic sections.
10. Run private acceptance on current canonical / `rev_120d` shard.
11. Update docs and code maps after implementation.

Do not start with frontend UI. The artifact contract and backend validation must land first.

## 11. Review Questions

These are the only intentional review points before implementation planning:

1. Should `neutralized_signal_diagnostics.csv` be v1 required or v1 optional?
2. Should v1 write diagnostics into the existing evaluation run directory, or create a sibling diagnostics extension directory linked from the evaluation run?
3. Which exposure columns are approved as first-class A 股 v1 inputs: industry, market cap, volatility, liquidity, beta proxy?
4. Should capacity diagnostics use timestamp-level rows first, bucket summary first, or both?

Default recommendation:

- `neutralized_signal_diagnostics.csv` is v1 optional;
- write into the existing evaluation run directory for v1;
- use industry, market cap, volatility and liquidity first; beta proxy only if already available or cheaply derived without external data;
- output capacity bucket summary first, with timestamp-level detail only if memory remains acceptable.
