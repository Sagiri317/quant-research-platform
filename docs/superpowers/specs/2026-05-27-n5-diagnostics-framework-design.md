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
  -> 读取 base evaluation payload + optional processed panel + assumptions
  -> 参与生成 schema 0.7 risk / regime / cost / capacity diagnostics tables
```

用户心智应是：

```text
一个 evaluation run
  -> 基础 signal / portfolio 评价
  -> N5 diagnostics 解释收益来源、风险暴露、成本侵蚀、容量压力和 A 股假设边界
```

这里的 `N5 diagnostics` 只是在规划文档中标记阶段目标；产品 UI、API、artifact、
display keys 和 tests 一律使用 `diagnostics` 命名。

工程心智应是：

```text
DiagnosticsSpec
  source EvaluationSpec output
  input bundle
  diagnostic policies
  artifact specs
  manifest.display exposure
```

N5 应支持《打开量化投资的黑箱》第 1-6 章提炼出的平台要求：signal evaluation、risk diagnostics、cost diagnostics、capacity diagnostics、基础 portfolio construction assumptions、A 股约束和 artifact/API/workbench 合同化。

Black Box Ch1-6 mapping：

| 平台要求 | schema `0.7` 承载位置 |
|---|---|
| signal evaluation：IC、RankIC、分组收益、long-short 诊断 | `metrics/ic.csv`、`metrics/grouped_returns.csv`、`metrics/evaluation_metrics_summary.csv` |
| risk diagnostics：beta、行业、市值、风格、波动率、流动性、中性化诊断、drawdown | `diagnostics/diagnostic_metrics.csv`，按需 `diagnostics/scenario_metrics.csv`、`diagnostics/exposure_capacity_summary.csv`、`diagnostics/portfolio_risk_summary.csv` |
| cost diagnostics：税费、佣金、滑点、impact、break-even 和 sensitivity | `portfolio/returns.csv` 的 baseline cost fields、`dimensions/diagnostic_policies.csv` 的 cost assumptions、`diagnostics/diagnostic_metrics.csv` / `scenario_metrics.csv` |
| capacity diagnostics：ADV / liquidity participation proxy | `diagnostics/diagnostic_metrics.csv`，按需 `diagnostics/exposure_capacity_summary.csv` |
| portfolio construction assumptions | `run_context.json`、`dimensions/diagnostic_policies.csv`，按需 `dimensions/portfolio_assumptions.csv` |
| A 股约束和执行假设边界 | `run_context.json.config.a_share_assumptions`、tradability diagnostics、cost/capacity warnings、report copy |

命名边界：

- `N5` 只是 roadmap / planning label，可以出现在本设计文档、路线图和阶段说明中；
- runtime code、API payload、artifact file names、frontend copy、test names 和 display keys 不应使用 `N5` / `n5_`；
- 运行时统一使用 `diagnostics`、`schema 0.7`、`Diagnostic*`、`diagnostic_*` 等术语。

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
  -> EvaluationSpec 或未来 PortfolioPolicySpec

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
- ADV / liquidity participation proxy 和 liquidity bucket；
- stress scenario definitions；
- 后续 capacity curve 和 crowding proxy definitions；v1 不包含这些 policy/output；
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

### 3.4 N5 v1 升级为 schema `0.7`，并生成完整新 run

N5 v1 不再把新增 diagnostics 静默塞进历史 artifact 合同。新增 dimension tables、
diagnostic roles、conditional artifacts 和 no-combination-filter 语义后，同版本含义会分裂。
因此 N5 v1 应升级为：

```text
artifact_type = evaluation_set
artifact_schema_version = 0.7
```

`0.7` 仍是完整 evaluation artifact，不是独立 `diagnostics_set`。它是新的
evaluation-set 合同，也是 N5 之后唯一的平台 artifact / API / workbench 合同。
历史 generated runs 不进入 N5 平台合同，不设计兼容 UI、运行时兼容、正式迁移器或长期比较组件。

默认输出策略是从 `EvaluationSpec + DiagnosticsSpec` 重新生成一个完整新 run：

```text
EvaluationSpec + DiagnosticsSpec
  -> regenerate base evaluation outputs
  -> generate diagnostics outputs
  -> write schema 0.7 evaluation_set run
  -> record diagnostics policy fingerprint
```

验收重点是 schema `0.7` run 自身的字段、状态、输入依赖、诊断数值和 artifact/API/workbench
合同正确。若本地临时需要对照历史结果，应使用一次性脚本或人工检查；该脚本不进入
QRP artifact schema、API、workbench、正式测试面或 implementation plan。

不做：

- 不提供历史 artifact 的 N5 API 兼容模式；
- 不把历史 run 原地升级成 schema `0.7`；
- 不实现 copy-on-write `DiagnosticsUpgradeRunner`。如果未来需要旧 run 原地升级或旁路扩展，
  应另写 migration / extension 设计。

不在 v1 新增独立 `artifact_type="diagnostics_set"`，原因：

- 当前 API-visible identity 是 `(source_alias, run_id)`；
- evidence workbench、download packaging 和 preview route 都已围绕 evaluation-set display contract 收敛；
- 独立 diagnostics artifact 会引入跨 artifact identity、run linking 和 stale extension 管理问题。

未来如果 diagnostics 需要多次追加、跨 evaluation run 复用或独立发布，可以设计独立
diagnostics artifact type，但不属于 N5 v1。

## 4. 版本路线

### 4.1 N5 v1: Post-Run Diagnostics Framework

目标：建立 `DiagnosticsSpec / DiagnosticPolicy / DiagnosticRunner`，并生成第一批 post-run diagnostics artifacts。

做：

- `DiagnosticsSpec`；
- `DiagnosticInputBundle`；
- `DiagnosticPolicy`；
- `DiagnosticRunner`；
- schema `0.7` evaluation-set artifact specs；
- `dimensions/diagnostic_policies.csv`；
- `diagnostics/diagnostic_summary.csv`；
- `diagnostics/diagnostic_metrics.csv`；
- `input_quality/coverage_summary.csv`；
- conditional `dimensions/portfolio_assumptions.csv`；
- conditional `diagnostics/tradability_diagnostics.csv`；
- conditional `diagnostics/tradability_summary.csv`；
- conditional `diagnostics/scenario_metrics.csv`；
- conditional `diagnostics/exposure_capacity_summary.csv` 和 `diagnostics/portfolio_risk_summary.csv`
  materialized views；
- optional `diagnostics/diagnostic_observations.parquet`；
- A 股假设 metadata 和 report 解释；
- generic diagnostic table display。

不做：

- `DiagnosticRunner` 不 post-hoc mutate staging base tables；
- schema `0.7` `ArtifactWriter` 会按新合同写出 base evaluation tables，包括
  `portfolio/returns.csv` 和 `metrics/evaluation_metrics_summary.csv`；
- 不让 diagnostics policy 改写 weights、returns 或基础 evaluation summary 的语义；
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
- `DiagnosticGateSpec` / `GateResult`；
- Factor Factory candidate risk/cost/capacity gates；
- diagnostic summary warning levels；
- `enforcement_mode=report_only` gates：只输出 pass / warn / fail，不改变 universe、weights 或 returns；
- policy validation：缺字段、非法阈值、覆盖率不足、约束过严。

不做：

- 不做优化求解器；
- 不做成交回放；
- 不让 gates 直接改变 universe、target weights、turnover path 或 net returns；
- 不写真实可交易规模承诺。

### 4.3 N5 v3: Rule-Based Portfolio Policy Layer

目标：把规则型组合构建从 evaluation runner 中抽象成可复用 policy，但仍不是优化器。

做：

- `PortfolioPolicySpec`；
- module ownership：建议放在 `src/quant_research_platform/research/portfolios/`
  或后续独立 policy layer，不放进 diagnostics 模块；
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
- portfolio policy 负责 materialize target weights；`EvaluationSpec` 只消费已生成权重或轻量
  weight config，不在 diagnostics 中隐式改权重；
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

行业里的完整研究到交易概念链路通常是：

```text
Signal / Prediction
  -> Evaluation
  -> Diagnostics
  -> PortfolioPolicy / Optimizer
  -> Simulation
  -> Paper Trading
  -> Live Trading
```

上图只是概念边界说明，不是 QRP 当前 roadmap 承诺。N5 长期职责停在 diagnostics、
constraints、portfolio policy inputs 和 optimizer bridge。Simulation、paper/live trading、
broker integration、portfolio accounting、controls 和 reconciliation 属于后续独立平台域。

## 5. N5 v1 架构组件

### 5.1 `DiagnosticsSpec`

`DiagnosticsSpec` 是 N5 v1 的主输入合同。

字段：

```text
evaluation_payload_ref
source_alias optional runtime-only
processed_panel_ref optional
timestamp_column
asset_column
policies
materialized_views
output_mode
metadata
```

入口层对象与 normalized spec 分开：

- `DiagnosticsRequest`：API / CLI / Pipeline Builder 提交的入口对象，包含 `enabled`、
  `policy_set_id`、`toggles` 和 `materialized_views`；
- `DiagnosticsSpec`：后端归一化后的执行合同，包含展开后的 `policies`、输入引用和输出请求；
- 所有入口必须先把 request normalize 成 `DiagnosticsSpec`，不得绕过 policy registry。

规则：

- `evaluation_payload_ref` 指向当前生成中的 schema `0.7` base evaluation payload；
- `source_alias` 是 API / run-index runtime identity，不写入 portable `manifest.json`、
  `run_context.json` 或 downloadable metadata；`run_context.json.config.diagnostics_spec`
  只能包含脱敏后的 normalized diagnostics fields；
- `processed_panel_ref` 可用于读取行业、市值、流动性、ADV、beta proxy 等列；它必须解析到
  approved input roots，不要求位于 run dir；
- artifact output、display download 和 ZIP packaging 只能解析 run dir 内的 artifact-relative paths；
- approved input roots 与 artifact output roots 必须分开配置和校验，不得为了读取私有输入扩大可下载 artifact root；
- `materialized_views` 是机器可读的 conditional output 请求；conditional 表必须同时被
  `DiagnosticPolicy.output_tables` 允许，并被 `DiagnosticsSpec.materialized_views[]` 请求，才可以生成；
- 缺少 optional exposure/capacity columns 时，应输出 unavailable/warning，而不是静默填补；
- diagnostics output metadata 必须记录 diagnostics policy ids、生成时间和 policy fingerprint。

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
uses_future_information
timing_semantics
decision_usable
changes_universe
changes_signal
changes_weights
changes_returns
output_tables
missing_input_mode
```

N5 v1 `diagnostic_only` policy 必须满足 `changes_universe=false`、`changes_signal=false`、
`changes_weights=false` 且 `changes_returns=false`。任一为 `true` 时，不得由
`DiagnosticRunner` 执行，必须进入 `evaluation_axis`、`PortfolioPolicySpec` 或 simulation policy。

`missing_input_mode` 是缺输入时的机器可读处理模式，允许值：

```text
fail_fast
emit_unavailable
emit_partial
skip_with_summary
```

含义：

- `fail_fast`：必需输入缺失时整个 run 失败；
- `emit_unavailable`：生成状态行，指标值不可用；
- `emit_partial`：可计算部分继续输出，其余写 `partial` / warning；
- `skip_with_summary`：不生成对应 conditional 表，但在 `diagnostic_summary.csv` 写 skipped 行。

Regime policy 必须额外声明 timing semantics：

```text
ex_post_analysis
  may use hindsight market labels for explanation
  decision_usable=false

ex_ante_policy
  must use point-in-time observable inputs
  decision_usable may be true only after leakage checks
```

约束：

- `decision_usable=true` 时，`timing_semantics` 必须为 `ex_ante_policy`；
- `decision_usable=true` 时，`uses_future_return` 和其他 future information 标记必须为 `false`；
- `ex_post_analysis` 可以用于研究解释，但不得用于调仓、因子切换或组合权重调整。

### 5.3 `DiagnosticInputBundle`

`DiagnosticInputBundle` 统一读取当前生成中的 schema `0.7` base evaluation staging payload，
以及用户已批准的本地 processed panel 列。它不读取同一个最终 run 目录中尚未生成的
diagnostics output，避免输入/输出自引用。

生命周期：

```text
EvaluationRunner
  -> BaseEvaluationPayload in memory or staging directory
  -> DiagnosticRunner reads BaseEvaluationPayload + approved panel columns
  -> ArtifactWriter writes final run directory atomically
  -> Loader/API/workbench read only the final complete run directory
```

staging payload 可以是内存对象或临时目录，但必须满足：

- 不作为 API-visible run；
- 不写入 `manifest.display`；
- 不暴露 direct download / ZIP download；
- 只包含 diagnostics 需要读取的 base evaluation tables 和 sanitized context；
- final run directory 写完并通过 manifest validation 后才进入 run index。

`BaseEvaluationPayload` 必须提供的逻辑输入：

- sanitized base evaluation config/context；
- combinations table；
- IC / RankIC results；
- grouped returns / long-short diagnostic results；
- evaluation metrics summary；
- portfolio returns；
- optional portfolio weights when available and required by a policy；
- optional cost breakdown when available；
- selected processed panel columns when approved and available.

`DiagnosticInputBundle` 不读取同一个最终 run 目录里的 `manifest.json`、`summary.json`、
`run_context.json` 或 diagnostics output。上述文件只由 `ArtifactWriter` 在 final run
directory 写出，并由 loader / API / workbench 在 run 完成后读取。

低内存规则：

- 不为 metadata-only diagnostics 全量读取 `weights.parquet`；
- 对 weights、returns 和 panel join 应列裁剪；
- chart/table preview 继续通过 API bounded preview，不要求前端拉全表。

### 5.4 `DiagnosticRunner`

`DiagnosticRunner` 负责：

- 接收 schema `0.7` evaluation payload 和 `DiagnosticsSpec`；
- 校验 schema `0.7` base evaluation payload 满足当前 diagnostics 合同；
- 解析 policies；
- 建立 input bundle；
- 执行 post-run diagnostics；
- 产出 validated diagnostics table objects、summary metadata 和 display intent；
- 调用 schema `0.7` `ArtifactWriter` 写出完整 artifact tables、`manifest.display`、
  `summary.json`、`run_context.json` 和 report；
- 不直接绕过 `ArtifactWriter` 写 run directory。

失败规则：

- base evaluation payload 缺少 schema `0.7` required fields：fail fast；
- required panel columns 缺失：按 policy 的 `missing_input_mode` 执行；
- optional columns 缺失：输出 warning，不静默补字段；
- artifact output path 超出 run dir 或 configured output roots：拒绝；
- input path 不在 approved input roots 内：拒绝；
- private absolute path 不写入 public-downloadable metadata。

## 6. Schema `0.7` Artifact 合同

### 6.1 发散方案与收敛结论

本节是 N5 v1 的最新 artifact 合同。发散时考虑过四种方案：

| 方案 | 结论 | 原因 |
|---|---|---|
| 继续沿用历史 schema，只追加 diagnostics CSV | 放弃 | 历史 schema 是强合同；新增 no-combination-filter dimension 表和 diagnostic roles 会让同版本语义分裂。 |
| 新建 `artifact_type="diagnostics_set"` | 暂缓 | 会引入跨 artifact identity、stale extension、run linking 和下载打包复杂度。 |
| 每个诊断一张表，例如 regime/cost/risk/capacity/neutralized 分散落表 | 放弃 | 文件数量膨胀，API/workbench 难做通用展示，很多字段同质。 |
| schema `0.7` 仍为完整 `evaluation_set`，用少量语义表承载 N5 | 采用 | 保留 evaluation-set identity，但不继承历史字段/API 兼容；把诊断规则、诊断总览、情景表现和暴露摘要制度化。 |

收敛原则：

- `combination_id` 只表示会改变基础 evaluation 结果身份的维度；
- post-run diagnostics 不膨胀 `combination_id`；
- 改变 universe、weights、主收益路径或正式 baseline cost 的 policy 不能伪装成 diagnostics；
- 所有用户可见结果必须进入 `manifest.display`；
- 诊断表必须有 machine-readable status 和 warning，不能只写在 report；
- capacity 只能写成 proxy / diagnostic，不写真实可交易规模承诺。

### 6.2 术语表

| 术语 | 中文解释 |
|---|---|
| artifact | 一次 run 生成的可审计结果目录，包含表、元数据和报告。 |
| schema version | artifact 合同版本，定义目录、文件、字段和验证规则。 |
| evaluation_set | QRP 当前 evaluation run 的 artifact 类型；schema `0.7` 仍使用该类型。 |
| combination_id | 一个 run 内的组合身份键；只对应真正改变基础 evaluation 结果的维度。 |
| DiagnosticPolicy | 诊断规则或诊断假设，例如 regime 切片、成本情景、风险暴露、容量 proxy。 |
| policy_family | policy 的类别，例如 `regime`、`cost`、`risk`、`capacity`、`tradability`、`neutralization`、`market_segment`、`diagnostics_contract`。 |
| identity_scope | policy 是否影响结果身份的枚举。用于判断它是否应进入 `combination_id`。 |
| diagnostic_only | 只解释已有结果，不改变 universe、weights 或主收益。 |
| evaluation_axis | 会改变基础 evaluation 结果，应作为 EvaluationSpec 维度进入新 run identity。 |
| portfolio_policy | 会改变 target weights，应进入后续 PortfolioPolicySpec。 |
| simulation_policy | 会改变成交、fill 或 realized returns，属于后续 simulation，不属于 N5。 |
| scenario | 情景或切片，例如不同 cost model、market regime、benchmark relative view。 |
| exposure | 暴露，例如 beta、行业、市值、风格、波动率、流动性、ADV participation proxy。 |
| proxy | 代理指标；表示近似诊断，不是可交易规模或真实成交承诺。 |
| decision_usable | 是否允许用于事前决策。事后解释型 regime 必须为 `false`。 |
| timing_semantics | 时间语义，说明是否事后回看、是否点时可观测。 |
| status | policy、input quality 或 base evaluation summary 状态，例如 available、partial、unavailable、skipped、error。 |
| diagnostic_status | diagnostics fact table 状态；diagnostic fact tables 使用它而不是 `status`。 |
| status_reason | 状态原因，用机器可读短码解释为什么 available、partial、unavailable、skipped 或 error。 |
| required_inputs_met | 必需输入是否满足；用于防止缺字段时静默硬算。 |
| warning | 人读警告；解释数据不足、A 股限制、未来函数风险或 proxy 边界。 |

### 6.2.1 字段合同通用规则

每个 schema `0.7` 表在代码实现时必须注册机器可读字段合同，不能只依赖本文的人读字段列表。

字段合同至少包含：

```text
name
dtype
nullable
required
enum_values
unit
unique_key_group
sort_key
description
```

通用规则：

- required 表缺 required column 必须 fail validation；
- conditional 表一旦生成，必须按同一字段合同完整校验；
- nullable 字段必须显式声明，不能用空字符串冒充 unknown；
- enum 字段必须有白名单，例如 status、policy_family、identity_scope、timing_semantics；
- unit 必须区分 decimal return、bps、count、rate、currency 和 proxy；
- 每张 fact table 必须声明唯一键或允许重复的原因；
- 每张时间序列表必须声明稳定 sort key；
- 字段说明进入 schema registry / display metadata，供 API 和 workbench 读取。

### 6.3 顶层文件

#### `manifest.json`

粒度：每个 run 一个文件。

内容：

- `artifact_type="evaluation_set"`；
- `artifact_schema_version="0.7"`；
- `artifacts`：所有 artifact-relative paths；
- `display`：API、download、workbench 的唯一展示合同；
- 不写本机私有绝对路径。

验收：

- `manifest.display.schema_version` 必须等于 `0.7`；
- 每个用户可见表都必须在 `manifest.display.tables[]` 中声明；
- display download item 必须指向 manifest 声明的安全相对路径；
- 不允许未知 artifact group 被 API 静默接受。

#### `summary.json`

粒度：每个 run 一个文件。

内容：

- run 摘要：run_id、project、kind、data_stage、n_rows_evaluated、n_combinations；
- base evaluation 摘要；
- diagnostics 启用状态；
- schema `0.7` 标记；
- 可公开的 aggregate warnings。

用途：

- run list 和 lightweight index；
- 快速判断 run 是否包含 schema `0.7` diagnostics 合同；
- 不承载完整配置，也不承载输入 fingerprint 细节。

required keys / types：

- `run_id`: string；
- `project`: string；
- `kind`: string；
- `data_stage`: string；
- `n_rows_evaluated`: integer；
- `n_combinations`: integer；
- `base_evaluation`: object；
- `diagnostics`: object；
- `artifact_schema_version`: string，必须为 `"0.7"`；
- `warnings`: array of strings。

`summary.json.run_id` 是 lightweight identity，必须与 `run_context.json.provenance.run_id`
一致；validator 必须检查二者相等。

#### `run_context.json`

粒度：每个 run 一个文件。

中文解释：`run_context.json` 是“本次打算怎么跑”和“结果实际从哪里来”的脱敏运行上下文。
它把原先分开的 `config.json` 与 `lineage.json` 物理合并，但逻辑上保留两个 section：
`config` 和 `provenance`。

内容：

- `config.evaluation_spec`：EvaluationSpec 的脱敏字段，例如 feature、label、kernel、
  portfolio mode、baseline cost config；
- `config.diagnostics_spec`：DiagnosticsSpec 的脱敏字段，例如 enabled policy ids、
  market context inputs、exposure columns；
- `config.a_share_assumptions`：涨跌停、停牌、T+1、印花税、收盘成交假设、小盘流动性处理；
- `provenance.input_panel_fingerprint` 或脱敏数据版本标识；
- `provenance.run_id`；
- `provenance.diagnostic_policy_fingerprint`；
- `provenance.generated_at`；
- `provenance.generator`：generator module 和 code version；
- `provenance.staging_payload_fingerprint`。

验收：

- 能审计 schema `0.7` run 的输入、policy 和生成器；
- 不依赖目录名猜测来源；
- API 展示配置时可以只暴露 `config` 的安全子集；
- direct download / ZIP packaging 必须使用同一份脱敏文件；
- 不写私有绝对路径、API key、credential、cookie 或 token。

required keys / types：

- `config`: object；
- `config.evaluation_spec`: object；
- `config.diagnostics_spec`: object；
- `config.a_share_assumptions`: object；
- `provenance`: object；
- `provenance.run_id`: string；
- `provenance.input_panel_fingerprint`: string；
- `provenance.diagnostic_policy_fingerprint`: string；
- `provenance.generated_at`: ISO-8601 string；
- `provenance.generator`: object；
- `provenance.staging_payload_fingerprint`: string。

#### `report.md`

状态：optional。

粒度：每个 run 一个文件。

内容：

- 人读结论；
- A 股限制解释；
- long-short 诊断口径提示；
- capacity proxy 边界提示；
- diagnostics warning 摘要。

约束：

- report 不能成为唯一数据源；
- 所有用于 API/workbench 的结果必须存在于 machine-readable table。
- 未生成 `report.md` 时，不得进入 `manifest.display.downloads.items[]` 或 ZIP presets；
  API evidence payload 应返回 empty/absent report state，并使用 `diagnostic_summary.csv` 展示状态。

### 6.4 `dimensions/` 定义表

#### `dimensions/combinations.csv`

状态：required。

粒度：每个 `combination_id` 一行。

内容：

- combination_id；
- feature；
- label；
- kernel；
- portfolio_mode；
- baseline cost config id；
- universe id when the run explicitly evaluates multiple universes；
- weight policy id when it changes target weights；
- identity_payload_json。

不得放入：

- post-run cost sensitivity policy；
- ex-post regime split；
- risk exposure diagnostic policy；
- capacity proxy diagnostic policy；
- neutralized IC diagnostic-only policy。

验收：

- `combination_id` 唯一；
- 同一 run 内的组合语义可复现；
- 额外身份维度必须进入 `identity_payload_json`，不得动态增加未声明列；
- `diagnostic_only` policy 不得进入 `combinations.csv` 造成组合膨胀。

#### `dimensions/diagnostic_policies.csv`

状态：required。

粒度：每个 diagnostic policy 一行。

作用：统一承载 regime、cost、risk、capacity、neutralization、market segment、benchmark comparison 和 A 股执行假设的定义，替代分散的 `regime_definitions.csv`、`cost_profiles.csv` 等多张定义表。

显式字段：

```text
policy_id
policy_family
identity_scope
timing_semantics
decision_usable
uses_future_return
uses_future_information
changes_universe
changes_signal
changes_weights
changes_returns
status
status_reason
warning
policy_payload_json
missing_input_mode
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `policy_id` | policy 唯一标识。 |
| `policy_family` | policy 类别，例如 `regime`、`cost`、`risk`、`capacity`、`tradability`、`neutralization`、`benchmark`、`market_segment`、`a_share_assumption`、`diagnostics_contract`。 |
| `identity_scope` | 是否影响 run 身份；取值为 `diagnostic_only`、`evaluation_axis`、`portfolio_policy`、`simulation_policy`。 |
| `timing_semantics` | 时间语义；取值如 `ex_post_analysis`、`ex_ante_policy`、`static_assumption`。 |
| `decision_usable` | 是否允许用于事前决策。 |
| `uses_future_return` | 是否使用未来收益定义标签或 regime。 |
| `uses_future_information` | 是否使用点时不可见信息。 |
| `changes_universe` | 是否改变股票池或 eligible mask。 |
| `changes_signal` | 是否改变原始因子信号。 |
| `changes_weights` | 是否改变 target weights。 |
| `changes_returns` | 是否改变基础 returns / net returns 主路径。 |
| `status` | policy 本身是否可用。 |
| `status_reason` | policy 可用性状态原因，例如缺少输入、未来函数风险、被配置跳过。 |
| `warning` | 缺字段、未来函数、诊断边界等警告。 |
| `policy_payload_json` | policy 的版本、名称、模型类型、参数、必需表/列、可选列、输出表和描述。使用 JSON 字符串避免为每类 policy 增加稀疏列。 |
| `missing_input_mode` | 缺输入处理模式：`fail_fast`、`emit_unavailable`、`emit_partial` 或 `skip_with_summary`。 |

关键规则：

- `identity_scope=diagnostic_only` 时，policy 不进入 `combination_id`；
- `identity_scope=evaluation_axis` 时，应在 EvaluationSpec 中重跑新 run，并进入组合身份；
- `identity_scope=portfolio_policy` 时，不属于 N5 v1 post-run diagnostics，应推迟到 N5 v3；
- `identity_scope=simulation_policy` 时，不属于 N5，应推迟到 simulation；
- `decision_usable=true` 必须满足 `timing_semantics=ex_ante_policy` 且不使用未来信息；
- 成本 policy 如果只做 sensitivity，是 `diagnostic_only`；如果成为正式 baseline net return，是 `evaluation_axis`；如果改变成交或权重，是 `portfolio_policy` 或 `simulation_policy`。

disabled run 规则：

- schema `0.7` 即使 `diagnostics.enabled=false`，也必须写出 required diagnostics contract files；
- `dimensions/diagnostic_policies.csv` 至少包含一行 reserved policy：
  `policy_id="__diagnostics_disabled__"`、`policy_family="diagnostics_contract"`、
  `identity_scope="diagnostic_only"`、`status="disabled_by_config"`；
- `diagnostics/diagnostic_summary.csv` 对每个 `combination_id` 写一行 disabled summary；
- `diagnostics/diagnostic_metrics.csv` 可以只有表头、无 metric rows；
- disabled run 不生成 conditional materialized views 或 optional detail files。

#### `dimensions/portfolio_assumptions.csv`

状态：conditional。

触发条件：必须同时由 `DiagnosticPolicy.output_tables` 允许，并由
`DiagnosticsSpec.materialized_views[]` 请求。它通常只在组合间组合假设不同，或
workbench 需要按 `combination_id` 展示机器可读组合假设时生成。若整次 run 只有一套组合假设，
放入 `run_context.json.config` 和 `dimensions/diagnostic_policies.csv` 的 policy payload 即可，
不强制展开成表。

粒度：`combination_id x portfolio_assumption_id`。

作用：机器可读地记录组合构建和收益解释假设，避免这些信息只存在于 report prose。

字段：

```text
combination_id
portfolio_assumption_id
weight_source
weight_policy_id
rebalance_frequency
signal_to_weight_rule
portfolio_diagnostic_mode
long_short_executable
gross_exposure_target
net_exposure_target
cash_assumption
leverage_assumption
trade_price_assumption
trade_timing_assumption
baseline_cost_policy_id
benchmark_id
universe_policy_id
eligibility_policy_id
tradability_policy_id
warning
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `weight_source` | 权重来源，例如 evaluation runner、uploaded weights、future portfolio policy。 |
| `rebalance_frequency` | 调仓频率。 |
| `signal_to_weight_rule` | 信号转权重规则，例如 equal weight、top quantile、long-only top N。 |
| `portfolio_diagnostic_mode` | 组合诊断口径，例如 long-only、long-short diagnostic。 |
| `long_short_executable` | long-short 是否被声明为真实可执行；A 股 schema `0.7` diagnostics 默认应为 `false`。 |
| `gross_exposure_target` | 目标总暴露。 |
| `net_exposure_target` | 目标净暴露。 |
| `cash_assumption` | 现金假设。 |
| `leverage_assumption` | 杠杆假设。 |
| `trade_price_assumption` | 成交价格假设，例如 close-to-close diagnostic。 |
| `trade_timing_assumption` | 交易时点假设，例如 next close / same close diagnostic。 |
| `baseline_cost_policy_id` | 基础收益路径使用的成本 policy。 |
| `benchmark_id` | 基准或指数标识；仅引用已批准本地输入。 |
| `universe_policy_id` | 股票池 policy。 |
| `eligibility_policy_id` | eligible mask policy。 |
| `tradability_policy_id` | 可交易性 policy。 |

验收：

- 生成该表时，每个相关 `combination_id` 至少有一行；
- A 股 long-short 诊断若不具备真实做空能力，`long_short_executable=false`；
- A 股 v1 强制 `long_short_executable=false`。未来如允许 `true`，必须提供可融券名单、券源、
  借券成本、保证金/费用、short-side tradability 和 policy provenance；否则 validation reject；
- `baseline_cost_policy_id` 如果改变主 `net_portfolio_return`，必须进入 evaluation identity；
- 不写 broker、exchange、paper/live trading 或实盘自动化假设。

### 6.5 `metrics/` 基础信号评价

这些表保留当前基础 signal evaluation 语义，但属于 schema `0.7` 新合同，
不承诺 API、loader 或 required-column 兼容历史 schema。

#### `metrics/ic.csv`

状态：required。

粒度：`combination_id x timestamp`。

字段：

```text
combination_id
timestamp
feature
label
ic
rank_ic
n_observations
```

#### `metrics/grouped_returns.csv`

状态：required。

粒度：`combination_id x timestamp x group_id`。

字段：

```text
combination_id
timestamp
group_id
group_label
n_assets
mean_forward_return
weighted_forward_return
```

#### `metrics/evaluation_metrics_summary.csv`

状态：required。

粒度：`combination_id x metric_family x metric_name`。

作用：统一承载 IC summary、group spread summary、performance summary 和 run-level
evaluation summary，替代多个窄 summary 表，避免 summary 文件膨胀。

字段：

```text
combination_id
metric_family
metric_name
metric_value
unit
metric_payload_json
n_periods
n_observations
aggregation_method
portfolio_diagnostic_mode
long_short_executable
status
status_reason
warning
```

规则：

- `metric_family=ic` 承载 mean IC、mean RankIC、ICIR、positive IC rate 等摘要；
- `metric_family=group_spread` 承载 top-bottom / long-short 诊断摘要；
- `metric_family=portfolio_performance` 承载 return、volatility、drawdown、turnover 等基础摘要；
- `metric_payload_json` 承载 family-specific 摘要口径；
- group spread / long-short summary 的 `metric_payload_json` 必须声明 `n_groups`、group sort order、
  top/bottom group id、long-short formula、factor direction、label horizon 和 return alignment；
- A 股 top-bottom / long-short 是因子诊断口径，不等于真实融券做空策略；
- A 股 long-short 相关 summary/diagnostic 行必须显式写 `long_short_executable=false`，并可附加 warning。

### 6.6 `portfolio/` 基础组合结果

这些表保留当前基础 portfolio result 语义，但属于 schema `0.7` 新合同。

#### `portfolio/returns.csv`

状态：required。

粒度：`combination_id x timestamp`。

字段：

```text
combination_id
timestamp
gross_return
baseline_cost
net_return
turnover
buy_turnover
sell_turnover
cost_model_id
cost_assumption_id
stamp_duty_cost
transfer_fee_cost
commission_cost
slippage_cost
impact_cost
portfolio_diagnostic_mode
long_short_executable
status
status_reason
warning
```

单位和符号约定：

- `gross_return`、`baseline_cost`、`net_return` 和所有 `*_cost` 列都是单期 decimal return，不是 bps；若字段用 bps，字段名必须显式带 `_bps`；
- cost component 列为非负成本扣减值，允许 null 表示 not modeled / unavailable / disabled；
- `net_return = gross_return - baseline_cost`；
- `baseline_cost` 应等于已启用 baseline cost components 的合计，例如印花税、过户费、佣金、滑点、impact 和其他基准成本项；
- no-cost / disabled-cost run 必须使用统一 sentinel：`cost_model_id="zero_cost_baseline"`
  且 `cost_assumption_id="__disabled__"`；此时 `baseline_cost=0`，各 cost component
  为 null，并在 policy/context status 中声明成本模型被禁用；
- `buy_turnover` / `sell_turnover` 是单边 turnover；`turnover` 的单边/双边口径必须写入 `run_context.json.config.evaluation_spec`；
- `cost_assumption_id` 必须能追溯到 `dimensions/diagnostic_policies.csv` 或 `run_context.json.config` 中的成本假设定义；
- 成本假设定义必须声明 `cost_component_source`、`cost_component_status`、`cost_component_unit`、`cost_component_scope` 和 `is_execution_simulated=false`；
- 成本假设定义必须声明 `effective_start`、`effective_end`、`applicable_side`、
  `rate_as_of` 和 `rate_source`，用于审计印花税、过户费、佣金等跨期费率口径；
- v1 没有 fill / execution simulation 时，`slippage_cost` 和 `impact_cost` 只能来自 assumption 或 proxy；缺失时必须写 status/warning，不能静默置 0 并解释成真实成交成本。
- `0` 只允许表示“该 component 已建模且结果为 0”；未建模、禁用或不可用必须使用 null 和 policy/context status。

#### `portfolio/weights.parquet`

状态：conditional。需要单票权重、集中度、行业权重、ADV participation 或其他
持仓级 diagnostics 时生成或保留。使用 Parquet 是因为行数可能较大。

字段：

```text
combination_id
timestamp
asset_id
target_weight
```

约束：

- diagnostics post-run policies 不改写这些表；
- cost sensitivity 只能进入 diagnostics 表，不覆盖 `net_return`；
- 如果某 policy 改变这些表，它不是 diagnostics-only，应进入新的 evaluation / portfolio policy 阶段。
- `portfolio/cumulative_returns.csv` 不作为 schema `0.7` 落盘 artifact；累计收益由 API/chart 从
  `portfolio/returns.csv` 派生。未来若因性能需要缓存，应另行定义 materialized view 字段合同。

### 6.7 `diagnostics/` 诊断结果

所有 schema `0.7` diagnostic fact tables 必须统一状态字段：

```text
diagnostic_status
status_reason
required_inputs_met
warning
```

含义：

- `diagnostic_status`：available / partial / unavailable / skipped / error；
- `status_reason`：机器可读短码，例如 `missing_required_column`、`insufficient_history`、
  `future_information_rejected`、`disabled_by_config`；
- `required_inputs_met`：必需输入是否满足；
- `warning`：人读解释。

状态字段命名规则：

- policy registry、input quality、base evaluation summary 使用 `status`；
- diagnostics fact tables 使用 `diagnostic_status`；
- display / API summary 不发明第三套状态字段，只透传上述字段或聚合成 warning level。

conditional artifact 机器触发规则：

- `DiagnosticPolicy.output_tables[]` 声明 policy 允许输出的 conditional 表；
- `DiagnosticsSpec.materialized_views[]` 声明本次 run 实际请求 materialize 的 optional/conditional views；
- writer 只能生成同时满足 policy 输出声明和 spec materialized view 请求的 conditional 表；
- conditional 表一旦出现在 `manifest.artifacts` 或 `manifest.display`，必须通过完整字段合同校验；
- conditional 表未生成时，不写空文件；缺失原因由 `diagnostics/diagnostic_summary.csv`
  输出 `skipped` / `unavailable` / `partial` 状态行。

`DiagnosticPolicy.output_tables[]` 最小 schema：

```text
table_key
path
required_inputs
optional_inputs
display_required
report_required
missing_input_mode
default_materialized
```

`required_inputs[]` / `optional_inputs[]` entry schema：

```text
kind
name
columns
required
```

字段解释：

- `kind`：`artifact_table`、`panel_column` 或 `context_field`；
- `name`：artifact path、panel input name 或 context dotted path；
- `columns`：需要的列名数组；对单个 context field 可为空；
- `required`：是否缺失即触发当前 policy 的 `missing_input_mode`。

字段解释：

- `table_key`：conditional view id，例如 `scenario_metrics`、`tradability_summary`；
- `path`：artifact-relative path；
- `display_required`：一旦该表生成，必须进入 `manifest.display.tables[]`；
- `report_required`：一旦该表生成，optional `report.md` 可以引用它；未生成时 report 必须引用 summary status；
- `missing_input_mode`：table-level override；若为空，继承 `dimensions/diagnostic_policies.csv`
  的 policy-level `missing_input_mode`；
- `default_materialized`：v1 默认为 `false`，除非本 spec 明确列为 required table。

触发真值表：

| policy allows | spec requests | required inputs | missing_input_mode | 输出行为 |
|---:|---:|---|---|---|
| false | any | any | any | validation error if requested; otherwise no output |
| true | false | any | any | no file; optional summary row may say skipped_by_config |
| true | true | met | any | generate file, validate fields, register display if display_required |
| true | true | missing | fail_fast | fail run |
| true | true | missing | emit_unavailable | generate status rows when table grain allows; otherwise summary unavailable |
| true | true | partial | emit_partial | generate partial rows with warnings |
| true | true | missing | skip_with_summary | no file; `diagnostic_summary.csv` row is skipped/unavailable |

对于 `dimensions/portfolio_assumptions.csv` 这类 conditional dimension，除了上表，还必须满足
“组合假设确实按 `combination_id` 变化或 display 明确需要组合级假设表”。若不满足，输出
`diagnostic_summary.csv` skipped row，不写空表。

所有可生成表，包括 conditional / optional 表，都必须进入代码 schema registry index。
disabled-by-default 的表可以延后实现完整逐字段 registry，但 path、status、unique key、sort key、
默认生成状态和 display/download eligibility 必须先注册；未注册表不得写入 `manifest.artifacts`
或 `manifest.display`。

policy input matrix 最小规则：

下表的缺失行为是 policy family 的默认建议；实现时最终以
`dimensions/diagnostic_policies.csv.missing_input_mode` 为机器可读准绳。

| `policy_family` | 必需 artifact / panel input | 输出 | 缺失行为 |
|---|---|---|---|
| `regime` | `portfolio/returns.csv`、regime label 或可观测 regime input | `diagnostic_metrics.csv`，可选 `scenario_metrics.csv` | ex-ante 必需字段缺失时 fail；ex-post label 缺失时 unavailable。 |
| `cost` | `portfolio/returns.csv`、turnover、cost policy parameters | `diagnostic_metrics.csv`，可选 `scenario_metrics.csv` | baseline cost 缺失 fail；sensitivity component 缺失 partial/unavailable。 |
| `neutralization` | `metrics/ic.csv`、factor/label panel、control columns | `diagnostic_metrics.csv`，可选 `scenario_metrics.csv` | required control 缺失 unavailable；optional control 缺失 warning。 |
| `benchmark` | `portfolio/returns.csv`、approved local benchmark return column | `diagnostic_metrics.csv`，可选 `scenario_metrics.csv` | benchmark column 缺失 unavailable，不自动下载。 |
| `risk` | `portfolio/returns.csv`，weights/exposure columns when required | `diagnostic_metrics.csv`，可选 path-risk/exposure materialized views | returns 缺失 fail；weights/exposure 缺失 partial/unavailable。 |
| `capacity` | weights, ADV/liquidity proxy, notional assumption | `diagnostic_metrics.csv`，可选 `exposure_capacity_summary.csv` | weights 或 ADV 缺失 unavailable，不静默置零。 |
| `tradability` | universe/eligible/tradability columns | `diagnostic_metrics.csv`，可选 tradability materialized views | A 股关键字段缺失 partial/unavailable。 |

#### `diagnostics/tradability_diagnostics.csv`

状态：conditional。必须同时由 `DiagnosticPolicy.output_tables` 允许，并由
`DiagnosticsSpec.materialized_views[]` 请求时生成。A 股 pipeline 如果需要输出可交易性诊断，
也必须通过 diagnostics spec 表达，不能绕过这条规则。
该表由 schema `0.7` diagnostics 生成，不作为同一 run 的 diagnostics 输入。

粒度：`combination_id x timestamp x feature x method x direction`。当 direction 不适用时使用
`direction="__all__"`。

字段：

```text
combination_id
timestamp
feature
method
direction
universe_count
eligible_count
can_enter_count
can_exit_count
entry_blocked_count
exit_blocked_count
long_count
short_count
eligible_rate
can_enter_rate
can_exit_rate
entry_blocked_rate
exit_blocked_rate
fallback_reason
reason_payload_json
diagnostic_status
status_reason
required_inputs_met
warning
```

作用：

- 它是可交易性覆盖诊断，不等同于 capacity；
- capacity 是流动性压力 proxy，两者不能合并。

验收：

- schema `0.7` 不继承历史 tradability table 字段合同；
- 必须包含统一状态字段；
- `reason_payload_json` 必须机器可读地拆分涨跌停、停牌、T+1、zero-volume/no-trade
  等逐期原因 count / rate；不得只用 prose fallback reason；
- 缺少 A 股特定可交易性字段时输出 `partial` 或 `unavailable`，不得静默当作可交易。

#### `diagnostics/tradability_summary.csv`

状态：conditional compact summary。默认可用 `diagnostics/diagnostic_metrics.csv`
的 tradability rows 表达；只有 `DiagnosticPolicy.output_tables` 允许，且
`DiagnosticsSpec.materialized_views[]` 请求 `tradability_summary` 时生成。

粒度：`combination_id`。

作用：A 股 universe / eligibility / tradability 三层摘要。它不替代
`tradability_diagnostics.csv` 明细，而是给 API、workbench 和 acceptance 一个稳定的
compact summary。
该表是每个 `combination_id` 的唯一全局汇总，不支持同一组合下多个 policy、method
或 direction 的并列表达；这些差异应进入 `diagnostic_metrics.csv` 或
`tradability_diagnostics.csv`。

字段：

```text
combination_id
start_date
end_date
n_periods
n_observations
aggregation_method
universe_count
eligible_count
eligible_rate
can_enter_count
can_enter_rate
can_exit_count
can_exit_rate
suspended_count
limit_up_count
limit_down_count
zero_volume_count
no_trade_count
t_plus_one_blocked_count
blocked_weight_rate
diagnostic_status
status_reason
required_inputs_met
warning
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `start_date` / `end_date` | summary 覆盖的日期范围。 |
| `n_periods` | 聚合覆盖的交易期数量。 |
| `n_observations` | 聚合覆盖的样本数量。 |
| `aggregation_method` | 聚合口径，例如 sum、mean、max、last 或 weighted_mean。 |
| `universe_count` | 原始股票池数量。 |
| `eligible_count` | 满足 eligible mask 的股票数量。 |
| `can_enter_count` | 可新买入样本数量。 |
| `can_exit_count` | 可卖出样本数量。 |
| `suspended_count` | 停牌样本数量。 |
| `limit_up_count` | 涨停导致不可买或受限样本数量。 |
| `limit_down_count` | 跌停导致不可卖或受限样本数量。 |
| `zero_volume_count` | 零成交量样本数量。 |
| `no_trade_count` | 无交易样本数量。 |
| `t_plus_one_blocked_count` | T+1 假设下受限样本数量。 |
| `blocked_weight_rate` | 被可交易性约束影响的权重比例。 |

验收：

- 必须覆盖 universe、eligibility、tradability 三层；
- 必须声明聚合口径，避免把累计值、均值、最大值或末期快照混淆；
- 缺少 weights 时，`blocked_weight_rate` 必须为 null，并将相关 weighted 指标标为
  `unavailable` 或 `partial`；不得静默置 0；
- 缺少某类 A 股约束字段时输出 `partial` 或 `unavailable`，并给出 `status_reason`；
- 不得把缺失字段静默当成全部可交易。

#### `diagnostics/diagnostic_summary.csv`

状态：required。

粒度：`combination_id x policy_family x policy_id`。

作用：diagnostics 总览。用于 API summary cards、workbench 表格 preview、下载和验收。

字段：

```text
combination_id
policy_family
policy_id
identity_scope
diagnostic_status
required_inputs_met
coverage_rate
primary_metric_name
primary_metric_value
secondary_metric_name
secondary_metric_value
warning_level
status_reason
warning
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `diagnostic_status` | 诊断结果状态：`available`、`partial`、`unavailable`、`skipped`、`error`。 |
| `required_inputs_met` | 必需输入是否满足。 |
| `coverage_rate` | 可用样本覆盖率。 |
| `primary_metric_*` | 该 policy 最重要的摘要指标。 |
| `secondary_metric_*` | 第二摘要指标。 |
| `warning_level` | 警告级别，例如 info、warning、blocker。 |
| `status_reason` | 诊断状态原因短码。 |

验收：

- 每个启用 policy 至少有一行 summary；
- 缺少输入时必须输出 `unavailable` 或 `partial`，不得静默跳过；
- 不得把 capacity proxy 写成真实容量承诺。

#### `diagnostics/diagnostic_metrics.csv`

状态：required。

粒度：`combination_id x policy_id x metric_family x metric_name x bucket_id x metric_variant_id`。
无 bucket 时 `bucket_id="__all__"`，避免唯一键包含空值。
无 metric variant 时 `metric_variant_id="__base__"`。

作用：统一承载低基数 diagnostics 指标，避免为 risk、cost、capacity、tradability、
path risk、neutralization 各建一张 required 宽表。

字段：

```text
combination_id
policy_id
policy_family
metric_family
metric_name
metric_variant_id
metric_value
unit
bucket_id
aggregation_method
metric_payload_json
decision_usable
diagnostic_status
status_reason
required_inputs_met
warning
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `metric_family` | 指标族，例如 `cost`、`path_risk`、`tradability`、`exposure`、`capacity`、`neutralization`。 |
| `metric_name` | 指标名，例如 `break_even_cost_bps`、`max_drawdown`、`adv_participation_proxy`。 |
| `metric_variant_id` | 指标变体 id，例如成本 component、notional assumption 或 neutralization control set；无变体时使用 `__base__`。 |
| `metric_payload_json` | family-specific 维度和方法说明，例如成本 component、neutralization controls、notional assumption、ADV 口径。 |
| `decision_usable` | 该行指标是否允许进入事前决策候选；diagnostic-only 默认 `false`。 |

验收：

- 每个启用 policy 至少在 `diagnostic_summary.csv` 中有状态行；若有可用低基数指标，应进入本表；
- 所有行必须包含统一状态字段；
- capacity 行的 `metric_name` 必须带 `_proxy` 后缀，且 `metric_payload_json` 必须声明
  `capacity_claim=false`；
- neutralization 行必须在 payload 中声明方法、控制变量集合、截面粒度、覆盖率、缺失处理和共线性处理；
- neutralization payload 必须声明 `control_as_of_timestamp`、`control_lag`、
  `control_point_in_time`、universe mask、standardization / winsorization 方法和 leakage check id；
  未通过 point-in-time / leakage check 时 `decision_usable=false`；
- cost 行必须能表达 buy/sell turnover、印花税、过户费、佣金、滑点和 impact component。
- 本表不得承载 scenario 维度；只要指标需要 `scenario_family` / `scenario_id`，
  必须进入 `diagnostics/scenario_metrics.csv`；
- `bucket_id` 只用于低基数 bucket，例如 liquidity bucket 或 size bucket；若 bucket 本身定义了
  一个可比较情景，应改用 `scenario_metrics.csv`。
- `metric_variant_id` 必须进入唯一键；payload 不能承载会改变唯一性的隐藏维度。

#### `diagnostics/portfolio_risk_summary.csv`

状态：conditional materialized summary。默认可用 `diagnostics/diagnostic_metrics.csv`
的 `metric_family=path_risk` 和 `metric_family=concentration` rows 表达。只有
`DiagnosticPolicy.output_tables` 允许，且 `DiagnosticsSpec.materialized_views[]` 请求
`portfolio_risk_summary` 时生成。

粒度：`combination_id x policy_id`。

作用：轻量路径风险摘要。它补足 exposure 之外的 returns path 和 concentration 诊断，
不做完整 portfolio risk model 或 stress test。

字段：

```text
combination_id
policy_id
n_periods
volatility
downside_volatility
max_drawdown
tail_quantile
tail_return
expected_shortfall_proxy
concentration_hhi
max_single_name_weight
exposure_coverage_rate
return_coverage_rate
diagnostic_status
status_reason
required_inputs_met
warning
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `volatility` | 组合收益波动率。 |
| `downside_volatility` | 下行波动率，只统计负收益或下行部分。 |
| `max_drawdown` | 最大回撤。 |
| `tail_quantile` | 尾部风险分位点，例如 5%。 |
| `tail_return` | 对应分位点的尾部收益。 |
| `expected_shortfall_proxy` | ES/CVaR 代理值；是诊断 proxy，不是正式风险模型承诺。 |
| `concentration_hhi` | 持仓集中度 HHI。 |
| `max_single_name_weight` | 最大单票权重。 |
| `exposure_coverage_rate` | 风险暴露输入覆盖率。 |
| `return_coverage_rate` | 收益序列覆盖率。 |

验收：

- 缺少 weights 时 concentration 指标为 `unavailable`，不能静默置零；
- 缺少足够 returns 时 volatility / drawdown / tail 指标为 `unavailable`；
- risk policy 的 `policy_payload_json` 必须声明 return frequency、annualization method、
  risk window、tail confidence level、tail method 和 ES proxy method；loader 必须机器校验这些口径；
- stress scenarios 仍 deferred，不在该表中伪造 stress result。

#### `diagnostics/scenario_metrics.csv`

状态：conditional。启用 regime、cost sensitivity、neutralized signal、benchmark comparison
或 market segment policy，且 `DiagnosticPolicy.output_tables` 允许，并由
`DiagnosticsSpec.materialized_views[]` 请求 `scenario_metrics` 时生成。

粒度：`combination_id x scenario_family x scenario_id x policy_id x metric_name`。
`scenario_id` 必须编码所有会改变同一 `metric_name` 数值的 scenario variant，例如
cost profile、neutralization control set、benchmark id 或 market segment；不得把会改变
唯一性的 variant 只藏在 `scenario_payload_json` 中。

作用：承载“同一基础结果在另一种情景、切片或参照下的指标”。该表采用长表结构，
避免 regime、cost、neutralization、benchmark 和 market segment 共用一个稀疏宽表。

required columns：

```text
combination_id
scenario_family
scenario_id
policy_id
metric_name
metric_value
unit
benchmark_id
scenario_label_type
timing_semantics
decision_usable
uses_future_information
uses_future_return
as_of_timestamp
label_available_at
lookback_end
data_lag
leakage_check_id
portfolio_diagnostic_mode
long_short_executable
metric_applicability
scenario_payload_json
label_source
formation_window
label_available_after
diagnostic_status
status_reason
required_inputs_met
warning
```

可承载内容：

- `scenario_family=regime`：不同市场环境下的 IC、RankIC、return、drawdown 等指标；
- ex-post regime 在 display/API 中应标记为 `retrospective_segment` 或
  `ex_post_label`，不得标成可决策 regime；
- `scenario_family=cost`：flat、linear、piecewise-linear、quadratic cost 下的
  `break_even_cost_bps`、`total_cost`、`stamp_duty_cost`、`slippage_cost`、`impact_cost`
  等指标；成本 component 应用多行 `metric_name` 表达，不塞进 payload；
  每行必须通过 `policy_id` 或 `scenario_payload_json` 追溯到带 `effective_start` /
  `effective_end`、applicable side、`rate_as_of` 和 `rate_source` 的 cost policy；
- `scenario_family=neutralization`：只承载 neutralized IC / RankIC / delta IC /
  delta RankIC 等信号诊断；不得生成 neutralized portfolio return；
- `scenario_family=benchmark`：与指数或基准收益对比；
- `scenario_family=market_segment`：行业、市值、流动性 segment 下的指标。

验收：

- cost sensitivity 不改写 `portfolio/returns.csv`；
- ex-post regime 可以回看市场环境解释结果，但 `decision_usable=false`；
- ex-post regime 如果用未来收益或完整样本期定义标签，必须写 `uses_future_return=true`；
- ex-post regime 也必须声明 `label_source`、`formation_window` 和 `label_available_after`；
- `timing_semantics=ex_ante_policy` 必须写出 `as_of_timestamp`、`label_available_at`、
  `lookback_end`、`data_lag` 和 `leakage_check_id`；`decision_usable=true`
  只是进一步声明该 ex-ante result 可以进入决策候选；
- `label_available_at` 是 point-in-time 可见的具体 timestamp；`label_available_after`
  是 regime 标签形成规则或滞后表达，例如 `after_window_close_plus_1_trading_day`，
  两者不得混用；
- `ex_post_analysis` 的结果不得进入 v2 gates、constraints 或 v3 portfolio policy；
- long-only run 的 short-side 指标必须使用 `metric_applicability=not_applicable`
  或 `diagnostic_status=unavailable`，不得用 0 表示不适用；
- A 股 long-short diagnostic 必须在本表行级声明 `long_short_executable=false`；
- neutralization rows 的 return 类 `metric_name` 必须 unavailable，除非该指标明确是未改写的 baseline 对照；
- `scenario_family=neutralization` 的 `scenario_payload_json` 必须声明 controls、control as-of
  timestamp、control lag、point-in-time flag、universe mask、standardization / winsorization、
  missing handling、collinearity handling 和 leakage check id；未满足时必须 `unavailable`
  或 `decision_usable=false`；
- benchmark comparison 需要在 `scenario_payload_json` 声明 benchmark return convention、
  alignment、as-of time、频率、币种和交易日对齐口径，不自动下载外部指数数据。

family-specific applicability：

- `benchmark_id` required only when `scenario_family=benchmark`，其他 family 必须 null 或 `__none__`；
- `as_of_timestamp`、`label_available_at`、`lookback_end`、`data_lag` 和 `leakage_check_id`
  required when `timing_semantics=ex_ante_policy`；
- `label_source`、`formation_window`、`label_available_after` 和 `uses_future_return` required for
  `scenario_family=regime`；
- 不适用字段必须使用 null / sentinel 并由 `metric_applicability` 解释，不得填伪造值。

#### `diagnostics/exposure_capacity_summary.csv`

状态：conditional materialized summary。默认可用 `diagnostics/diagnostic_metrics.csv`
表达低基数 exposure/capacity 指标；只有 `DiagnosticPolicy.output_tables` 允许，且
`DiagnosticsSpec.materialized_views[]` 请求 `exposure_capacity_summary` 时生成。

粒度：`combination_id x exposure_family x exposure_name x bucket_id x policy_id x notional_assumption_id`。
无 bucket 时 `bucket_id="__all__"`；无 notional assumption 时 `notional_assumption_id="__none__"`。

作用：承载风险暴露和容量压力摘要。

字段：

```text
combination_id
exposure_family
exposure_name
bucket_id
policy_id
start_date
end_date
n_periods
notional_assumption_id
notional_currency
notional_value_basis
notional_side_basis
notional_period_basis
liquidity_measure
adv_window
adv_unit
participation_cap
rebalance_frequency
participation_direction
participation_basis
long_exposure
short_exposure
net_exposure
gross_exposure
weight_delta_proxy
trade_value_proxy
coverage_rate
aggregation_basis
aggregation_method
adv_participation_proxy
max_adv_participation_proxy
p95_adv_participation_proxy
assumed_notional_for_proxy
over_adv_cap_weight_proxy
blocked_weight_rate_proxy
capacity_claim
diagnostic_status
status_reason
required_inputs_met
warning
```

可承载内容：

- `exposure_family=market_beta`：市场 beta proxy；
- `exposure_family=industry`：行业暴露；
- `exposure_family=size`：市值暴露；
- `exposure_family=style`：风格暴露；
- `exposure_family=volatility`：波动率暴露；
- `exposure_family=liquidity`：流动性暴露；
- `exposure_family=capacity`：ADV participation、blocked weight、over-cap proxy。

验收：

- 缺少行业、市值、beta、波动率或流动性列时输出 unavailable/warning；
- capacity proxy 行必须显式写出 `notional_assumption_id`、`liquidity_measure`、
  `adv_window`、`adv_unit` 和 `participation_cap`；
- capacity proxy 行必须写出 `start_date`、`end_date`、`n_periods`、`aggregation_basis`、
  `max_adv_participation_proxy`、`p95_adv_participation_proxy` 和 `capacity_claim=false`；
- notional assumption 必须显式写出币种、数值口径、单边/双边口径和日度/调仓期口径；
- capacity proxy 必须区分持仓占 ADV 和交易额占 ADV；ADV participation 口径必须声明
  `rebalance_frequency`、`participation_direction`、`participation_basis`、
  `weight_delta_proxy` 或 `trade_value_proxy`；
- timestamp-level ADV participation 必须有 timestamped ADV/liquidity proxy 和 weight change、
  `trade_value_proxy` 或 `assumed_notional_for_proxy`；
- 字段名必须带 `proxy`，避免真实规模承诺；
- long-only run 不得产生真实 short exposure。

#### `diagnostics/diagnostic_observations.parquet`

状态：optional，默认不生成。

粒度：`combination_id x policy_id x grain x timestamp x asset_id x observation_family x observation_name`。
portfolio-level rows 使用 `grain="portfolio"` 且 `asset_id="__all__"`；asset-level rows 使用
`grain="asset"` 和真实 asset id。

作用：保存高基数明细，例如逐日逐股票暴露、逐日 ADV participation proxy、逐日 regime label。

字段：

```text
combination_id
policy_id
grain
timestamp
asset_id
observation_family
observation_name
observation_value
unit
diagnostic_status
status_reason
required_inputs_met
warning
```

约束：

- 仅在用户需要审计明细且资源允许时生成；
- 不进入默认 summary cards；
- preview 必须有严格 row limit；
- 不替代 summary 表。

### 6.8 `input_quality/` 输入质量

这些表保留当前输入质量诊断语义，但属于 schema `0.7` 新合同。schema `0.7`
把原先多张覆盖率表收敛为一张长表。

#### `input_quality/coverage_summary.csv`

状态：required。

粒度：`combination_id x policy_id x input_family x input_name`。
run-level input quality 使用 `policy_id="__run__"`。

字段：

```text
combination_id
policy_id
input_family
input_name
required
coverage_rate
n_missing
n_observations
status
status_reason
warning
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `input_family` | 输入族，例如 `label`、`factor`、`tradability`、`exposure`、`capacity`、`benchmark`。 |
| `input_name` | 具体列名或输入名。 |
| `required` | 是否为当前 run / policy 必需输入。 |
| `coverage_rate` | 覆盖率。 |
| `n_missing` | 缺失样本数。 |
| `status` | available / partial / unavailable / skipped / error。 |

### 6.9 `summaries/`

schema `0.7` 不再新增 `summaries/evaluation_summary.csv` 或
`summaries/diagnostics_summary.csv`。基础 evaluation 汇总进入
`metrics/evaluation_metrics_summary.csv`；diagnostics 总览进入
`diagnostics/diagnostic_summary.csv`。

### 6.10 `breakdowns/`

延续当前 time breakdown 机制：

```text
breakdowns/year/*
breakdowns/quarter/*
breakdowns/month/*
```

不新增 `breakdowns/regime/*`。regime 属于诊断情景，统一进入
`diagnostics/scenario_metrics.csv`，减少目录和表数量。

### 6.11 前置数据能力与归属

市场环境研究需要前置输入，但不等于 N5 要自动下载数据。

允许作为 N5 输入的本地已批准字段：

- benchmark / index return columns；
- industry classification；
- market cap / size bucket；
- liquidity / ADV proxy columns；
- volatility columns；
- market beta proxy columns；
- point-in-time regime labels；
- ex-post research regime labels。

归属规则：

| 研究问题 | 归属 |
|---|---|
| 事后看因子在牛市/熊市、高波动/低波动、大盘强/小盘强中的表现 | `DiagnosticsSpec`，输出到 `scenario_metrics.csv`。 |
| 事后看因子在行业、市值、流动性 segment 中的表现 | `DiagnosticsSpec`，输出到 `scenario_metrics.csv` 或 `exposure_capacity_summary.csv`。 |
| 与指数或 benchmark 对比 | `DiagnosticsSpec`，输出到 `scenario_metrics.csv`。 |
| 从一开始就只在某指数成分股、某行业或某市值段里建组合 | `EvaluationSpec`，进入 `combinations.csv` 的 universe identity。 |
| 根据 market regime 调整因子、仓位或权重 | 后续 `PortfolioPolicySpec` 或新 evaluation run，不能作为 ex-post diagnostics。 |

不得做：

- 不自动下载外部指数或板块数据；
- 不接 vendor API；
- 不把事后 regime label 标成可决策；
- 不把 A 股 long-short 诊断写成真实可融券做空策略。

### 6.12 Deferred v1 artifacts

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

v1 capacity acceptance 只覆盖一个或少数 `notional_assumption_id` 下的 ADV / liquidity
proxy 诊断；不覆盖 capacity curve、crowding proxy，也不承诺真实可交易容量。

### 6.13 V2/V3 接口占位边界

V2 `DiagnosticGateSpec` 与 V1 `DiagnosticPolicy` 必须分开：

- `DiagnosticPolicy` 负责计算诊断结果；
- `DiagnosticGateSpec` 负责读取诊断结果并输出 pass / warn / fail；
- V2 gate 必须 `enforcement_mode=report_only`；
- V2 gate 不得改变 universe、signals、weights、turnover path 或 returns；
- 任何会改变 universe、weights 或 returns 的规则必须进入 V3 `PortfolioPolicySpec`。

V3 `PortfolioPolicySpec` 的最小交接链路：

```text
PortfolioPolicySpec
  -> TargetWeightsArtifact
  -> EvaluationSpec consumes target weights
  -> schema 0.7 evaluation_set run
```

`TargetWeightsArtifact` 最小列：

```text
timestamp
asset
target_weight
weight_policy_id
as_of_timestamp
weight_status
status_reason
```

边界：

- `PortfolioPolicySpec` materialize target weights；
- `EvaluationSpec` 只消费已生成权重或轻量 weight config；
- `DiagnosticsSpec` 不得隐式修改 target weights；
- tradability、cost 和 capacity 约束如果只是诊断，留在 N5 v1 diagnostics；
- 如果这些约束改变权重或交易路径，必须进入 V3 或后续 simulation。

### 6.14 Schema `0.7` 文件与字段数量统计

统计口径：

- 表格文件按本 spec 列出的 schema `0.7` contract fields 统计；
- JSON / Markdown 顶层文件按本 spec 列出的机器可读字段组或内容组统计；
- 不统计历史 writer output columns；schema `0.7` 是新的独立合同；
- `breakdowns/year|quarter|month` 是可生成视图模板，不进入最小 schema 字段总账；
- `diagnostic_observations.parquet` 默认不生成；
- `portfolio/cumulative_returns.csv` 不落盘，不进入字段总账。

| 目录 | 文件 | 状态 | 字段 / 指标数 | 说明 |
|---|---|---:|---:|---|
| root | `manifest.json` | required | 4 | artifact type、schema version、artifact paths、display contract。 |
| root | `summary.json` | required | 10 | run 摘要、base evaluation 摘要、diagnostics 状态、schema 标记、aggregate warnings。 |
| root | `run_context.json` | required | 11 | config / provenance 合并后的脱敏运行上下文。 |
| root | `report.md` | optional | 5 | 人读结论、A 股限制、long-short 口径、capacity proxy 边界、diagnostics warnings。 |
| `dimensions/` | `combinations.csv` | required | 9 | schema `0.7` evaluation identity fields。 |
| `dimensions/` | `diagnostic_policies.csv` | required | 16 | 统一 policy / assumption registry，family-specific 内容进 payload。 |
| `dimensions/` | `portfolio_assumptions.csv` | conditional | 20 | 只有组合假设按 combination 不同时生成。 |
| `metrics/` | `ic.csv` | required | 7 | schema `0.7` contract fields。 |
| `metrics/` | `grouped_returns.csv` | required | 7 | schema `0.7` contract fields。 |
| `metrics/` | `evaluation_metrics_summary.csv` | required | 14 | IC、group spread、performance、run summary 的长表汇总。 |
| `portfolio/` | `returns.csv` | required | 20 | 逐期收益、turnover 和 A 股成本 component。 |
| `portfolio/` | `weights.parquet` | conditional | 4 | 需要持仓级 diagnostics 时生成或保留。 |
| `diagnostics/` | `diagnostic_summary.csv` | required | 14 | diagnostics policy-level 总览。 |
| `diagnostics/` | `diagnostic_metrics.csv` | required | 16 | 低基数 diagnostics 指标长表。 |
| `diagnostics/` | `tradability_diagnostics.csv` | conditional | 24 | schema `0.7` tradability diagnostic output。 |
| `diagnostics/` | `tradability_summary.csv` | conditional | 24 | A 股 universe / eligibility / tradability materialized summary。 |
| `diagnostics/` | `portfolio_risk_summary.csv` | conditional | 17 | 轻量路径风险 materialized summary。 |
| `diagnostics/` | `scenario_metrics.csv` | conditional | 29 | regime、cost、neutralization、benchmark、market segment 情景指标长表。 |
| `diagnostics/` | `exposure_capacity_summary.csv` | conditional | 40 | risk exposure 和 capacity proxy materialized summary。 |
| `diagnostics/` | `diagnostic_observations.parquet` | optional | 13 | 高基数明细，默认不生成。 |
| `input_quality/` | `coverage_summary.csv` | required | 11 | label、factor、tradability、exposure、capacity、benchmark 覆盖率长表。 |

目录级统计：

| 目录 | required 文件数 | conditional 文件数 | optional 文件数 | required 字段 / 指标数 | conditional 字段 / 指标数 | optional 字段 / 指标数 |
|---|---:|---:|---:|---:|---:|---:|
| root | 3 | 0 | 1 | 25 | 0 | 5 |
| `dimensions/` | 2 | 1 | 0 | 25 | 20 | 0 |
| `metrics/` | 3 | 0 | 0 | 28 | 0 | 0 |
| `portfolio/` | 1 | 1 | 0 | 20 | 4 | 0 |
| `diagnostics/` | 2 | 5 | 1 | 30 | 134 | 13 |
| `input_quality/` | 1 | 0 | 0 | 11 | 0 | 0 |
| `breakdowns/` | 0 | 0 | view templates | 0 | 0 | not counted |

总计：

| 统计口径 | 文件数 | 字段 / 指标数 |
|---|---:|---:|
| required 最小 schema `0.7` | 12 | 139 |
| required + conditional materialized outputs，不含 optional detail/report/views | 19 | 297 |
| required + conditional + optional report/detail views | 21 | 315 |
| breakdowns year / quarter / month | view templates | not counted |

### 6.15 Required Schema Registry Appendix

本 appendix 只覆盖 required 最小 schema 的 registry index / implementation checklist。
正文各文件的字段列表是 canonical contract；字段级机器合同必须落在代码中的 schema registry。
本文不在 Markdown 中展开 139 个字段的完整 `name/dtype/nullable/unit/enum` 表，避免文档和代码漂移。

通用 dtype：

- `string`：UTF-8 string；
- `date`：ISO-8601 date（`YYYY-MM-DD`），不含日内时间；
- `timestamp`：ISO-8601 timestamp；A 股日频可用本地交易日日期，但 as-of / availability 字段必须声明时区或交易日闭区间语义；
- `integer`：整数；
- `float`：double precision numeric；
- `boolean`：true / false；
- `json`：JSON object encoded as string or native JSON in metadata file；
- `array[string]`：string array。

通用枚举：

- `status` / `diagnostic_status`：`available`、`partial`、`unavailable`、`skipped`、`error`；
- `warning_level`：`info`、`warning`、`blocker`；
- `identity_scope`：`diagnostic_only`、`evaluation_axis`、`portfolio_policy`、`simulation_policy`；
- `timing_semantics`：`ex_post_analysis`、`ex_ante_policy`、`static_assumption`；
- `missing_input_mode`：`fail_fast`、`emit_unavailable`、`emit_partial`、`skip_with_summary`。

Required file registry：

| file | unique key | sort key | field groups |
|---|---|---|---|
| `manifest.json` | n/a | n/a | `artifact_type:string required enum=evaluation_set`; `artifact_schema_version:string required enum=0.7`; `artifacts:json required`; `display:json required` |
| `summary.json` | n/a | n/a | `run_id:string required`; `project:string required`; `kind:string required`; `data_stage:string required`; `n_rows_evaluated:integer required unit=count`; `n_combinations:integer required unit=count`; `base_evaluation:json required`; `diagnostics:json required`; `artifact_schema_version:string required enum=0.7`; `warnings:array[string] required` |
| `run_context.json` | n/a | n/a | `config:json required`; `config.evaluation_spec:json required`; `config.diagnostics_spec:json required`; `config.a_share_assumptions:json required`; `provenance:json required`; `provenance.run_id:string required`; `provenance.input_panel_fingerprint:string required`; `provenance.diagnostic_policy_fingerprint:string required`; `provenance.generated_at:timestamp required`; `provenance.generator:json required`; `provenance.staging_payload_fingerprint:string required` |
| `dimensions/combinations.csv` | `combination_id` | `combination_id` | required fields follow the正文清单；`universe_id` / `weight_policy_id` absent in simple runs use `__base__` / `__none__` sentinel if implemented as required columns; `identity_payload_json:json nullable` |
| `dimensions/diagnostic_policies.csv` | `policy_id` | `policy_family,policy_id` | id/family/scope/timing/status fields required strings with enum validation; `decision_usable`, `uses_future_return`, `uses_future_information`, `changes_*` required boolean; `warning` nullable string; `policy_payload_json:json required`; `missing_input_mode:string required enum` |
| `metrics/ic.csv` | `combination_id,timestamp` | `combination_id,timestamp` | exact 7 fields from正文清单；IC metrics are float; `n_observations:integer unit=count`; no status fields in this base table |
| `metrics/grouped_returns.csv` | `combination_id,timestamp,group_id` | `combination_id,timestamp,group_id` | exact 7 fields from正文清单；return fields are decimal returns; long-short executable/applicability belongs in summary/diagnostics, not this base table |
| `metrics/evaluation_metrics_summary.csv` | `combination_id,metric_family,metric_name` | `combination_id,metric_family,metric_name` | exact 14 fields from正文清单；id/family/name/unit required string; `metric_value:float nullable when unavailable`; `metric_payload_json:json nullable but required for group_spread/long-short rows`; status fields follow common enum |
| `portfolio/returns.csv` | `combination_id,timestamp` | `combination_id,timestamp` | return/cost/turnover fields float decimal return or rate; `cost_model_id:string required`; `cost_assumption_id:string required`; `portfolio_diagnostic_mode:string required`; `long_short_executable:boolean required`; status fields follow common enum |
| `diagnostics/diagnostic_summary.csv` | `combination_id,policy_family,policy_id` | `combination_id,policy_family,policy_id` | id/family/scope required string; metric names nullable string; metric values float nullable; `coverage_rate:float unit=rate`; `warning_level:string required enum`; diagnostic status fields follow common enum |
| `diagnostics/diagnostic_metrics.csv` | `combination_id,policy_id,metric_family,metric_name,bucket_id,metric_variant_id` | `combination_id,policy_id,metric_family,metric_name,bucket_id,metric_variant_id` | id/family/name/unit required string; `bucket_id:string required` using `__all__` when no bucket applies; `metric_variant_id:string required` using `__base__` when no variant applies; `metric_value:float nullable when unavailable`; `metric_payload_json:json nullable`; `decision_usable:boolean required`; diagnostic status fields follow common enum |
| `input_quality/coverage_summary.csv` | `combination_id,policy_id,input_family,input_name` | `combination_id,policy_id,input_family,input_name` | id/family/name required string; run-level rows use `policy_id=__run__`; `required:boolean required`; `coverage_rate:float unit=rate`; `n_missing,n_observations:integer unit=count`; status fields follow common enum |

实现时的代码 schema registry 必须把正文 canonical field lists 展开成逐字段规则；不得只检查列名存在。
schema registry 必须支持 `required_if`、`nullable_if`、`sentinel_if` 和 family-specific
validator hook。`scenario_metrics.csv`、`diagnostic_metrics.csv` 和 conditional materialized
views 的条件字段不得退化成分散手写 if/else。

Generated file registry index：

| file | status | unique key | sort key | default materialized |
|---|---|---|---|---:|
| `dimensions/portfolio_assumptions.csv` | conditional | `combination_id,portfolio_assumption_id` | `combination_id,portfolio_assumption_id` | false |
| `portfolio/weights.parquet` | conditional | `combination_id,timestamp,asset_id` | `combination_id,timestamp,asset_id` | false |
| `diagnostics/tradability_diagnostics.csv` | conditional | `combination_id,timestamp,feature,method,direction` | `combination_id,timestamp,feature,method,direction` | false |
| `diagnostics/tradability_summary.csv` | conditional | `combination_id` | `combination_id` | false |
| `diagnostics/portfolio_risk_summary.csv` | conditional | `combination_id,policy_id` | `combination_id,policy_id` | false |
| `diagnostics/scenario_metrics.csv` | conditional | `combination_id,scenario_family,scenario_id,policy_id,metric_name` | `combination_id,scenario_family,scenario_id,policy_id,metric_name` | false |
| `diagnostics/exposure_capacity_summary.csv` | conditional | `combination_id,exposure_family,exposure_name,bucket_id,policy_id,notional_assumption_id` | `combination_id,exposure_family,exposure_name,bucket_id,policy_id,notional_assumption_id` | false |
| `diagnostics/diagnostic_observations.parquet` | optional/detail | `combination_id,policy_id,grain,timestamp,asset_id,observation_family,observation_name` | `combination_id,policy_id,grain,timestamp,asset_id,observation_family,observation_name` | false |
| `report.md` | optional | n/a | n/a | false |

Conditional / optional files can remain disabled in v1 implementation, but if generated they must have full
field registry, loader validation and display/download registration before becoming API-visible.
Generated file registry entries must also declare `display_when_absent=false` and
`summary_when_skipped=true/false` so loader, API and workbench use the same absent-file semantics.

## 7. `manifest.display` 合同

schema `0.7` 不让前端写死每张诊断表。后端应在 `manifest.display.tables[]` 中声明诊断 role。

`manifest.display` 必须是 API preview、direct download、ZIP download 和 workbench
展示的唯一合同。实现时不允许前端或 API route 用硬编码路径绕过 display。

每个 display table metadata 必须包含：

```text
key
label
path
format
role
diagnostic_family conditional
section
section_order
display_order
preview.default_limit
preview.max_limit
preview.row_count_capability
preview.route
preview.supports_offset
download.enabled
download.preset_tags
join_key
supports_combination_filter
dimension_table
conditional
required
empty_state
```

`diagnostic_family` 对 `role=diagnostic` 或 `role=detail` 是 required；对其他
role 必须为 null 或 absent。

最小 display JSON shape：

```json
{
  "schema_version": "0.7",
  "tables": [
    {
      "key": "diagnostic_metrics",
      "label": "Diagnostic Metrics",
      "path": "diagnostics/diagnostic_metrics.csv",
      "format": "csv",
      "role": "diagnostic",
      "diagnostic_family": "multi_family",
      "section": "diagnostics",
      "section_order": 60,
      "display_order": 20,
      "preview": {
        "default_limit": 200,
        "max_limit": 5000,
        "row_count_capability": "response_only",
        "route": "table-preview",
        "supports_offset": true
      },
      "download": {
        "enabled": true,
        "preset_tags": ["diagnostics", "all"]
      },
      "join_key": "combination_id",
      "supports_combination_filter": true,
      "dimension_table": false,
      "conditional": false,
      "required": true,
      "empty_state": "No diagnostic metrics are available for this run."
    }
  ],
  "downloads": {
    "items": [
      {
        "key": "run_context",
        "label": "Run Context",
        "kind": "metadata_file",
        "path": "run_context.json",
        "format": "json",
        "enabled": true
      }
    ],
    "presets": []
  },
  "charts": []
}
```

`run_context.json` 是 required metadata download item。direct download 和 ZIP packaging
必须使用同一份脱敏文件。

规则：

- `tables[].key`、`downloads.items[].key`、`downloads.presets[].key` 和 `charts[].key`
  在各自 namespace 内必须唯一；
- `downloads.presets[].items[]` 必须使用 `{kind, key}`，避免 table、metadata file 和 chart
  在不同 namespace 中同名时解析歧义；
- `tables[].path` 必须指向 `manifest.artifacts` 中声明的 run-relative artifact path；
  API preview 和 chart generation 不得读取 display 之外或 artifacts 未登记的路径；
- `downloads.items[].path` 必须指向 `manifest.artifacts` 中声明的相对路径；
- metadata file downloads 通过 `downloads.items[]` 暴露，不伪装成 table；
- `charts[].table_key` 必须引用 `tables[].key`；
- conditional table 未生成时，不得出现在 `tables[]` 或 downloadable item 中；
- conditional table 的 skipped/unavailable 原因必须在 `diagnostic_summary.csv` 中表达。
- `preview.row_count_capability` 只描述 preview route 是否能返回 row-count metadata；真实
  `total_rows_exact` 只能出现在 preview API response 中，不得作为静态 manifest 事实。

`role` 枚举：

```text
dimension
metric
portfolio
input_quality
summary
diagnostic
detail
```

`diagnostic_family` 只在 `role=diagnostic` 或 `role=detail` 时使用，取值如
`multi_family`、`tradability`、`path_risk`、`scenario`、`exposure`、`capacity`、`neutralization`。

前端规则：

- generic diagnostic table sections 读取 role 分组；
- definition-only dimension tables 不随组合切换过滤，展示为定义/配置表；
- conditional table 未生成时，display 不得声明为 downloadable；若需要暴露缺失原因，
  应通过 `diagnostics/diagnostic_summary.csv` 输出 unavailable/skipped 行；
- `dimensions/portfolio_assumptions.csv` 生成且带 `combination_id` 时，可以声明
  `supports_combination_filter=true`；
- validator 必须有正负测试：no-combination dimension table 不需要 `combination_id`，
  但不得为了通过校验填假的 `combination_id`；
- `diagnostics/diagnostic_summary.csv` 是默认 diagnostics summary surface；
- `diagnostics/diagnostic_metrics.csv` 是默认低基数 diagnostics metrics surface；
- `diagnostics/tradability_diagnostics.csv` 是 A 股 universe / eligibility / tradability time-series fact；
- `diagnostics/tradability_summary.csv` 如果生成，是 A 股 universe / eligibility / tradability compact materialized view；
- `diagnostics/portfolio_risk_summary.csv` 如果生成，只是轻量路径风险 materialized view；
- `diagnostics/scenario_metrics.csv` 承载 regime、cost、neutralization、benchmark 和 market segment 情景指标；
- `diagnostics/exposure_capacity_summary.csv` 如果生成，只是 risk exposure 和 capacity proxy 的 materialized view；
- `diagnostics/diagnostic_observations.parquet` 默认不进入 summary cards；
- 组合切换时，只刷新声明 `supports_combination_filter=true` 的表；
- CSV/detail table 使用 display preview row limit；
- chart table 才允许 bounded preview pagination 拼接完整当前组合时间轴；
- 前端不得重新计算 risk、cost、capacity、neutralization 或 regime results。

下载规则：

- `tables[].key` 必须唯一；
- direct download 只能下载 `download.enabled=true` 的 display item；
- ZIP preset 只能包含 display 已声明、且 manifest artifact paths 中存在的文件；
- API preview 必须使用 `preview.default_limit` 和 `preview.max_limit`，不得无界读取；
- Parquet detail preview 必须列裁剪和行限制，不能为了显示元数据全量载入。

## 8. Diagnostics 生成入口合同

schema `0.7` diagnostics 必须支持后端直接运行、CLI/config 运行和前端 Pipeline Builder
三类入口。三类入口只是提交方式不同，最终都必须收敛为同一个
`EvaluationSpec + DiagnosticsSpec -> DiagnosticRunner -> schema 0.7 ArtifactWriter`
链路。

Pipeline Builder 只是前端里创建 run 的表单入口，不是新的量化模块，也不是唯一生成路径。
前端只提交入口选择；后端负责 policy expansion、输入校验、artifact writing 和
`manifest.display` registration。

等价入口：

| 入口 | 用途 | 必须收敛到 |
|---|---|---|
| Python/backend runner | 开发者或脚本在后端直接构造 `EvaluationSpec` 和 `DiagnosticsSpec`。 | 同一个 `DiagnosticRunner`。 |
| CLI/config | 命令行或配置文件把 diagnostics 参数解析成 `DiagnosticsSpec`。 | 同一个 `DiagnosticRunner`。 |
| Frontend Pipeline Builder | 用户通过前端表单创建 pipeline/evaluation run。 | `POST /api/pipeline-runs` 后端再构造 `DiagnosticsSpec`。 |

后端直接运行示意：

```python
run_evaluation(
    evaluation_spec,
    diagnostics=DiagnosticsSpec(
        enabled=True,
        policy_set_id="a_share_default_v1",
        toggles={"regime": True, "cost": True, "capacity": True},
        materialized_views=["scenario_metrics", "tradability_summary"],
    ),
)
```

CLI/config 入口可以有自己的参数名或配置格式，但解析后的语义必须等价于
`DiagnosticsSpec`。CLI 不应绕过 policy registry、schema validation、`manifest.display`
registration 或 artifact writer。

前端入口：现有 executable A-share pipeline request `POST /api/pipeline-runs`。不新增
`/api/n5/*` 或独立 diagnostics run endpoint。

配置快照入口：现有 `POST /api/pipeline-snapshots` 可以保存和校验 diagnostics config，
但不运行 `DiagnosticRunner`，不生成 schema `0.7` artifact，也不进入 run index。

命名迁移规则：schema `0.7` 迁移时删除现有 `PipelineConfig.diagnostics: tuple[str, ...]`
旧字段。之后 API / frontend request 中的 `diagnostics` 只表示本节定义的 diagnostics request
object，不再允许 list/tuple 旧语义。实现必须添加负测，拒绝旧 list/tuple 形状。

最小 request fields：

下面只展示现有 `PipelineRunRequest` 的 `diagnostics` 子对象。`POST /api/pipeline-runs`
仍必须包含现有 A-share pipeline 字段和 `start_from`，不能实现为 diagnostics-only request。

```json
{
  "start_from": "existing_factors",
  "diagnostics": {
    "enabled": true,
    "policy_set_id": "a_share_default_v1",
    "toggles": {
      "regime": true,
      "cost": true,
      "capacity": false,
      "neutralization": true,
      "benchmark": false
    },
    "materialized_views": [
      "scenario_metrics",
      "tradability_summary"
    ]
  }
}
```

字段解释：

| 字段 | 中文解释 |
|---|---|
| `diagnostics.enabled` | 是否启用 post-run diagnostics policies。省略或为 `false` 时，仍写 schema `0.7` required diagnostics contract files，但只记录 disabled/skipped 状态，不生成 conditional materialized views。 |
| `diagnostics.policy_set_id` | 后端预设 policy set id，例如 A 股默认诊断包。 |
| `diagnostics.toggles` | 允许用户在预设包内开关 policy family；未知 key 必须拒绝。 |
| `diagnostics.materialized_views` | 请求额外生成哪些 conditional 表；未知 view 必须拒绝。 |

v1 默认 `capacity=false`。用户显式开启 capacity 时，如果缺少 weights、ADV 或 liquidity proxy，
输出 unavailable / warning summary，不 fail 整个 run。

`diagnostics.enabled` 真值表：

| request shape | normalized enabled | validation |
|---|---:|---|
| `diagnostics` absent | false | valid; write required diagnostics files with disabled status; no conditional views generated |
| `diagnostics.enabled=false` and no sibling fields | false | valid; write required diagnostics files with disabled status; no conditional views generated |
| `diagnostics.enabled` missing/false with non-empty `policy_set_id` / `toggles` / `materialized_views` | false | invalid, `error_code=diagnostics_disabled_with_options` |

共同后端规则：

- policy set registry 由后端配置或代码拥有，前端不得提交任意 policy JSON；
- allowed toggles、allowed materialized views 和默认 policy set 必须由现有 pipeline
  configuration/options response 返回或在 shared API type 中声明；不得使用阶段专用 endpoint；
- 缺少必需输入时，后端可以 fail fast，也可以生成 `diagnostic_summary.csv`
  unavailable 行，取决于 policy 的 `missing_input_mode`；
- 错误响应必须机器可读，至少包含 `error_code`、`message`、`field` 和 `details`；
- Pydantic validation、pipeline validation 和 policy validation 必须转换成同一个 API error
  envelope；不得让同一表单字段在不同失败路径返回不同错误形状；
- Pipeline Builder 入口只定义 diagnostics 相关字段，不重新设计完整 pipeline 表单；
- `diagnostics.enabled=false` 时，如果 `policy_set_id`、`toggles` 或 `materialized_views`
  非空，必须返回机器可读 validation error；不能静默忽略，也不能生成半截 diagnostics artifact。
- 后端直接 run、CLI/config 和前端 Pipeline Builder 不能各自实现一套 diagnostics 计算逻辑；
  它们只能作为 `DiagnosticsSpec` 的构造入口。

## 9. A 股假设边界

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
- sanitized `run_context.json.config`；
- `run_context.json.provenance`；
- `dimensions/diagnostic_policies.csv` 中的 `a_share_assumption` policy rows；
- cost、tradability 和 capacity diagnostics warnings；
- workbench copy。

不得进入：

- broker/exchange API；
- order model；
- fill model；
- paper/live trading；
- secrets or credentials。

## 10. Verification Strategy

### 10.1 Unit and contract tests

Python tests:

- `DiagnosticsSpec` validation；
- `DiagnosticPolicy` validation；
- schema `0.7` artifact path contract validation；
- schema `0.7` required column validation for every required file；
- schema `0.7` API/display/download paths accept only schema `0.7` artifacts；
- no historical artifact compatibility path is part of diagnostics runtime；
- stale run rejection；
- display manifest validation for new diagnostic roles；
- display manifest validation for unique table keys, preview limit schema, download items,
  ZIP preset membership, conditional table registration and display-approved paths；
- display manifest validation for `role=dimension` tables with no combination filter；
- negative display validation that rejects fake `combination_id` added only to make a definition table filterable；
- schema registry supports `required_if` / `nullable_if` / `sentinel_if` and family-specific validators；
- `dimensions/diagnostic_policies.csv` identity_scope validation；
- conditional `dimensions/portfolio_assumptions.csv` rows only when assumptions vary by combination
  or display needs combination-level assumptions；
- `diagnostics/diagnostic_summary.csv` has one status row per enabled policy；
- `diagnostics/diagnostic_metrics.csv` uses payload for family-specific metrics without sparse required columns；
- conditional `diagnostics/tradability_diagnostics.csv` covers universe / eligibility / tradability layers when enabled；
- conditional path-risk metrics mark unavailable when returns or weights are insufficient；
- all diagnostic fact tables include `diagnostic_status`, `status_reason`,
  `required_inputs_met`, and `warning`；
- invalid cost policy rejection；
- cost component validation for buy/sell turnover, stamp duty, transfer fee, commission,
  slippage and impact when A-share cost policy is enabled；
- cost validation that `cost_assumption_id` resolves to policy/context metadata, `is_execution_simulated=false`,
  and every enabled component declares source, status, unit and scope；
- cost policy validation for `effective_start` / `effective_end`, buy/sell side, `rate_as_of` and `rate_source`；
- regime `decision_usable=true` rejection when future information is used；
- ex-ante regime requires as-of timestamp, label availability timestamp, lookback end,
  data lag and leakage check id；
- neutralization scenario validation requires point-in-time control metadata, control lag, universe mask,
  missing/collinearity handling and leakage check id；
- capacity proxy validation distinguishes holdings participation from trade-value / weight-delta participation；
- risk summary validation requires machine-readable frequency, annualization, window, tail and ES proxy methods；
- ex-post regime rows exposed as retrospective segments and rejected from gates / constraints / portfolio policy inputs；
- missing optional exposure columns produce warnings；
- missing required columns fail or mark unavailable by policy；
- cost/capacity required-input gating；
- schema `0.7` generated run self-consistency and artifact/API/workbench contract validation；
- private absolute paths not written to public metadata。
- `source_alias` is not serialized into `manifest.json`, `run_context.json`,
  direct-download metadata or ZIP metadata；it remains API / run-index runtime context only。

API tests:

- generic table preview exposes diagnostics tables through `manifest.display`；
- legacy diagnostics route is removed and no API payload advertises it；
- direct downloads reject non-display items；
- ZIP presets include only display-approved items；
- `(source_alias, run_id)` identity preserved。

Frontend tests:

- Pipeline Builder form can submit only allowed evaluation fields and optional
  `diagnostics.policy_set_id` / allowed diagnostics toggles；
- backend expands policy sets, validates columns, writes artifacts, and owns
  `manifest.display` / `run_context.json`；
- Pipeline Builder request schema rejects unknown diagnostics toggles and returns machine-readable
  errors for missing policy inputs；
- frontend cannot synthesize artifact files, hidden download items, or diagnostics results；
- generic diagnostic sections render by role；
- combination switching refreshes combination-aware diagnostic previews；
- frontend does not infer latest rows from non-exact preview counts；
- download picker uses display metadata only。

### 10.2 Private acceptance

schema `0.7` diagnostics private acceptance should use current small, stable evidence:

- user-approved current small private acceptance runs；
- configured private run roots。

Do not expand to a single 32-combination 2023-2025 full run until low-memory weights/remap writing is designed.

### 10.3 Verification commands

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

Browser acceptance is required only after workbench behavior changes. API-backed workbench acceptance must run both FastAPI and Vite; an API-offline state is not acceptance evidence for diagnostics.

## 11. Implementation Plan Outline

Implementation v1 默认 `DiagnosticsSpec.materialized_views=[]`。第一刀验收优先依赖
`diagnostics/diagnostic_summary.csv` 和 `diagnostics/diagnostic_metrics.csv`；conditional
materialized views 只在用户、CLI/config 或验收明确请求时生成。

The implementation plan should split the work into reviewable vertical slices:

1. Add schema `0.7` path contract, schema registry index, display adapter/types and loader validation shell.
2. Add schema `0.7` API surface migration: run index, run evidence payloads, comparison service,
   download service, frontend shared API types, fixtures, summary-card aggregation and chart table selection.
3. Add `summary.json.run_id` and `run_context.json.provenance.run_id`; validate equality and preserve
   `(source_alias, run_id)` through API routes, downloads and frontend route state.
4. Add `run_context.json`; remove separate schema `0.7` `config.json` / `lineage.json` from artifact contract.
5. Remove old `PipelineConfig.diagnostics: tuple[str, ...]` semantics and add negative tests that reject
   list/tuple diagnostics payloads.
6. Add `DiagnosticsRequest -> DiagnosticsSpec` normalization for backend runner, CLI/config and Pipeline Builder.
7. Remove the legacy `/api/runs/{source_alias}/{run_id}/diagnostics/{metric_name}` route entirely.
   Schema `0.7` diagnostics access must go through `manifest.display` table preview / download
   surfaces only; no compatibility wrapper should infer `diagnostics/{metric_name}.csv` paths.
8. Add schema `0.7` base evaluation writer slice: `dimensions/combinations.csv`,
   `metrics/ic.csv`, `metrics/grouped_returns.csv`, `metrics/evaluation_metrics_summary.csv`,
   `portfolio/returns.csv`, and `input_quality/coverage_summary.csv`, with writer/loader/display/API tests.
9. Add `dimensions/diagnostic_policies.csv`, `diagnostics/diagnostic_summary.csv` and
   `diagnostics/diagnostic_metrics.csv` in the same writer/loader/display/API slice.
10. Add generic frontend diagnostics sections backed only by `manifest.display`.
11. Add conditional `dimensions/portfolio_assumptions.csv` only with full registry/display/download coverage.
12. Add conditional `diagnostics/tradability_diagnostics.csv` and `diagnostics/tradability_summary.csv` only with full registry/display/download coverage.
13. Add conditional `diagnostics/scenario_metrics.csv` only with separate family validators for regime,
    cost, neutralization, benchmark and market segment metrics; family-specific required fields must be
    expressed through schema registry `required_if` / `nullable_if` rules, not hidden prose.
14. Add conditional exposure/capacity/path-risk materialized views only when `DiagnosticPolicy.output_tables`
    allows them and `DiagnosticsSpec.materialized_views[]` requests them.
15. Add optional `diagnostics/diagnostic_observations.parquet` only behind an explicit detail flag.
16. Run private acceptance on user-approved small private runs / configured private run roots.
17. Update docs and code maps after implementation.

Do not start with frontend UI. The artifact contract and backend validation must land first.

## 12. Remaining Product Questions

These are non-blocking product / research choices. Schema identity, API migration,
display/download coverage, and field-count accounting must be resolved before implementation planning;
these choices can be finalized during the first implementation slice if needed.

1. Which market context columns are first-class diagnostics v1 inputs: benchmark return, industry, market cap, volatility, liquidity, beta proxy?
2. Should benchmark comparison be enabled only when benchmark returns already exist locally, or should it wait for a separate approved benchmark data preparation step?
3. Should optional `diagnostics/diagnostic_observations.parquet` be disabled by default for private acceptance runs?

Default recommendation:

- use benchmark return, industry, market cap, volatility and liquidity first; beta proxy only if already available or cheaply derived without external data;
- do not download benchmark/index data inside the diagnostics runner; require approved local benchmark columns or a separate data preparation step;
- keep `diagnostic_observations.parquet` disabled by default and rely on summary tables first.
