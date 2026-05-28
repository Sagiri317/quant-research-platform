# Engineering Standards / 工程规范

本文定义 `quant-research-platform` 的横切工程规则。它回答模块如何依赖、合同如何版本化、错误如何分类、测试如何守边界，以及未来扩展如何接入；目录归属以 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md) 为准。

本文的工程规则与 [ai-native-engineering-principles.md](ai-native-engineering-principles.md) 的长期原则一致，但具体执行规则仍以本文为准。

Current phase stop rule: 当前阶段是 Local Research Platform，架构风格是 Stable Core + High-Velocity Research Edge。本文中的 `apps/`、`workers`、`schedulers`、`storage`、`jobs` 等词汇是 long-term boundary vocabulary；除非当前任务已经明确批准并落在当前阶段，否则不得据此创建 speculative infrastructure、future capability scaffolding 或部署/调度/存储/作业系统。

## 1. 模块边界

```text
apps/                                                # 可部署应用入口，只做装配、路由、进程启动、依赖注入和运行配置
interfaces/                                          # 用户入口，只通过服务层和公开合同访问内部模块
pipelines/                                           # 编排流程，可以调用 research、markets、datasets、evaluations 和 artifacts
research/                                            # 研究生产层，可以调用 markets、datasets、evaluations 和 artifacts 合同
evaluations/                                         # 评价系统，可以调用 kernels，不直接依赖 interfaces 或 trading
kernels/                                             # 数值计算内核，不依赖业务模块
markets/                                             # 市场事实层，不依赖 interfaces、web、api 或 trading
datasets/                                            # 数据读取层，不定义市场口径，不依赖 research 结果
artifacts/                                           # 产物合同层，负责 schema、manifest、writer、loader 和 package
runtime/                                             # 本地运行上下文，不承载业务计算
```

硬规则：

- `apps/*` 不承载核心研究、评价、市场、交易或风控算法；这些逻辑必须下沉到 `src/quant_research_platform/` 的领域模块。
- `web` 不重新计算 IC、cumulative returns、turnover、cost、weights 或 risk metrics。
- API 层不绕过 `artifacts/display_manifest.py` 暴露下载和预览文件。
- `research` 不能直接触发 `trading` 或真实外部连接。
- `markets` 不能读取凭据、连接 vendor API 或承担用户权限逻辑。
- 跨目录调用优先依赖 `Spec`、`Schema`、`Manifest`、`Payload`、`Result` 等合同对象。
- `evaluations/diagnostics/` 属于 evaluation layer，可以依赖 evaluation contracts、diagnostic input bundle 和纯计算 helper；不得依赖 `artifacts/`、`interfaces/api/` 或 `web/`。
- Artifact writer 可以消费 diagnostic frames，并通过 artifact registry 校验它们；API 和 frontend 只能消费 artifact/display outputs，不能消费 diagnostics calculation internals。

## 2. 合同对象

```text
Spec                                                  # 调用方输入配置，例如 EvaluationSpec、PipelineSpec、ModelSpec
Config                                                # 运行时或内部配置，例如 RuntimeConfig
Schema                                                # 持久化或序列化结构，例如 ArtifactSchema
Manifest                                              # 一组产物或模型的元数据，例如 DisplayManifest、ModelManifest
Payload                                               # 模块间传递的内存数据包，例如 EvaluationPayload
Result                                                # 计算完成后的返回对象，例如 EvaluationResult
Record                                                # 表格、日志或数据库记录中的一行对象
Snapshot                                              # 某一时点状态，例如 PipelineSnapshot
Checkpoint                                            # 可恢复节点，例如 PipelineCheckpoint
Repository                                            # run、artifact、model 等对象的存取索引
Service                                               # API 层业务服务，必须带具体领域前缀
Error                                                 # 异常类型
```

所有可持久化、可下载、可复现、可跨版本读取的对象都必须有 schema 或 version 字段。

## 3. 注册表

需要扩展的对象使用注册表，不使用散落的 `if market == ...` 分支。

```text
factor_registry.py                                   # 因子 primitive、模板和候选来源注册
market_registry.py                                   # 市场实现和 capability 注册
kernel_registry.py                                   # dataframe、tensor、GPU、distributed 计算内核注册
pipeline_step_registry.py                            # pipeline step 注册
artifact_type_registry.py                            # artifact type 和 display contract 注册
schema_registry.py                                   # artifact table/file contract、字段、grain、key、条件规则和 display/download 可见性注册
```

注册对象必须声明：

```text
name                                                  # 稳定名称
version                                               # 对象版本
inputs                                                # 输入字段、schema 或依赖
outputs                                               # 输出字段、schema 或产物
capabilities                                          # 支持能力，例如 daily bars、options chain、live trading
```

Schema registry 归属：

- 可机械校验的 artifact table / file contract 必须落在代码 registry 和测试里，不能只靠 Markdown 维护。
- Registry 负责字段名、dtype、nullable 规则、grain、unique key、sort key、required files、conditional required files、sentinel values、generated-file index、display/download 可见性。
- Writer、loader、lightweight run index、API payload builder、download packaging 和 frontend shared types 必须消费或测试 registry contract，不能各自维护平行字段清单。
- `manifest.display` 仍然是用户可见暴露边界。Registry-backed contract 不授权 frontend-only shortcut，也不授权 backend-only hidden download。

## 4. 版本和迁移

```text
artifact_schema_version                              # artifact 持久化版本
dataset_schema_version                               # dataset manifest 或 schema 版本
factor_definition_version                            # 因子定义版本
model_manifest_version                               # 模型元数据版本
pipeline_spec_version                                # pipeline 配置版本
evaluation_spec_version                              # 评价输入合同版本
api_version                                          # HTTP / SDK 公开接口版本
```

迁移边界：

```text
storage/migrations/                                  # 数据库迁移，网站化后使用
artifacts/migrations/                                # artifact schema 迁移，只有明确设计后才允许
datasets/migrations/                                 # dataset manifest 迁移，只有明确设计后才允许
```

当前默认策略：unsupported artifact schema 作为重生成候选，loader 不做宽松兼容。

## 5. 错误分类

错误必须区分可行动原因，避免统一抛出 `ValueError`。

```text
ContractError                                         # 合同不满足，例如 schema version、字段名或 display manifest 错误
DataQualityError                                      # 数据质量问题，例如缺失、重复、异常、不可交易样本
MarketRuleError                                       # 市场规则问题，例如合约到期、停牌、涨跌停或 calendar 无法对齐
EvaluationError                                       # 评价失败，例如输入组合不合法或指标无法计算
ArtifactError                                         # artifact 读取、写入、校验、打包失败
PipelineError                                         # pipeline step、checkpoint、重试或依赖失败
PermissionError                                       # 权限不足，网站化后使用
EnvironmentError                                      # 环境、路径、依赖、运行时配置错误
```

网站化后错误应有稳定错误码，例如：

```text
QRP-DATA-001                                          # 缺少必需字段
QRP-ARTIFACT-001                                      # unsupported artifact schema
QRP-PIPELINE-001                                      # pipeline checkpoint invalid
QRP-AUTH-001                                          # permission denied
```

## 6. 配置和密钥

```text
config/                                               # checked-in 默认配置和模板
runtime/                                              # 本地运行上下文和路径解析
pipelines/                                            # pipeline spec 和 checkpoint
secrets/                                              # 长期预留，管理 broker key、vendor key、OAuth secret 和 webhook secret
```

规则：

- 私有路径、token、cookie、broker key、vendor key 不进入 checked-in 配置。
- pipeline config 必须可版本化、可审计、可复现。
- `.env`、密钥和凭据文件不得被读取、打印、提交或写入 artifact。

## 7. 标识符

平台合同中使用稳定 ID，不让原始 vendor 字段泄漏到跨模块合同。

```text
asset_id                                              # 平台资产 ID
instrument_id                                         # 金融工具 ID
contract_id                                           # 合约 ID
venue_id                                              # 交易场所 ID
dataset_id                                            # 数据集 ID
run_id                                                # 单个 run 的稳定 ID
source_alias                                          # run root 或 run source 别名
workspace_id                                          # 网站化后的 workspace ID
project_id                                            # 网站化后的 project ID
artifact_id                                           # artifact ID
model_id                                              # 模型 ID
experiment_id                                         # 实验 ID
job_id                                                # 后台任务 ID
```

市场内部可以保留 `stock_code`、`symbol`、`ticker`、`secid` 等原始字段，但进入平台合同后必须映射为稳定 ID。

## 8. 测试分层

```text
tests/unit/                                           # 单元测试，覆盖纯函数、schema、validation 和小对象
tests/contracts/                                      # 合同测试，覆盖 schema、display manifest、API payload 和版本规则
tests/integration/                                    # 跨模块主链路测试，例如 evaluation -> artifacts -> API
tests/regression/                                     # 历史 bug 和退休行为测试
tests/markets/                                        # 市场字段、calendar、合约和数据源映射测试
tests/browser/                                        # 前端浏览器验收测试，覆盖 stale evidence 和下载流程
tests/performance/                                    # 性能 smoke，覆盖行数、内存、耗时和 artifact 大小预算
```

关键边界必须有测试：

- API 和 Web 只能通过 `manifest.display` 暴露用户可见 artifact。
- run identity 必须保留 `(source_alias, run_id)`。
- loader 必须拒绝 unsupported schema。
- research 不能直接调用 trading。
- market module 不能读取凭据或启动外部连接。

## 9. 生成物和缓存

生成物默认不进入 Git，除非是明确设计的小型 fixture。

```text
reports/runs/                                         # 评价 run 产物
reports/private_real/                                 # 私有真实数据报告
reports/profiles/                                     # profiling 输出
download cache                                        # ZIP、临时下载和浏览器下载
screenshots                                           # 浏览器验收截图
logs                                                  # 运行日志和服务日志
```

所有用户可见输出必须通过 artifact/display 合同进入 API 和前端，不添加前端私有 shortcut 或后端隐藏下载。

## 10. Apps 和 Package 边界

未来网站化后采用 repo-level `apps/`，避免可部署应用和可复用平台代码混在一起。该段只定义长期边界词汇，不授权当前 Local Research Platform 阶段创建 repo-level `apps/`、workers、schedulers 或部署装配层。

```text
apps/web                                             # 前端应用入口，负责路由、页面、状态管理、API client 和可视化
apps/api                                             # HTTP 应用入口，负责 FastAPI app、middleware、auth、routes 装配和错误响应
apps/workers                                         # 后台 worker 入口，负责读取 job、装配运行上下文、调用领域服务和回写状态
apps/schedulers                                      # 调度入口，负责定时触发、依赖触发、数据到达触发和任务创建
src/quant_research_platform                          # 平台领域代码，负责研究、评价、市场、数据、产物、流水线和平台域
```

规则：

- `apps/*` 可以依赖 `src/quant_research_platform/*`，反向依赖禁止。
- `apps/*` 里只允许薄装配代码，不能出现因子公式、评价指标、撮合逻辑、交易风控或数据清洗主逻辑。
- 当前 Local Research Platform 阶段保留 `web/` 作为当前 workbench；本地 FastAPI 适配迁到 `src/quant_research_platform/interfaces/api/`，不再以旧顶层 API 包作为目标位置。
- 未来网站化阶段才创建 repo-level `apps/api`，它是可部署 HTTP 应用装配层；当前阶段的 `interfaces/api` 是本地接口适配层，二者功能不重叠。
- `interfaces/` 是平台代码中的交互合同和适配层，不是长期可部署应用目录。
- 当前阶段不得为了 future capability 创建空 `apps/*`、workers、schedulers 或 package shells。

## 11. Storage 和 Repository 规则

`storage/` 是长期持久化基础设施词汇，不是所有数据库对象的归宿；当前阶段不因该词汇创建持久化基础设施目录或数据库层。

```text
storage/database_session.py                          # 数据库连接、事务和 session 生命周期
storage/migrations/                                  # 数据库 migration
storage/object_store.py                              # artifact、报告和大型对象存储适配
storage/cache_store.py                               # 可重建缓存和失效策略
storage/search_index.py                              # 搜索索引适配
```

归属规则：

- `User` 属于 `identity/`，不属于 `storage/`。
- `Workspace` 和 `Project` 属于 `tenancy/`，不属于 `storage/`。
- `ArtifactRecord` 属于 `artifacts/`，不属于 `storage/`。
- `PipelineJob` 属于 `jobs/` 或 `pipelines/`，不属于 `storage/`。
- Repository 可以封装持久化访问，但对象语义仍归属于对应领域。
- 事务边界必须显式；跨 artifact、job、audit 的写入不能靠隐式全局 session。

## 12. Jobs、Workers 和调度规则

长任务必须通过 job 合同管理，不能让 HTTP request 直接执行重计算。该规则属于长期平台边界；当前阶段不创建 job system、worker runtime 或 scheduler，除非有单独批准的当前阶段任务。

```text
JobSpec                                              # 用户或系统提交的任务输入
JobRecord                                            # 持久化任务状态、租户、资源、进度和错误
JobResult                                            # 任务完成后的领域结果或 artifact 引用
WorkerHeartbeat                                      # worker 存活、负载和当前任务状态
RetryPolicy                                          # 可重试错误、最大次数、退避和人工干预规则
```

规则：

- job 必须有稳定 `job_id`、状态机、创建者、租户、输入摘要、输出引用和 audit record。
- worker step 必须尽量幂等；重复执行不能覆盖私有数据或破坏已有 run。
- 长任务必须支持取消、超时、失败原因、checkpoint 和资源预算。
- 调度器只创建 job，不直接绕过 job/worker 边界执行领域逻辑。
- GPU、分布式、tick 回放和大型模型训练必须有单独资源队列和并发限制。

## 13. API、SDK 和前端合同

公开接口必须版本化，并且和 artifact/display 合同保持一致。

```text
api_version                                          # HTTP / SDK 版本
request_schema_version                               # 请求 schema 版本
response_schema_version                              # 响应 schema 版本
error_code                                           # 稳定错误码
pagination                                           # 大列表分页和游标
filter_spec                                          # 列表过滤和排序合同
```

规则：

- API 不返回内部 Python 对象 repr，不泄漏本地路径、私有字段或异常栈。
- 前端列表必须分页或虚拟化，不能假设 run、artifact、candidate 或 job 数量很小。
- SDK 只能依赖公开 API 或稳定 Python public contract，不能 import 私有实现文件。
- 破坏性 API 变更必须有迁移说明、测试和版本标记。
- 用户可见表格、图表和下载继续以 `manifest.display` 为唯一展示合同。
- screening、fundamental analysis、ranking、watchlists、alerts、portfolio analytics、research cards 等用户产品功能必须经过权限、审计、数据许可、合规和计费边界，不能只是前端私有状态或本地 shortcut。

## 14. 安全和隔离规则

长期平台必须假设用户、数据源、插件和生成代码都可能不可信。

```text
secrets                                              # broker key、vendor key、OAuth secret 和 webhook secret 管理
access_control                                       # 资源权限、策略判断和策略执行
audit                                                # 访问、下载、运行、审批、权限变化和交易事件
sandboxes                                            # 用户代码、notebook、插件和 AI 生成代码隔离执行
approvals                                            # 高风险操作审批，例如数据接入、模型上线、实盘启用
```

规则：

- secrets 只通过专门接口按最小权限读取，不进入 artifact、日志、报告、聊天、浏览器状态或异常消息。
- 用户代码、插件、notebook 和 AI 生成代码必须在 sandbox 中执行，不能直接访问 host 文件系统、凭据或交易连接。
- 权限校验必须在 API、worker 和下载路径都执行，不能只靠前端隐藏按钮。
- 审计记录必须包含 actor、tenant、resource、action、time、result 和 request id。
- 实盘、下载私有报告、发布公开链接、模型上线和权限提升必须进入 approvals 或等价审批边界。

## 15. 性能和规模规则

规模相关能力必须有预算，不靠“能跑一次”作为完成标准。

```text
row_count_budget                                     # 输入行数预算
memory_budget                                        # 内存预算
runtime_budget                                       # 运行时间预算
artifact_size_budget                                 # 产物大小预算
download_size_budget                                 # 下载包大小预算
frontend_render_budget                               # 前端首屏和表格渲染预算
```

规则：

- 大 panel、tick、order book、模型训练和批量候选必须先有 small smoke，再跑全量。
- artifact 产物要列裁剪、分表、压缩和索引，不能把所有中间结果塞进单个巨大 CSV。
- API 大响应必须分页、采样或生成可下载 artifact，不直接返回全量表。
- 前端不能一次性渲染超大表格；必须用分页、虚拟滚动或 server-side preview。
- 性能测试优先覆盖行数、内存、耗时、artifact 大小和浏览器渲染，而不是只看功能测试。

## 16. 扩展边界

长期可以支持插件和外部扩展，但必须满足：

- 插件声明 name、version、inputs、outputs、capabilities。
- 插件不能绕过 access control、audit、controls 或 artifact/display 合同。
- 插件不能直接读取 secrets。
- 插件如果生成用户可见输出，必须写成 artifact/display 可消费结构。
- 插件如果触发交易，必须走 trading、controls 和 approvals 边界。
