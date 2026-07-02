# QRP Agent 指令

本文件是 QRP 仓库中面向 coding agent 的根指令。它只保存长期稳定的执行规则，不保存当前分支、当前阶段、最近运行结果、PR 状态或下一步任务。所有 agent 在计划、审查、修改、验证前，必须先从 live git baseline 重建现场事实。

## 1. 项目边界

QRP 是本地量化研究平台，目标是支持数据合同、因子研究、特征处理、模型训练、预测、评估、组合构建、报告和 evidence 管理。QRP 不是生产交易系统，不是 paper trading 系统，不是 live trading 系统，不接 broker，不接 exchange，不提供 investment advice。

未经用户明确批准，禁止新增 public API、CLI、registry、store、facade、scheduler、worker、deployment、external LLM API、vendor API、broker connector、exchange connector。未经用户明确批准，禁止访问 private data 或生成 row-level private-derived review-safe output。

## 2. live git baseline

任何任务开始时，必须运行并记录：

```bash
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
git branch -a
```

如果任务涉及远端分支或 merge，还必须检查 upstream、divergence 和目标分支。当前事实只来自本轮用户指令、live git baseline 和当前仓库文件。

仓库文档不得声明当前分支、当前阶段、当前 PR 状态或当前 merge gate。需要当前事实时，使用 live git baseline 和当前用户指令。

## 3. 事实源优先级

执行优先级如下：

1. 用户本轮明确指令和 approved task envelope
2. live git baseline 与当前仓库文件
3. 本文件
4. `.agents/skills/quant-research-platform-maintenance/SKILL.md`
5. `.agents/briefs/repo-map.md` 和 `.agents/briefs/user-principles.md`
6. `.agents/skills/quant-research-platform-maintenance/references/**`
7. `docs/**` 的当前合同文档
如果文档和代码、tests、contracts、schemas、runtime behavior 冲突，以可执行合同和当前仓库状态为准；停止并报告冲突，不要用叙事文档覆盖现场事实。

## 4. `.agents` 与 `.run`

`.agents/skills/quant-research-platform-maintenance/scripts/` 是 tracked governance tooling。handoff builder、verifier、governance check 等工具必须从这里读取。

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源，集中定义 active governance docs、current-contract-only 扫描规则、版本面规则、negative-test allowlist、source-tree scan allowlist、共享安全路径规则和 status token。其他治理 gate 可以分层执行，但不得复制一份平行规则表。

`.agents` handoff verifier 必须校验正式 handoff 的 REVIEW_SAFE evidence 完整性，并拒绝 status-only 摘要。handoff package layout、sidecar shape、task metadata shape 和 validation summary 合同由 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_contract.py` 拥有；receipt schema、receipt invariant 和 receipt 聚合由 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/receipt_contract.py` 拥有；governance scan、source-tree scan、共享安全路径和 status token 规则由 `governance_contract.py` 拥有。handoff 使用 compact replayable receipts：可从 `final-branch.bundle` 重放的检查只保存 receipt，由 verifier 在 bundle clone 中重放并比对；不可重放的 private/local 检查只通过 check receipt 记录 REVIEW_SAFE 状态。当前 validation 状态由 builder 从 receipts 派生为 `handoff_evidence/validation-summary.json`。Codex 不应手写 handoff evidence，也不应把 raw grep output、完整 pytest log、完整 stdout/stderr、wheel、sdist、完整 subagent transcript 或本地临时安装目录打入 handoff。

`.run/` 只是 output layer 和 evidence layer。它可以保存运行产物、handoff zip、manifest、logs、reports、temporary evidence，但不能反向定义架构、合同、工具入口或当前状态。

正式 pytest profile runner 位于 `tests/tools/run_profile.py`。`.run/` 只保存输出、临时文件、profile 和 handoff/review artifacts。测试入口保持 Python-only；`.ps1` 和 `.sh` 不属于官方测试入口层。测试 profile 的输出路径和临时路径必须通过参数指定，未运行项状态为 `NOT_RUN`。

## 5. 验证状态语义

所有最终报告、sidecar、handoff、review、prompt 和 PR 总结只能使用以下状态：

- `PASS`：实际运行并通过，必须附 command、scope、result 或 artifact path。
- `FAIL`：实际运行并失败，必须说明失败范围和下一步。
- `NOT_RUN`：未运行，不得伪装为 PASS。
- `BLOCKED`：因环境、依赖、权限、超时或边界无法运行，必须说明阻塞条件。

pytest PASS、governance PASS、handoff verifier PASS、share-safety PASS、source-tree scan PASS、compare PASS、sidecar PASS 不能互相替代。每个 PASS 只覆盖对应命令和对应 scope。

## 6. 架构职责

`contracts/` 只定义结构合同、枚举、ID、时间窗口和基础校验。`catalog/` 只负责注册描述与组合入口。`sources/` 负责数据源摄取、raw schema、字段映射和 raw/interim 转换，不承载市场语义。

`markets/` 是通用市场领域层；`markets/implementations/a_share/` 承载 A 股 calendar、processed data、mask、factor/label binding 和 profile/scenario。通用 session window 的唯一 owner 是 `markets/session_windows.py`。

`policy/` 只负责共享安全规则、ID 安全判断和 PASS/FAIL/NOT_RUN/BLOCKED 状态聚合，不 import QRP 业务层。`contracts/` 只定义结构合同、枚举、ID、时间窗口和基础校验，不保存共享安全规则事实。`catalog/` 只负责 generic registry class。

`profiles/` 是 API、CLI、agent 和 workflow 共用的组合根，可以组装 catalog、source、market implementation 和 materializer。`research/` 是通用研究计算层，只构建 result，不写文件，不依赖具体市场实现或 sources。`compute/` 是 dataframe/tensor、portfolio math 和 performance 数值核心。`storage/` 只负责 dataset IO、manifest、fingerprint、compare 和 data/raw、data/interim、data/processed、data/research layout，不定义 run outputs、reports 或 review-safe file writers。

`runtime/` 只提供 resources、logging、workflow records 和 execution metrics payload，不写报告内容。`reports/` 只 build/render evidence/report 内容，不写文件，不重算研究逻辑。`artifacts/` 只负责把 LOCAL_ONLY run outputs 和 REVIEW_SAFE evidence 写成文件。`workflows/` 只编排 research -> reports -> artifacts，不定义 policy regex，不直接写底层文件。`interfaces/` 只提供已批准的 API、CLI 和 service adapter。`agents/` 保持 proposal-only。`.agents/` 只承载治理、handoff 和 prompt tooling。

## 7. current-contract-only

QRP 采用 current-contract-only。普通 tests 只保护当前合同；边界回归由对应 owner-layer negative tests 和治理 gate 控制。未经用户批准，禁止新增兼容读取、兼容字段、alias 字段或 silent drop。

新实现应替代被替换实现，不应并行保留两个语义相同的入口。确需并行时，必须说明语义差异、生命周期、owner layer 和未来收敛方式。

Current-contract path rules are owned by `governance_contract.py` and enforced by `verify_governance`.

版本面和命名规则由 `governance_contract.py`、owner-layer negative tests 和治理 gate 控制。QRP-owned generated JSON 顶层身份只使用 `schema_id`；Python internal model version 字段不能作为 generated JSON identity。

内部接口、owner-local 命名、parallel implementation 和 valuable deferred feature 的具体规则由 references、`governance_contract.py` 和 owner-layer gates 执行；顶层文件只保留边界原则，不复制细则。

## 8. private safety

禁止把 private data、vendor marker、本地绝对路径、股票代码样本、raw traceback、row-level sample 写入 REVIEW_SAFE docs、handoff、reports、logs 或 final response。需要引用本地验证时，只写摘要、计数、hash、schema、command status 和 REVIEW_SAFE evidence。run output 主轴是 `LOCAL_ONLY` 和 `REVIEW_SAFE`。

## 9. 修改边界

默认允许修改与任务直接相关的最小文件集合。修改 docs、governance scripts、handoff scripts、tests 时，必须同步更新相关验证。修改业务逻辑时，必须有用户明确授权。

禁止为了让测试通过而恢复被替换的文档事实。遇到脚本引用被替换的 governance path 时，应修脚本和测试。

## 10. 推荐验证命令

按任务范围选择最小充分命令，不要把未运行项写成 PASS：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
```

如果完整测试因环境或超时不可行，必须分段运行并明确写 `BLOCKED` 或 `NOT_RUN`。

## 11. 提交流程

commit 前必须运行 `git status --short --branch` 并检查 diff。禁止 force push、main push、未经批准的 rebase、amend、squash、merge main。最终报告必须列出 branch、HEAD、changed files、validation results、remaining issues 和 working tree status。
