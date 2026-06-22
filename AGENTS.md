# QRP 智能体执行规则

本文件给 Codex、GPT、Claude 和其他 AI agents 使用。它只放硬边界、任务路由和停止条件，故意保持短小稳定。详细流程只在任务需要时从 `.agents/` 按需加载。

## 1. 项目边界

QRP / Quant Research Platform 是本地量化研究平台，不是投资建议、simulated order execution（模拟交易）、real-money order execution（实盘交易）、external execution-venue connector（券商/交易所连接器）或生产部署系统。

当前开发基线是 `feature/qrp-clean-rewrite`。除非用户明确提出 merge/release 任务，不要把 `main` 当作当前实现目标。

当前源码拓扑：

```text
src/quant_research_platform/
  contracts/ storage/ catalog/ sources/ markets/ research/
  workflows/ compute/ agents/ interfaces/ reports/ runtime/
```

当前 public surface（公共接口面）只有：

```text
GET /api/0.1/catalog/factors
GET /api/0.1/catalog/labels
GET /api/0.1/research-datasets/{dataset_id}/requirements
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

当前 approved private local CLI（获批私有本地命令行）只有：

```text
qrp-private schema-inventory --project-root . --output <path>
qrp-private evidence run
```

`qrp-private evidence run` 是 private local verification harness，用于本地
smoke/evidence generation；不是 public CLI；不是 public execution mode；不是
public evaluate、predict 或 report command；不是 simulated or real-money order execution；不承诺收益；
不输出 row-level private data。

## 2. 不可违反原则

- Semantic closure（语义收口）：contracts、命名、owner layer、tests、docs、artifacts、handoff 证据必须一致闭合。
- Naming first-class（命名一等）：命名是架构问题，不是 cosmetic（表面修饰）。名称必须表达归属层和生命周期，不暴露临时实现细节。
- Replace over parallel（替代优先）：能替代就删除旧路径；默认不加 compatibility alias、legacy shim、wrapper、旧 re-export。
- Minimal public API（最小公共接口）：未经用户明确批准，不新增 public API、CLI、registry、store、facade、artifact type 或 execution mode。
- Evidence over claims（证据优先）：只有实际运行并通过的检查才能写 `PASS`；失败写 `FAIL`；未运行写 `NOT_RUN` 或 `BLOCKED`。
- Private-data safety（私有数据安全）：禁止泄露 row-level private data、vendor marker、stock-code sample、本地绝对路径、credential、token、含私有上下文的 raw traceback。
- Layer-local change（局部改动）：新功能应主要改 owner layer 的少量文件；跨 storage/evaluation/report/workflow/interfaces 大改前必须先做 architecture closure（架构收口）。
- No handoff-driven architecture（handoff 不反向驱动架构）：`.run` 证据可用于验证和回传，但不定义架构。
- Extensible without abstraction creep（可扩展但不抽象膨胀）：为 ML/DL、diagnostics、factor factory、多市场、高频、frontend 预留方向，但不提前造通用框架。

## 3. 治理路径权威

- 项目治理根是 `.agents/`，保持 agent-neutral（智能体中立），适配 Codex / GPT / Claude / 其他 agents。
- global `.codex/` 是用户级 Codex 状态；只有用户显式提供时可作为只读证据源。不要修改它，不要把 raw 内容复制进 repo。
- repo-local `.codex/` 默认不创建。Codex-specific config、hooks、custom agents 只能作为 draft-only adapter（草案适配器）文档或 `.example`，不能默认启用。
- 默认不新增 `CLAUDE.md`、`.claude/`、hooks、MCP server、plugin、subagent runtime config。确有必要时先写 draft-only 方案并等待用户批准。

## 4. 短提示词任务路由

用户输入很短时，也必须先精准路由，不要猜成实现任务。编辑前先选择一个 mode ID：

```text
small-fix | standard-implementation | goal-closure | independent-review |
handoff-packaging | merge-gate | release-gate | docs-status-sync |
governance-redesign | apply-only-validation | ambiguous
```

最小加载规则：

- 每次任务只读本文件和用户本轮指令。
- QRP 维护任务再读 `.agents/skills/quant-research-platform-maintenance/SKILL.md`。
- 当前事实按需读 `.agents/briefs/current-qrp-status.md`、`.agents/briefs/repo-map.md`、`.agents/briefs/user-principles.md`。
- mode 和 gate 按需读 `references/task-modes.md`、`references/governance-gates.md`。
- 实现或审查归属按需读 `context-routing.md`、`owner-layer-routing.md`、代码和 tests。
- 默认一次只加载一个 primary template（主模板）；不要一次读完所有 templates。

如果任务仍是 `ambiguous`，只在 blocking ambiguity（阻塞性歧义）时提问；不要把 review、handoff、merge、release、governance 任务静默改成业务实现。

## 5. 停止条件

出现以下情况且用户没有明确批准时，停止并报告：

- 工作树、分支、base branch 或用户已有改动归属不清。
- public API/CLI/schema/report display contract 变化。
- 新增 public registry、store、facade、service、compatibility layer、execution mode、artifact type 或 dependency。
- 新增顶层包、目录搬迁、跨层重构。
- 需要访问 private data，但任务授权不清。
- 涉及 deployment、scheduler、worker、model registry、feature store、external LLM API、vendor API、external execution-venue connector、simulated or real-money order execution 或交易自动化。
- 验证失败或未知，但继续推进必须把它写成 `PASS`。
- 同类测试失败反复不清，或出现性能/内存回退。

## 6. Token economy（token 节约）

- 采用 diff-first、file-second、repo-last：先看 `git diff --name-only` / `git diff --stat`，再读相关文件，最后才扩到 repo 搜索。
- same-issue-family scan（同族问题扫描）先用 `rg -n` 看命中摘要，最多打开最相关的少数文件。
- `.run` 证据先读 summary/manifest/index，不读 raw logs 或完整 stdout，除非 summary 指向具体失败。
- 外部工具先受 QRP gate 约束；Context7/GitHub MCP、plugins、superpowers skills 可辅助，但不能绕过本文件。
- Final report（最终报告）按任务规模压缩；小任务只列变更、验证、风险，不粘贴长日志。

## 7. 验证真实性

中等及以上任务至少报告：

```text
changed files | checks actually run | checks not run | PASS/FAIL/NOT_RUN/BLOCKED |
private-safety result | public-surface result | docs/status sync result | remaining risks
```

提交或 handoff 前必须运行 `git diff --check` 和相关确定性治理脚本。未运行的检查必须说明原因，不得写成通过。

## 8. 数据与 artifact 安全

私有根目录只允许本地使用：

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_research/
```

handoff package、public-safe artifacts 禁止包含 parquet、csv、pkl、pickle、feather、arrow、raw vendor exports、row-level private data、credentials、tokens、旧 handoff zip/json，除非 package rule 明确允许脱敏摘要。

## 9. Git 安全

编辑前运行 `git status --short --branch`。不要丢弃用户改动。除非用户明确批准，不要 amend、rebase、squash、force-push、删除分支、merge 到 `main` 或 push `main`。实现和治理任务默认使用 `feature/<task-name>`。
