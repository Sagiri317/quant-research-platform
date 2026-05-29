# Next Stage Worktree Plan / 下一阶段分支与工作树计划

Historical boundary: this file is a frozen operations planning record from
2026-05. Current Git/worktree/task-mode execution rules are maintained in
`.codex/templates/*`, the quant research platform maintenance skill, and its
context-routing reference. Do not update this archived plan into current truth.

本文只记录当前 active branches、future branch suggestions 和 current worktree strategy。
已完成分支历史、旧 smoke run stories 和 implementation-heavy notes 不在本文展开。
功能验收仍以后续各分支的测试和 code review 为准。

## 当前基线

- 当前主线是 `main`。
- N1/N2/N3/N4/N7/N8 已合入 `main`，后续新任务应基于当前 schema `0.7`、
  `combination_id`、portable display preview metadata 和 artifact/API/frontend 合同。
- 当前本地 `reports/private_real/` 保留 schema `0.7` 2023-2025 canonical、
  `rev_120d` parity shard 和当前 Factor Factory smoke；旧 schema runs 和早期 smoke
  只作为历史记录。
- Pipeline 对外语言固定为 snapshot vs run：
  - `POST /api/pipeline-snapshots` 只保存配置。
  - `POST /api/pipeline-runs` 执行到 evaluation。
  - executable run 只用 `start_from` 控制开始层。
- 当前没有记录中的进行中功能分支。

## 原则

- 每个分支只交付一个可 review 的主题，避免把 artifact、API、pipeline、frontend 和 docs 大范围混在一起。
- 新增用户可见 artifact、API、download、frontend table/chart 时，同步 `manifest.display`、API tests、frontend tests 和 code-map。
- 不在这些分支里接入外部 API、自动下载 CSMAR、读取凭据、部署服务、paper trading 或 live trading。
- 工作树创建前必须确认当前 checkout 干净，并确认 `.worktrees/` 被 Git 忽略。
- Vite 默认资源清理仍是独立低优先级 hygiene，不混进功能分支。

## 建议后续分支

| 分支 | 路线图任务 | 范围 | 依赖 | 冲突风险 |
|---|---|---|---|---|
| `codex/tradability-diagnostics-summary` | N2 follow-up / diagnostics summary | `tradability_diagnostics_summary` artifact、display/API exposure、frontend summary card copy/tests | 依赖当前 display preview contract；前端只消费 artifact/API summary，不从 preview 计算全交易日平均 | 中。涉及 artifact/API/frontend 合同，但研究语义窄 |
| `codex/risk-regime-diagnostics` | N5 full 风险、regime、成本和容量扩展 | research diagnostics artifacts、API exposure、report/docs | 依赖 schema `0.7`、current canonical sample 和 N8 display/API/frontend helper | 中到高。需要先写详细计划 |
| `codex/ml-prediction-handoff` | N6 ML prediction handoff | prediction manifest、handoff schema、evaluation adapter-like input、tests/docs | 可独立设计，但应复用 evaluation run identity，不能混入 N2/N5 | 中到高。合同设计优先 |
| `codex/pipeline-history-ui` | Pipeline history UI | 本地 workbench history visibility，不改变 snapshot/run/start_from 语义 | 依赖当前 pipeline API 和 evidence run selection 行为 | 中。容易和 pipeline semantics 或 frontend state 混淆 |

## 推荐顺序

1. `codex/risk-regime-diagnostics`
   - 先写独立计划。
   - 不混入 N6 prediction handoff。
   - 复用当前 display preview、download allowlist、A-share service entry 和 current evidence samples。

2. `codex/ml-prediction-handoff`
   - 单独设计 prediction input contract。
   - 不把模型训练、model registry 或 feature store 混入当前阶段。

3. `codex/tradability-diagnostics-summary` 或 `codex/pipeline-history-ui`
   - 前者是窄 diagnostics summary。
   - 后者是本地 workbench 可见性改善，不改变 pipeline runtime 语义。

## 不建议并行的组合

- 不要把 N5 风险/regime 扩展混进 N2 tradability summary。两者都属于 diagnostics，但研究语义和验收口径不同。
- 不要把 N6 ML prediction handoff 混进现有 factor evaluation runner。Prediction 是新的输入合同，应单独设计。
- N5/N6 不应重新打开 N8 已收敛的 display preview、download allowlist、factor-template API、A-share service entry 和 CSV upload legacy/dev 边界。
- Pipeline history UI 不应和 pipeline runtime semantics 改动混在同一分支。

## 工作树创建策略

后续新任务从干净 `main` 创建新工作树：

```powershell
git status --short
git branch --show-current
git worktree list
git check-ignore -q .worktrees
git worktree add ".worktrees/<task-name>" -b "codex/<task-name>"
```

如果 `.worktrees/` 未被忽略，先更新 `.gitignore` 并单独提交该治理变更，再创建工作树。
不要把工作树目录加入 Git。

## 每个分支的最低验收

Artifact/API/backend 分支：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api tests/pipelines/a_share tests/research/factor_factory
conda run -n analysis python -m ruff check src tests scripts
git diff --check
```

Frontend 分支：

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

跨 artifact/API/frontend 合同分支应补跑：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests scripts
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

需要真实浏览器 UI 抽查时，按项目维护 skill 的浏览器验收流程启动本地 API/Vite
并运行 Playwright，避免生成顶层 `output/`：

```powershell
scripts\start_private_workbench.ps1
scripts\stop_workbench.ps1
```

## 相关路线图

- Active roadmap: [next-stage-roadmap.md](../../architecture/next-stage-roadmap.md)
- Current status and valid samples: [../../PROJECT_STATUS.md](../../../PROJECT_STATUS.md)
