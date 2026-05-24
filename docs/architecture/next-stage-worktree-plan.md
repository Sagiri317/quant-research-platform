# Next Stage Worktree Plan / 下一阶段分支与工作树计划

本文把 [next-stage-roadmap.md](next-stage-roadmap.md) 的近期任务拆成适合隔离开发的分支和工作树。它不是功能验收记录；功能验收仍以后续各分支的测试和 code review 为准。

## 当前基线

- `main` 已包含 N1 A 股 `start_from` runner。
- `main` 已包含 N3 Artifact ZIP packaging 下沉。
- Pipeline 对外语言固定为 snapshot vs run：
  - `POST /api/pipeline-snapshots` 只保存配置。
  - `POST /api/pipeline-runs` 执行到 evaluation。
  - executable run 只用 `start_from` 控制开始层。
- 当前没有保留已完成 N1/N3 的本地工作树；新任务应从干净 `main` 创建新工作树。

## 原则

- 每个分支只交付一个可 review 的主题，避免把 artifact、API、pipeline、frontend 和 docs 大范围混在一起。
- 新增用户可见 artifact、API、download、frontend table/chart 时，同步 `manifest.display`、API tests、frontend tests 和 code-map。
- 不在这些分支里接入外部 API、自动下载 CSMAR、读取凭据、部署服务、paper trading 或 live trading。
- 工作树创建前必须确认当前 checkout 干净，并确认 `.worktrees/` 被 Git 忽略。
- Vite 默认资源清理仍是独立低优先级 hygiene，不混进功能分支。

## 已完成分支

| 分支 | 路线图任务 | 状态 | 结果 |
|---|---|---|---|
| `codex/a-share-build-runner` | N1 A 股 start-from runner | merged to `main` | snapshot/run split、`start_from` runner、完整 checkpoint 链和 Factor Factory existing-factors start layer 已落地 |
| `codex/artifact-zip-packaging` | N3 ZIP packaging 下沉 | merged to `main` | ZIP item validation、cache identity、file collection 和 dynamic item handling 已移入 `artifacts/artifact_package.py` |
| `codex/frontend-feature-split` | N4 前端 feature split | complete in worktree | `web/src` 已拆为 `shell/`、`features/` 和 `shared/`，未新增 diagnostics UI，未改变 API contract |

## 建议后续分支

| 分支 | 路线图任务 | 范围 | 依赖 | 冲突风险 |
|---|---|---|---|---|
| `codex/tradability-artifacts` | N2 后端 artifact/API 诊断 | evaluation artifact display、A 股 adapter diagnostics、API preview/download tests、docs/code-map | 基于当前 `main`；消费 N1 runner 和 N3 packaging 边界 | 中。会改 artifact/display/API |
| `codex/tradability-workbench` | N2 frontend 展示 | frontend evidence/workbench 展示新增 diagnostics | 依赖 `codex/tradability-artifacts`；最好基于 N4 后结构 | 中。会碰 frontend evidence state |
| `codex/risk-regime-diagnostics` | N5 风险、regime、成本和容量扩展 | research diagnostics artifacts、API exposure、report/docs | 依赖 N2 诊断 artifact/display 模式稳定 | 中到高。需要先写详细计划 |
| `codex/ml-prediction-handoff` | N6 ML prediction handoff | prediction manifest、handoff schema、evaluation adapter-like input、tests/docs | 可独立设计，但不能混入 N2/N5 | 中到高。合同设计优先 |

## 推荐顺序

1. `codex/tradability-artifacts`
   - 先做后端 artifact/API 合同。
   - 新增表必须通过 `manifest.display.tables` 和 downloads 暴露。
   - API preview、direct download、ZIP download 不得绕过 display contract。

2. `codex/tradability-workbench`
   - 消费 N2 后端 diagnostics artifact/API。
   - 前端只展示，不重新计算 eligibility、tradability、IC、PnL、positions 或 turnover。

3. `codex/risk-regime-diagnostics` 或 `codex/ml-prediction-handoff`
   - 两者都应先写独立计划。
   - 不建议和 N2 frontend 同时抢同一批 artifact/API/frontend 文件。

## 不建议并行的组合

- 不要同时做 `codex/frontend-feature-split` 和 `codex/tradability-workbench`，除非已经约定 disjoint ownership。一个移动文件，一个加 UI，冲突概率高。
- 不要把 `codex/tradability-artifacts` 和 `codex/tradability-workbench` 混到一个大分支。先把 artifact/API 合同稳定下来，再让前端消费。
- 不要把 N5 风险/regime 扩展混进 N2 tradability 诊断。两者都属于 diagnostics，但研究语义和验收口径不同。
- 不要把 N6 ML prediction handoff 混进现有 factor evaluation runner。Prediction 是新的输入合同，应单独设计。

## 工作树创建策略

从干净 `main` 创建新工作树：

```powershell
git status --short
git branch --show-current
git worktree list
git check-ignore -q .worktrees
git worktree add ".worktrees/tradability-artifacts" -b "codex/tradability-artifacts"
```

如果 `.worktrees/` 未被忽略，先更新 `.gitignore` 并单独提交该治理变更，再创建工作树。不要把工作树目录加入 Git。

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

需要真实浏览器 UI 抽查时，用项目 wrapper，避免生成顶层 `output/`：

```powershell
& scripts/playwright_cli_project.ps1 open http://127.0.0.1:5173
& scripts/playwright_cli_project.ps1 snapshot
& scripts/playwright_cli_project.ps1 screenshot
```

## 相关计划文件

- N1 归档计划：[../superpowers/plans/2026-05-24-a-share-build-runner.md](../superpowers/plans/2026-05-24-a-share-build-runner.md)
- N3 归档计划：[../superpowers/plans/2026-05-24-artifact-zip-packaging.md](../superpowers/plans/2026-05-24-artifact-zip-packaging.md)
- N4 计划草案：[../superpowers/plans/2026-05-24-frontend-feature-split.md](../superpowers/plans/2026-05-24-frontend-feature-split.md)
