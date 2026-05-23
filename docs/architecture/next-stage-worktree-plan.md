# Next Stage Worktree Plan / 下一阶段分支与工作树计划

本文把 [next-stage-roadmap.md](next-stage-roadmap.md) 中的近期任务拆成可隔离的分支和工作树。它不是功能验收记录；功能验收仍以后续各分支的测试和代码 review 为准。

## 原则

- 先拆边界，再开工作树。只有任务的文件范围、依赖和验收命令清楚后才创建分支。
- 每个分支只交付一个可 review 的主题，避免把 pipeline、artifact、API 和 frontend 重构混在同一个提交链里。
- 保留当前 Phase 1 合同：`processed_existing` 轻量 runner、artifact schema `0.5`、`manifest.display`、API-visible `(source_alias, run_id)`、A 股私有数据边界。
- 不在这些分支里接入外部 API、自动下载 CSMAR、读取凭据、部署服务、paper trading 或 live trading。
- 工作树创建前必须确认当前 checkout 干净，并确认 `.worktrees/` 或目标工作树目录已被 Git 忽略。

## 建议分支

| 分支 | 路线图任务 | 范围 | 依赖 | 冲突风险 |
|---|---|---|---|---|
| `codex/a-share-build-runner` | N1 完整 A 股 build runner | `pipelines/a_share/`、A 股 build/evaluation entrypoints、pipeline API service、tests、docs | 无；先做 | 中。会定义后续 diagnostics/checkpoint 口径 |
| `codex/artifact-zip-packaging` | N3 ZIP packaging 下沉 | `artifacts/`、API download service、download tests、contract docs | 可与 N1 并行 | 低到中。主要冲突在 docs 和 API service |
| `codex/frontend-feature-split` | N4 前端 feature split | `web/src` 目录重组、frontend tests、code-map | 最好在 N2 frontend 前完成 | 中。会移动大量 frontend 文件 |
| `codex/tradability-artifacts` | N2 后端 artifact/API 诊断 | evaluation artifact display、A 股 adapter diagnostics、API preview/download tests | 建议等 N1 的 checkpoint/runtime summary 口径稳定后做 | 中。会改 artifact/display/API |
| `codex/tradability-workbench` | N2 frontend 展示 | frontend evidence/workbench 展示新增 diagnostics | 依赖 `codex/tradability-artifacts`；最好基于 N4 后结构 | 中。会碰 frontend evidence state |

## 推荐顺序

1. `codex/a-share-build-runner`
   - 先交付完整 A 股 build/evaluate orchestration。
   - 输出稳定的 step registry、checkpoint、runtime summary 和 failure code。
   - 保留 `processed_existing`，不把完整 build 伪装成已有轻量模式。

2. `codex/artifact-zip-packaging`
   - 可在 N1 期间并行，但只在确认 API download tests 能独立 review 时并行。
   - 如果 N1 改动 API service 边界较多，则等 N1 合并后再开。

3. `codex/frontend-feature-split`
   - 在继续给 workbench 加 diagnostics UI 前做。
   - 只移动和重组 frontend，不改变 API contract，不新增业务计算。

4. `codex/tradability-artifacts`
   - 先做后端 artifact/API 合同。
   - 新增表必须通过 `manifest.display.tables` 和 downloads 暴露。
   - API preview、direct download、ZIP download 不得绕过 display contract。

5. `codex/tradability-workbench`
   - 消费后端 diagnostics artifact/API。
   - 前端只展示，不重新计算 eligibility、tradability、IC、PnL、positions 或 turnover。

## 不建议并行的组合

- 不要同时做 `codex/frontend-feature-split` 和 `codex/tradability-workbench`，除非已经约定 disjoint ownership。一个移动文件，一个加 UI，冲突概率高。
- 不要把 N1 与 N2 合并到同一分支。N1 定义 pipeline 执行和 checkpoint 口径，N2 消费其结果；混在一起会让失败定位困难。
- 不要把 Vite 默认资源清理混入这些分支。`web/src/assets/hero.png`、`react.svg`、`vite.svg` 是否删除应单独确认引用和路径后处理。

## 工作树创建策略

默认从当前 `main` 创建工作树：

```powershell
git status --short
git branch --show-current
git worktree list
```

如果使用项目内 `.worktrees/`：

```powershell
git check-ignore -q .worktrees
git worktree add ".worktrees/a-share-build-runner" -b "codex/a-share-build-runner"
```

如 `.worktrees/` 未被忽略，先更新 `.gitignore` 并单独提交该治理变更，再创建工作树。不要把工作树目录加入 Git。

## 每个分支的最低验收

Python 后端分支：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share tests/markets/implementations/a_share
conda run -n analysis python -m ruff check src tests
git diff --check
```

Artifact/API download 分支：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api
conda run -n analysis python -m ruff check src tests
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
conda run -n analysis python -m ruff check src tests
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

## 当前执行计划

- N1 详细实施计划：[../superpowers/plans/2026-05-24-a-share-build-runner.md](../superpowers/plans/2026-05-24-a-share-build-runner.md)。
- N3 详细实施计划：[../superpowers/plans/2026-05-24-artifact-zip-packaging.md](../superpowers/plans/2026-05-24-artifact-zip-packaging.md)。
- N4 详细实施计划：[../superpowers/plans/2026-05-24-frontend-feature-split.md](../superpowers/plans/2026-05-24-frontend-feature-split.md)。
