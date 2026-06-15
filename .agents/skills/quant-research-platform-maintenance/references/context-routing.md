# 上下文路由（Context routing）

本文件说明 QRP 任务应该读什么、不读什么。原则是：先读最小上下文，只有证据显示不够时才扩展。

## 基础规则

1. 每次任务先读 `AGENTS.md` 和用户本轮指令。
2. 用 `task-modes.md` 选 mode ID。
3. 用 `token-economy.md` 定预算。
4. 用 `repo-map.md` 判断 owner layer。
5. 按 mode 只读一个 primary template。
6. 用 `rg` 摘要做 issue-family scan，再决定是否打开更多文件。

## 任务到上下文

| 任务 | 首读 | 按需扩展 | 禁止默认读取 |
|---|---|---|---|
| 小修 | 目标文件、最小测试 | 相邻文件 | 全 repo、全部 docs |
| 标准实现 | owner layer 文件、相关 tests | contracts/docs | private data、无关 layer |
| 独立审查 | diff/stat、changed files | issue-family 命中文件 | 自动改代码 |
| handoff | `.run` summary、package rules | package manifest、safety result | raw logs、row-level data |
| merge gate | git status/log/merge-base | compare summary | 源码全文 |
| docs sync | active status/docs | code facts | private roots |
| governance | `AGENTS.md`、`.agents/**`、治理 docs | 官方 docs、代码事实 | 业务实现大范围读取 |

## 代码事实校验

治理文件写到具体数字、路径或 public surface 时，必须回到代码或 tests 校验。例：内存预算应以 `src/quant_research_platform/runtime/resources.py` 的 `DEFAULT_MEMORY_BUDGET_FRACTION` 为准。

## 停止扩展条件

当已能明确回答任务、定位 owner layer、给出验证结果时，停止继续读取。不要为了“更全面”读取旧 handoff、全量 docs 或无关业务代码。
