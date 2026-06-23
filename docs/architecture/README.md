# Architecture Docs

本目录只保存当前合同文档。它不保存当前分支、当前阶段、当前 PR、最近运行结果或历史完成记录。

## 文件

```text
governance.md
engineering-contract.md
research-contract.md
platform-architecture.md
open-issues.md
```

## 读取规则

- 当前事实来自 live git baseline 和当前任务。
- 当前问题来自 `open-issues.md`。
- 历史追溯使用 git history。
- 不从 `.run/`、旧 reports 或旧 handoff 中读取当前合同。

## 文档边界

architecture docs 只描述长期稳定边界、contracts、owner layers 和 gates。不得记录“下一步”“当前阶段”“已完成历史”这类易过期内容。
