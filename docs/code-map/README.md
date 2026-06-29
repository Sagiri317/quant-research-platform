# Code Map

本目录提供当前代码地图。

## 文件

```text
platform-flow.md
  平台总体流程和模块边界。

a-share-data-flow.md
  A-share 内部供应商数据路径和 source/dataset 边界。
```

## 使用规则

代码地图用于减少 agent 搜索成本。若代码地图与源码、contracts 或 tests 冲突，以可执行合同为准，并修正文档。

代码地图只描述当前合同。治理规则事实源是 `.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py`；`.run/` 只是 output/evidence/input artifact layer，不是 tracked tool source。API 版本面使用 `/api/0.1`。
