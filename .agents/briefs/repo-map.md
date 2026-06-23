# Repo Map

本文件只给 agent 提供仓库导航，不记录当前状态。

## 主要路径

```text
AGENTS.md
README.md
.agents/briefs/user-principles.md
.agents/skills/quant-research-platform-maintenance/SKILL.md
.agents/skills/quant-research-platform-maintenance/references/
.agents/skills/quant-research-platform-maintenance/scripts/
docs/architecture/
docs/code-map/
docs/api/
docs/data/
src/quant_research_platform/
tests/
```

## 源码分层

```text
contracts/       字段、schema、ID、time、privacy、run id、workflow 合同
catalog/         dataset、factor、label、model、report 等注册描述
sources/         A-share 内部 provider source 层和 raw/interim 映射
storage/         dataset 读写、manifest、fingerprint、compare、layout
research/        factors、features、labels、models、predictions、evaluations、portfolios、signals
compute/         dataframe/tensor 计算工具和 portfolio math
runtime/         logging、privacy、resources、execution metrics
reports/         public-safe evidence 与 report artifact
interfaces/      API、CLI、service facade
agents/          workflow planning 和 proposal 组件
```

## 文档分层

```text
docs/architecture/   当前架构合同、治理合同、open issues
docs/code-map/       代码流程地图和 A-share 数据流
docs/api/            API 边界
docs/data/           数据目录合同
```

## agent skill 分层

```text
SKILL.md                  skill 入口和路由
references/current-contract.md
references/task-routing.md
references/verification-and-gates.md
references/handoff-and-evidence.md
references/tooling-and-portability.md
references/prompt-patterns.md
scripts/verify_governance.py
scripts/verify_code_map_flow.py
scripts/handoff/
schemas/handoff/
templates/handoff/
```

## 读取规则

先用 live git baseline 确认现场，再按任务选择最小文件集合。不要为了找当前状态而读取历史 reports、旧 handoff 或 `.run/` 产物。
