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
contracts/       结构合同、枚举、ID、TimeWindow 和基础校验
catalog/         dataset、factor、label、model、report 等注册描述
sources/         provider/source 摄取、raw schema、字段映射和 raw/interim 转换
markets/         通用市场领域、session window 和市场实现容器
markets/implementations/a_share/  A 股 calendar、processed、mask 和研究绑定
storage/         dataset 读写、manifest、fingerprint、compare、layout
research/        通用研究算法，不依赖具体市场实现或 sources
compute/         dataframe/tensor 数值核心、portfolio math、performance
runtime/         privacy、resources、logging、workflow records、metrics payload
reports/         public-safe report/evidence artifact 写出
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
