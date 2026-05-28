# Code Map 总览

本目录是 `quant-research-platform` 的源码阅读地图。`docs/code-map/` is the Current Implementation Learning Layer.

它记录当前代码事实、runtime flow、entrypoints、module relations、artifact/API/frontend tracing，以及新人接手时需要的实现阅读路径。它不替代研究报告、数据手册、架构蓝图、roadmap、治理文件或项目状态。

入口文档：

- 项目边界：[../../README.md](../../README.md)
- 当前事实：[../../PROJECT_STATUS.md](../../PROJECT_STATUS.md)
- 架构边界：[../architecture/README.md](../architecture/README.md)
- A 股 CSMAR 字段口径：[../data/a-share-csmar-data-manual.md](../data/a-share-csmar-data-manual.md)
- A 股字段可用性：[../data/a-share-field-inventory.md](../data/a-share-field-inventory.md)

## 当前文件

| 文档 | 用途 |
|---|---|
| [platform-flow.md](platform-flow.md) | 当前平台全局概览；用少量节点描述主链路、入口和边界 |
| [platform-flow-interactive.html](platform-flow-interactive.html) | 当前平台详细交互图；用细粒度节点和边定位具体模块、入口和文件 |
| [a-share-data-and-pipeline.md](a-share-data-and-pipeline.md) | A 股 CSMAR raw/interim/panel/factor/start_from pipeline、Factor Factory 和 validation/runtime 的代码地图 |
| [evaluation-artifacts-api.md](evaluation-artifacts-api.md) | Evaluation contract、schema `0.7` artifacts、diagnostics registry、`manifest.display`、API/download/frontend 合同地图 |

## 推荐阅读顺序

按新人接手顺序：

1. [platform-flow.md](platform-flow.md)：先建立全局链路、入口和边界。
2. [a-share-data-and-pipeline.md](a-share-data-and-pipeline.md)：看懂 A 股私有数据如何进入 processed factor table，以及 start_from runner 的真实范围。
3. [evaluation-artifacts-api.md](evaluation-artifacts-api.md)：看懂 `EvaluationSpec -> artifacts -> API -> workbench` 的合同。
4. [platform-flow-interactive.html](platform-flow-interactive.html)：需要查细粒度节点、边或入口时用浏览器浏览。

## 按问题定位

| 现象 | 先读 |
|---|---|
| 不知道当前平台哪些能力已经实现 | `platform-flow.md` |
| 原始 CSV、interim、panel 或 factor table 口径不清楚 | `a-share-data-and-pipeline.md` |
| Pipeline Builder 到底是 snapshot 还是 runner | `platform-flow.md` -> `a-share-data-and-pipeline.md` |
| Factor Factory 候选、失败日志或本地浏览入口不清楚 | `a-share-data-and-pipeline.md` |
| evaluation run 没有 schema `0.7` 或 `manifest.display` | `evaluation-artifacts-api.md` |
| cumulative returns、turnover、download、ZIP 或 workbench 展示不一致 | `evaluation-artifacts-api.md` |
| API 里 `source_alias + run_id` 身份丢失 | `evaluation-artifacts-api.md` -> `platform-flow.md` |
| 想从数据入口一路追到 workbench 展示 | `platform-flow.md` -> `a-share-data-and-pipeline.md` -> `evaluation-artifacts-api.md` |

## 职责边界

code-map 只回答“当前系统怎么工作”：

- current implementation reality。
- runtime flow / entrypoints / module relations。
- artifact table -> `manifest.display` -> API route -> frontend consumer 的追踪。
- onboarding / developer learning 阅读路径。

code-map 不承担：

- roadmap、future architecture 或 architecture debt 管理。
- governance、project status 或 root handoff。
- historical closeout、completed milestone story 或研究叙事。

## 维护规则

- `platform-flow.md` 和 `platform-flow-interactive.html` 保持语义一致，但粒度不同；Markdown 版是 overview，HTML 版是 detailed map。
- 新增用户可见 artifact、API、download、frontend table/chart 时，同步 `evaluation-artifacts-api.md`。
- 修改 A 股 raw/interim/panel/factor/runner/Factor Factory 时，同步 `a-share-data-and-pipeline.md`。
- 修改 platform flow 图时，按项目维护 skill 的 code-map flow reference 做 UTF-8、节点/边、环、重叠和 Bezier edge 静态检查。
- 不在 code-map 中记录一次性 run id、临时文件、完整私有路径或不可公开的 CSMAR 明细。
