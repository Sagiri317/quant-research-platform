# 归属层路由（Owner layer routing）

本文件帮助把任务路由到正确 owner layer，避免 P2/P3 或未来能力被顺手跨层修。

## 核心规则

1. 先定位 owner layer，再改文件。
2. 能在 owner layer 局部完成，就不要跨 storage/evaluation/report/workflow/interfaces 大改。
3. 如果必须跨三层以上，先停下写 architecture closure（架构收口）说明。
4. P2/P3 默认登记到 owner layer，不在无关任务中顺手修。

## 常见路由

| 需求 | 首选 owner | 典型触及 | 不应默认触及 |
|---|---|---|---|
| public API/CLI | `interfaces/` + contracts | API/CLI adapters, tests | storage internals |
| dataset IO/manifest | `storage/` | layout, dataset IO, manifest | evaluation/report |
| A-share 交易语义 | `markets/implementations/a_share/` | annotations, masks, calendar | labels/evaluation 重算规则 |
| factor | `research/factors/` | factor spec/materializer/tests | interfaces/public API |
| label | `research/labels/` | label spec/materializer/tests | market semantics 重写 |
| research dataset | `research/datasets/` | assembler/spec/tests | source ingestion |
| evaluation diagnostics | `research/evaluations/` | metrics/diagnostics/tests | report storage API 扩张 |
| model/prediction | `research/models/`, `research/predictions/` | model/pred contract/tests | live trading |
| reports | `reports/` | report writer/catalog/private-safe output | public artifact lifecycle API |
| runtime memory | `runtime/resources.py` | runtime policy/chunk planning | market session semantics |
| workflow execution | `workflows/` | DAG/plan/checkpoint | broker/scheduler |
| agent governance | `.agents/`, `AGENTS.md`, governance docs | skill/templates/scripts | business implementation |

## Future capability routing

ML/DL、time-series、factor factory、多市场、高频、frontend、website、speed/memory 优化都要先落 owner layer。不能因为“未来会用”而新增 universal registry/store/facade。
