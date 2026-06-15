# Owner Layer Routing / 归属层路由

本文件用于防止 P2/P3 顺手乱修和跨层扩张。先确认 owner layer，再决定是否改代码。

## Owner map

| Owner layer | Owns | 不应顺手拥有 |
|---|---|---|
| `contracts/` | stable specs, IDs, column contracts, run/workflow/report contracts | market-specific logic, storage IO |
| `storage/` | layout, dataset IO, manifests, checksums, partitions, storage privacy | research semantics, A-share trading rules |
| `catalog/` | queryable registries, research profile composition | private path helpers, dynamic source fallback |
| `sources/` | provider/source ingestion into interim canonical data | A-share eligibility/trading semantics |
| `markets/` | market-specific semantics, A-share annotations, calendars, IDs | generic research assembler logic |
| `research/` | factors, labels, research datasets, evaluation/model/prediction/portfolio research workflows | public API/CLI adapters, report path layout |
| `workflows/` | local DAG specs/executor/checkpoints | production scheduler/worker |
| `compute/` | dataframe/tensor math helpers | storage planning, public contract ownership |
| `agents/` | proposal-only specs/planners | external LLM calls, private row-level writes, execution authority |
| `interfaces/` | local API/CLI adapters | business semantics beyond adapters |
| `reports/` | report catalog and private report artifact writers | public run repository/facade/store |
| `runtime/` | privacy, sanitization, logging, profiling, resource policy, errors | market-session semantics |

## P2/P3 routing rule

- P2/P3 issue 默认不在当前任务顺手修。
- 如果 task owner layer 与 P2/P3 owner layer 不一致：记录 follow-up，不实现。
- 如果修复需要新增 public API / registry / store / facade / compatibility shim：停止并升级为 architecture closure。
- 如果修复需要跨 storage/evaluation/report/workflow/interfaces：先产出 architecture decision，再实现。
- 如果命名问题影响 contract/API/docs/tests：命名是架构问题，不能当作 cosmetic commit。

## Same issue family scan / 同族问题扫描

高风险改动完成前，按同族而不是同文件搜索：

- status truth：`PASS`, `FAIL`, `NOT_RUN`, `NOT_APPLICABLE`, `BLOCKED`
- private safety：`private`, `raw`, `interim`, `processed`, `reports/private_real`, `stock_code`, `trade_date`
- public surface：`api`, `cli`, `registry`, `store`, `facade`, `Repository`, `run`, `workflow`
- compatibility：`legacy`, `compat`, `alias`, `wrapper`, `re-export`, `retired`
- path governance：`.agents`, `.codex`, `.claude`, `hooks`, `MCP`, `plugins`, `subagent`
- platform/script：`.ps1`, `powershell`, `bash`, `sh`, `conda`, `python`

同族扫描发现 owner mismatch 时，报告 candidate，不扩大当前实现范围。
