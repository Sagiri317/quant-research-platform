# QRP Agent 指令

本文件只保存长期执行边界。当前 branch、phase、review、PR 和运行结果必须从本轮用户指令、live Git 与当前文件重建，不写入仓库文档。

## 1. Live baseline

任何修改或审查前运行：

```text
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
git branch -a
```

涉及远端或 merge 时再检查 upstream 与 divergence。事实优先级是：用户本轮指令、live Git/代码/测试/配置、本文、当前合同文档；叙事与可执行行为冲突时停止并报告。

## 2. 产品与安全边界

QRP 是本地量化研究平台，不是 live/paper trading、broker/exchange connector、deployment、SaaS 或 investment-advice 系统。未经明确批准，不新增 public API/CLI、依赖、framework、registry、scheduler、worker、deployment、connector 或 external LLM/vendor API。

不读取、复制、提交或公开 private data、secret、vendor raw marker、本地绝对路径、股票代码样本、row-level sample 或 raw traceback。REVIEW_SAFE 内容只包含 schema、计数、hash、状态、逻辑路径结构与无私有行的摘要。

## 3. 执行角色与修改范围

受控 Goal/Phase 同时至多一个 writable implementer；controller 只管理控制状态、派发与只读 reconciliation，不修改或提交 tracked implementation；reviewer 必须独立只读。实现者不得自审，review finding 由 controller 复现并按唯一 owner family 回交修正。

默认只修改任务直接相关的最小文件。修改 public/machine contract、schema、CLI、持久化布局、跨层依赖或业务语义前停止并请求批准。新实现替代旧实现，不保留 compatibility shim、alias、parallel path、tests-only API 或 proof-only framework。

## 4. Current contract 与 owner

QRP 采用 current-contract-only。行为、public machine contract、owner tests 与 dependency direction 定义当前事实；历史名称、目录或实现形状不构成兼容要求。

- `contracts/`：结构合同、枚举、ID、时间窗口和基础校验。
- `sources/`：摄取、raw schema、字段映射和 raw/interim 转换。
- `markets/`：市场领域；具体市场语义位于对应 implementation。
- `policy/`：共享安全规则与 `PASS`/`FAIL`/`NOT_RUN`/`BLOCKED` 聚合，不依赖业务层。
- `profiles/`：API、CLI 与 workflow 的组合根。
- `research/`、`compute/`：纯研究与数值计算，不写文件。
- `storage/`：dataset IO、identity、exact compare 与 data layout。
- `runtime/`：资源 policy、logical paths 与 execution metrics。
- `reports/`：构建内容，不写文件、不重算研究逻辑。
- `artifacts/`：写 LOCAL_ONLY outputs 与 REVIEW_SAFE evidence。
- `workflows/`：编排 research -> reports -> artifacts。
- `interfaces/`：已批准的 API、CLI 与 service adapter。

## 5. 资源生命周期硬合同

每个 top-level resource-bearing operation 在首次数据规模 read、decode 或 allocation 前只 admission 一次：

```text
captured_available_memory_bytes
= min(OS available memory, applicable cgroup/container/job/process headroom)

task_memory_budget_bytes
= min(floor(captured_available_memory_bytes * 0.80), valid explicit cap if any)
```

显式 cap 只能降低预算。适用平台上限无法可靠读取时使用可证明的更保守上限；无法建立安全上限则在数据规模分配前 fail closed。Policy 与 budget 捕获后 immutable；available memory 增加不能上调预算，已观察到 pressure 后只能 shrink、release、spill 或 fail closed。

一个 process 默认只有一个 active root policy、一个 execution owner/lease 和一个 total reservation ledger。Nested owner 继承同一 policy identity；different-policy nesting/parallel root fail closed。Same-policy reentry 只允许同 owner 真嵌套，copied context 跨 thread/task 不继承 authority，child 不得越过 root lifetime。

Resource-bearing child process 不得重新捕获 80% 预算；父 operation 必须从 root budget 划分并显式传递 child cap。跨进程共享 reservation 未实现前，production resource-bearing execution 保持 single-process。

raw/decoded/Arrow/Pandas/copy/index/sort/join/reducer/planner/model/native/output/writer/readback/spill/overlap state 共用一个预算。每次数据规模 allocation 前 reserve conservative owner-local upper bound，planned reservation 绝不超过 budget；不足则 shrink、release、spill 或 fail closed。不可分割结果超预算时必须在完整创建前失败。不建立 universal deep-size framework，不用 probe 波动为计划超额分配发明容忍线。

## 6. Identity、原子性与异常

保持 factor/label/mask、registered schema/dtype/PK、逻辑时间与输出范围、确定性和已批准 public machine contracts。QRP-owned generated JSON 顶层身份只使用 `schema_id`。

Storage 拥有唯一 physical traversal/session。Fingerprint 只能 quick reject；只有 exact comparison 可以产生 `MATCH`。发布必须 staged/atomic 并保留 old-good；partial-run 状态必须真实。Cleanup 只处理 owner-qualified resources，`BaseException` active primary 不得被 cleanup、metrics 或 secondary failure 覆盖。

## 7. 验证

只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`；每个 `PASS` 绑定实际 command、scope 与 result。先运行最窄相关测试，再按任务要求运行 direct full suite。环境名从 `environment.yml` 动态读取。

```text
git diff --check
conda run -n <env> ruff check .
conda run -n <env> python -m pytest -q -o "addopts=--import-mode=importlib --strict-markers" tests
conda run -n <env> python -m build --no-isolation --sdist --wheel --outdir <run-dir>
```

构建 smoke 不安装 package：用标准库解压 wheel，再以解压目录作为 `PYTHONPATH` import。不得把未运行或阻塞的检查写成 `PASS`。

## 8. Git 与环境安全

编辑和 commit 前检查 `git status` 与 diff，不丢弃或覆盖无法解释的 WIP。未经明确批准，不执行 reset、restore、clean、stash、rebase、amend、force push、main push、branch deletion、递归/通配删除、依赖安装或系统/环境配置修改。删除必须绑定已批准的精确路径。

最终报告列出 branch、HEAD、changed/deleted files、每条验证状态、remaining issues 与 working tree status。
