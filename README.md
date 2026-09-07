# QRP

QRP 是本地量化研究平台，用于数据合同、因子与特征研究、评估、模型训练、预测、
组合与历史执行模拟、报告和证据。它不是实盘交易、模拟账户交易或经纪商/交易所执行系统，
也不提供投资建议。

## 目录与责任模块

```text
src/         QRP 生产代码
tests/       当前行为、合同与架构测试
docs/        当前合同文档
data/        本地数据占位；不提交私有数据
.run/        被忽略的本地输入、输出和证据；不定义代码事实
```

- `sources/` 负责摄取、原始结构及原始/中间层转换；`markets/` 负责市场领域实现。
- `contracts/` 定义结构合同；`policy/` 定义共享安全规则；`profiles/` 绑定 API、CLI 与工作流组合根。
- `research/` 和 `compute/` 负责研究与数值计算；`storage/` 负责数据集 IO、持久化标识、精确比较与数据布局。
- `runtime/` 负责资源准入与执行指标；`artifacts/` 写 LOCAL_ONLY/REVIEW_SAFE 文件；`reports/` 构建内容。
- `workflows/` 只编排；`interfaces/` 暴露已批准的 API/CLI。

## 托管操作

一个托管的数据规模操作先取得主机本地锁，只捕获一次 `available0`，
默认规划目标约为 `available0` 的 70%；显式上限只能进一步降低它。
每个数据规模读取/解码/分配在发生前做保守准入，流程保持有界；
聚合持久化输出可以超过 RAM 预算而精确完成。主要工作流阶段按顺序执行，
前一阶段释放临时状态后，后一阶段复用同一个操作预算；
只为当前阶段中真正同时存活的分配准入。新载荷、工作区或输出只有一个逻辑归属，
在生产者/消费者边界转移，不重复计费；责任模块的本地上下文退出时释放资源。

默认使用单进程加原生线程，只使用一层主要并行，线程数不超过物理核并可因规划目标下调。
70% 及派生出的 `memory_budget_bytes` 都只是相对安全的软规划参考，
不是进程 RSS 或真实分配的硬上限；租约的 bytes 是可相加的保守粗估，不要求字节级精确。
约 80% 只描述工程上的大致安全包络，也不是第二硬阈值。
仓库外全新子进程的 RSS/CPU/I/O 观测只用于校准和验收，不是运行时规划权威。
小幅、稳定且可解释的估算或包络超出应标为风险供用户判断，
不在 70% 或 80% 的精确点自动判死；借用基线、原生工作区和这些观测都不参与生产规划或重复计费。

## 存储与发布

数据集标识由持久化结构、dtype、PK、逻辑窗口与精确内容定义。
指纹只用于快速拒绝；只有精确比较可以产生 `MATCH`。
输出在事务内磁盘预检后写入同父目录 UUID 暂存，再通过平台原生不覆盖移动，
或责任归属明确的替换发布；成功才原子发布，失败保留原有已验证产物，
并只清理本次操作能证明创建的资源。存在主 `BaseException` 时，释放或验证失败不得覆盖它。

QRP 只采用一套活动合同：替换旧实现时删除旧入口，不保留别名、并行读取器或平行实现。

## 公共接口面

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
GET  /api/0.1/status
GET  /api/0.1/inspect/{target_id}
GET  /api/0.1/runs/{run_id}/reports/{report_identity}
GET  /api/0.1/runs/{run_id}/portfolios/{target_id}

qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
qrp data materialize --profile a_share_csmar_choice
qrp dataset build <dataset_id> --profile a_share_csmar_choice --start-date <date> --end-date <date>
qrp research run --profile a_share_csmar_choice --start-date <date> --end-date <date> [--holdout]
qrp report build --run-id <research-run-id> --profile a_share_csmar_choice
qrp status --profile a_share_csmar_choice [--run-id <research-run-id>]
qrp inspect <target_id> --profile a_share_csmar_choice [--run-id <research-run-id>]
```

研究运行返回不可变的尝试标识 `run_id`；报告必须指定该标识，并且只消费记录中引用的持久化产物。
研究运行默认使用开发区间；在冻结的开发标识可用之前，`--holdout` 必须明确拒绝执行。

目录及查询接受既有 `a_share_csmar_choice` 与 `crypto_binance_hft` profile。
API 通过 `profile` 查询参数选择；CLI 目录查询同样接受 `--profile`。
`status` 不再规划研究：未指定 `run_id` 时只列出已有运行，空目录返回空列表；
指定时投影该记录，不能把目录状态当成研究就绪状态。

文件查询的项目根目录由本地应用绑定，HTTP 客户端不能传路径切换项目：

```python
from pathlib import Path
from quant_research_platform.interfaces.api.app import create_app

app = create_app(project_root=Path("YOUR_PROJECT"))
```

未绑定项目的应用只提供目录查询。报告视图必须给定已发布的 `run_id` 和
`report_identity`，并校验二者来源关系；同一运行存在多份报告时，先用
`inspect <scenario_id>_report_artifact --run-id <run_id>` 查看全部身份，再明确选择。
读取旧报告使用其已经保存的 Markdown，不按当前软件重渲染或重新发布。

[可复现演示 Notebook](notebooks/research-interview-demo.ipynb)
使用 `quantlab313` kernel。填写项目、profile、运行及报告身份后从头执行；
源码不含执行输出或本地路径，验证副本使用合成产物。
它仅读取已有诊断和聚合 NAV：A-share 标注交易日和 CNY，Crypto 标注 UTC、
USDT 与已记录 OOS/空档；事件 NAV 不替代 365 日频统计，日数不足仍为 `UNDEFINED`。
如显式增大 `NAV_STRIDE`，仅改变图形采样，源点数与显示点数会同时列出。
核心 CLI/API 不需要导入 Notebook 的 Matplotlib/Jupyter 开发依赖。

Choice 是唯一正式财务 PIT 来源，CSMAR 财务仅用于验证。
PIT 适用性按已证明的数据范围与内容判断，数据质量和后续处理分别说明。
真实财务处理遵循必需原始资料齐备的操作顺序，并明确报告实际缺口。

新增公开 API/CLI、依赖、框架、连接器或部署能力需要明确批准。

## 直接验证

环境名以 `environment.yml` 的 `name` 为准：

```text
conda run -n <env> ruff format --check .
conda run -n <env> ruff check . --no-cache
conda run -n <env> pytest -q
conda run -n <env> python -m build --no-isolation --sdist --wheel --outdir <run-dir>
```

默认 `pytest -q` 运行全部测试。包验收必须从仓库外干净导出构建全新 sdist，
再从解出的 sdist 构建 wheel，在仓库外隔离环境安装，最后从空工作目录做 CLI/API 冒烟检查；
可编辑安装的检出目录或直接从源码路径导入不算通过。

## 数据安全

不提交私有数据、秘密、供应商载荷、本地绝对路径、股票代码样本、行级样本或原始异常回溯。
REVIEW_SAFE 证据只包含结构、计数、哈希、状态、逻辑路径结构和不含私有行的摘要。
