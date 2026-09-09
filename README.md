# QRP

QRP 是本地量化研究平台，主线为 A 股低频研究和 Binance BTC/ETH 高频研究。
它覆盖来源数据、因子与标签、模型与预测、组合及历史执行模拟、诊断和报告。
项目不提供实盘交易、模拟账户、经纪商/交易所服务、SaaS 或投资建议。

## 从这里开始

- [A 股：终局能力、已实现功能、短期建议与研究约定](docs/a-share.md)
- [Crypto：终局能力、已实现功能、短期建议与录制合同](docs/crypto.md)
- [架构与工程合同](docs/architecture.md)
- [数据布局与来源边界](docs/data.md)
- [只读研究演示 Notebook](notebooks/research-interview-demo.ipynb)

长期文档随代码版本管理；实现、合成验证和真实数据验收分别表述。
本机采集操作、SSH、当前验收摘要和交付副本放在被忽略的 .run/，不提交 Git。
两份市场文档中的未来候选不授权预建接口、依赖或第二套研究框架。

## 环境与操作

环境定义见 environment.yml，Python 与依赖合同见 pyproject.toml。
使用已配置的 quantlab313；安装项目时复用当前 Conda 环境，不使用 base 运行研究。
下面的日期、项目根和 run 标识由调用者提供。

~~~text
qrp catalog factors --profile a_share_csmar_choice
qrp catalog labels --profile crypto_binance_hft
qrp research-datasets requirements <dataset_id> --profile <profile>
qrp data materialize --profile <profile> --project-root <project>
qrp dataset build <dataset_id> --profile <profile> --project-root <project> --start-date <date> --end-date <date>
qrp research run --profile <profile> --project-root <project> --start-date <date> --end-date <date>
qrp status --profile <profile> --project-root <project> [--run-id <run_id>]
qrp inspect <target_id> --profile <profile> --project-root <project> [--run-id <run_id>]
qrp report build --profile <profile> --project-root <project> --run-id <run_id>
~~~

Crypto 的 research run 还必须使用 --research-plan <plan.json> 提供显式研究/执行计划。
研究运行返回不可变尝试标识 run_id；报告只消费指定运行引用的已发布产物。
默认运行开发区间；没有合法开发冻结产物时，--holdout 必须明确拒绝。
status 未指定 run_id 时只列已有运行；空目录返回空列表，不规划或启动研究。

本地 API 使用 create_app(project_root=Path("YOUR_PROJECT")) 绑定项目。
HTTP 客户端不能传路径切换项目；未绑定时仅提供目录查询。
已有 API 为 /api/0.1 下的 catalog/factors、catalog/labels、
research-datasets/{dataset_id}/requirements、status、inspect/{target_id}、
runs/{run_id}/reports/{report_identity} 和 runs/{run_id}/portfolios/{target_id}。
API 使用 profile 查询参数选择市场。报告查询核对 run 与 report_identity 来源关系，
读取已经保存的 Markdown，不按当前软件重渲染历史报告。

Notebook 使用 quantlab313 kernel，填写项目、profile、run_id 和 report_identity 后运行。
它只读取已有诊断和聚合 NAV；A 股采用交易日/CNY，Crypto 采用 UTC/USDT 并区分 OOS 与空档。
NAV_STRIDE 只改变图形采样，同时显示源点数和显示点数；短区间不冒充正式日频统计。
核心 CLI/API 不需要导入 Notebook 的 Matplotlib/Jupyter 开发依赖。

## 开发验证

~~~text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
~~~

修改后先验证受影响行为。完整包验收必须从仓库外干净导出构建 sdist，
从解出的 sdist 构建 wheel，再隔离安装并在空工作目录验证 CLI/API 和代表性工作流。
直接从检出目录或 editable 安装导入不算包验收。实际执行范围和未执行项写入本机摘要，
不把旧提交的 PASS 移记到当前版本。
