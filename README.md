# QRP

QRP 是本地量化研究平台，主线为 A 股低频研究和 Binance BTC/ETH 高频研究。
它覆盖来源数据、因子与标签、模型与预测、组合及历史执行模拟、诊断和报告。
项目不提供实盘交易、模拟账户、经纪商/交易所服务、SaaS 或投资建议。

## 从这里开始

- [A 股：研究范围、功能与经济合同](docs/a-share.md)
- [Crypto：研究范围、功能与事件合同](docs/crypto.md)
- [架构与工程合同](docs/architecture.md)
- [数据布局与来源边界](docs/data.md)
- [只读研究演示 Notebook](notebooks/research-interview-demo.ipynb)

代码、配置和长期文档一并纳入 Git；功能实现、合成验证和真实数据验收分别表述。
本机采集操作、SSH 配置、验收记录和交付副本保存在被忽略的 .run/，不提交 Git。
两份市场文档中的未来候选不授权预建接口、依赖或第二套研究框架。

## 环境与操作

环境定义见 environment.yml，Python 与依赖合同见 pyproject.toml。
使用已配置的 quantlab313 运行研究，不使用 Conda base。
下面的日期、项目根和 run 标识由调用者提供。

~~~text
qrp catalog factors --profile a_share_csmar_choice
qrp catalog labels --profile crypto_binance_hft
qrp research-datasets requirements <dataset_id> --profile <profile>
qrp data materialize --profile <profile> --project-root <project>
qrp dataset build <dataset_id> --profile <profile> --project-root <project> --start-date <date> --end-date <date>
qrp research run --project-root <project> --research-plan <configuration.json>
qrp status --profile <profile> --project-root <project> [--run-id <run_id>]
qrp inspect <target_id> --profile <profile> --project-root <project> [--run-id <run_id>]
qrp report build --profile <profile> --project-root <project> --run-id <run_id>
~~~

两市场的 research run 均通过 --research-plan <configuration.json> 读取显式研究配置。
profile、scenario、研究窗口、因子、模型与组合选择都来自该文件；示例位于 configs/research/。
Crypto 配置的 research_plan 还须给出时间切分与执行假设；可选择连续回测或遇真实缺口独立分段，午夜不重置账户。
研究运行返回不可变尝试标识 run_id；报告只消费指定运行引用的已发布产物。
默认运行开发区间；没有合法开发冻结产物时，--holdout 必须明确拒绝。
status 未指定 run_id 时只列已有运行；空目录返回空列表，不规划或启动研究。

本地 API 使用 create_app(project_root=Path("YOUR_PROJECT")) 绑定项目。
HTTP 客户端不能传路径切换项目；未绑定时仅提供目录查询。
已有 API 为 /api 下的 catalog/factors、catalog/labels、
research-datasets/{dataset_id}/requirements、status、inspect/{target_id}、
runs/{run_id}/reports/{report_identity} 和 runs/{run_id}/portfolios/{target_id}。
API 使用 profile 查询参数选择市场。报告查询核对 run 与 report_identity 来源关系，
直接返回保存的 Markdown；查询不重新生成报告正文。

Notebook 使用 quantlab313 kernel，填写项目、profile、run_id 和 report_identity 后运行。
它只读取已有诊断和聚合 NAV；A 股采用交易日/CNY，Crypto 采用 UTC/USDT 并区分 OOS 与空档。
独立分段各自绘图，不连接成跨缺口资金曲线。
NAV_STRIDE 只改变图形采样，同时显示源点数和显示点数；短区间不冒充正式日频统计。
核心 CLI/API 不需要导入 Notebook 的 Matplotlib/Jupyter 开发依赖。

## 开发验证

~~~text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
~~~

修改后先验证受影响行为。检查结论绑定受测代码、环境和输入；
实际执行范围、结果与未执行项写入本机验收记录，不复用其他提交的 PASS。
