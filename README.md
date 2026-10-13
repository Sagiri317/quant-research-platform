# QRP

QRP 是本地量化研究平台，主线为 A 股低频研究和 Binance BTC/ETH 高频研究。
它覆盖来源数据、因子与标签、模型与预测、组合及历史执行模拟、诊断和报告。
项目不提供实盘交易、模拟账户、经纪商/交易所服务、SaaS 或投资建议。

## 从这里开始

- [A 股：研究范围、功能与经济合同](docs/a-share.md)
- [Crypto：研究范围、功能与事件合同](docs/crypto.md)
- [架构与工程合同](docs/architecture.md)
- [数据布局与来源边界](docs/data.md)

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
`stage` 可选 `dataset`、`evaluation`、`models`、`research`，每次在所选阶段停止；
比较多个配置后再由用户或获授权 agent 选择下一阶段，精确来源引用见[研究配置合同](docs/architecture.md#研究配置)。
Crypto 配置的 research_plan 还须给出时间切分与执行假设；可选择连续回测或遇真实缺口独立分段，午夜不重置账户。
研究运行返回不可变尝试标识 run_id；报告只消费指定运行引用的已发布产物。
默认运行开发区间；没有合法开发冻结产物时，--holdout 必须明确拒绝。
status 未指定 run_id 时只列已有运行；空目录返回空列表，不规划或启动研究。
结构化状态的退出码、`--rebuild` 的终端目标范围及已发布上游复用规则见
[计划、身份与持久化](docs/architecture.md#计划身份与持久化)。

本地 API 使用 create_app(project_root=Path("YOUR_PROJECT")) 绑定项目。
HTTP 客户端不能传路径切换项目；未绑定时仅提供目录查询。
已有 API 为 /api 下的 catalog/factors、catalog/labels、
research-datasets/{dataset_id}/requirements、status、inspect/{target_id}、
runs/{run_id}/reports/{report_identity} 和 runs/{run_id}/portfolios/{target_id}。
API 使用 profile 查询参数选择市场。报告查询核对 run 与 report_identity 来源关系，
直接返回保存的 Markdown；查询不重新生成报告正文。

研究者或获授权 agent 通过现有只读查询读取明确的 profile、run_id 和 report_identity，
按需制作研究图片与分析。正式数值、适用范围、状态及缺失原因以已发布证据为准。
A 股采用交易日/CNY，Crypto 采用 UTC/USDT 并区分 OOS 与空档；独立分段不连接。
曲线采样不改变正式统计，事件级 NAV 不替代日频收益；零 NAV 后的金额恢复不使未定义收益率变为有效。
Matplotlib 是可选开发依赖，核心 CLI/API 不依赖绘图环境。

`reports.plots` 提供纯绘图函数，返回 `(figure, axes)`，不读取磁盘、不生成研究结论、
不改变正式指标。绘图时才加载 Matplotlib；使用本机已有 Noto Sans CJK SC、思源黑体、
微软雅黑等中文字体，缺少可用字体时明确报错，不静默输出方框字。

- `plot_portfolio_curve(curve, access_window=..., timezone=...)`：直接接收
  `queries.portfolio_curve(...)` 的返回值，绘制金额净值、OOS/空档与逐段回撤。
  仅完整未抽样曲线提供描述性回撤路径；正式最大回撤仍以统计产物为准。
- `plot_metric_comparison(rows, metric_id="sharpe", units="无量纲", ...)`：
  显示已发布 Sharpe、风险、成本或换手指标，保留不可用状态；调用方须先选择可比较的范围。
  不同货币、执行模式、覆盖区间不能仅因画在一起就解释为可比。
- `plot_returns(rows, ...)`：各期净收益散点与描述性分布，不补零、不复利、不推断缺失日期。
- `plot_factor_ic(rows, summary=True, ...)`：IC/Rank IC 因子横向比较；
  `summary=False` 消费真实逐时点评估行，不能把摘要伪装成时序。
- `plot_portfolio_activity(rows, ...)`：A 股已发布日表的成本拖累与实际换手，分轴标明比例。

除净值函数外须显式传入 `source_identity` 与 `access_window`；含时间的函数还须传 `timezone`。
时区应来自对应市场合同：A 股通常 `Asia/Shanghai`，Crypto 为 `UTC`。
无时区时间戳保留来源时钟含义，只标注传入时区，不擅自转换；有时区时间戳才转换。
调用方使用已校验报告/产物中的相同身份与访问范围，且传入行的有效值必须有限。
极端有限值仅按图中标明的十进制倍率显示，不改变原始统计。

报告直接提供 A 股指标/IC 摘要及 Crypto 完整 UTC 日收益；A 股日收益、成本/换手与 IC 时序
应从已有校验读取器消费 `portfolio_daily.parquet` 或 `metrics.parquet`，转换为记录后传入。
这些数据不在报告摘要中时不回退拼造，也不新增查询端点。

~~~python
from quant_research_platform.reports.plots import plot_portfolio_curve

figure, axes = plot_portfolio_curve(
    curve, access_window="holdout", timezone="UTC"
)
figure.savefig("nav.png", dpi=220, bbox_inches="tight")
figure.savefig("nav.svg", bbox_inches="tight")
# 批量生成时由调用方 plt.close(figure)，及时释放图形资源。
~~~


## 开发验证

~~~text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
~~~

修改后先验证受影响行为。检查结论绑定受测代码、环境和输入；
实际执行范围、结果与未执行项写入本机验收记录，不复用其他提交的 PASS。
