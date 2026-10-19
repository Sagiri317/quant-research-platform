# quant-research-platform

quant-research-platform 是一个本地量化研究项目，覆盖 A 股日频选股与 Binance BTC/ETH 永续合约的高频预测和回测。它把数据准备、因子构建、模型训练、样本外评价和交易核算连接起来，便于检查一项预测改善能否转化为费用后的收益。

## 研究报告

- [A 股量价研究报告](reports/A股量价研究报告.md)：比较 Core6、PV25、PV52 的信息增量与四类经济块消融，解释样本筛选、缺失处理、中性化、停牌涨跌停和交易成本。
- [加密资产高频研究报告](reports/加密资产高频研究报告.md)：比较多尺度、跨币和跨场所信息的预测价值，并展示已完成的延迟实验与首批交易结果。完整执行矩阵仍在计算。

两篇报告的图表与汇总数据一同保存在 `reports/`。市场规则与计算方法见 `docs/`，小型操作示例见 `configs/`。供应商原始数据和本地运行目录不随仓库分发。

## 直接操作

Python 函数与 `qrp` 命令行使用同一套计算实现。研究配置明确输入、参数、时间窗口和输出位置；每次操作生成新产物，已有结果保持不变。

```text
qrp prepare-source --config configs/prepare-source.json
qrp prepare --config configs/prepare-financial.json
qrp assemble --config <组装配置.json>
qrp fit --config configs/fit-a-share.json
qrp predict --config configs/predict-a-share.json
qrp explain --config configs/explain-model.json
qrp score --config <验证评分配置.json>
qrp select --config <候选选择配置.json>
qrp calibrate --config <A股校准配置.json>
qrp backtest --config <A股组合配置.json>
qrp calibrate-crypto-maker --config <maker校准配置.json>
qrp backtest-crypto-taker --config <taker配置.json>
qrp backtest-crypto-maker --config <maker配置.json>
qrp evaluate --config <评估配置.json>
qrp statistics-model --config <模型统计配置.json>
qrp statistics-factors --config <因子统计配置.json>
qrp statistics-portfolio --config <A股组合统计配置.json>
qrp statistics-crypto --config <Crypto组合统计配置.json>
qrp plot --config configs/plot-risk.json
qrp inspect <已发布产物目录>
```

例如 Python 直接调用：

```python
from quant_research_platform.storage.manifests import read_qrp_json_file
from quant_research_platform.research.models.production import fit

model_path = fit(read_qrp_json_file("configs/fit-a-share.json")[1])
```

score 明确提供完整 validation_windows；select 明确提供 refit_windows 与 refit_cutoff，
最终拟合只消费明确给定窗口内截至拟合截止已成熟的样本，可加入合法新日期，不填补窗口间空档。
模型可以显式加载、多次预测。预测不拟合，评分不重训，选择不自动最终重新拟合。
walk-forward 明确每次拟合、决策、预测与评价窗口，样本不永久分组。
每个窗口说明训练、验证或评价用途；计算检查拟合截止、标签成熟和校准时点。
直接使用 Parquet 时，需提供列角色、单位、主键和信息可用时间。

[操作示例](configs/) 中的路径需替换为实际已准备输入。
[同族多批因子装配](configs/assemble-factor-batches.json)明确实例及列投影；
[已有特征复用](configs/assemble-existing-features.json)只选择物化列，不重复处理，
已声明的观测/缺失伴随列自动保留。前者仅拼接原始因子，后者保留 eligible，
用于无标签特征输入；训练时另选实际标签及其成熟列。
[ON/ID准备](configs/prepare-on-id.json)、[共同域分层评价](configs/evaluate-stratified-a-share.json)与
[持久化日表归约](configs/statistics-stratified-a-share.json)分别调用现有 prepare、evaluate、statistics-factors；
own 评价需另选新输出、sample_scope=own 并移除 comparison_inputs。两份评价的焦点信号、输入和分层规格应保持一致。
[冻结模型解释](configs/explain-model.json)的背景必须落在该模型实际拟合域内；
[组合基准比较](configs/statistics-benchmark-a-share.json)需替换真实指数代码、时钟、币种及价格/全收益口径，并通过 inputs.calendar 提供覆盖收益区间的完整交易日历。
组合统计默认只计算绩效，额外分析通过 analyses 显式选择。
[科研选择参考](docs/research-rosters/) 保留有序模型、参数候选、因子、资产与研究窗口；
它们用于设计实验，实际运行仍通过各操作的配置给出。
复杂配置按各操作合同校验字段、类型和必要数值，拒绝重复 JSON key。
未选择的统计分析不读取其专用输入；解释等严格接口拒绝未消费选项。
候选及输入列顺序具有含义，不能自行排序去重。

## 只读查询与图表

```text
qrp catalog factors --profile a_share_csmar_choice
qrp catalog labels --profile crypto_binance_hft
qrp catalog dataset pv_all
```

HTTP `create_app(project_root=...)` 提供 catalog、数据集要求、
`/api/inspect?path=...` 与 `/api/portfolio-curve?path=...`。
查询限定在绑定的项目根内，只读取已有产物；原始数据和计算任务由本地接口管理。

`reports.plots.plot(config)` 原子写入新的 PNG/SVG/PDF，按需加载 Matplotlib 和本机中文字体。
缺中文字体明确报错。纯绘图函数仍返回 `(figure, axes)`，由调用者关闭。
图表消费正式数值：金额净值与回撤、IC/RankIC、成本换手、收益分布和指标比较；
`series` 明确选择现有数值列，可展示基准、年度/月度、滚动风险、分组、暴露归因、覆盖、暂估、ON/ID及小时序列。
`explanation_summary` 与 `explanation_waterfall` 读取已发布解释产物，不加载模型或 SHAP。
不从采样曲线重建正式指标，不补零、不连接未定义缺口；独立 episode 不拼接。
Crypto 的负权益保持原值。图表同时保留实际窗口和样本用途。

## 合同与验证

[研究合同索引](docs/specifications.md)按主题列出接口、字段与市场规则。

- [架构、API 与读写边界](docs/architecture.md)
- [来源与数据资格](docs/data.md)
- [A 股数学和经济合同](docs/a-share.md)
- [Crypto 事件、单位和执行合同](docs/crypto.md)

环境配置见 [environment.yml](environment.yml) 和 [pyproject.toml](pyproject.toml)，项目运行使用独立 Conda 环境。

```text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

测试覆盖公式、时间边界、持久化和下游读取；真实研究的样本范围、结果与局限分别记录在报告中。

## 许可证

代码和文档采用 [MIT License](LICENSE)。供应商原始数据不随项目分发，使用时需自行取得相应数据权限。
