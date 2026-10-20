# quant-research-platform

quant-research-platform 是一个本地量化研究项目，覆盖 A 股日频选股与 Binance BTC/ETH 永续合约的高频预测和回测。它把数据准备、因子构建、模型训练、样本外评价和交易核算连接起来，便于检查一项预测改善能否转化为费用后的收益。

## 研究报告

- [A 股量价研究报告](reports/A股量价研究报告.md)：比较 Core6、PV25、PV52 的信息增量与四类经济块消融，解释样本筛选、缺失处理、中性化、停牌涨跌停和交易成本。
- [加密资产高频研究报告](reports/加密资产高频研究报告.md)：比较多尺度、跨币和跨场所信息的预测价值，并展示已完成的延迟实验与首批交易结果。

两篇报告的图表与汇总数据一同保存在 `reports/`。市场规则与计算方法见 `docs/`，小型操作示例见 `configs/`。供应商原始数据和本地运行目录不随仓库分发。

## 如何使用

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

各阶段需要分别配置。评分操作（score）通过 validation_windows 指定完整验证窗口；
选择操作（select）通过 refit_windows 与 refit_cutoff 记录最终重新拟合的窗口和截止时刻。
最终拟合只使用这些窗口中、截至拟合时已成熟的样本；可以加入符合条件的新日期，但不会填平窗口间的空档。

保存的模型可以加载后多次预测。预测不会重新训练，评分不会重新训练，选出模型也不会自动执行最终拟合。
滚动向前研究（walk-forward）需要逐轮明确拟合、决策、预测和评价窗口。
同一日期的样本可以先用于评价，待标签成熟后再进入后续训练；窗口用途随本次操作记录。
计算会检查拟合截止、标签成熟和校准时点。直接使用 Parquet 时，还需声明列角色、单位、主键和信息可用时间。

[操作示例](configs/) 中的路径需替换为实际已准备输入。
[分批因子组装](configs/assemble-factor-batches.json)指定各批输入和所需列，只拼接原始因子。
[已有特征复用](configs/assemble-existing-features.json)读取已经生成的特征，不重复处理；
保留 eligible 资格列，并自动带上已声明的观测/缺失标志。该示例生成无标签特征输入，训练时需另选标签及其成熟时间列。

[隔夜与日内收益准备](configs/prepare-on-id.json)、[共同样本分层评价](configs/evaluate-stratified-a-share.json)与
[已保存日表的统计汇总](configs/statistics-stratified-a-share.json)分别调用 prepare、evaluate、statistics-factors。
若要同时报告各方案自身样本的结果，需另建输出，设置 sample_scope=own，并移除 comparison_inputs。
自身样本与共同样本两份评价应使用相同的焦点信号、输入和分层设置。
[冻结模型解释](configs/explain-model.json)的背景必须落在该模型实际拟合域内；
[组合基准比较](configs/statistics-benchmark-a-share.json)需替换真实指数代码、时钟、币种及价格/全收益口径，并通过 inputs.calendar 提供覆盖收益区间的完整交易日历。
组合统计默认只计算绩效，额外分析通过 analyses 选择。
[研究配置参考](docs/research-rosters/)列出有顺序的模型、候选参数、因子、资产和研究窗口，供设计实验时选用。
实际运行仍需填写各操作自己的配置。接口会校验字段、类型和必要数值，拒绝重复的 JSON 键。
未选择的统计分析不会读取其专用输入；模型解释等严格接口也会拒绝用不到的选项。
候选顺序可能决定同分时的选择，输入列顺序则决定模型接收特征的排列，不能自行排序或去重。

## 只读查询与图表

```text
qrp catalog factors --profile a_share_csmar_choice
qrp catalog labels --profile crypto_binance_hft
qrp catalog dataset pv_all
```

HTTP `create_app(project_root=...)` 提供 catalog、数据集要求、
`/api/inspect?path=...` 与 `/api/portfolio-curve?path=...`。
查询限定在绑定的项目根内，只读取已有结果；原始数据和计算任务由本地接口管理。

`reports.plots.plot(config)` 原子写入新的 PNG/SVG/PDF，按需加载 Matplotlib 和本机中文字体。
缺中文字体明确报错。纯绘图函数仍返回 `(figure, axes)`，由调用者关闭。
图表读取已经计算并保存的正式数值：金额净值与回撤、IC/RankIC、成本换手、收益分布和指标比较；
`series` 明确选择现有数值列，可展示基准、年度/月度、滚动风险、分组、暴露归因、覆盖、暂估、ON/ID及小时序列。
`explanation_summary` 与 `explanation_waterfall` 读取已发布解释产物，不加载模型或 SHAP。
不会从采样曲线重算正式指标。缺失值不补零，未知区间不连线，独立分段的账户结果不拼接。
加密资产账户的负权益保持原值。图表同时标明实际窗口和样本用途。

## 常用术语

| 术语 | 本项目中的含义 |
|---|---|
| PIT（历史时点可得信息） | 在历史决策时，只使用那时已经公开、能够证明版本的信息。 |
| 标签成熟 | 预测目标依赖的未来价格或其他事实已经可用，标签此时才能用于训练或评分。 |
| OOS（样本外） | 预测发生在该次模型拟合范围之后；是否还受到选型影响，需结合实际选择过程说明。 |
| IC / RankIC | 预测与目标的相关系数 / 排名相关系数，用来衡量预测关联和排序能力。 |
| Taker / Maker | 主动吃单 / 挂单等待成交；两者使用不同的费用与执行假设。 |
| 数据产物 | 操作保存的数据表、模型、预测、账本或统计结果，每次写到新的输出位置。 |
| 元数据与清单 | 元数据说明输入、参数、单位和时间范围；清单还记录产物成员及必要的物理信息。 |
| 约定 | 接口、数据和计算必须遵守的规则，包括时间、单位、缺失和失败处理。 |

## 文档与验证

[研究方法与接口索引](docs/specifications.md)按主题列出接口、字段与市场规则。
想了解研究结论，可先读前面的两篇报告；要复现实验或修改实现，再查下列详细文档。

- [架构、API 与读写边界](docs/architecture.md)
- [来源与数据资格](docs/data.md)
- [A 股计算方法与交易规则](docs/a-share.md)
- [加密资产事件、单位与执行规则](docs/crypto.md)

环境配置见 [environment.yml](environment.yml) 和 [pyproject.toml](pyproject.toml)，项目运行使用独立 Conda 环境。

```text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

测试覆盖公式、时间边界、持久化和下游读取；真实研究的样本范围、结果与局限分别记录在报告中。

## 许可证

代码和文档采用 [MIT License](LICENSE)。供应商原始数据不随项目分发，使用时需自行取得相应数据权限。
