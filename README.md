# quant-research-platform

quant-research-platform 是一个本地量化研究项目，覆盖 A 股日频选股与 Level-2 高频预测和库存回测。它把数据准备、因子构建、模型训练、样本外评价和交易核算连接起来，便于检查一项预测改善能否转化为费用后的收益。

也支持执行、库存与组合的 PPO 策略学习、公式因子搜索、冻结不确定性估计和成本感知配置。
独立研究 agent 支持连续对话，也可在固定协议与预算下调用这些操作。默认终端在宿主机以当前用户权限运行；无网络容器 Python 工具需要显式配置。
入口、示例和证据边界见[策略学习与研究 agent](docs/research-learning.md)。

[全球宏观流动性](docs/macro-liquidity.md)支持版本化源数据、明确盘前决策时钟、宏观组合与个股交互。
高频扩展包含逐笔与盘口因子、Hawkes、DeepLOB、因果信号滤波、AC 执行进度及被动队列假设，均通过现有原生操作接入。

## 研究报告

- [A 股日频量价研究](reports/A股量价研究报告.md)：6/25/52 因子、四块消融、九类模型与 PV25 组合回测。
- [A 股 ETF 高频研究](reports/A股高频研究报告.md)：五个历史窗口的预测、因子增量、执行与费用后账户比较，含 51 项因子说明。
- [量化研究 Agent](reports/量化研究Agent报告.md)：实验反馈驱动的研究流程、100 只真实 A 股的小样本演示、自由追问验收与证据限制。
- [美国流动性与 A 股研究](reports/美国流动性与A股研究.md)：截至 2026 年的数据条件、波动率与偏度因子、七个待检验假说；尚未运行正式实证。

报告记录研究问题、方法、结果或待检验假说及限制；技术规则见 `docs/`，操作示例见 `configs/`。来源证据随原始数据保存，供应商原始数据和本地运行目录不随仓库分发。

## 如何使用

Python 函数与 `qrp` 命令行使用同一套计算实现。研究配置明确输入、参数、时间窗口和输出位置；每次操作生成新产物，已有结果保持不变。

```text
qrp prepare-source --config configs/prepare-source.json
qrp prepare --config <数据准备配置.json>
qrp assemble --config <组装配置.json>
qrp fit --config configs/fit-a-share.json
qrp run-quarterly --config configs/quarterly-a-share.json
qrp backtest-hft --config configs/backtest-hft.json
qrp predict --config configs/predict-a-share.json
qrp explain --config configs/explain-model.json
qrp score --config <验证评分配置.json>
qrp select --config <候选选择配置.json>
qrp calibrate --config <A股校准配置.json>
qrp backtest --config <A股组合配置.json>
qrp evaluate --config <评估配置.json>
qrp statistics-model --config <模型统计配置.json>
qrp statistics-factors --config <因子统计配置.json>
qrp statistics-portfolio --config <A股组合统计配置.json>
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
`run-quarterly` 用 `training_start` 与 `rolling_window` 按季度扩展训练并执行预测、评价，示例为 2014 起训练、2018—2023 滚动。
若二者同日开始，首季度积累初始训练样本，从第二季度产生样本外预测；2014 年以前的原始数据只用于因子预热。
同一日期的样本可以先用于评价，待标签成熟后再进入后续训练；窗口用途随本次操作记录。
计算会检查拟合截止、标签成熟和校准时点。直接使用 Parquet 时，还需声明列角色、单位、主键和信息可用时间。

[操作示例](configs/) 中的路径需替换为实际已准备输入。
[分批因子组装](configs/assemble-factor-batches.json)指定各批输入和所需列，只拼接原始因子。
[已有特征复用](configs/assemble-existing-features.json)读取已经生成的特征，不重复处理；
保留 eligible 资格列与特征中的实际空值。该示例生成无标签特征输入，训练时需另选标签及其成熟时间列。

[隔夜与日内收益准备](configs/prepare-on-id.json)、[共同样本分层评价](configs/evaluate-stratified-a-share.json)与
[已保存日表的统计汇总](configs/statistics-stratified-a-share.json)分别调用 prepare、evaluate、statistics-factors。
若要同时报告各方案自身样本的结果，需另建输出，设置 sample_scope=own，并移除 comparison_inputs。
自身样本与共同样本两份评价应使用相同的焦点信号、输入和分层设置。
[冻结模型解释](configs/explain-model.json)的背景必须落在该模型实际拟合域内；
[组合基准比较](configs/statistics-benchmark-a-share.json)需替换真实指数代码、时钟、币种及价格/全收益口径，并通过 inputs.calendar 提供覆盖收益区间的完整交易日历。
组合统计默认只计算绩效，额外分析通过 analyses 选择。
[研究选择](docs/research.md)说明主线、比较方法与证据边界；运行参数在各项操作中明确指定。
实际运行仍需填写各操作自己的配置。接口会校验字段、类型和必要数值，拒绝重复的 JSON 键。
未选择的统计分析不会读取其专用输入；模型解释等严格接口也会拒绝用不到的选项。
候选顺序可能决定同分时的选择，输入列顺序则决定模型接收特征的排列，不能自行排序或去重。

监督预测模型包括 OLS、LASSO、XGBoost、LSTM、Transformer，以及 DeepLOB 收益回归和三分类；支持 GPU 的模型在 CUDA 不可用时使用 CPU，也可显式选择 CPU。
策略学习的 PPO、公式搜索和成本感知网络采用相同的设备回退约定。
高频数据接入、因子、标签、T+1 库存账户与使用限制见 [A 股高频](docs/a-share-hft.md)。

## 只读查询与图表

```text
qrp catalog factors --profile a_share_csmar_choice
qrp catalog labels --profile a_share_level2
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
想了解研究结论，可先读前面的研究报告；要复现实验或修改实现，再查下列详细文档。

- [架构、API 与读写边界](docs/architecture.md)
- [来源与数据资格](docs/data.md)
- [A 股计算方法与交易规则](docs/a-share.md)
- [策略学习、公式搜索、不确定性与研究 agent](docs/research-learning.md)

环境配置见 [environment.yml](environment.yml) 和 [pyproject.toml](pyproject.toml)，项目运行使用独立 Conda 环境。

```text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

测试覆盖公式、时间边界、持久化和下游读取；真实研究的样本范围、结果与局限分别记录在报告中。

## 许可证

代码和文档采用 [MIT License](LICENSE)。供应商原始数据不随项目分发，使用时需自行取得相应数据权限。
