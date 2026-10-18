# QRP

QRP 用于本地 A 股低频和 Binance BTC/ETH 高频量化研究，不提供实盘交易、经纪服务或投资建议。

agent 决定研究问题、显式输入、操作顺序与重算范围；代码计算并检查科学资格。
每次操作接收实际输入和新输出路径；调用者负责输入选择及其适用范围。

## 直接操作

Python 是默认接口；CLI 严格读取同一 JSON，调用相同函数，不另设参数覆盖层。
下面每个命令只做命名的操作，输入应已经存在，正式输出拒绝覆盖。

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
窗口用途由调用者声明；代码核实际拟合截止、标签成熟与必要校准时点，研究者负责选择纪律。
裸 Parquet 需要本操作要求的列角色、单位、主键和可用时间依据；列名不证明 PIT。

[操作示例](configs/) 中的路径需替换为实际已准备输入。
[同族多批因子装配](configs/assemble-factor-batches.json)明确实例及列投影；
[已有特征复用](configs/assemble-existing-features.json)只选择物化列，不重复处理，
已声明的观测/缺失伴随列自动保留。前者仅拼接原始因子，后者保留 eligible，
用于无标签特征输入；训练时另选实际标签及其成熟列。
[ON/ID准备](configs/prepare-on-id.json)、[共同域分层评价](configs/evaluate-stratified-a-share.json)与
[持久化日表归约](configs/statistics-stratified-a-share.json)分别调用现有 prepare、evaluate、statistics-factors；
own 评价需另选新输出、sample_scope=own 并移除 comparison_inputs。两份评价的焦点信号、输入和分层规格应保持一致。
[冻结模型解释](configs/explain-model.json)的背景必须落在该模型实际拟合域内；
[组合基准比较](configs/statistics-benchmark-a-share.json)需替换真实指数代码、时钟、币种及价格/全收益口径，不能只改显示名称。
组合统计默认只计算绩效，额外分析通过 analyses 显式选择。
[科研选择参考](docs/research-rosters/) 保留有序模型、参数候选、因子、资产与研究窗口；
它们供研究者选用，不是运行默认值、配置模板或自动编排入口。
复杂配置按各操作合同校验字段、类型和必要数值，拒绝重复 JSON key。
未选择的统计分析不读取其专用输入；解释等严格接口拒绝未消费选项。
候选及输入列顺序具有含义，不能自行排序去重。

## 只读查询与图表

```text
qrp catalog factors --profile a_share_csmar_choice
qrp catalog labels --profile crypto_binance_hft
qrp catalog dataset pv_all
```

现有 HTTP `create_app(project_root=...)` 只提供 catalog、数据集要求、
`/api/inspect?path=...` 与 `/api/portfolio-curve?path=...`。
路径限定于服务绑定的项目根，不能切换项目、访问 raw 或触发计算。
查询只读取指定产物，数值不以完整研究状态为前提。

`reports.plots.plot(config)` 原子写入新的 PNG/SVG/PDF，按需加载 Matplotlib 和本机中文字体。
缺中文字体明确报错。纯绘图函数仍返回 `(figure, axes)`，由调用者关闭。
图表消费正式数值：金额净值与回撤、IC/RankIC、成本换手、收益分布和指标比较；
`series` 明确选择现有数值列，可展示基准、年度/月度、滚动风险、分组、暴露归因、覆盖、暂估、ON/ID及小时序列。
`explanation_summary` 与 `explanation_waterfall` 读取已发布解释产物，不加载模型或 SHAP。
不从采样曲线重建正式指标，不补零、不连接未定义缺口；独立 episode 不拼接。
Crypto 合法负权益保持原值。用途说明与实际窗口一起保存，窗口名称本身不证明样本外资格。

## 合同与验证

[研究合同索引](docs/specifications.md)按主题列出接口、字段与市场规则。

- [架构、API 与读写边界](docs/architecture.md)
- [来源与数据资格](docs/data.md)
- [A 股数学和经济合同](docs/a-share.md)
- [Crypto 事件、单位和执行合同](docs/crypto.md)

环境见 environment.yml 和 pyproject.toml；使用明确项目解释器，勿用 Conda base。

```text
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

先定向验证再全量验证；通过记录只属于实际受测代码、环境和输入。
代码验证通过不等于真实研究完成。私有行、原始载荷、秘密、本机路径及原始回溯不提交。
