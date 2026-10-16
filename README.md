# QRP

QRP 用于本地 A 股低频和 Binance BTC/ETH 高频量化研究，不提供实盘交易、经纪服务或投资建议。

agent 决定研究问题、显式输入、操作顺序与重算范围；代码计算并检查科学资格。
每次操作接收实际输入和新输出路径，不找最新上游、不自动补跑、不自动判断旧结果是否仍适用。

## 直接操作

Python 是默认接口；CLI 严格读取同一 JSON，调用相同函数，不另设参数覆盖层。
下面每个命令只做命名的操作，输入应已经存在，正式输出拒绝覆盖。

```text
qrp prepare-source --config configs/prepare-source.json
qrp prepare --config configs/prepare-financial.json
qrp assemble --config <组装配置.json>
qrp fit --config configs/fit-a-share.json
qrp predict --config configs/predict-a-share.json
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
声明冻结最终评价时检查实际冻结事实，不以目录名或窗口名称证明资格。
裸 Parquet 需要本操作要求的列角色、单位、主键和可用时间依据；列名不证明 PIT。

[操作示例](configs/) 中的路径需替换为实际已准备输入。
[科研选择参考](docs/research-rosters/) 保留有序模型、参数候选、因子、资产与研究窗口；
它们供研究者选用，不是运行默认值、配置模板或自动编排入口。
复杂配置拒绝未知字段、重复 JSON key、非法类型、非有限值、重复成员和未消费选项。
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
不从采样曲线重建正式指标，不补零、不连接未定义缺口；独立 episode 不拼接。
Crypto 合法负权益保持原值。用途说明与实际窗口一起保存，窗口名称本身不证明样本外资格。

## 合同与验证

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
