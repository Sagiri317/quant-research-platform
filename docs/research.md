# A 股研究选择

研究主线为低频量价与高频微观结构，因子分析的范围与期限见[因子假设清单](factor-hypotheses.md)。训练、预测与回测分别配置，复用账户、费用、交易约束和风险模型。分钟研究暂缓，财务与分析师预期能力独立保留。

## 数据与目标

- `pv_all` 提供量价、路径、市场模型及多因子残差因子。实际输入列由操作配置选择，不再为同一数据域维护“基础版/扩展版”场景。
- 高频按观测事件与报价时钟计算，不能将报价间隔解释成固定分钟 K 线，也不能以快照参与率证明真实队列成交。
- 默认风险风格为规模、Beta、动量、残差波动和非流动性，加上行业暴露。财务缺失不阻塞这些计算。
- `risk_residual` 用信号时点已知的行业及五风格暴露，对显式`target_label_column`的未来收益做平方根市值加权回归（默认示例为20会话，也可指定1／5日、次日日内或高频收益）。它复用共同风险的风格预处理和回归实现；残差保留原收益单位，原始收益也保留供评价。残差标签须等参与回归的实际收益全部成熟后才可用于分析或训练。
- `industry_size_residual` 是等权行业与规模基线；`raw`、`date_zscore`、`percentile` 分别保留原值、按日标准化或排名。目标裁剪由 `target_clip` 显式指定。缺失风格或不可识别的回归不以零填补。

历史残差因子由 `prepare` 的 `risk_residual_factors` 操作产生，显式输入 `market_panel_1d`、`price_volume_facts`、`market_model_factors` 和 `trading_calendar`。每日收益用前一个自身市场交易会话已知的行业、五风格和总市值拟合，复用平方根市值加权回归；当日收盘才能得到残差。输出为：

- `risk_residual_reversal_5d`：最近 5 个会话残差之和的相反数。
- `risk_residual_momentum_20d`：最近 20 个会话残差之和。
- `risk_residual_volatility_60d`：最近 60 个会话残差的总体标准差，不年化。
- `risk_residual_drawdown_20d`／`risk_residual_drawdown_60d`：完整20／60会话加法残差路径距此前峰值的当前回撤，包含路径初始零；不是复利财富回撤。

窗口包含当日，须走满名义跨度；反转、动量、波动的有效残差至少达到`ceil(min_fresh_fraction × 窗口)`，默认比例0.75；当前回撤须有完整残差路径，不跨缺失压缩路径。停牌或收益缺失不产生零残差，生命周期变更重置历史。输出保留小数收益量纲，下一会话可用；残差之和不解释为可交易资产的复利收益。它们与使用未来收益的风险残差标签分别计算，样本外有效性仍待检验。

## 比较方式

先固定样本资格、训练起点、季度窗口、目标和成本假设，再比较新增因子的增量价值。风格中性化、残差标签与组合风险约束分别回答不同问题，不能相互替代。

可比较的候选包括风险调整后的动量/反转、隔夜与日内收益分解、成交量条件下的价格变化，以及订单流、撤单和盘口恢复。保留因子以样本外残差 RankIC、相关性、稳定性、换手和扣费后收益为依据，不能以模型分数提升推断稳定交易利润。

模型能力为 OLS、Ridge、LASSO、XGBoost、LSTM、Transformer，以及使用十档盘口序列的 DeepLOB 回归/分类。支持 GPU 的模型在 CUDA 不可用时使用 CPU。每项实验只选择回答问题所需的模型和参数，不展开“全部因子 × 全部模型 × 全部组合”的固定矩阵。

组合比较保留明确基线与需要的风险约束组合。高低频默认维护单一目标组合，不按预测期限自动建立重叠子账户。调仓间隔、实际持有期、分位范围、风险限额、基准和预测路径分别配置；未设置的限额不视为已经控制，指数增强需要真实历史权重。研究说明不作为运行配置或测试夹具。

## 因子子区间检验

股票截面复用 `research.factors.statistics.factor_statistics` 的 `evaluation_inputs` 与 `window`，读取已发布EVAL的每日指标，发布新的STAT产物。它不重新读取因子面板、计算因子或重新划分每日多空组，也不能从全区间平均IC反推子区间。窗口必须在原评估范围内；原信号时点、筛选方式、标签及成熟截止不变。因子或检验口径已经修正时，先使用修正后的EVAL。

[日表统计配置](../configs/statistics-stratified-a-share.json)复用原有 `statistics-factors --config` CLI及同实现Python入口。调用方明确 `evaluation_inputs`、新 `output` 路径及 `window`；历史近期检验使用2022-01-01至2026-10-30，其他子区间直接修改窗口。

`analyses=["stratified_ic"]` 复用已有全体、long、short的IC和RankIC，输出均值、有效日数、配对观测数及可定义时的HAC推断与BY校正；原评估已有额外分层时同时保留。缺失导致推断未定义时，仍保留独立可用的均值和样本数。多重检验范围是本次明确输入的原生评估，不冒称覆盖其他研究。

非零点时间戳（例如盘前09:00）的EVAL需要完整交易会话轴；调用方从原生交易日历生成 `sessions`，不从有指标的日期猜测交易日，也不填补缺失IC。日历属于本次显式输入，不另建日历解析器或统计入口。例如：

```python
import json
from pathlib import Path

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.research.factors.statistics import factor_statistics
from quant_research_platform.storage.datasets import read_dataset

config = json.loads(Path("configs/statistics-stratified-a-share.json").read_text())
config["evaluation_inputs"] = ["results/factor-evaluation"]
config["output"] = "results/factor-recent"
config["window"] = {"start": "2022-01-01", "end": "2026-10-30"}
calendar = read_dataset(
    "data/interim/csmar/trading_calendar",
    columns=["timestamp", "calendar_date", "source_market_type", "is_trading_day"],
    time_window=TimeWindow(**config["window"]),
)
config["sessions"] = unified_trading_sessions(calendar).strftime("%Y-%m-%d").tolist()
factor_statistics(config)
```

示例中的评估、日历和输出路径由调用方替换。需要 CLI 时将同一份运行配置写入 JSON，再调用 `qrp statistics-factors --config runtime.json`；`sessions` 与 `window` 都是原生统计接口的配置项，交易日列表无需提交到研究配置模板。交易会话选择由市场适配层负责，统计层只读取这些日期，不依赖 CSMAR。

ETF日频的全区间时序相关不能由一个总相关系数切出子区间，应复用现有因子与标签，在 `research.evaluations.runner.evaluate` 中明确设置同一个子区间及原成熟截止，重新求相关；无需重新计算因子。比较全历史与近期结果时分别保留原生产物和真实可用样本，近期观察用于检查衰减，不将事后选择窗口当作独立样本外证明。

## 独立操作与证据

研究过程使用独立 Python 操作，不使用 Notebook。数据准备、组装、训练、预测、评价、组合和统计分别指定输入、窗口和新输出。主线按时间滚动训练、验证选型和后续预测，不要求另设独立最终留出集；显式冻结的历史研究模板仍保留。季度窗口以前方已成熟数据训练；起点及窗口由调用者设定。因子预热历史不等于提前开始训练。

操作示例见 [configs](../configs/)，精确市场及经济语义见 [A 股约定](a-share.md)。历史报告属于当时版本的结果，代码更新不会自动重算研究，也不证明新因子的有效性。
