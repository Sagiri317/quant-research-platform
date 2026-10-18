# 信息配方的共同训练／共同评价对照

主实验让每个配方使用声明训练窗口内的全部合法样本。
额外对照只改变样本域：同一币、同一目标期限、同一折、预先选定的有限配方集合，
分别取合法训练键和合法评价键的交集。BTC、ETH 分开运行；不要把多个币的时间和资产分别
做 `in` 筛选，因为那会产生并不存在于共同域的键组合。

[common_domain.py](common_domain.py) 是可直接载入的实验配方，不是新公共 API 或 CLI。
它按共享输入逐批输出精确主键和两个布尔列，通过现有 `write_dataset` 保存 mask 表，
再由 `assemble` 合回原输入。配置只使用 `row_filters=[["common_train", "==", true]]`
或 `common_eval`，无需向 JSON 写入数百万个时间戳。

## 样本规则

- 合法训练行复用 `fitting._legal_training_rows`，与正式拟合共用 owner：
  eligible、所选过滤条件、标签已成熟且非空、至少一个真实可用的原始输入。
  部分特征缺失仍可插补或使用模型原生 missing；不能对特征并集做 `dropna`。
  技术填补的零和 missingness flag 本身不算原始观测；声明的 observed flag 必须有效。
- 每个配方先在自己的完整窗口中验证，再按真实 PK 取交集。不能先删掉另一配方的缺失行，
  以此跳过本配方的无效数据检查。需要训练中位数插补的模型若有选中输入列在共同训练域全空，
  仍由真实 fit 明确失败；原生支持 NaN 的模型保留该列及缺失值。
  不偷偷删列、补标签、缩配方集合或改为其他币。
- 训练 mask 只看训练窗、拟合截止与各配方输入；评价数据不参与训练 mask。
  评价 mask 使用单独的评价窗与成熟截止。两者均只依据缺失和可用时点，
  不按标签正负、大小、预测表现或费用后收益筛行。离散窗口之间的空档不填平。
- 每折另存 mask 表。之后折的截止前已成熟的历史行可正常进入后来训练窗；
  数据没有永久的 train/test 身份。只给 train/dev 操作传实际指定的数据，
  不把独立最终评价数据传给调参或选择。
- mask 产物保存每个配方的 own_rows 和共同域 common_rows。空共同域如实保留零覆盖，
  不自动退回 own 或删掉困难配方。mask 必须通过原生 writer 发布；主键空值和
  包括跨批次在内的主键唯一性由 writer 统一检查，配方不去重或丢弃无效键。

## 显式计算示例

下面代码只展示步骤，路径、模型、窗口由本次研究明确给出。
先完成原有 `assemble`，得到包含本次所需信息列的共享数据集；稀疏信息保留缺失。
例子只计算 mask，不训练或运行研究。

```python
from pathlib import Path
from runpy import run_path

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.research.models.fitting import model_training_read_columns
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options, iter_dataset_chunks, open_dataset_session,
)

common_domain_chunks = run_path(
    "docs/research-rosters/crypto/common_domain.py"
)["common_domain_chunks"]
source_selection = {"path": ".run/research/shared_inputs"}
models = (
    ModelSpec(model_id="ofi", model_algorithm="linear_regression",
              input_factor_columns=("ofi_100ms",),
              target_column="future_mid_return_500ms"),
    ModelSpec(model_id="microstructure", model_algorithm="ridge_regression",
              input_factor_columns=("ofi_100ms", "relative_spread", "imbalance_1"),
              target_column="future_mid_return_500ms"),
)
plan = dict(
    asset_id="btc_usdt",
    fit_windows=[{"start": "2024-01-01T00:00:00", "end": "2024-01-02T23:59:59.900"}],
    fit_cutoff="2024-01-02T23:59:59.900",
    evaluation_windows=[{"start": "2024-01-03T00:00:00", "end": "2024-01-03T23:59:59.900"}],
    evaluation_cutoff="2024-01-04T00:00:05",
)
options = _dataset_input_options(source_selection)
options.setdefault("columns", ("timestamp", "asset_id"))
session = open_dataset_session(**options)
declared = session.spec
keys = tuple(declared["primary_key"])
target = models[0].target_column
available = declared["label_availability_columns"][target]
columns = tuple(dict.fromkeys(
    column for model in models
    for column in model_training_read_columns(
        keys, model.input_columns, label_column=target,
        sample_filter_mode=model.sample_filter_mode,
        availability_column=available, declared_columns=declared["column_roles"],
    )
))
coverage = {}
mask_spec = DataSpec(
    data_id="btc_fold_1_common_masks", frequency=declared["frequency"],
    grain=declared["grain"], primary_key=keys,
    partition_columns=tuple(declared["partition_columns"]),
    column_roles={**{key: declared["column_roles"][key] for key in keys},
                  "common_train": "mask", "common_eval": "mask"},
)
mask_path = Path(".run/research/btc_fold_1_common_masks")
write_dataset(
    mask_path,
    common_domain_chunks(
        iter_dataset_chunks(session.projected_view(columns)), models,
        primary_key=keys, declared_columns=declared["column_roles"],
        availability_column=available, coverage=coverage, **plan,
    ),
    spec=mask_spec,
    metadata={
        "inputs": {"dataset": source_selection},
        "parameters": {**plan, "models": [model_spec_payload(m) for m in models]},
        "producer_diagnostics": coverage,
    },
)
```

用原有 `assemble` 把 `source_selection` 与 mask_path 作为两个输入，基表仍为原输入。
保留原有 factor/feature/label/context 选择，在 `filter_columns` 中加入
`common_train`、`common_eval`；两个输入使用相同 PK 和原窗口，不重做已有特征变换。
原表标签成熟映射、单位和标签语义继续由 assemble 保留。下面的 inputs 对象可直接用于
现有 `fit`、`predict`、`score.labels` 和 `evaluate.inputs`：

```python
train_inputs = {"path": ".run/research/btc_fold_1_with_masks",
                "row_filters": [["common_train", "==", True]]}
eval_inputs = {"path": ".run/research/btc_fold_1_with_masks",
               "row_filters": [["common_eval", "==", True]]}
```

主实验的 fit 继续使用原 inputs。对照 fit 使用 train_inputs 与相同模型、窗口、截止、
权重和 seed。分别保存到新位置。common_eval 是事后可评价域，不是实盘预测域；
对照可保存相同 eval_inputs 的 PRED，也可直接复用覆盖更广的已保存预测。
EVAL 以 eval_inputs 的实际键对齐预测并披露缺失，预测中的域外键不参与本次评价。
执行/经济回测继续使用原 own-domain PRED，不能因未来标签缺失而删掉决策时点。

同币、同 horizon 的完整候选列表交给现有 `score`，其本来就在合法验证 PK 交集计算指标。
默认选择规则为 `selection_metric="validation_mse"`、`selection_direction="minimize"`，
`tie_break_order` 使用预声明顺序；须确保预测与目标单位一致。IC、覆盖率为辅助；
费用后收益、换手与库存等经济指标另报，不混进 MSE。
own-domain 主实验和 common-train/common-eval 对照各自保留 N 与结果，不改写历史选择。

合成端到端示例与边界反例见
[测试](../../../tests/research/test_crypto_common_domain_recipe.py)：
原生 mask 写入、assemble、own/common fit、PRED、score、EVAL，以及部分缺失、
仅有缺失标记、无真实输入、无效观测标记、标签未成熟、跨币、离散折窗、重复 PK。
这些测试不代表真实数据研究已经运行。
