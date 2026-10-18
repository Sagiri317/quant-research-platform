"""实验局部共同域配方；只生成掩码，训练 mask 不依赖评价窗口。"""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.models import _fit_windows
from quant_research_platform.contracts.time import prediction_window_mask
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.models.fitting import _legal_training_rows


def common_domain_chunks(
    frames,
    models,
    *,
    primary_key,
    declared_columns,
    availability_column,
    asset_id,
    fit_windows,
    fit_cutoff,
    evaluation_windows,
    evaluation_cutoff,
    coverage,
):
    """从共享输入逐批生成精确键与布尔掩码；跨批主键由原生 writer 核验。"""
    models = tuple(models)
    if not models or len({model.model_id for model in models}) != len(models):
        raise ValueError("common_domain_models_must_be_nonempty_unique")
    if len({model.target_column for model in models}) != 1:
        raise ValueError("common_domain_requires_same_target")
    if type(asset_id) is not str or not asset_id:
        raise ValueError("common_domain_requires_one_asset")
    cutoffs = strict_timestamp_series(
        pd.Series([fit_cutoff, evaluation_cutoff]), error="common_domain_cutoff_invalid"
    )
    domains = (
        ("common_train", _fit_windows(fit_windows, cutoffs.iloc[0]), cutoffs.iloc[0]),
        (
            "common_eval",
            _fit_windows(evaluation_windows, cutoffs.iloc[1]),
            cutoffs.iloc[1],
        ),
    )
    if cutoffs.iloc[0] >= pd.Timestamp(domains[1][1][0].start):
        raise ValueError("common_domain_training_evaluation_overlap")
    coverage.update(
        {
            name: {
                "common_rows": 0,
                "own_rows": dict.fromkeys((model.model_id for model in models), 0),
            }
            for name, _, _ in domains
        }
    )
    with closing_iterator(iter(frames)) as source:
        for frame in source:
            frame = frame.reset_index(drop=True)
            frame["timestamp"] = strict_timestamp_series(
                frame.timestamp, error="common_domain_timestamp_invalid"
            )
            result = frame.loc[:, list(primary_key)].copy()
            for name, windows, cutoff in domains:
                in_windows = pd.Series(False, index=frame.index)
                for window in windows:
                    in_windows |= prediction_window_mask(frame.timestamp, window)
                scoped = frame.loc[in_windows & frame.asset_id.eq(asset_id)]
                common = pd.Series(True, index=scoped.index)
                for model in models:
                    # 每个配方核自己的域，不能先被另一个配方的缺失筛掉。
                    usable, _ = _legal_training_rows(
                        scoped,
                        model,
                        fitting_cutoff=cutoff,
                        availability_column=availability_column,
                        declared_columns=declared_columns,
                    )
                    common &= scoped.index.isin(usable.index)
                    coverage[name]["own_rows"][model.model_id] += len(usable)
                result[name] = result.index.isin(common.index[common])
                coverage[name]["common_rows"] += int(common.sum())
            yield result
