"""Reader-facing unified research smoke report content."""

from __future__ import annotations


def build_unified_research_markdown(payload: dict[str, object]) -> str:
    status = payload["status"]
    assert isinstance(status, dict)
    return "\n".join(
        (
            "# QRP A股与 BTC/ETH 统一研究面试演示",
            "",
            "SMOKE_ONLY / SYNTHETIC_DATA / REAL_DATA_PENDING",
            "",
            "## 摘要",
            "",
            f"- run ID: `{payload['run_id']}`",
            f"- 状态: `{status['implementation']} / {status['data']} / "
            f"{status['holdout']} / {status['readiness']}`",
            "- 本材料证明仓库能力，不证明市场优势或投资收益。",
            "",
            "## 背景与方法",
            "",
            "共同 ExperimentPlan 使用 calendar/grid fold、purge、validation、"
            "development test；固定小候选集，没有自动因子发现或大范围调参。",
            "",
            "### 关键假设",
            "",
            "- A股财务公告仅按日期可知时，从下一交易日可用。",
            "- Crypto pooled 模型每行只使用本资产特征，并保留 symbol identity。",
            "- 交易成本、风险和 holdout 状态均来自显式合同。",
            "",
            "## 数据",
            "",
            "真实 CSMAR 财务导出尚未到达；展示使用确定性合成事实和可审查 fixture。",
            "",
            "## A股结果",
            "",
            "PIT、更正、effective-date benchmark/industry、因子形式风险、"
            "SLSQP long-only 组合与日期感知成本链已执行。",
            "真实 A股 model comparison、IC/RankIC 与 horizon decay："
            "`REAL_DATA_PENDING`，未用合成数值替代实证结果。",
            "",
            "## Crypto 结果",
            "",
            "BTC/ETH 同 run 完整性以及独立、pooled、leave-BTC-out、leave-ETH-out "
            "均通过 persisted model/prediction/evaluation artifacts 执行。",
            "执行诊断包含实际 taker/maker、markout、fee/spread additive sensitivity "
            "与 full-book latency sweep。",
            "",
            "## 稳健性与过拟合诊断",
            "",
            "包含 HAC、stationary bootstrap、BH/BY FDR、SPA/Reality Check、"
            "StepM、MCS、PSR/DSR、CSCV/PBO 与 Fama-MacBeth v0；"
            "合成样本不作显著性结论。",
            "",
            "## 局限",
            "",
            "真实财务 schema、行业历史和长期 BTC/ETH 共同录制覆盖仍等待真实数据验收。",
            "",
            "## 结论",
            "",
            "`IMPLEMENTATION_COMPLETE / REAL_DATA_PENDING / HOLDOUT_LOCKED / "
            "READY_FOR_REAL_DATA`。最终 holdout 未读取。",
            "",
        )
    )
