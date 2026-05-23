# 用途：集中维护本机私有 evidence API 可读取的 run roots。
# start_api_private_runs.ps1 和 start_private_workbench.ps1 会默认加载这里。
# 注意：这些路径指向 private reports，不应写入公开文档或提交生成数据。
$QrpPrivateRunRoots = @(
    "smoke_tensor_long_only=reports/private_real/factor_evaluation/smoke_2023_2025_tensor_long_only/core_runs",
    "smoke_tensor_long_short=reports/private_real/factor_evaluation/smoke_2023_2025_tensor_long_short/core_runs",
    "smoke_dataframe_long_only=reports/private_real/factor_evaluation/smoke_2023_2025_dataframe_long_only/core_runs",
    "smoke_dataframe_long_short=reports/private_real/factor_evaluation/smoke_2023_2025_dataframe_long_short/core_runs",
    "acceptance_dataframe_long_only=reports/private_real/factor_evaluation/acceptance_2024_dataframe_long_only/evaluation_runs",
    "acceptance_dataframe_long_short=reports/private_real/factor_evaluation/acceptance_2024_dataframe_long_short/evaluation_runs",
    "acceptance_tensor_long_only=reports/private_real/factor_evaluation/acceptance_2024_tensor_long_only/evaluation_runs",
    "acceptance_tensor_long_short=reports/private_real/factor_evaluation/acceptance_2024_tensor_long_short/evaluation_runs"
) -join [System.IO.Path]::PathSeparator

$QrpPipelineOutputRoot = "reports/private_real/pipeline_runs"
$QrpFactorFactoryRoot = "reports/private_real/ff/0523/factor_factory"
