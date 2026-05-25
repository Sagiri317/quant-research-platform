# 用途：集中维护本机私有 evidence API 可读取的 run roots。
# start_api_private_runs.ps1 和 start_private_workbench.ps1 会默认加载这里。
# 注意：这些路径指向 private reports，不应写入公开文档或提交生成数据。
$QrpPrivateRunRoots = @(
    "n7_private=reports/private_real/factor_evaluation"
) -join [System.IO.Path]::PathSeparator

$QrpPipelineOutputRoot = "reports/private_real/pipeline_runs"
$QrpFactorFactoryRoot = "reports/private_real/ff/20260525_n7_ff_smoke_v2/factor_factory"
