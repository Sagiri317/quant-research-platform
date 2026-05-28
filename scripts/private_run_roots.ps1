# 用途：集中维护本机私有 evidence API 可读取的 run roots。
# start_private_workbench.ps1 会默认加载这里。
# 注意：这些路径指向 private reports，不应写入公开文档或提交生成数据。
# pipeline runs 不需要写进 $QrpPrivateRunRoots；API 会使用
# $QrpPipelineOutputRoot 自动挂载 source_alias=pipeline，避免生成后还要手动刷新配置。
$QrpPrivateRunRoots = @(
    "n7_private=reports/private_real/factor_evaluation"
) -join [System.IO.Path]::PathSeparator

$QrpPipelineOutputRoot = "reports/private_real/pipeline_runs"
$QrpFactorFactoryRoot = "reports/private_real/ff/20260527_rev120_factory_smoke_v1/factor_factory"
